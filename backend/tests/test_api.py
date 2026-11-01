from __future__ import annotations

import io
import json
import os
import sqlite3
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


# --- sample media helpers --------------------------------------------------------------

def make_png(width: int = 320, height: int = 240) -> bytes:
    rng = np.random.default_rng(1)
    array = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def make_mp4(path: Path, frames: int = 48, size: tuple[int, int] = (160, 120)) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, size)
    assert writer.isOpened(), "OpenCV could not open an MP4 writer in this environment"
    rng = np.random.default_rng(2)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def make_wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes((np.sin(np.linspace(0, 440 * 2 * np.pi, 16000)) * 12000).astype("<i2").tobytes())
    return buffer.getvalue()


def post(client, kind: str, name: str, content: bytes, mime: str):
    return client.post(f"/api/analyze/{kind}", files={"file": (name, content, mime)})


# --- health / hosting ------------------------------------------------------------------

def test_health_reports_real_configuration(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["analyzers"]["audio"]["status"] == "not_configured"
    assert body["analyzers"]["image"]["repository"] == "sakshamkr1/deitfake-v2"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


def test_frontend_files_are_served_but_project_folder_is_not(client):
    assert client.get("/").status_code == 200
    assert "DeepTruth" in client.get("/").text
    assert client.get("/style.css").status_code == 200
    assert client.get("/script.js").status_code == 200
    for hidden in ("/main.py", "/storage.py", "/data/history.sqlite3", "/requirements.txt", "/../main.py"):
        assert client.get(hidden).status_code == 404, hidden


# --- image -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def image_result(client):
    response = post(client, "image", "sample.png", make_png(), "image/png")
    assert response.status_code == 200, response.text
    return response.json()


def test_image_analysis_returns_real_model_output(image_result):
    assert image_result["analysis_status"] == "completed"
    assert 0.0 <= image_result["model_score"] <= 1.0
    evidence = {item["type"]: item for item in image_result["evidence"]}
    classification = evidence["image_classification"]
    assert classification["fake_class_score"] + classification["real_class_score"] == pytest.approx(1.0, abs=1e-4)
    assert image_result["model_score"] == pytest.approx(classification["fake_class_score"])
    assert image_result["interpretation"]["band"] in {"low", "inconclusive", "review"}
    assert "not calibrated" in image_result["interpretation"]["note"]
    assert len(image_result["upload_information"]["sha256"]) == 64


def test_image_scope_check_warns_when_no_face_found(image_result):
    face = next(item for item in image_result["evidence"] if item["type"] == "face_detection")
    assert face["status"] == "ok" and face["faces_detected"] == 0
    assert any("No frontal face" in warning for warning in image_result["warnings"])
    assert any("centre-cropped" in warning for warning in image_result["warnings"])  # 320x240 is non-square


def test_gradcam_runs_on_the_deit_architecture(image_result):
    explain = next(item for item in image_result["evidence"] if item["type"] == "explainability")
    assert explain["status"] == "generated", explain
    assert explain["patch_grid"] == "14x14"
    assert image_result["visuals"]["heatmap"]["image"].startswith("data:image/jpeg;base64,")
    assert image_result["visuals"]["model_input"]["image"].startswith("data:image/jpeg;base64,")


def test_gradcam_declines_unsupported_architecture(monkeypatch):
    import image_analyzer
    monkeypatch.setattr(image_analyzer, "_gradcam_layer", lambda: None)
    result = image_analyzer.analyze_image(Image.open(io.BytesIO(make_png())))
    explain = next(item for item in result["evidence"] if item["type"] == "explainability")
    assert explain["status"] == "unavailable" and "not a DeiT/ViT" in explain["reason"]
    assert "heatmap" not in result["visuals"]
    assert result["model_score"] is not None  # prediction itself is unaffected


def test_exif_rotation_is_applied(tmp_path):
    import image_analyzer
    picture = Image.new("RGB", (300, 100), "white")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees
    path = tmp_path / "rotated.jpg"
    picture.save(path, exif=exif)
    result = image_analyzer.analyze_image(path, explain=False)
    assert result["analysis_status"] == "completed"
    # After rotation the image is 100x300 (still non-square) so the crop warning must be present.
    assert any("centre-cropped" in warning for warning in result["warnings"])


def test_score_bands_are_descriptive_not_verdicts():
    from image_analyzer import describe_score
    assert describe_score(0.99)["band"] == "review"
    assert describe_score(0.01)["band"] == "low"
    assert describe_score(0.5)["band"] == "inconclusive"
    assert describe_score(None)["band"] == "none"
    for score in (0.01, 0.5, 0.99):
        assert "not" in describe_score(score)["summary"].lower()


# --- validation ------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kind,name,content,mime,expected",
    [
        ("image", "notes.txt", b"hello", "text/plain", 415),            # bad extension
        ("image", "fake.png", b"this is not a png", "image/png", 415),  # bad content
        ("image", "empty.png", b"", "image/png", 400),                  # empty
        ("image", "sample.png", make_png(), "application/pdf", 415),    # bad MIME
        ("video", "fake.mp4", b"not a video at all, just text...", "video/mp4", 415),
        ("audio", "fake.wav", b"RIFFxxxxNOTWAVE-data-here", "audio/wav", 415),
        ("audio", "program.exe", b"MZ", "audio/wav", 415),
    ],
)
def test_invalid_uploads_are_rejected_without_a_prediction(client, kind, name, content, mime, expected):
    response = post(client, kind, name, content, mime)
    assert response.status_code == expected
    body = response.json()
    assert body["model_score"] is None
    assert body["analysis_status"] in {"rejected", "invalid_media"}
    assert any("not evidence" in item for item in body["limitations"])


def test_oversized_upload_is_rejected(client, monkeypatch):
    import main
    monkeypatch.setitem(main.MAX_BYTES, "image", 1024)
    response = post(client, "image", "big.png", make_png(400, 400), "image/png")
    assert response.status_code == 413
    assert response.json()["analysis_status"] == "rejected"


def test_filename_path_components_are_stripped(client):
    response = post(client, "image", "../../etc/evil.png", make_png(), "image/png")
    assert response.status_code == 200
    assert response.json()["filename"] == "evil.png"


def test_temp_files_are_cleaned_up(client):
    import tempfile
    before = set(Path(tempfile.gettempdir()).glob("deeptruth_*"))
    post(client, "image", "a.png", make_png(), "image/png")
    post(client, "image", "bad.png", b"junk", "image/png")
    after = set(Path(tempfile.gettempdir()).glob("deeptruth_*"))
    leftovers = {p for p in after - before if p.is_file()}
    assert not leftovers


# --- video -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def video_result(client, tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    response = post(client, "video", "clip.mp4", make_mp4(path), "video/mp4")
    assert response.status_code == 200, response.text
    return response.json()


def test_video_uses_real_extracted_frames(video_result):
    assert video_result["analysis_status"] == "completed"
    frames = video_result["evidence"][0]["sampled_frames"]
    assert len(frames) == 8
    assert [f["frame_index"] for f in frames] == sorted(f["frame_index"] for f in frames)
    for frame in frames:
        assert 0.0 <= frame["model_score"] <= 1.0
        assert str(frame["frame_index"]) in video_result["visuals"]["frame_thumbnails"]
    assert video_result["model_information"]["video_metadata"]["frame_count"] > 0


def test_video_does_not_claim_temporal_detection(video_result):
    text = " ".join(video_result["limitations"]).lower()
    assert "not temporal deepfake detection" in text
    assert "audio is not analyzed" in text
    assert video_result["evidence"][0]["aggregate_basis"] == "all_sampled_frames"  # no faces in noise frames
    assert any("No frontal face" in w for w in video_result["warnings"])


def test_video_review_candidates_only_from_real_scores(video_result):
    threshold = video_result["evidence"][0]["review_threshold"]
    scores = {f["frame_index"]: f["model_score"] for f in video_result["evidence"][0]["sampled_frames"]}
    for flagged in video_result["suspicious_timestamps"]:
        assert flagged["model_score"] >= threshold
        assert scores[flagged["frame_index"]] == flagged["model_score"]


# --- audio -----------------------------------------------------------------------------

def test_audio_reports_that_no_detector_is_configured(client):
    before = client.get("/api/stats").json()["total_investigations"]
    response = post(client, "audio", "voice.wav", make_wav(), "audio/wav")
    assert response.status_code == 200
    body = response.json()
    assert body["analysis_status"] == "not_configured"
    assert body["model_score"] is None
    assert body["evidence"] == [] and body["suspicious_timestamps"] == []
    assert any("No score" in item for item in body["limitations"])
    assert client.get("/api/stats").json()["total_investigations"] == before  # not counted as an investigation
    pdf = client.get(body["report_url"])  # a no-result report is still downloadable
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")


# --- reports ---------------------------------------------------------------------------

def test_pdf_report_downloads_and_contains_expected_sections(client, image_result):
    response = client.get(image_result["report_url"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF") and len(response.content) > 8000
    assert "attachment" in response.headers["content-disposition"]


def test_pdf_text_content(client, image_result):
    fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed; text extraction skipped")
    document = fitz.open(stream=client.get(image_result["report_url"]).content, filetype="pdf")
    text = " ".join(page.get_text() for page in document)
    for expected in ("Investigation report", image_result["investigation_id"], "Limitations and uncertainty", "Final interpretation", "Ethical use"):
        assert expected in text


def test_pdf_survives_hostile_and_non_latin_filenames(client):
    markup = post(client, "image", "<b>&amp;</b>.png", make_png(), "image/png").json()
    assert client.get(markup["report_url"]).status_code == 200
    gurmukhi = post(client, "image", "ਤਸਵੀਰ.png", make_png(), "image/png").json()
    assert client.get(gurmukhi["report_url"]).status_code == 200


@pytest.mark.parametrize("bad_id", ["0" * 32, "../../main", "not-hex", "A" * 32])
def test_unknown_or_malformed_report_ids_return_404(client, bad_id):
    assert client.get(f"/api/reports/{bad_id}.pdf").status_code == 404


# --- history ---------------------------------------------------------------------------

def test_history_lists_completed_investigations(client, image_result, video_result):
    items = client.get("/api/history").json()["items"]
    ids = {item["investigation_id"] for item in items}
    assert image_result["investigation_id"] in ids and video_result["investigation_id"] in ids
    item = next(i for i in items if i["investigation_id"] == image_result["investigation_id"])
    assert item["media_type"] == "image" and item["analysis_status"] == "completed"
    assert item["model_score"] == pytest.approx(image_result["model_score"])
    assert item["report_url"].endswith(".pdf")


def test_stats_are_counted_from_stored_investigations(client, image_result, video_result):
    stats = client.get("/api/stats").json()
    items = client.get("/api/history").json()["items"]
    assert stats["total_investigations"] == len(items)
    assert stats["images_analyzed"] == sum(i["media_type"] == "image" for i in items)
    assert stats["videos_analyzed"] == sum(i["media_type"] == "video" for i in items)
    assert stats["flagged_for_review"] == sum(i["flagged"] for i in items)
    assert sum(stats["band_counts"].values()) == stats["total_investigations"]


def test_open_stored_investigation(client, image_result):
    body = client.get(f"/api/history/{image_result['investigation_id']}").json()
    assert body["investigation_id"] == image_result["investigation_id"]
    assert body["model_score"] == pytest.approx(image_result["model_score"])
    assert body["visuals_retained"] is True  # still in the short-lived memory cache


def test_database_never_stores_image_pixels(image_result, video_result):
    import storage
    with sqlite3.connect(storage.DB_PATH) as db:
        blobs = [row[0] for row in db.execute("SELECT result_json FROM investigations")]
    assert blobs
    for blob in blobs:
        assert "data:image" not in blob and "base64" not in blob
        assert "visuals" not in json.loads(blob)


def test_history_and_pdf_survive_a_backend_restart(image_result):
    """Run a brand-new Python process against the same database file."""
    code = (
        "import json, storage, report_generator;"
        f"rid = '{image_result['investigation_id']}';"
        "items = storage.list_investigations();"
        "report = storage.get_investigation(rid);"
        "pdf = report_generator.generate_pdf_report(report);"
        "print(json.dumps({'found': any(i['investigation_id'] == rid for i in items), 'pdf': pdf[:4].decode()}))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == {"found": True, "pdf": "%PDF"}


def test_delete_one_and_clear_all(client):
    created = post(client, "image", "temp.png", make_png(), "image/png").json()
    rid = created["investigation_id"]
    assert client.delete(f"/api/history/{rid}").status_code == 200
    assert client.get(f"/api/history/{rid}").status_code == 404
    assert client.delete(f"/api/history/{rid}").status_code == 404
    assert client.delete("/api/history/not-an-id").status_code == 404
    assert client.delete("/api/history").json()["deleted_count"] >= 0
    assert client.get("/api/history").json()["items"] == []
    assert client.get("/api/stats").json()["total_investigations"] == 0


# --- model failure ---------------------------------------------------------------------

def test_model_failure_yields_no_prediction_and_is_not_called_fake(client, monkeypatch):
    import image_analyzer
    monkeypatch.setattr(image_analyzer, "_MODEL", None)
    monkeypatch.setattr(image_analyzer, "_PROCESSOR", None)
    monkeypatch.setattr(image_analyzer, "MODEL_ID", "this/model-does-not-exist-offline")
    response = post(client, "image", "sample.png", make_png(), "image/png")
    assert response.status_code == 503
    body = response.json()
    assert body["analysis_status"] == "model_unavailable"
    assert body["model_score"] is None and body["evidence"] == []
    assert any("not evidence that the media is fake" in item for item in body["limitations"])
    assert client.get("/api/stats").json()["total_investigations"] == 0  # failures are not counted


# --- Grad-CAM math ---------------------------------------------------------------------

@pytest.mark.parametrize("special_tokens", [1, 2])
def test_gradcam_peak_lands_on_the_planted_patch(special_tokens):
    """A strong positive activation x gradient at patch (row 3, col 9) must be the CAM maximum,
    regardless of whether the architecture has 1 (ViT) or 2 (DeiT) leading special tokens."""
    import torch
    from image_analyzer import _cam_from_tensors
    grid, channels = 14, 8
    activations = torch.zeros(1, special_tokens + grid * grid, channels)
    gradients = torch.zeros_like(activations)
    gradients[:, special_tokens:, :] = 0.1                      # uniform weak positive gradient
    planted = special_tokens + 3 * grid + 9
    activations[0, planted, :] = 5.0                            # the patch that "matters"
    activations[0, 0, :] = 100.0                                # CLS token must be ignored
    cam = _cam_from_tensors(activations, gradients, grid)
    assert cam.shape == (grid, grid)
    assert np.unravel_index(cam.argmax(), cam.shape) == (3, 9)
    assert cam.min() == 0.0 and cam.max() == pytest.approx(1.0)


def test_gradcam_rejects_unexpected_token_layout():
    import torch
    from image_analyzer import _cam_from_tensors
    with pytest.raises(ValueError, match="unexpected token layout"):
        _cam_from_tensors(torch.ones(1, 50, 4), torch.ones(1, 50, 4), 14)
