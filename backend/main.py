"""DeepTruth FastAPI service for local image, video, and audio investigations."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

import audio_analyzer
import image_analyzer
import storage
import video_analyzer
from audio_analyzer import analyze_audio
from image_analyzer import AnalyzerUnavailable, analyze_image, describe_score
from report_generator import generate_pdf_report
from video_analyzer import analyze_video


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("deeptruth.api")

BASE_DIR = Path(__file__).resolve().parent
MAX_BYTES = {"image": 25 * 1024 * 1024, "video": 250 * 1024 * 1024, "audio": 50 * 1024 * 1024}
ALLOWED_EXTENSIONS = {
    "image": {".png", ".jpg", ".jpeg", ".webp"},
    "video": {".mp4", ".mov", ".webm"},
    "audio": {".wav", ".mp3", ".m4a"},
}
ALLOWED_MIME = {
    "image": {"image/png", "image/jpeg", "image/webp"},
    "video": {"video/mp4", "video/quicktime", "video/webm"},
    "audio": {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/mp4", "audio/x-m4a"},
}
MAX_IMAGE_PIXELS = 50_000_000
ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
WHATSAPP_NAME = re.compile(r"whatsapp|-WA\d{3,5}", re.IGNORECASE)

storage.init_db()

app = FastAPI(
    title="DeepTruth API",
    description="Media analysis API with explicit model limitations and uncertainty.",
    version="1.1.0",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "DEEPTRUTH_CORS_ORIGINS",
        "http://localhost:5500,http://127.0.0.1:5500,http://localhost:5501,http://127.0.0.1:5501",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


def _public_filename(filename: str | None) -> str:
    cleaned = (filename or "upload").replace("\\", "/").split("/")[-1]
    cleaned = "".join(char for char in cleaned if char.isprintable() and char not in "\r\n\0")
    return cleaned[:180] or "upload"


def _error_payload(
    status: str,
    message: str,
    *,
    limitations: list[str] | None = None,
    model_information: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "analysis_status": status,
        "model_score": None,
        "interpretation": describe_score(None),
        "warnings": [],
        "evidence": [],
        "suspicious_timestamps": [],
        "limitations": limitations or [message, "No prediction was produced; a failed request is not evidence that the media is fake."],
        "model_information": model_information or {"status": "unavailable"},
        "error": message,
    }


async def _save_upload(upload: UploadFile, kind: str) -> tuple[Path, int, str]:
    filename = _public_filename(upload.filename)
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS[kind]:
        raise HTTPException(status_code=415, detail=f"Unsupported {kind} file extension.")
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    if content_type and content_type not in ALLOWED_MIME[kind] and content_type != "application/octet-stream":
        raise HTTPException(status_code=415, detail=f"The uploaded MIME type is not valid for {kind} media.")

    temp_handle = tempfile.NamedTemporaryFile(prefix="deeptruth_", suffix=extension, delete=False)
    temp_path = Path(temp_handle.name)
    digest = hashlib.sha256()
    total = 0
    try:
        with temp_handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES[kind]:
                    raise HTTPException(status_code=413, detail=f"{kind.capitalize()} upload exceeds the {MAX_BYTES[kind] // (1024 * 1024)} MB limit.")
                digest.update(chunk)
                temp_handle.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        return temp_path, total, digest.hexdigest()
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _validate_signature(kind: str, path: Path) -> None:
    with path.open("rb") as media_file:
        header = media_file.read(64)
    if kind == "image":
        try:
            with Image.open(path) as image:
                width, height = image.size
                if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image dimensions exceed the 50 megapixel safety limit.")
                image.verify()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("The file content is not a supported, readable image.") from exc
    elif kind == "video":
        ext = path.suffix.lower()
        is_mp4_or_mov = len(header) >= 12 and header[4:8] == b"ftyp"
        is_webm = header.startswith(b"\x1aE\xdf\xa3")
        if not (is_mp4_or_mov if ext in {".mp4", ".mov"} else is_webm):
            raise ValueError("The file signature does not match an MP4, MOV, or WebM video.")
    elif kind == "audio":
        ext = path.suffix.lower()
        is_wav = len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE"
        is_mp3 = header.startswith(b"ID3") or (len(header) >= 2 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0)
        is_m4a = len(header) >= 12 and header[4:8] == b"ftyp"
        valid = {".wav": is_wav, ".mp3": is_mp3, ".m4a": is_m4a}[ext]
        if not valid:
            raise ValueError("The file signature does not match WAV, MP3, or M4A audio.")


def _apply_whatsapp_rule(kind: str, filename: str, result: dict[str, Any]) -> dict[str, Any]:
    """Rule-based provenance flag. It does not change the model score and is not a fake/real verdict."""
    if kind != "image" or not WHATSAPP_NAME.search(filename):
        return result
    result = dict(result)
    base = describe_score(result.get("model_score"))
    result["interpretation"] = {
        **base,
        "band": "review",
        "label": "Flagged: WhatsApp-sourced file",
        "summary": (
            "The file name matches WhatsApp's naming pattern, so it was probably forwarded through WhatsApp. "
            "That recompresses images and strips metadata, so it is flagged for human review. "
            "It does not mean the image is fake."
        ),
    }
    result["warnings"] = [*result.get("warnings", []), "File name matches WhatsApp's pattern; the original source is unknown."]
    result["evidence"] = [
        *result.get("evidence", []),
        {
            "type": "provenance_rule",
            "rule": "file name contains 'WhatsApp' or '-WA####'",
            "matched_filename": filename,
            "interpretation": "Rule-based flag for review, not a model finding.",
        },
    ]
    return result


def _build_report(kind: str, filename: str, upload_information: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Wrap an analyzer result with identity, timing and upload facts. Fills in consistent defaults."""
    investigation_id = uuid.uuid4().hex
    report = {
        "investigation_id": investigation_id,
        "media_type": kind,
        "filename": filename,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **result,
        "upload_information": upload_information,
        "report_url": f"/api/reports/{investigation_id}.pdf",
    }
    report.setdefault("interpretation", describe_score(report.get("model_score")))
    report.setdefault("warnings", [])
    return report


def _is_flagged(report: dict[str, Any]) -> bool:
    if report.get("analysis_status") != "completed":
        return False
    return (report.get("interpretation") or {}).get("band") == "review" or bool(report.get("suspicious_timestamps"))


def _record(report: dict[str, Any]) -> None:
    """Completed investigations go to persistent history; no-result reports stay in a short in-memory cache."""
    if report.get("analysis_status") == "completed":
        storage.save_investigation(report, flagged=_is_flagged(report))
        storage.remember_visuals(report["investigation_id"], report.get("visuals"))
    else:
        storage.remember_transient(report)


async def _run_analysis(kind: str, upload: UploadFile) -> JSONResponse:
    filename = _public_filename(upload.filename)
    temp_path: Path | None = None
    try:
        temp_path, size, sha256 = await _save_upload(upload, kind)
        _validate_signature(kind, temp_path)
        analyzers: dict[str, Callable[[str | Path], dict[str, Any]]] = {
            "image": analyze_image,
            "video": analyze_video,
            "audio": analyze_audio,
        }
        result = await run_in_threadpool(analyzers[kind], temp_path)
        result = _apply_whatsapp_rule(kind, filename, result)

        report = _build_report(kind, filename, {"filename": filename, "size_bytes": size, "sha256": sha256}, result)
        _record(report)
        return JSONResponse(content=report)
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content=_error_payload("rejected", str(exc.detail)))
    except ValueError as exc:
        return JSONResponse(status_code=415, content=_error_payload("invalid_media", str(exc)))
    except AnalyzerUnavailable as exc:
        logger.warning("Model unavailable for %s: %s", kind, exc)
        model_info = image_analyzer.model_information() if kind in {"image", "video"} else {"status": "unavailable"}
        payload = _error_payload(
            "model_unavailable",
            str(exc),
            limitations=[str(exc), "No prediction was produced. This service error is not evidence that the media is fake."],
            model_information=model_info,
        )
        report = _build_report(kind, filename, {"filename": filename}, payload)
        _record(report)
        return JSONResponse(status_code=503, content=report)
    except Exception:
        logger.exception("Unhandled analysis failure for %s", kind)
        payload = _error_payload(
            "error",
            "The analysis request could not be completed. No prediction was produced.",
        )
        payload["upload_information"] = {"filename": filename}
        return JSONResponse(status_code=500, content=payload)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        await upload.close()


# --- analysis endpoints --------------------------------------------------------------

@app.post("/api/analyze/image", summary="Analyze an uploaded image")
async def post_image(file: UploadFile = File(...)) -> JSONResponse:
    return await _run_analysis("image", file)


@app.post("/api/analyze/video", summary="Analyze selected frames from an uploaded video")
async def post_video(file: UploadFile = File(...)) -> JSONResponse:
    return await _run_analysis("video", file)


@app.post("/api/analyze/audio", summary="Prepare an uploaded audio file for model analysis")
async def post_audio(file: UploadFile = File(...)) -> JSONResponse:
    return await _run_analysis("audio", file)


# --- reports and history -------------------------------------------------------------

def _load_report(report_id: str) -> dict[str, Any]:
    if not ID_PATTERN.fullmatch(report_id):
        raise HTTPException(status_code=404, detail="Report not found.")
    report = storage.get_investigation(report_id) or storage.recall_transient(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found. It may have been deleted, or it was a no-result report that has expired.")
    return report


@app.get("/api/reports/{report_id}.pdf", summary="Download a PDF investigation report")
async def get_report(report_id: str) -> Response:
    report = _load_report(report_id)
    visuals = storage.recall_visuals(report_id)
    if visuals:
        report = {**report, "visuals": visuals}
    pdf = await run_in_threadpool(generate_pdf_report, report)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="deeptruth-report-{report_id[:12]}.pdf"'},
    )


@app.get("/api/history", summary="List stored investigations (newest first)")
async def get_history(limit: int = Query(200, ge=1, le=1000)) -> dict[str, Any]:
    items = await run_in_threadpool(storage.list_investigations, limit)
    return {"items": items, "count": len(items)}


@app.get("/api/history/{investigation_id}", summary="Open one stored investigation")
async def get_history_item(investigation_id: str) -> dict[str, Any]:
    report = _load_report(investigation_id)
    visuals = storage.recall_visuals(investigation_id)
    return {**report, "visuals": visuals or {}, "visuals_retained": bool(visuals)}


@app.delete("/api/history/{investigation_id}", summary="Delete one stored investigation")
async def delete_history_item(investigation_id: str) -> dict[str, Any]:
    if not ID_PATTERN.fullmatch(investigation_id) or not storage.delete_investigation(investigation_id):
        raise HTTPException(status_code=404, detail="Investigation not found.")
    return {"deleted": investigation_id}


@app.delete("/api/history", summary="Delete all stored investigations")
async def delete_all_history() -> dict[str, Any]:
    return {"deleted_count": storage.clear_history()}


@app.get("/api/stats", summary="Real counts computed from stored investigations")
async def get_stats() -> dict[str, Any]:
    return await run_in_threadpool(storage.stats)


@app.get("/api/health", summary="Check API and analyzer configuration")
async def get_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "DeepTruth API",
        "version": app.version,
        "analyzers": {
            "image": {
                "configured": True,
                **image_analyzer.model_information(),
                "explainability": "Grad-CAM (DeiT/ViT patch tokens)",
                "scope_check": "OpenCV Haar frontal-face detector",
            },
            "video": {
                "configured": True,
                "mode": "frame-wise image classification (no temporal analysis)",
                "frame_analyzer": image_analyzer.model_information(),
            },
            "audio": audio_analyzer.detector_information(),
        },
        "limits_bytes": MAX_BYTES,
        "history_stored": storage.stats()["total_investigations"],
    }


# --- frontend (explicit files only; the project folder is never exposed) --------------

def _frontend_file(name: str, media_type: str) -> FileResponse:
    path = BASE_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Frontend file not found.")
    return FileResponse(path, media_type=media_type)


@app.get("/", include_in_schema=False)
async def frontend_index() -> FileResponse:
    return _frontend_file("index.html", "text/html")


@app.get("/style.css", include_in_schema=False)
async def frontend_css() -> FileResponse:
    return _frontend_file("style.css", "text/css")


@app.get("/script.js", include_in_schema=False)
async def frontend_js() -> FileResponse:
    return _frontend_file("script.js", "application/javascript")