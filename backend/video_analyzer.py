"""Video sampling and frame scoring using the configured image model.

This is *frame-wise image classification*, not temporal deepfake detection. Each
sampled frame is scored independently; motion, blink patterns, lip-sync and
frame-to-frame consistency are not analysed.
"""

from __future__ import annotations

import base64
import io
import math
import os
from pathlib import Path
from statistics import median
from typing import Any

import cv2
from PIL import Image

from image_analyzer import REVIEW_THRESHOLD, analyze_image, describe_score, model_information


MAX_SAMPLED_FRAMES = max(2, min(int(os.getenv("DEEPTRUTH_VIDEO_SAMPLE_COUNT", "8")), 24))
THUMBNAIL_WIDTH = 192


def _thumbnail(image: Image.Image) -> str:
    height = max(1, round(image.height * THUMBNAIL_WIDTH / image.width))
    buffer = io.BytesIO()
    image.resize((THUMBNAIL_WIDTH, height), Image.Resampling.LANCZOS).save(buffer, format="JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _faces_in(result: dict[str, Any]) -> int | None:
    for item in result.get("evidence", []):
        if item.get("type") == "face_detection":
            return item.get("faces_detected")
    return None


def analyze_video(video_path: str | Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("The uploaded file is not a readable video or its codec is unsupported.")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count < 1 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("The video does not expose a usable frame count or frame rate.")
        if width <= 0 or height <= 0 or width * height > 50_000_000:
            raise ValueError("Video frame dimensions exceed the 50 megapixel safety limit.")

        sample_total = min(MAX_SAMPLED_FRAMES, frame_count)
        if sample_total == 1:
            frame_indices = [0]
        else:
            frame_indices = sorted({round(i * (frame_count - 1) / (sample_total - 1)) for i in range(sample_total)})

        frame_results: list[dict[str, Any]] = []
        thumbnails: dict[str, str] = {}
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            picture = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = analyze_image(picture, explain=False)
            frame_results.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_index / fps, 3),
                    "model_score": result["model_score"],
                    "model_output": result["evidence"][0]["model_output"],
                    "faces_detected": _faces_in(result),
                }
            )
            thumbnails[str(frame_index)] = _thumbnail(picture)

        if not frame_results:
            raise ValueError("No decodable frames were available for analysis.")

        face_info_available = all(frame["faces_detected"] is not None for frame in frame_results)
        frames_with_faces = [frame for frame in frame_results if frame["faces_detected"]] if face_info_available else []
        scored = frames_with_faces or frame_results
        aggregate_basis = "frames_with_detected_face" if frames_with_faces else "all_sampled_frames"
        aggregate_score = float(median(frame["model_score"] for frame in scored))

        # A high score on a frame with no detectable face is out of the model's scope, so it is not offered as a review candidate.
        suspicious_timestamps = [
            {
                "timestamp_seconds": frame["timestamp_seconds"],
                "frame_index": frame["frame_index"],
                "model_score": frame["model_score"],
                "reason": f"Sampled-frame score met the human-review threshold ({REVIEW_THRESHOLD:.2f}).",
            }
            for frame in frame_results
            if frame["model_score"] >= REVIEW_THRESHOLD and frame["faces_detected"] != 0
        ]

        interpretation = describe_score(aggregate_score)
        if suspicious_timestamps:
            interpretation["summary"] += (
                f" {len(suspicious_timestamps)} sampled frame(s) crossed the review band individually; inspect them directly."
            )

        warnings: list[str] = []
        if face_info_available and not frames_with_faces:
            warnings.append(
                "No frontal face was found in any sampled frame. This checkpoint is face-focused, so the scores may not be meaningful for this video."
            )
        elif face_info_available and len(frames_with_faces) < len(frame_results):
            warnings.append(
                f"A face was found in {len(frames_with_faces)} of {len(frame_results)} sampled frames; the aggregate uses only those frames."
            )
        if len(frame_results) < len(frame_indices):
            warnings.append(f"{len(frame_indices) - len(frame_results)} requested frame(s) could not be decoded and were skipped.")

        scores = [frame["model_score"] for frame in frame_results]
        return {
            "analysis_status": "completed",
            "model_score": aggregate_score,
            "interpretation": interpretation,
            "warnings": warnings,
            "evidence": [
                {
                    "type": "sampled_video_frames",
                    "sampled_frames": frame_results,
                    "frames_requested": len(frame_indices),
                    "frames_analyzed": len(frame_results),
                    "frames_with_detected_face": len(frames_with_faces) if face_info_available else None,
                    "aggregate_method": "median fake-class softmax score across the frames named in aggregate_basis",
                    "aggregate_basis": aggregate_basis,
                    "lowest_frame_score": min(scores),
                    "highest_frame_score": max(scores),
                    "review_threshold": REVIEW_THRESHOLD,
                }
            ],
            "suspicious_timestamps": suspicious_timestamps,
            "limitations": [
                "This is frame-wise image classification, not temporal deepfake detection: motion, blinking, lip-sync and frame-to-frame consistency are not analysed.",
                "Only a limited, evenly spaced sample of frames is analyzed; manipulation between sampled frames may be missed.",
                "The underlying checkpoint is face-focused. Non-face footage and unseen manipulation methods may not generalize.",
                "The aggregate is a median of uncalibrated model scores and is not a calibrated probability or proof of authenticity.",
                "A timestamp is flagged only as a human-review candidate when its sampled-frame score crosses the configured threshold; this does not establish that the frame is fake.",
                "Face detection uses a basic frontal-face Haar cascade and can miss faces that are turned, small or occluded.",
                "Audio is not analyzed by this video endpoint.",
            ],
            "visuals": {"frame_thumbnails": thumbnails},
            "model_information": {
                **model_information(),
                "usage": "Same image checkpoint applied independently to selected RGB frames.",
                "frame_sampling": "Evenly spaced frames, capped by DEEPTRUTH_VIDEO_SAMPLE_COUNT.",
                "aggregation": "Median of frame-level fake-class scores.",
                "video_metadata": {
                    "frame_count": frame_count,
                    "fps": fps,
                    "duration_seconds": round(frame_count / fps, 3),
                    "width": width,
                    "height": height,
                },
            },
        }
    finally:
        capture.release()
