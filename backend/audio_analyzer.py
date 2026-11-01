"""Integration seam for a future pretrained synthetic-audio detector.

Waveform statistics can describe signal properties but cannot establish that a
voice is synthetic. Until a suitably documented checkpoint is selected, this
module deliberately returns no score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class SyntheticAudioDetector(Protocol):
    """Adapter contract for a separately reviewed pretrained audio model."""

    @property
    def information(self) -> dict[str, Any]: ...

    def predict(self, audio_path: str | Path) -> dict[str, Any]:
        """Return model_score, evidence, limitations, and model_information."""
        ...


_DETECTOR: SyntheticAudioDetector | None = None


def register_detector(detector: SyntheticAudioDetector) -> None:
    """Register a separately reviewed adapter from application startup code."""
    if not callable(getattr(detector, "predict", None)):
        raise TypeError("An audio detector must provide predict(audio_path).")
    global _DETECTOR
    _DETECTOR = detector


def detector_information() -> dict[str, Any]:
    if _DETECTOR is None:
        return {"status": "not_configured", "detail": "No synthetic-audio detector is registered."}
    return {"status": "configured", "model_information": _DETECTOR.information}


def analyze_audio(
    audio_path: str | Path,
    detector: SyntheticAudioDetector | None = None,
) -> dict[str, Any]:
    """Return an explicit unconfigured response unless a reviewed model exists."""
    detector = detector or _DETECTOR
    if detector is None:
        return {
            "analysis_status": "not_configured",
            "model_score": None,
            "evidence": [],
            "suspicious_timestamps": [],
            "limitations": [
                "No pretrained synthetic-audio detector has been selected or installed.",
                "No score or synthetic-speech finding was produced. Basic waveform features are not used as a substitute detector.",
            ],
            "model_information": {
                "name": None,
                "status": "integration_interface_only",
                "expected_interface": "SyntheticAudioDetector.predict(audio_path)",
            },
        }

    result = detector.predict(audio_path)
    required = {"model_score", "evidence", "limitations", "model_information"}
    missing = required.difference(result)
    if missing:
        raise ValueError(f"Audio detector response is missing required fields: {', '.join(sorted(missing))}")
    return {
        "analysis_status": "completed",
        "model_score": result["model_score"],
        "evidence": result["evidence"],
        "suspicious_timestamps": result.get("suspicious_timestamps", []),
        "limitations": result["limitations"],
        "model_information": result["model_information"],
    }
