"""Image analysis using the published DeiTFake v2 image classifier.

The model is intentionally lazy-loaded: importing the API does not download
model weights. The first image/video request loads the configured checkpoint.

What this module returns is the checkpoint's own output (a softmax score for its
"Fake" class), a face-presence check, and - when the architecture supports it - a
Grad-CAM view of which image patches pushed that score up. Nothing here is a
calibrated probability, and nothing is invented when a step is unavailable.
"""

from __future__ import annotations

import base64
import io
import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import AutoFeatureExtractor, AutoModelForImageClassification


MODEL_ID = os.getenv("DEEPTRUTH_IMAGE_MODEL", "sakshamkr1/deitfake-v2")
MODEL_CARD = "https://huggingface.co/sakshamkr1/deitfake-v2"
MODEL_LICENSE = "Apache-2.0 (as declared by the model card)"


def _env_fraction(name: str, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(os.getenv(name, default))))
    except ValueError:
        return default


# Presentation bands only. They are NOT calibrated or validated decision thresholds.
REVIEW_THRESHOLD = _env_fraction("DEEPTRUTH_REVIEW_THRESHOLD", 0.75)
LOW_THRESHOLD = min(_env_fraction("DEEPTRUTH_LOW_THRESHOLD", 0.25), REVIEW_THRESHOLD)

_MODEL = None
_PROCESSOR = None
_DEVICE = "cpu"
_LOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_FACE_CASCADE = None
_FACE_LOCK = threading.Lock()


class AnalyzerUnavailable(RuntimeError):
    """Raised when a prediction cannot safely be produced."""


def model_information() -> dict[str, Any]:
    return {
        "name": "DeiTFake v2",
        "repository": MODEL_ID,
        "model_card": MODEL_CARD,
        "architecture": "DeiT / Vision Transformer, base patch size 16, 224 input",
        "task": "binary fake-vs-real facial image classification",
        "license": MODEL_LICENSE,
        "status": "loaded" if _MODEL is not None else "lazy_load_on_first_request",
    }


def describe_score(score: float | None) -> dict[str, Any]:
    """Plain-language band for a model score, with an explicit statement of what it is not."""
    thresholds = {"low": LOW_THRESHOLD, "review": REVIEW_THRESHOLD}
    note = (
        "Bands are presentation choices for readability. They are not calibrated or "
        "validated decision thresholds and carry no statistical guarantee."
    )
    if score is None:
        return {"band": "none", "label": "No model score", "summary": "No score was produced.", "thresholds": thresholds, "note": note}
    if score >= REVIEW_THRESHOLD:
        return {
            "band": "review",
            "label": "Elevated model score",
            "summary": "The checkpoint gave this media a high fake-class score. Treat that as a reason for closer human review, not as a finding that the media is fake.",
            "thresholds": thresholds,
            "note": note,
        }
    if score <= LOW_THRESHOLD:
        return {
            "band": "low",
            "label": "Low model score",
            "summary": "The checkpoint gave this media a low fake-class score. That is not proof the media is authentic: the model only knows the manipulation styles in its training data.",
            "thresholds": thresholds,
            "note": note,
        }
    return {
        "band": "inconclusive",
        "label": "Inconclusive model score",
        "summary": "The score sits between the low and elevated bands. The checkpoint did not separate this media clearly; rely on other evidence.",
        "thresholds": thresholds,
        "note": note,
    }


def _load_model() -> None:
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is not None and _PROCESSOR is not None:
        return
    with _LOAD_LOCK:
        if _MODEL is not None and _PROCESSOR is not None:
            return
        try:
            processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
            model = AutoModelForImageClassification.from_pretrained(MODEL_ID, use_safetensors=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
        except Exception as exc:
            raise AnalyzerUnavailable(
                "The configured image model could not be loaded. No prediction was produced. "
                "Check model access, installed dependencies, and available memory."
            ) from exc
        _PROCESSOR, _MODEL, _DEVICE = processor, model, device


def _label_map() -> dict[int, str]:
    labels = getattr(_MODEL.config, "id2label", None) or {}
    result: dict[int, str] = {}
    for index, label in labels.items():
        try:
            result[int(index)] = str(label)
        except (TypeError, ValueError):
            continue
    return result


# --- face presence -------------------------------------------------------------------

def detect_faces(image: Image.Image) -> dict[str, Any]:
    """Count frontal faces with OpenCV's bundled Haar cascade.

    This is a coarse scope check, not a forensic step: it misses profile, tilted,
    small or occluded faces, so "0 faces" means "none found", not "none present".
    """
    global _FACE_CASCADE
    method = "OpenCV Haar cascade (frontal faces)"
    try:
        with _FACE_LOCK:
            if _FACE_CASCADE is None:
                cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
                if cascade.empty():
                    raise RuntimeError("Haar cascade file could not be loaded")
                _FACE_CASCADE = cascade
            gray = np.asarray(image.convert("L"))
            height, width = gray.shape
            scale = min(1.0, 640 / max(height, width))
            if scale < 1.0:
                gray = cv2.resize(gray, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
            min_side = max(24, int(min(gray.shape) * 0.08))
            boxes = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
        return {"status": "ok", "faces_detected": int(len(boxes)), "method": method}
    except Exception:
        return {"status": "unavailable", "faces_detected": None, "method": method}


# --- inference + Grad-CAM ------------------------------------------------------------

def _gradcam_layer():
    """Last encoder block's pre-attention LayerNorm for DeiT/ViT; None for other architectures."""
    for attribute in ("deit", "vit"):
        base = getattr(_MODEL, attribute, None)
        encoder = getattr(base, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is not None and len(layers) and hasattr(layers[-1], "layernorm_before"):
            return layers[-1].layernorm_before
    return None


def _cam_from_tensors(activations: torch.Tensor, gradients: torch.Tensor, grid: int) -> np.ndarray:
    """Grad-CAM over patch tokens: channel weights = mean gradient, map = ReLU(sum(weight * activation))."""
    activations, gradients = activations[0].float(), gradients[0].float()
    special_tokens = activations.shape[0] - grid * grid
    if special_tokens < 1 or special_tokens > 2:
        raise ValueError(f"unexpected token layout ({activations.shape[0]} tokens for a {grid}x{grid} patch grid)")
    patch_activations, patch_gradients = activations[special_tokens:], gradients[special_tokens:]
    weights = patch_gradients.mean(dim=0)
    cam = torch.relu((patch_activations * weights).sum(dim=-1)).reshape(grid, grid)
    peak = float(cam.max())
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("the fake-class logit had no positive patch attribution")
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.cpu().numpy()


def _run_model(inputs: dict[str, torch.Tensor], fake_index: int, explain: bool, grid: int):
    """Return (probabilities, cam | None, explain_note | None). One forward pass either way."""
    if not explain:
        with _INFERENCE_LOCK, torch.inference_mode():
            logits = _MODEL(**inputs).logits[0]
            return torch.softmax(logits.float(), dim=-1).cpu(), None, None

    layer = _gradcam_layer()
    if layer is None:
        with _INFERENCE_LOCK, torch.inference_mode():
            logits = _MODEL(**inputs).logits[0]
            probabilities = torch.softmax(logits.float(), dim=-1).cpu()
        return probabilities, None, "The loaded checkpoint is not a DeiT/ViT architecture that this Grad-CAM implementation supports."

    captured: dict[str, torch.Tensor] = {}
    handle = layer.register_forward_hook(lambda _module, _inputs, output: captured.__setitem__("activations", output))
    try:
        with _INFERENCE_LOCK, torch.enable_grad():
            logits = _MODEL(**inputs).logits[0]
            probabilities = torch.softmax(logits.detach().float(), dim=-1).cpu()
            try:
                activations = captured["activations"]
                (gradients,) = torch.autograd.grad(logits[fake_index], activations)
                cam = _cam_from_tensors(activations.detach(), gradients.detach(), grid)
                return probabilities, cam, None
            except Exception as exc:  # explanation must never break the prediction
                return probabilities, None, f"Grad-CAM could not be computed: {exc}"
    finally:
        handle.remove()


def _model_view(pixel_values: torch.Tensor) -> Image.Image:
    """Rebuild the exact (resized, cropped) picture the network received."""
    mean = torch.tensor(list(getattr(_PROCESSOR, "image_mean", None) or [0.5] * 3)).view(3, 1, 1)
    std = torch.tensor(list(getattr(_PROCESSOR, "image_std", None) or [0.5] * 3)).view(3, 1, 1)
    pixels = (pixel_values[0].detach().cpu().float() * std + mean).clamp(0, 1)
    return Image.fromarray((pixels.permute(1, 2, 0).numpy() * 255).astype(np.uint8))


def _overlay(view: Image.Image, cam: np.ndarray, size: int = 448) -> Image.Image:
    base = np.asarray(view.resize((size, size), Image.Resampling.BICUBIC)).astype(np.float32)
    heat = np.clip(cv2.resize(cam, (size, size), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)
    colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1].astype(np.float32)
    alpha = (0.10 + 0.60 * heat)[..., None]
    return Image.fromarray(np.clip(base * (1 - alpha) + colored * alpha, 0, 255).astype(np.uint8))


def _data_url(image: Image.Image, size: int | None = None, quality: int = 82) -> str:
    if size is not None:
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def analyze_image(image_input: str | Path | Image.Image, *, explain: bool = True) -> dict[str, Any]:
    """Return the model's fake-class score, scope checks, and (optionally) a Grad-CAM view.

    The scalar is a model probability, not a calibrated probability that an
    entire media item is fake. No decision threshold is applied to the score itself.
    """
    try:
        if isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            with Image.open(image_input) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc

    _load_model()
    labels = _label_map()
    fake_indices = [index for index, label in labels.items() if "fake" in label.casefold()]
    real_indices = [index for index, label in labels.items() if "real" in label.casefold()]
    if len(fake_indices) != 1 or len(real_indices) != 1:
        raise AnalyzerUnavailable(
            "The loaded checkpoint does not expose unambiguous Real/Fake class labels. "
            "No prediction was produced."
        )
    fake_index, real_index = fake_indices[0], real_indices[0]

    face_check = detect_faces(image)
    try:
        inputs = _PROCESSOR(images=image, return_tensors="pt")
        pixel_values_cpu = inputs["pixel_values"].detach().cpu()
        patch_size = getattr(_MODEL.config, "patch_size", 16)
        patch_size = patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size
        grid = int(pixel_values_cpu.shape[-1] // patch_size)
        inputs = {key: value.to(_DEVICE) for key, value in inputs.items()}
        probabilities, cam, explain_note = _run_model(inputs, fake_index, explain, grid)
        if fake_index >= len(probabilities) or real_index >= len(probabilities):
            raise AnalyzerUnavailable("The model label map does not match its output logits.")
        fake_score = float(probabilities[fake_index].item())
        real_score = float(probabilities[real_index].item())
    except AnalyzerUnavailable:
        raise
    except Exception as exc:
        raise AnalyzerUnavailable("Image inference failed. No prediction was produced.") from exc

    output_label = labels[fake_index] if fake_score >= real_score else labels[real_index]
    width, height = image.size
    crops = bool(getattr(_PROCESSOR, "do_center_crop", False))

    warnings: list[str] = []
    if face_check["status"] == "ok" and face_check["faces_detected"] == 0:
        warnings.append(
            "No frontal face was found (OpenCV Haar detector). This checkpoint is face-focused, so its score may not be meaningful for this image."
        )
    if crops and abs(width - height) / max(width, height) > 0.05:
        warnings.append("The model analyses a resized, centre-cropped square; the outer edges of this non-square image were not scored.")

    limitations = [
        "The model card describes research/educational use and does not recommend production forensic verification without further validation.",
        "Training data is face-focused and derived from OpenForensics; results may not generalize to non-face imagery or unseen manipulation methods.",
        "The score is the checkpoint's softmax output, not a calibrated forensic probability or proof of authenticity.",
        "This model returns image-level classification only; it does not localize altered pixels.",
        "Face detection uses a basic Haar cascade; it can miss non-frontal, small or occluded faces, so 'no face found' is not 'no face present'.",
    ]
    if crops:
        limitations.append("The checkpoint's preprocessing resizes and centre-crops the input, so content near the edges may not influence the score.")

    visuals: dict[str, Any] = {}
    model_view = _model_view(pixel_values_cpu)
    visuals["model_input"] = {
        "image": _data_url(model_view, size=448),
        "caption": "The picture the network actually received after resizing/cropping.",
    }
    if cam is not None:
        visuals["heatmap"] = {
            "image": _data_url(_overlay(model_view, cam)),
            "method": "Grad-CAM over final-layer patch tokens",
            "caption": "Brighter regions pushed the fake-class score up more for this checkpoint.",
        }
        limitations.append(
            "The Grad-CAM view explains where this model's score came from. It does not show which pixels were edited, and bright regions are not proof of manipulation."
        )

    evidence: list[dict[str, Any]] = [
        {
            "type": "image_classification",
            "model_output": output_label,
            "fake_class_score": fake_score,
            "real_class_score": real_score,
            "interpretation": "Uncalibrated model output for review; not a finding of fact.",
        },
        {
            "type": "face_detection",
            "status": face_check["status"],
            "faces_detected": face_check["faces_detected"],
            "method": face_check["method"],
            "interpretation": "Scope check only: the checkpoint was trained on face imagery.",
        },
    ]
    if explain:
        evidence.append(
            {
                "type": "explainability",
                "method": "Grad-CAM over final-layer ViT patch tokens",
                "status": "generated" if cam is not None else "unavailable",
                "target_class": labels[fake_index],
                "patch_grid": f"{grid}x{grid}",
                "reason": explain_note,
                "interpretation": "Shows which patches raised the fake-class score. It explains model behaviour; it does not localize edits.",
            }
        )

    return {
        "analysis_status": "completed",
        "model_score": fake_score,
        "interpretation": describe_score(fake_score),
        "warnings": warnings,
        "evidence": evidence,
        "suspicious_timestamps": [],
        "limitations": limitations,
        "visuals": visuals,
        "model_information": {
            **model_information(),
            "input_requirements": (
                "RGB image; the checkpoint's processor resizes and centre-crops to its 224x224 input."
                if crops
                else "RGB image; the checkpoint's processor resizes to its 224x224 input."
            ),
            "device": _DEVICE,
            "class_labels": labels,
        },
    }
