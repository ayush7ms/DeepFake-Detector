"""Check the REAL DeiTFake v2 checkpoint on your machine.

Run this once after installing requirements (it downloads the weights, ~350 MB):

    python verify_model.py                 # uses a generated test image
    python verify_model.py C:/path/to/photo.jpg

It confirms the things that cannot be checked without the real weights:
  1. the model downloads and loads
  2. its class labels are exactly one "Real" and one "Fake"
  3. inference returns two probabilities that sum to 1
  4. Grad-CAM works on this checkpoint's architecture
"""
from __future__ import annotations

import sys
import time

from PIL import Image


def main() -> int:
    import image_analyzer

    print(f"Model: {image_analyzer.MODEL_ID}")
    started = time.time()
    try:
        image_analyzer._load_model()
    except image_analyzer.AnalyzerUnavailable as exc:
        print(f"FAIL  could not load the model: {exc}")
        cause = exc.__cause__
        if cause:
            print(f"      underlying error: {type(cause).__name__}: {cause}")
        print("      Check your internet connection, free disk space (~1 GB) and that torch/transformers installed correctly.")
        return 1
    print(f"PASS  model loaded in {time.time() - started:.1f}s on {image_analyzer._DEVICE}")

    labels = image_analyzer._label_map()
    print(f"      id2label = {labels}")
    fake = [i for i, name in labels.items() if "fake" in name.casefold()]
    real = [i for i, name in labels.items() if "real" in name.casefold()]
    if len(fake) != 1 or len(real) != 1:
        print("FAIL  labels are not exactly one Real and one Fake. DeepTruth refuses to guess which class is which.")
        return 1
    print("PASS  labels are unambiguous")

    if len(sys.argv) > 1:
        picture = Image.open(sys.argv[1])
        print(f"      using your image: {sys.argv[1]}")
    else:
        picture = Image.effect_noise((256, 256), 60).convert("RGB")
        print("      using a generated noise image (scores on it are meaningless)")

    result = image_analyzer.analyze_image(picture)
    classification = result["evidence"][0]
    total = classification["fake_class_score"] + classification["real_class_score"]
    print(f"PASS  inference ran: fake={classification['fake_class_score']:.4f} real={classification['real_class_score']:.4f} (sum {total:.4f})")
    if abs(total - 1.0) > 1e-3:
        print("FAIL  class scores do not sum to 1")
        return 1

    explain = next(item for item in result["evidence"] if item["type"] == "explainability")
    if explain["status"] == "generated":
        print(f"PASS  Grad-CAM generated on a {explain['patch_grid']} patch grid")
    else:
        print(f"WARN  Grad-CAM unavailable on this checkpoint: {explain['reason']}")
        print("      Scores still work; the heat-map panel will explain why it is missing.")
    faces = next(item for item in result["evidence"] if item["type"] == "face_detection")
    print(f"      face check: {faces['faces_detected']} face(s) found")
    print("\nAll essential checks passed. Start the app with: python -m uvicorn main:app --reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
