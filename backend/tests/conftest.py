"""Shared test setup.

IMPORTANT: these tests exercise the application plumbing (uploads, validation,
face check, Grad-CAM code path, video sampling, PDF, history). They use a real
`DeiTForImageClassification` with RANDOM weights built locally, so no download is
needed. Scores from this stand-in are meaningless and are never used to judge
detection quality. Run `python verify_model.py` to test the real checkpoint.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Isolated, throwaway history database for the whole test session.
_TMP = tempfile.mkdtemp(prefix="deeptruth_test_")
os.environ["DEEPTRUTH_DATA_DIR"] = _TMP
os.environ["DEEPTRUTH_DB"] = str(Path(_TMP) / "history.sqlite3")

import pytest  # noqa: E402
import torch  # noqa: E402
from transformers import DeiTConfig, DeiTForImageClassification, DeiTImageProcessor  # noqa: E402

import image_analyzer  # noqa: E402


def install_stand_in_model() -> None:
    torch.manual_seed(0)
    config = DeiTConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        image_size=224, patch_size=16, num_labels=2, id2label={0: "Real", 1: "Fake"}, label2id={"Real": 0, "Fake": 1},
    )
    model = DeiTForImageClassification(config).eval()
    image_analyzer._MODEL = model
    image_analyzer._PROCESSOR = DeiTImageProcessor()
    image_analyzer._DEVICE = "cpu"


@pytest.fixture(scope="session", autouse=True)
def _stand_in():
    install_stand_in_model()
    yield


@pytest.fixture(scope="session")
def client(_stand_in):
    from fastapi.testclient import TestClient
    import main
    return TestClient(main.app)
