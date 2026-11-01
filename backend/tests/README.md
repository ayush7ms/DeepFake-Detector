# DeepTruth — AI-Based Deepfake Detection & Digital Media Forensics

> We don't just tell you whether media is suspicious. We show you **why**, **where** and **how much you should trust the result**.

DeepTruth analyses images and videos with a pretrained deepfake classifier, explains its image scores with Grad-CAM, records every investigation, and produces a professional PDF report. It never invents results: when something is unavailable (a model that can't load, audio detection, a face that isn't there) the interface and the report say so.

## What is real and what is not

| Feature | Status |
|---|---|
| Image analysis with **DeiTFake v2** (`sakshamkr1/deitfake-v2`) | Real model inference |
| Face-presence scope check (OpenCV Haar cascade) | Real, coarse (frontal faces only) |
| Grad-CAM explanation for images | Real, computed from this model's final transformer block; disabled automatically if the architecture isn't DeiT/ViT |
| Video analysis | **Frame-wise image classification** of evenly spaced frames. *Not* temporal deepfake detection |
| Audio analysis | **Not supported.** No detector is installed; the app says so and produces no score |
| Investigation history | Real, stored in a local SQLite file; survives restarts |
| PDF reports | Real, include SHA-256 of the upload, evidence, limitations, ethical notice |

The score is the checkpoint's own softmax output for its "Fake" class. **It is not a calibrated probability.** The low / inconclusive / elevated bands are readability aids, not validated thresholds.

## Run it on Windows (VS Code + PowerShell)

Use **Python 3.10, 3.11 or 3.12**.

```powershell
cd path\to\deeptruth
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If PowerShell blocks the script, run this once and retry:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
pip install -r requirements.txt
python verify_model.py          # downloads the model (~350 MB) and checks it really works
python -m uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000** — the backend serves the frontend itself, so there is nothing else to start.

*Using VS Code Live Server instead?* Keep uvicorn running and open the page from Live Server (port 5500 or 5501). The page finds the backend at `127.0.0.1:8000` automatically. To allow other origins set `DEEPTRUTH_CORS_ORIGINS` (comma-separated).

**First image/video run is slow** — it loads (and, the very first time, downloads) the model. The UI shows elapsed time.

## Project layout

```
main.py              FastAPI app: uploads, validation, history API, PDF route, serves the frontend
image_analyzer.py    DeiTFake v2 inference, face check, score bands, Grad-CAM
video_analyzer.py    Frame sampling + per-frame scoring + thumbnails
audio_analyzer.py    Adapter interface only (no detector registered)
report_generator.py  Branded PDF report
storage.py           SQLite history + short-lived in-memory visuals
index.html / style.css / script.js   Dashboard (plain HTML/CSS/JS, no build step)
verify_model.py      One-command check of the real checkpoint
tests/               pytest suite
data/                Created automatically; holds history.sqlite3 (git-ignored)
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/analyze/image` · `/video` · `/audio` | multipart field `file` |
| GET | `/api/reports/{id}.pdf` | PDF report |
| GET | `/api/history` · `/api/history/{id}` | list / open stored investigations |
| DELETE | `/api/history/{id}` · `/api/history` | delete one / all |
| GET | `/api/stats` | counts computed from stored investigations |
| GET | `/api/health` | analyser status (does not load the model) |

Every analysis response has the same shape: `analysis_status`, `model_score`, `interpretation`, `warnings`, `evidence`, `suspicious_timestamps`, `limitations`, `model_information`, `upload_information` (name, size, SHA-256), `investigation_id`, `report_url`. Interactive docs: http://127.0.0.1:8000/docs

Optional environment variables: `DEEPTRUTH_IMAGE_MODEL`, `DEEPTRUTH_VIDEO_SAMPLE_COUNT` (2–24, default 8), `DEEPTRUTH_REVIEW_THRESHOLD` (default 0.75), `DEEPTRUTH_LOW_THRESHOLD` (default 0.25), `DEEPTRUTH_DATA_DIR`, `DEEPTRUTH_CORS_ORIGINS`, `LOG_LEVEL`.

## Privacy and security

- Uploads go to a temp file, are hashed, analysed, and **deleted immediately** (also on errors).
- History stores scores, evidence and limitations only. **No pixels are ever written to disk**: Grad-CAM overlays and frame thumbnails live in a 50-item in-memory cache, so a PDF made right after analysis includes them, while one rebuilt after a restart says they were not retained.
- Extension, MIME type, real file signature, pixel count and size limits (image 25 MB, video 250 MB, audio 50 MB) are all enforced server-side.
- Only `index.html`, `style.css`, `script.js` are served; the project folder and database are not exposed. Responses carry `nosniff` and `no-store` headers; IDs are validated as 32 hex characters.
- The server is intended for **local use**. Add authentication and HTTPS before exposing it to a network.

## Testing

```powershell
pip install -r requirements-dev.txt
python -m pytest tests -q
```

The 39 tests use a **randomly initialised DeiT** built locally (same class as the real model) so they run offline. They verify plumbing, validation, Grad-CAM mechanics, video sampling, PDF output, history, a real subprocess "restart" and that no image data reaches the database. Their scores are meaningless; `verify_model.py` is what tests the real checkpoint.

### Manual checklist (do this before presenting)

1. `python verify_model.py` → all PASS (labels printed, Grad-CAM generated).
2. Start the server; the sidebar says **API connected**; `/api/health` returns 200.
3. Analyse a clear photo of a face → score, band, Grad-CAM overlay, PDF downloads and opens.
4. Analyse a landscape photo → a **no-face warning** appears.
5. Upload a `.txt` renamed to `.png` → rejected, no score.
6. Analyse a short MP4 → timeline, 8 frame cards, PDF has the chart.
7. Audio page → says analysis is unavailable; the button is disabled.
8. History → open an item, download its PDF, delete one.
9. Stop uvicorn (the UI shows **Backend offline**), start it again → history is still there.
10. Resize the browser to phone width; check the sidebar menu, results and history rows.

## Known limitations

- DeiTFake v2 is face-focused and trained on OpenForensics-derived data; its card does not recommend production forensic use. Expect poor generalisation to unseen manipulations, non-face images and heavily compressed media.
- The heat-map explains the *model*, not the *edit*; it is not a tamper map.
- The processor resizes and centre-crops to 224×224, so image edges may not influence the score (the UI warns for non-square images).
- Haar face detection misses turned, small or occluded faces.
- Video: sampled frames only; no motion, lip-sync or audio analysis. Some WebM files don't report a frame count and are rejected with a clear message.
- Not benchmarked: this project reports model outputs; it has not measured accuracy, false-positive or false-negative rates.

## Audio: why it is off

A suitable detector needs a clear licence, evaluation on data like yours, and a standard loading path. The public candidates reviewed were not drop-in: some use custom architectures that need their own code, one widely cited checkpoint is deprecated and licensed non-commercial/no-derivatives, and others are small unreviewed community repos. To enable one later, implement `SyntheticAudioDetector.predict()` (see `audio_analyzer.py`) and call `register_detector()` at startup; the UI switches on automatically.

## Ethical use

DeepTruth supports careful human review. Don't use a score as the sole basis for accusing or identifying anyone, only analyse media you have a right to examine, and corroborate with source verification and expert review.
