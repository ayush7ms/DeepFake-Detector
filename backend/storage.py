"""Investigation history for DeepTruth.

Design choices (kept deliberately small and privacy-minimal):

* History lives in a local SQLite file (standard library only, no extra install).
* Only *metadata and model output* are persisted: file name, size, SHA-256, score,
  evidence, limitations. The uploaded media itself is never written to history
  (the temporary upload is deleted as soon as analysis finishes).
* Visual evidence (Grad-CAM overlays, video frame thumbnails) contains pixels from
  the uploaded media, so it is NOT persisted. It is kept in a small in-memory cache
  so a freshly generated PDF can include it; after a restart, reports are rebuilt
  from the stored text results and say that visuals were not retained.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

DATA_DIR = Path(os.getenv("DEEPTRUTH_DATA_DIR", Path(__file__).resolve().parent / "data"))
DB_PATH = Path(os.getenv("DEEPTRUTH_DB", DATA_DIR / "history.sqlite3"))
VISUAL_CACHE_LIMIT = 50
TRANSIENT_REPORT_LIMIT = 50

_DB_LOCK = threading.Lock()
_CACHE_LOCK = threading.Lock()
_VISUALS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_TRANSIENT: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with _DB_LOCK, _connect() as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS investigations (
                id              TEXT PRIMARY KEY,
                created_at      TEXT NOT NULL,
                media_type      TEXT NOT NULL,
                filename        TEXT NOT NULL,
                size_bytes      INTEGER,
                analysis_status TEXT NOT NULL,
                model_score     REAL,
                flagged         INTEGER NOT NULL DEFAULT 0,
                result_json     TEXT NOT NULL
            )
            """
        )
        db.execute("CREATE INDEX IF NOT EXISTS idx_investigations_created ON investigations(created_at DESC)")


def _persistable(report: dict[str, Any]) -> dict[str, Any]:
    """Strip pixel-bearing visuals before anything touches disk."""
    return {key: value for key, value in report.items() if key != "visuals"}


def save_investigation(report: dict[str, Any], *, flagged: bool) -> None:
    upload = report.get("upload_information") or {}
    with _DB_LOCK, _connect() as db:
        db.execute(
            "INSERT OR REPLACE INTO investigations VALUES (?,?,?,?,?,?,?,?,?)",
            (
                report["investigation_id"],
                report["generated_at"],
                report["media_type"],
                report["filename"],
                upload.get("size_bytes"),
                report.get("analysis_status", "unknown"),
                report.get("model_score"),
                1 if flagged else 0,
                json.dumps(_persistable(report), ensure_ascii=False, default=str),
            ),
        )


def _summary(row: sqlite3.Row) -> dict[str, Any]:
    stored = json.loads(row["result_json"])
    interpretation = stored.get("interpretation") or {}
    return {
        "investigation_id": row["id"],
        "generated_at": row["created_at"],
        "media_type": row["media_type"],
        "filename": row["filename"],
        "size_bytes": row["size_bytes"],
        "analysis_status": row["analysis_status"],
        "model_score": row["model_score"],
        "flagged": bool(row["flagged"]),
        "band": interpretation.get("band"),
        "band_label": interpretation.get("label"),
        "report_url": f"/api/reports/{row['id']}.pdf",
    }


def list_investigations(limit: int = 200) -> list[dict[str, Any]]:
    with _DB_LOCK, _connect() as db:
        rows = db.execute("SELECT * FROM investigations ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_summary(row) for row in rows]


def get_investigation(investigation_id: str) -> dict[str, Any] | None:
    with _DB_LOCK, _connect() as db:
        row = db.execute("SELECT result_json FROM investigations WHERE id = ?", (investigation_id,)).fetchone()
    return json.loads(row["result_json"]) if row else None


def delete_investigation(investigation_id: str) -> bool:
    with _DB_LOCK, _connect() as db:
        deleted = db.execute("DELETE FROM investigations WHERE id = ?", (investigation_id,)).rowcount > 0
    with _CACHE_LOCK:
        _VISUALS.pop(investigation_id, None)
        _TRANSIENT.pop(investigation_id, None)
    return deleted


def clear_history() -> int:
    with _DB_LOCK, _connect() as db:
        count = db.execute("DELETE FROM investigations").rowcount
    with _CACHE_LOCK:
        _VISUALS.clear()
        _TRANSIENT.clear()
    return count


def stats() -> dict[str, Any]:
    """Real counts computed from stored investigations. Nothing is estimated."""
    with _DB_LOCK, _connect() as db:
        rows = db.execute(
            "SELECT media_type, COUNT(*) AS n, SUM(flagged) AS flagged FROM investigations "
            "WHERE analysis_status = 'completed' GROUP BY media_type"
        ).fetchall()
        bands = db.execute(
            "SELECT result_json FROM investigations WHERE analysis_status = 'completed'"
        ).fetchall()
    by_type = {row["media_type"]: {"count": row["n"], "flagged": int(row["flagged"] or 0)} for row in rows}
    band_counts = {"low": 0, "inconclusive": 0, "review": 0}
    for row in bands:
        band = (json.loads(row["result_json"]).get("interpretation") or {}).get("band")
        if band in band_counts:
            band_counts[band] += 1
    return {
        "total_investigations": sum(item["count"] for item in by_type.values()),
        "images_analyzed": by_type.get("image", {}).get("count", 0),
        "videos_analyzed": by_type.get("video", {}).get("count", 0),
        "audio_analyzed": by_type.get("audio", {}).get("count", 0),
        "flagged_for_review": sum(item["flagged"] for item in by_type.values()),
        "band_counts": band_counts,
        "source": "Counted from completed investigations stored by this backend.",
    }


# --- transient (in-memory only) caches -------------------------------------------------

def remember_visuals(investigation_id: str, visuals: dict[str, Any] | None) -> None:
    if not visuals:
        return
    with _CACHE_LOCK:
        _VISUALS[investigation_id] = visuals
        _VISUALS.move_to_end(investigation_id)
        while len(_VISUALS) > VISUAL_CACHE_LIMIT:
            _VISUALS.popitem(last=False)


def recall_visuals(investigation_id: str) -> dict[str, Any] | None:
    with _CACHE_LOCK:
        return _VISUALS.get(investigation_id)


def remember_transient(report: dict[str, Any]) -> None:
    """Keep a no-result report (e.g. audio detector not configured) available for download."""
    with _CACHE_LOCK:
        _TRANSIENT[report["investigation_id"]] = report
        _TRANSIENT.move_to_end(report["investigation_id"])
        while len(_TRANSIENT) > TRANSIENT_REPORT_LIMIT:
            _TRANSIENT.popitem(last=False)


def recall_transient(investigation_id: str) -> dict[str, Any] | None:
    with _CACHE_LOCK:
        return _TRANSIENT.get(investigation_id)
