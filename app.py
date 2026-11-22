"""Application composition root and backwards-compatible public exports."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, abort

import inference
import storage
from inference import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    BASE_DIR,
    CORE_CLASS_IDS,
    CORE_CLASS_THRESHOLD,
    MODEL_DIR,
    MODEL_LABELS,
    MODEL_LOAD_LOCK,
    MODEL_LOCK,
    MODEL_PATH,
    MODELS,
    available_models,
    choose_device,
    clamp_float,
    current_device,
    detections_from_result,
    get_model,
    media_type_for,
    model_names,
    process_image,
    process_video,
    public_models,
    resolve_model,
    run_prediction,
    selected_class_ids,
    selected_class_names,
)
from routes import (
    JOBS,
    JOBS_LOCK,
    RuntimeConfig,
    register_routes,
)


UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
DATABASE_PATH = BASE_DIR / "detections.db"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024


def init_database() -> None:
    """Compatibility wrapper; the schema and migrations live in storage.py."""
    storage.init_database(DATABASE_PATH)


def safe_result_path(filename: str) -> Path:
    """Compatibility wrapper used by existing callers and tests."""
    path = storage.result_path(RESULT_DIR, filename)
    if path is None:
        abort(404)
    return path


def _runtime_config() -> RuntimeConfig:
    # Read module globals on each request to preserve test and deployment overrides.
    return {
        "UPLOAD_DIR": UPLOAD_DIR,
        "RESULT_DIR": RESULT_DIR,
        "DATABASE_PATH": DATABASE_PATH,
    }


init_database()
register_routes(app, _runtime_config)


def __getattr__(name: str):
    """Expose the mutable inference device state through the legacy module."""
    if name == "ACTIVE_DEVICE":
        return inference.ACTIVE_DEVICE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    init_database()
    app.run(host="127.0.0.1", port=5000, debug=True)
