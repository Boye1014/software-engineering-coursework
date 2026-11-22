"""SQLite 历史记录与检测结果文件存储。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "detections.db"


def init_database(database_path: Path = DATABASE_PATH) -> None:
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id TEXT PRIMARY KEY,
                original_name TEXT NOT NULL,
                media_type TEXT NOT NULL,
                result_filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                conf REAL NOT NULL,
                iou REAL NOT NULL,
                selected_classes TEXT NOT NULL,
                detection_count INTEGER NOT NULL,
                elapsed_seconds REAL NOT NULL,
                device TEXT NOT NULL,
                model TEXT NOT NULL DEFAULT ''
            )
            """
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(detections)")}
        if "model" not in columns:
            connection.execute(
                "ALTER TABLE detections ADD COLUMN model TEXT NOT NULL DEFAULT ''"
            )


def insert_detection(
    database_path: Path,
    *,
    record_id: str,
    original_name: str,
    media_type: str,
    result_filename: str,
    created_at: str,
    conf: float,
    iou: float,
    selected_classes: list[str],
    detection_count: int,
    elapsed_seconds: float,
    device: str,
    model: str,
) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO detections
            (id, original_name, media_type, result_filename, created_at, conf, iou,
             selected_classes, detection_count, elapsed_seconds, device, model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                original_name,
                media_type,
                result_filename,
                created_at,
                conf,
                iou,
                json.dumps(selected_classes, ensure_ascii=False),
                detection_count,
                elapsed_seconds,
                device,
                model,
            ),
        )


def list_detections(database_path: Path, limit: int = 50) -> list[dict[str, object]]:
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM detections ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    items = []
    for row in rows:
        items.append(
            {
                "id": row["id"],
                "original_name": row["original_name"],
                "media_type": row["media_type"],
                "created_at": row["created_at"],
                "conf": row["conf"],
                "iou": row["iou"],
                "selected_classes": json.loads(row["selected_classes"]),
                "detection_count": row["detection_count"],
                "elapsed_seconds": row["elapsed_seconds"],
                "device": row["device"],
                "model": row["model"] if "model" in row.keys() else "",
                "result_filename": row["result_filename"],
            }
        )
    return items


def delete_detection(database_path: Path, result_dir: Path, record_id: str) -> bool:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT result_filename FROM detections WHERE id = ?", (record_id,)
        ).fetchone()
        if row is None:
            return False
        connection.execute("DELETE FROM detections WHERE id = ?", (record_id,))
    result_path = Path(result_dir) / Path(row[0]).name
    result_path.unlink(missing_ok=True)
    return True


def result_path(result_dir: Path, filename: str) -> Path | None:
    """Return an existing result path after stripping any supplied directory."""
    path = Path(result_dir) / Path(filename).name
    return path if path.exists() else None
