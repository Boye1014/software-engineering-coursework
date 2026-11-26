"""SQLite 历史记录与检测结果文件存储。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "detections.db"

# 支撑历史记录分页、按类型/时间/模型筛选的索引
INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS idx_detections_created_at ON detections(created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_detections_media_type ON detections(media_type)",
    "CREATE INDEX IF NOT EXISTS idx_detections_model ON detections(model)",
)


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
        # 历史记录分页与筛选依赖的索引，重复执行不会报错
        for statement in INDEX_STATEMENTS:
            connection.execute(statement)


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


def _record_from_row(row: sqlite3.Row) -> dict[str, object]:
    """把一行查询结果转换成接口使用的记录结构。"""
    return {
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


def _filter_clause(
    media_type: str | None,
    date_from: str | None,
    date_to: str | None,
    keyword: str | None,
) -> tuple[str, list[object]]:
    """按传入条件拼出 WHERE 子句与参数，空条件不参与过滤。"""
    conditions: list[str] = []
    params: list[object] = []
    if media_type:
        conditions.append("media_type = ?")
        params.append(media_type)
    if date_from:
        conditions.append("created_at >= ?")
        params.append(f"{date_from} 00:00:00")
    if date_to:
        conditions.append("created_at <= ?")
        params.append(f"{date_to} 23:59:59")
    if keyword:
        conditions.append("original_name LIKE ?")
        params.append(f"%{keyword}%")
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    return where, params


def query_detections(
    database_path: Path,
    *,
    page: int = 1,
    page_size: int = 12,
    media_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    keyword: str | None = None,
) -> tuple[list[dict[str, object]], int]:
    """分页查询检测记录，返回当前页记录与满足条件的总条数。"""
    where, params = _filter_clause(media_type, date_from, date_to, keyword)
    offset = max(page - 1, 0) * page_size
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        total = int(
            connection.execute(
                f"SELECT COUNT(*) AS total FROM detections{where}", params
            ).fetchone()["total"]
        )
        rows = connection.execute(
            f"SELECT * FROM detections{where} ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
            [*params, page_size, offset],
        ).fetchall()
    return [_record_from_row(row) for row in rows], total


def get_detection(database_path: Path, record_id: str) -> dict[str, object] | None:
    """按记录 ID 查询单条检测记录，不存在时返回 None。"""
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM detections WHERE id = ?", (record_id,)
        ).fetchone()
    return _record_from_row(row) if row is not None else None


def list_detections(database_path: Path, limit: int = 50) -> list[dict[str, object]]:
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM detections ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_record_from_row(row) for row in rows]



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
