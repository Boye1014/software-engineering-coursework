"""历史记录分页、筛选与详情接口测试。"""

from __future__ import annotations

import json
import sqlite3

import app as app_module


def _insert_record(record_id: str, *, media_type: str = "image", created_at: str, name: str = None):
    """直接写入一条检测记录，便于构造分页与筛选场景。"""
    with sqlite3.connect(app_module.DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT INTO detections
            (id, original_name, media_type, result_filename, created_at, conf, iou,
             selected_classes, detection_count, elapsed_seconds, device, model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                name or f"{record_id}.jpg",
                media_type,
                f"{record_id}.jpg" if media_type == "image" else f"{record_id}.mp4",
                created_at,
                0.25,
                0.45,
                json.dumps(["person"]),
                1,
                0.1,
                "cpu",
                "yolo11n",
            ),
        )


def _seed(count: int):
    """按时间递增写入 count 条图片记录。"""
    for index in range(count):
        _insert_record(
            f"rec-{index:02d}",
            created_at=f"2026-11-{index + 1:02d} 10:00:00",
            name=f"sample-{index:02d}.jpg",
        )


def test_history_returns_pagination_metadata(client):
    _seed(15)
    payload = client.get("/api/history").get_json()

    assert payload["total"] == 15
    assert payload["page"] == 1
    assert payload["page_size"] == 12
    assert payload["pages"] == 2
    assert len(payload["items"]) == 12


def test_history_second_page_contains_rest(client):
    _seed(15)
    first = client.get("/api/history?page=1&page_size=12").get_json()
    second = client.get("/api/history?page=2&page_size=12").get_json()

    assert len(second["items"]) == 3
    assert second["page"] == 2
    first_ids = {item["id"] for item in first["items"]}
    second_ids = {item["id"] for item in second["items"]}
    assert first_ids.isdisjoint(second_ids)


def test_history_page_beyond_range_is_empty(client):
    _seed(5)
    payload = client.get("/api/history?page=9&page_size=12").get_json()

    assert payload["items"] == []
    assert payload["total"] == 5
    assert payload["pages"] == 1


def test_history_filters_by_media_type(client):
    _insert_record("img-1", media_type="image", created_at="2026-11-01 09:00:00")
    _insert_record("vid-1", media_type="video", created_at="2026-11-02 09:00:00")

    images = client.get("/api/history?media_type=image").get_json()
    videos = client.get("/api/history?media_type=video").get_json()

    assert [item["id"] for item in images["items"]] == ["img-1"]
    assert [item["id"] for item in videos["items"]] == ["vid-1"]


def test_history_filters_by_keyword(client):
    _insert_record("a", created_at="2026-11-01 09:00:00", name="street-car.jpg")
    _insert_record("b", created_at="2026-11-02 09:00:00", name="street-bus.jpg")
    _insert_record("c", created_at="2026-11-03 09:00:00", name="indoor-cat.jpg")

    payload = client.get("/api/history?keyword=street").get_json()

    assert payload["total"] == 2
    assert {item["id"] for item in payload["items"]} == {"a", "b"}


def test_history_filters_by_date_range(client):
    _insert_record("old", created_at="2026-10-01 09:00:00")
    _insert_record("mid", created_at="2026-11-10 09:00:00")
    _insert_record("new", created_at="2026-11-20 09:00:00")

    payload = client.get("/api/history?date_from=2026-11-01&date_to=2026-11-15").get_json()

    assert payload["total"] == 1
    assert payload["items"][0]["id"] == "mid"


def test_history_rejects_invalid_media_type(client):
    response = client.get("/api/history?media_type=audio")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_history_rejects_invalid_date(client):
    response = client.get("/api/history?date_from=2026-13-01")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_history_detail_returns_record(client, sandbox):
    _insert_record("detail-1", created_at="2026-11-05 12:00:00", name="demo.jpg")
    (sandbox / "results" / "detail-1.jpg").write_bytes(b"fake")

    response = client.get("/api/history/detail-1")
    body = response.get_json()

    assert response.status_code == 200
    assert body["id"] == "detail-1"
    assert body["original_name"] == "demo.jpg"
    assert body["result_url"] == "/media/detail-1.jpg"
    assert body["result_exists"] is True


def test_history_detail_marks_missing_result_file(client):
    _insert_record("detail-2", created_at="2026-11-06 12:00:00")

    body = client.get("/api/history/detail-2").get_json()

    assert body["result_exists"] is False


def test_history_detail_missing_record_returns_404(client):
    response = client.get("/api/history/not-exists")
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_history_indexes_are_created(client):
    with sqlite3.connect(app_module.DATABASE_PATH) as connection:
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'detections'"
            )
        }
    assert {
        "idx_detections_created_at",
        "idx_detections_media_type",
        "idx_detections_model",
    } <= names
