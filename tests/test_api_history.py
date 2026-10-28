"""历史记录接口测试：写入、列表、单条删除与旧库兼容。"""

from __future__ import annotations

import json
import sqlite3

import app as app_module
from conftest import IMAGE_CAR, IMAGE_MAN_BICYCLE


def test_history_empty_initially(client):
    response = client.get("/api/history")
    assert response.status_code == 200
    assert response.get_json() == []


def test_history_records_image_detection(client, post_image):
    payload = post_image(IMAGE_MAN_BICYCLE, conf="0.35").get_json()
    history = client.get("/api/history").get_json()

    assert len(history) == 1
    record = history[0]
    assert record["id"] == payload["id"]
    assert record["media_type"] == "image"
    assert record["original_name"].endswith(".jpg")
    assert record["detection_count"] == payload["detection_count"]
    assert record["model"] == payload["model"]
    assert record["result_url"].startswith("/media/")
    assert record["download_url"].startswith("/download/")
    assert isinstance(record["selected_classes"], list)


def test_history_records_video_detection(client, short_clip, wait_job):
    with open(short_clip, "rb") as handle:
        response = client.post(
            "/api/detect",
            data={"file": (handle, "clip.mp4"), "conf": "0.4"},
            content_type="multipart/form-data",
        )
    job_id = response.get_json()["job_id"]
    wait_job(job_id)

    history = client.get("/api/history").get_json()
    assert len(history) == 1
    assert history[0]["media_type"] == "video"
    assert history[0]["id"] == job_id


def test_history_is_newest_first(client):
    """按 created_at 倒序返回（同一秒内的插入顺序不参与断言）。"""
    with sqlite3.connect(app_module.DATABASE_PATH) as connection:
        for record_id, created_at in (
            ("old", "2026-10-28 09:00:00"),
            ("new", "2026-10-28 18:00:00"),
        ):
            connection.execute(
                """
                INSERT INTO detections
                (id, original_name, media_type, result_filename, created_at, conf, iou,
                 selected_classes, detection_count, elapsed_seconds, device, model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    f"{record_id}.jpg",
                    "image",
                    f"{record_id}.jpg",
                    created_at,
                    0.25,
                    0.45,
                    json.dumps(["person"]),
                    1,
                    0.1,
                    "cpu",
                    "YOLO11n",
                ),
            )
    history = client.get("/api/history").get_json()
    assert [record["id"] for record in history] == ["new", "old"]


def test_history_limits_to_fifty(client, post_image):
    """超过 50 条时只返回最近 50 条。"""
    with sqlite3.connect(app_module.DATABASE_PATH) as connection:
        for index in range(55):
            connection.execute(
                """
                INSERT INTO detections
                (id, original_name, media_type, result_filename, created_at, conf, iou,
                 selected_classes, detection_count, elapsed_seconds, device, model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"id-{index:03d}",
                    f"f{index}.jpg",
                    "image",
                    f"f{index}.jpg",
                    f"2026-10-28 10:{index // 60:02d}:{index % 60:02d}",
                    0.25,
                    0.45,
                    json.dumps(["person"]),
                    1,
                    0.1,
                    "cpu",
                    "YOLO11n",
                ),
            )
    assert len(client.get("/api/history").get_json()) == 50


def test_delete_history_record(client, sandbox, post_image):
    payload = post_image(IMAGE_CAR).get_json()
    result_filename = payload["result_url"].rsplit("/", 1)[-1]
    stored = sandbox / "results" / result_filename
    assert stored.exists()

    response = client.delete(f"/api/history/{payload['id']}")
    assert response.status_code == 200
    assert response.get_json()["ok"] is True

    assert client.get("/api/history").get_json() == []
    assert not stored.exists(), "删除记录时应同时清理结果文件"


def test_delete_missing_record_returns_404(client):
    response = client.delete("/api/history/not-exists")
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_delete_is_idempotent_only_first_time(client, post_image):
    record_id = post_image(IMAGE_CAR).get_json()["id"]
    assert client.delete(f"/api/history/{record_id}").status_code == 200
    assert client.delete(f"/api/history/{record_id}").status_code == 404


def test_history_survives_missing_result_file(client, sandbox, post_image):
    """结果文件被手工删掉后，历史列表仍应可用（只是链接失效）。"""
    payload = post_image(IMAGE_CAR).get_json()
    for item in (sandbox / "results").iterdir():
        item.unlink()

    response = client.get("/api/history")
    assert response.status_code == 200
    assert len(response.get_json()) == 1
    assert payload["id"] == response.get_json()[0]["id"]
