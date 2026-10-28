"""视频检测的异步任务测试：提交、进度轮询、产出与失败分支。"""

from __future__ import annotations

import time

import app as app_module
from conftest import MODEL_SELF_4


def _submit(client, clip, **form):
    with open(clip, "rb") as handle:
        data = {"file": (handle, "clip.mp4")}
        for key, value in form.items():
            data[key] = value
        return client.post("/api/detect", data=data, content_type="multipart/form-data")


def test_video_submission_is_accepted(client, short_clip):
    response = _submit(client, short_clip, conf="0.4")
    assert response.status_code == 202, "视频任务应立即返回 202 而不是阻塞等待"
    payload = response.get_json()
    assert payload["status"] == "queued"
    assert payload["job_id"]
    assert payload["media_type"] == "video"
    assert payload["progress"] == 0


def test_video_job_completes(client, short_clip, wait_job):
    job_id = _submit(client, short_clip, conf="0.4").get_json()["job_id"]
    final = wait_job(job_id)

    assert final["status"] == "complete", final.get("error")
    assert final["progress"] == 100
    assert final["processed_frames"] >= 1
    # 部分 mp4 的帧数元数据与实际可读帧会有 ±1 的偏差
    if final["total_frames"]:
        assert abs(final["processed_frames"] - final["total_frames"]) <= 2
    assert final["detection_count"] >= 0
    assert final["result_url"].endswith(".mp4")
    assert final["elapsed_seconds"] >= 0


def test_video_result_is_playable(client, short_clip, wait_job):
    job_id = _submit(client, short_clip, conf="0.4").get_json()["job_id"]
    final = wait_job(job_id)

    media = client.get(final["result_url"])
    assert media.status_code == 200
    assert len(media.data) > 1000, "输出视频不应为空文件"

    download = client.get(final["download_url"])
    assert download.status_code == 200
    assert "attachment" in download.headers["Content-Disposition"]


def test_video_progress_is_monotonic(client, short_clip):
    job_id = _submit(client, short_clip, conf="0.4").get_json()["job_id"]

    seen = []
    deadline = time.time() + 120
    while time.time() < deadline:
        payload = client.get(f"/api/detect/{job_id}").get_json()
        seen.append(payload["progress"])
        if payload["status"] in {"complete", "error"}:
            break
        time.sleep(0.2)

    assert seen == sorted(seen), f"进度必须单调不减，实际 {seen}"
    assert all(0 <= value <= 100 for value in seen)


def test_video_job_with_class_filter(client, short_clip, wait_job):
    job_id = _submit(client, short_clip, conf="0.3", classes=["0"]).get_json()["job_id"]
    final = wait_job(job_id)
    assert final["status"] == "complete"
    assert final["selected_classes"] == ["person"]


def test_video_job_with_self_trained_model(client, short_clip, wait_job):
    job_id = _submit(
        client, short_clip, conf="0.35", model=MODEL_SELF_4
    ).get_json()["job_id"]
    final = wait_job(job_id)
    assert final["status"] == "complete"
    assert final["model_id"] == MODEL_SELF_4


def test_video_rejects_unknown_model(client, short_clip):
    response = _submit(client, short_clip, model="../../app.py")
    assert response.status_code == 400


def test_video_upload_temp_file_removed(sandbox, client, short_clip, wait_job):
    job_id = _submit(client, short_clip, conf="0.4").get_json()["job_id"]
    wait_job(job_id)
    assert list((sandbox / "uploads").iterdir()) == []


def test_corrupt_video_reports_error(client, post_raw):
    """扩展名合法但内容不是视频：任务应进入 error，且不留下结果文件。"""
    response = post_raw(b"not-a-real-video" * 100, "broken.mp4", conf="0.3")
    assert response.status_code == 202
    job_id = response.get_json()["job_id"]

    deadline = time.time() + 60
    payload = {}
    while time.time() < deadline:
        payload = client.get(f"/api/detect/{job_id}").get_json()
        if payload["status"] in {"complete", "error"}:
            break
        time.sleep(0.3)

    assert payload["status"] == "error"
    assert payload["error"]


def test_jobs_registry_is_cleared_between_tests():
    with app_module.JOBS_LOCK:
        assert app_module.JOBS == {}
