"""pytest 公共夹具。

三条隔离原则：
1. 每个用例用独立的临时 uploads/ results/ 和 SQLite 文件，不碰真实运行数据；
2. 测试默认加载最轻的 yolo11n 权重，避免反复加载 39MB 的 yolo11m；
3. JOBS 字典每个用例清空，视频任务的异步状态不会串场。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 必须在 import app 之前设置：app.py 在导入时就会读取这个环境变量
os.environ.setdefault("YOLO_MODEL", str(REPO_ROOT / "yolo11n.pt"))

import app as app_module  # noqa: E402

IMAGE_MAN_BICYCLE = REPO_ROOT / "test_images" / "06_man_bicycle_public_domain.jpg"
IMAGE_CAR = REPO_ROOT / "test_images" / "04_street_car_public_domain.jpg"
IMAGE_PEOPLE = REPO_ROOT / "test_images" / "05_people_bus_cc_by.jpg"
IMAGE_CAT = REPO_ROOT / "test_images" / "01_cat_cc0.jpg"
VIDEO_CAR = REPO_ROOT / "test_videos" / "04_car_detection.mp4"

MODEL_COCO = "yolo11n.pt"                       # 80 类，界面过滤成 12 个核心类
MODEL_SELF_4 = "models/traffic_yolo11n.pt"      # 自训 4 类
MODEL_SELF_12 = "models/traffic12_yolo11n.pt"   # 自训 12 类


@pytest.fixture(scope="session")
def client():
    """Flask 测试客户端，等价于在真实 HTTP 服务上发请求。"""
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """把上传目录、结果目录、数据库全部指向临时目录。"""
    upload_dir = tmp_path / "uploads"
    result_dir = tmp_path / "results"
    upload_dir.mkdir()
    result_dir.mkdir()

    monkeypatch.setattr(app_module, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(app_module, "RESULT_DIR", result_dir)
    monkeypatch.setattr(app_module, "DATABASE_PATH", tmp_path / "detections.db")
    app_module.init_database()

    with app_module.JOBS_LOCK:
        app_module.JOBS.clear()
    yield tmp_path
    with app_module.JOBS_LOCK:
        app_module.JOBS.clear()


@pytest.fixture
def post_image(client):
    """上传图片并调用 /api/detect，返回 Response。"""

    def _post(path: Path = IMAGE_MAN_BICYCLE, filename: str | None = None, **form):
        with open(path, "rb") as handle:
            data: dict[str, object] = {
                "file": (handle, filename or Path(path).name)
            }
            for key, value in form.items():
                data[key] = value
            return client.post(
                "/api/detect", data=data, content_type="multipart/form-data"
            )

    return _post


@pytest.fixture
def post_raw(client):
    """上传任意字节内容，用于伪造文件、超大文件和非法扩展名测试。"""

    def _post(payload: bytes, filename: str, **form):
        from io import BytesIO

        data: dict[str, object] = {"file": (BytesIO(payload), filename)}
        for key, value in form.items():
            data[key] = value
        return client.post(
            "/api/detect", data=data, content_type="multipart/form-data"
        )

    return _post


@pytest.fixture(scope="session")
def short_clip(tmp_path_factory):
    """从真实素材裁出 10 帧短视频，视频用例不必处理 300+ 帧。"""
    import cv2

    if not VIDEO_CAR.exists():
        pytest.skip("缺少 test_videos/04_car_detection.mp4，跳过视频用例")

    target = tmp_path_factory.mktemp("clip") / "short.mp4"
    capture = cv2.VideoCapture(str(VIDEO_CAR))
    if not capture.isOpened():
        pytest.skip("无法打开测试视频")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(
        str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    written = 0
    try:
        while written < 10:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        writer.release()

    if written == 0:
        pytest.skip("测试视频没有可读帧")
    return target


@pytest.fixture
def wait_job(client):
    """轮询视频任务直到结束，返回最后一次状态。"""

    def _wait(job_id: str, timeout: float = 180.0) -> dict:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            response = client.get(f"/api/detect/{job_id}")
            assert response.status_code == 200
            payload = response.get_json()
            if payload["status"] in {"complete", "error"}:
                return payload
            time.sleep(0.5)
        raise AssertionError(f"任务 {job_id} 在 {timeout}s 内没有结束")

    return _wait
