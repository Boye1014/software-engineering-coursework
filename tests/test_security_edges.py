"""安全与异常输入测试：非法文件类型、超大请求、伪造图片、路径穿越、文件名注入。"""

from __future__ import annotations

import pytest

import app as app_module
from conftest import IMAGE_CAR


# --------------------------------------------------------------------------- #
# 上传校验
# --------------------------------------------------------------------------- #
def test_detect_without_file_returns_400(client):
    response = client.post("/api/detect", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "请选择" in response.get_json()["error"]


def test_detect_with_empty_filename_returns_400(client, post_raw):
    response = post_raw(b"abc", "")
    assert response.status_code == 400


@pytest.mark.parametrize("filename", ["note.txt", "app.py", "archive.zip", "image.gif", "script.exe"])
def test_detect_rejects_unsupported_extension(client, post_raw, filename):
    response = post_raw(b"payload", filename)
    assert response.status_code == 400
    assert "只支持" in response.get_json()["error"]


def test_detect_rejects_double_extension_smuggling(client, post_raw):
    # 结尾是 .jpg 会被放行，但内容不是图片，应在解码阶段失败而不是崩溃进程
    response = post_raw(b"MZ\x90\x00 not an image", "evil.jpg")
    assert response.status_code == 500
    assert "无法读取图片" in response.get_json()["error"]


def test_oversized_upload_returns_413(client, post_raw, monkeypatch):
    monkeypatch.setitem(app_module.app.config, "MAX_CONTENT_LENGTH", 1024)
    response = post_raw(b"\xff\xd8" + b"0" * 5000, "big.jpg")
    assert response.status_code == 413
    assert "512" in response.get_json()["error"] or "过大" in response.get_json()["error"]


# --------------------------------------------------------------------------- #
# 文件名注入
# --------------------------------------------------------------------------- #
def test_malicious_filename_is_sanitized(post_raw):
    response = post_raw(b"x", "../../../etc/passwd.jpg")
    # 扩展名合法，内容非法 → 500，但绝不应该是路径写入成功
    assert response.status_code in {400, 500}
    assert "../../../etc" not in response.get_data(as_text=True)


def test_saved_original_name_has_no_directory(post_image):
    payload = post_image(IMAGE_CAR, filename="../../hack/../car.jpg").get_json()
    assert "/" not in payload["original_name"]
    assert "\\" not in payload["original_name"]


# --------------------------------------------------------------------------- #
# 路径穿越
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "url",
    [
        "/media/..%2F..%2Fapp.py",
        "/media/%2e%2e%2fapp.py",
        "/download/..%2F..%2Fapp.py",
        "/media/../../app.py",
    ],
)
def test_path_traversal_is_blocked(client, url):
    response = client.get(url)
    assert response.status_code in {302, 404}, f"{url} 不应返回 200"
    assert b"BASE_DIR" not in response.data, "不能把源码当静态资源返回"


def test_media_only_serves_existing_result(client, sandbox):
    target = sandbox / "results" / "ok.jpg"
    target.write_bytes(b"\xff\xd8demo")
    assert client.get("/media/ok.jpg").status_code == 200

    outside = sandbox / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    assert client.get("/media/secret.txt").status_code == 404


# --------------------------------------------------------------------------- #
# 模型标识白名单
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "model_id",
    ["/etc/passwd", "../../app.py", "C:/Windows/win.ini", "*", "../models/x.pt"],
)
def test_model_id_whitelist(client, post_image, model_id):
    response = post_image(IMAGE_CAR, model=model_id)
    assert response.status_code == 400


def test_classes_endpoint_never_returns_traceback(client):
    response = client.get("/api/classes?model=%2e%2e%2fapp.py")
    assert response.status_code == 400
    body = response.get_data(as_text=True)
    assert "Traceback" not in body


# --------------------------------------------------------------------------- #
# 并发与状态
# --------------------------------------------------------------------------- #
def test_two_image_requests_do_not_interfere(client, post_image):
    first = post_image(IMAGE_CAR, conf="0.3").get_json()
    second = post_image(IMAGE_CAR, conf="0.3").get_json()
    assert first["id"] != second["id"]
    assert first["result_url"] != second["result_url"]
    assert len(client.get("/api/history").get_json()) == 2
