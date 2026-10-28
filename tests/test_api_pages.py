"""页面渲染与元数据接口的测试：GET /、/api/health、/api/classes、404 处理。"""

from __future__ import annotations

import re

import pytest

import app as app_module
from conftest import MODEL_COCO, MODEL_SELF_12, MODEL_SELF_4

CLASS_BUTTON = re.compile(r'title="添加 ([^"]+)"')
MODEL_OPTION = re.compile(r'<option value="([^"]*)"')


def test_index_returns_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.mimetype == "text/html"


def test_index_renders_all_core_classes(client):
    html = client.get("/").data.decode("utf-8")
    names = CLASS_BUTTON.findall(html)
    assert len(names) == 12, f"核心类别应有 12 个，实际 {names}"
    assert names[0] == "person"
    assert {"cat", "dog", "laptop", "cell phone"} <= set(names)


def test_index_renders_model_selector(client):
    html = client.get("/").data.decode("utf-8")
    values = MODEL_OPTION.findall(html)
    assert values, "页面必须有检测模型下拉"
    available = {item["id"] for item in app_module.available_models()}
    assert set(values) <= available | {""}


def test_index_shows_active_model_and_device(client):
    html = client.get("/").data.decode("utf-8")
    assert 'id="activeModelLabel"' in html
    # 设备名直接渲染在 deviceLabel 里，必须非空
    match = re.search(r'id="deviceLabel">([^<]*)<', html)
    assert match and match.group(1).strip(), "页面应显示当前推理设备"


def test_index_contains_form_fields(client):
    html = client.get("/").data.decode("utf-8")
    for element in ('id="detectForm"', 'name="conf"', 'name="iou"', 'id="classPicker"'):
        assert element in html, f"缺少 {element}"


def test_health_reports_status_and_models(client):
    payload = client.get("/api/health").get_json()
    assert payload["status"] == "ok"
    assert payload["model"]
    assert payload["device"]
    assert isinstance(payload["models"], list) and payload["models"]
    assert all({"id", "label", "size_mb"} <= set(item) for item in payload["models"])


def test_health_does_not_leak_absolute_paths(client):
    body = client.get("/api/health").data.decode("utf-8")
    assert ":\\" not in body and "/home/" not in body, "接口不应暴露本机绝对路径"


# --------------------------------------------------------------------------- #
# /api/classes
# --------------------------------------------------------------------------- #
def test_classes_default_model_returns_core_classes(client):
    payload = client.get("/api/classes").get_json()
    assert payload["class_count"] == 12
    assert {item["id"] for item in payload["classes"]} == set(app_module.CORE_CLASS_IDS)


def test_classes_for_coco_model_is_filtered(client):
    payload = client.get(f"/api/classes?model={MODEL_COCO}").get_json()
    assert payload["model"] == MODEL_COCO
    assert payload["class_count"] == 12


def test_classes_for_self_trained_12(client):
    payload = client.get(f"/api/classes?model={MODEL_SELF_12}").get_json()
    assert payload["class_count"] == 12
    assert [item["name"] for item in payload["classes"]][:3] == ["person", "bicycle", "car"]
    assert payload["model_label"].endswith("自训权重")


def test_classes_for_self_trained_4(client):
    payload = client.get(f"/api/classes?model={MODEL_SELF_4}").get_json()
    assert payload["class_count"] == 4
    assert [item["name"] for item in payload["classes"]] == [
        "person",
        "bicycle",
        "car",
        "bus",
    ]


@pytest.mark.parametrize(
    "bad_model", ["nope.pt", "../../app.py", "/etc/passwd", "models/../../app.py"]
)
def test_classes_rejects_unknown_model(client, bad_model):
    response = client.get("/api/classes", query_string={"model": bad_model})
    assert response.status_code == 400
    assert "error" in response.get_json()


# --------------------------------------------------------------------------- #
# 静态资源与错误页
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", ["/static/style.css", "/static/app.js"])
def test_static_assets_served(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.data


def test_unknown_api_returns_json_404(client):
    response = client.get("/api/not-exists")
    assert response.status_code == 404
    assert response.get_json()["error"]


def test_unknown_page_redirects_home(client):
    response = client.get("/no-such-page")
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")


def test_detect_status_unknown_job(client):
    response = client.get("/api/detect/deadbeef")
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_media_missing_file(client):
    assert client.get("/media/none.jpg").status_code == 404


def test_download_missing_file(client):
    assert client.get("/download/none.jpg").status_code == 404
