"""图片检测接口 /api/detect 的功能、参数与结果文件测试。"""

from __future__ import annotations

import pytest

import app as app_module
from conftest import (
    IMAGE_CAR,
    IMAGE_CAT,
    IMAGE_MAN_BICYCLE,
    IMAGE_PEOPLE,
    MODEL_COCO,
    MODEL_SELF_12,
    MODEL_SELF_4,
)


def test_detect_image_returns_detections(post_image):
    response = post_image(IMAGE_MAN_BICYCLE, conf="0.35")
    assert response.status_code == 200
    payload = response.get_json()

    assert payload["media_type"] == "image"
    assert payload["detection_count"] >= 1
    assert payload["detection_count"] == len(payload["detections"])
    assert payload["elapsed_seconds"] >= 0
    assert payload["result_url"].endswith(".jpg")
    assert payload["download_url"].startswith("/download/")


def test_detect_image_detection_fields(post_image):
    payload = post_image(IMAGE_MAN_BICYCLE, conf="0.35").get_json()
    for item in payload["detections"]:
        assert isinstance(item["class_id"], int)
        assert isinstance(item["class_name"], str) and item["class_name"]
        assert 0.0 <= item["confidence"] <= 1.0
        box = item["box"]
        assert len(box) == 4
        x1, y1, x2, y2 = box
        assert x2 > x1 and y2 > y1, "边界框坐标必须满足左上小于右下"


def test_detect_image_finds_expected_classes(post_image):
    payload = post_image(IMAGE_MAN_BICYCLE, conf="0.35").get_json()
    names = {item["class_name"] for item in payload["detections"]}
    assert "person" in names or "bicycle" in names


def test_detect_cat_image(post_image):
    payload = post_image(IMAGE_CAT, conf="0.25").get_json()
    assert payload["detection_count"] >= 1
    assert "cat" in {item["class_name"] for item in payload["detections"]}


def test_detect_people_bus_image(post_image):
    payload = post_image(IMAGE_PEOPLE, conf="0.3").get_json()
    assert payload["detection_count"] >= 3
    assert "person" in {item["class_name"] for item in payload["detections"]}


def test_detect_result_file_is_written_and_served(client, post_image):
    payload = post_image(IMAGE_CAR, conf="0.35").get_json()
    filename = payload["result_url"].rsplit("/", 1)[-1]

    media = client.get(payload["result_url"])
    assert media.status_code == 200
    assert media.data[:2] == b"\xff\xd8", "输出应是 JPEG 文件"

    download = client.get(payload["download_url"])
    assert download.status_code == 200
    assert "attachment" in download.headers["Content-Disposition"]
    assert filename in download.headers["Content-Disposition"]


# --------------------------------------------------------------------------- #
# 类别筛选
# --------------------------------------------------------------------------- #
def test_class_filter_only_person(post_image):
    payload = post_image(IMAGE_PEOPLE, conf="0.3", classes=["0"]).get_json()
    names = {item["class_name"] for item in payload["detections"]}
    assert names <= {"person"}
    assert payload["selected_classes"] == ["person"]


def test_class_filter_multiple_classes(post_image):
    payload = post_image(
        IMAGE_PEOPLE, conf="0.3", classes=["0", "1"]
    ).get_json()
    assert {item["class_name"] for item in payload["detections"]} <= {"person", "bicycle"}
    assert payload["selected_classes"] == ["person", "bicycle"]


def test_class_filter_only_car(post_image):
    payload = post_image(IMAGE_CAR, conf="0.3", classes=["2"]).get_json()
    assert {item["class_name"] for item in payload["detections"]} <= {"car"}


def test_no_class_filter_uses_all_core_classes(post_image):
    payload = post_image(IMAGE_MAN_BICYCLE, conf="0.35").get_json()
    assert len(payload["selected_classes"]) == 12


def test_invalid_class_id_falls_back_to_all(post_image):
    payload = post_image(IMAGE_MAN_BICYCLE, conf="0.35", classes=["999"]).get_json()
    assert len(payload["selected_classes"]) == 12


def test_non_numeric_class_id_is_ignored(post_image):
    payload = post_image(IMAGE_MAN_BICYCLE, conf="0.35", classes=["abc"]).get_json()
    assert len(payload["selected_classes"]) == 12


# --------------------------------------------------------------------------- #
# 参数边界
# --------------------------------------------------------------------------- #
def test_higher_conf_never_returns_more(post_image):
    low = post_image(IMAGE_PEOPLE, conf="0.01").get_json()
    high = post_image(IMAGE_PEOPLE, conf="0.99").get_json()
    assert high["detection_count"] <= low["detection_count"]


def test_non_numeric_conf_uses_default(post_image):
    response = post_image(IMAGE_MAN_BICYCLE, conf="abc", iou="xyz")
    assert response.status_code == 200


def test_out_of_range_conf_is_clamped(post_image):
    response = post_image(IMAGE_MAN_BICYCLE, conf="5")
    assert response.status_code == 200
    # conf 被夹到 0.99，结果通常为空但接口仍应成功
    assert response.get_json()["detection_count"] >= 0


# --------------------------------------------------------------------------- #
# 模型选择
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model", [MODEL_COCO, MODEL_SELF_4, MODEL_SELF_12])
def test_detect_with_each_model(post_image, model):
    payload = post_image(IMAGE_MAN_BICYCLE, conf="0.35", model=model).get_json()
    assert payload["model_id"] == model
    assert payload["detection_count"] >= 1


def test_detect_with_four_class_model_only_outputs_its_classes(post_image):
    payload = post_image(
        IMAGE_CAT, conf="0.25", model=MODEL_SELF_4
    ).get_json()
    names = {item["class_name"] for item in payload["detections"]}
    assert names <= {"person", "bicycle", "car", "bus"}
    assert len(payload["selected_classes"]) == 4


def test_detect_with_twelve_class_model_exposes_twelve_classes(post_image):
    payload = post_image(
        IMAGE_MAN_BICYCLE, conf="0.3", model=MODEL_SELF_12
    ).get_json()
    assert len(payload["selected_classes"]) == 12


def test_detect_with_unknown_model_returns_400(post_image):
    response = post_image(IMAGE_MAN_BICYCLE, model="../../app.py")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_detect_response_includes_model_label(post_image):
    payload = post_image(IMAGE_MAN_BICYCLE).get_json()
    assert payload["model"]
    assert isinstance(payload["model"], str)


# --------------------------------------------------------------------------- #
# 中间文件清理
# --------------------------------------------------------------------------- #
def test_upload_temp_file_is_removed(sandbox, post_image):
    post_image(IMAGE_CAR)
    leftovers = list((sandbox / "uploads").iterdir())
    assert leftovers == [], f"上传的临时文件应被删除，实际残留 {leftovers}"


def test_result_file_is_kept(sandbox, post_image):
    post_image(IMAGE_CAR)
    results = list((sandbox / "results").iterdir())
    assert len(results) == 1
    assert results[0].stat().st_size > 0


def test_app_module_database_path_is_temp(sandbox):
    # 防止夹具失效导致测试写入真实 detections.db
    assert app_module.DATABASE_PATH == sandbox / "detections.db"
