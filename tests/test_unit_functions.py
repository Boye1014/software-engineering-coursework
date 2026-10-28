"""纯函数与模型元数据的单元测试（不经过 HTTP 层）。"""

from __future__ import annotations

import sqlite3

import pytest
from werkzeug.exceptions import NotFound

import app as app_module
from conftest import MODEL_COCO, MODEL_SELF_12, MODEL_SELF_4


# --------------------------------------------------------------------------- #
# clamp_float：置信度 / IoU 的边界收敛
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.5", 0.5),        # 正常值原样通过
        ("0.25", 0.25),
        (None, 0.25),        # 未传参使用默认值
        ("", 0.25),          # 空字符串按默认处理
        ("abc", 0.25),       # 非数字按默认处理
        ("0", 0.01),         # 低于下限被夹紧
        ("-3", 0.01),
        ("1", 0.99),         # 高于上限被夹紧
        ("99", 0.99),
        ("0.001", 0.01),
    ],
)
def test_clamp_float(raw, expected):
    assert app_module.clamp_float(raw, 0.01, 0.99, 0.25) == pytest.approx(expected)


def test_clamp_float_custom_range():
    assert app_module.clamp_float("5", 0, 1, 0.5) == 1.0
    assert app_module.clamp_float("not-a-number", 0, 1, 0.7) == 0.7


# --------------------------------------------------------------------------- #
# media_type_for：扩展名白名单
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["a.jpg", "a.JPG", "a.jpeg", "a.png", "a.bmp", "a.webp"])
def test_media_type_image_extensions(name):
    assert app_module.media_type_for(name) == "image"


@pytest.mark.parametrize("name", ["a.mp4", "a.MOV", "a.avi", "a.mkv", "a.webm"])
def test_media_type_video_extensions(name):
    assert app_module.media_type_for(name) == "video"


@pytest.mark.parametrize("name", ["a.txt", "a.gif", "a.pdf", "a.exe", "noext", "a.jpg.exe"])
def test_media_type_rejects_unknown(name):
    assert app_module.media_type_for(name) is None


def test_media_type_with_directory_prefix():
    assert app_module.media_type_for("../../etc/passwd.jpg") == "image"


# --------------------------------------------------------------------------- #
# available_models / resolve_model：模型白名单
# --------------------------------------------------------------------------- #
def test_available_models_structure():
    models = app_module.available_models()
    assert models, "至少要能发现一个权重文件"
    for item in models:
        assert set(item) >= {"id", "label", "size_mb", "path"}
        assert isinstance(item["id"], str) and item["id"]
        assert item["size_mb"] > 0
        assert item["path"].is_file()


def test_available_models_no_duplicate_ids():
    ids = [item["id"] for item in app_module.available_models()]
    assert len(ids) == len(set(ids))


def test_available_models_id_is_relative_to_project():
    for item in app_module.available_models():
        assert not item["id"].startswith("/"), "模型标识应是项目内相对路径"


def test_resolve_model_defaults_to_first():
    first = app_module.available_models()[0]
    assert app_module.resolve_model(None)["id"] == first["id"]
    assert app_module.resolve_model("")["id"] == first["id"]


def test_resolve_model_accepts_known_id():
    target = app_module.available_models()[-1]["id"]
    assert app_module.resolve_model(target)["id"] == target


@pytest.mark.parametrize(
    "bad_id",
    [
        "../../app.py",
        "/etc/passwd",
        "yolo_does_not_exist.pt",
        "models/../../app.py",
        "C:/Windows/system32/drivers/etc/hosts",
    ],
)
def test_resolve_model_rejects_unknown(bad_id):
    with pytest.raises(ValueError):
        app_module.resolve_model(bad_id)


# --------------------------------------------------------------------------- #
# model_names：核心类别过滤
# --------------------------------------------------------------------------- #
def test_model_names_filters_coco_to_core_classes():
    names = app_module.model_names(MODEL_COCO)
    assert len(names) == len(app_module.CORE_CLASS_IDS) == 12
    assert {item["id"] for item in names} == set(app_module.CORE_CLASS_IDS)
    assert names[0]["name"] == "person"


def test_model_names_keeps_small_model_classes():
    names = app_module.model_names(MODEL_SELF_12)
    assert len(names) == 12
    assert [item["id"] for item in names] == list(range(12))


def test_model_names_four_class_model():
    names = app_module.model_names(MODEL_SELF_4)
    assert [item["name"] for item in names] == ["person", "bicycle", "car", "bus"]


# --------------------------------------------------------------------------- #
# selected_class_ids / selected_class_names
# --------------------------------------------------------------------------- #
def test_selected_class_ids_empty_returns_none():
    assert app_module.selected_class_ids([], MODEL_SELF_12) is None


def test_selected_class_ids_parses_and_sorts():
    assert app_module.selected_class_ids(["3", "1", "1"], MODEL_SELF_12) == [1, 3]


def test_selected_class_ids_drops_invalid_numbers():
    assert app_module.selected_class_ids(["abc", "2"], MODEL_SELF_12) == [2]


def test_selected_class_ids_drops_out_of_range():
    # 4 类模型只有 0~3，传 9 应当被过滤掉
    assert app_module.selected_class_ids(["9"], MODEL_SELF_4) is None


def test_selected_class_ids_rejects_non_core_coco_class():
    # COCO 的 id 6 (train) 不在 12 个核心类里
    assert app_module.selected_class_ids(["6"], MODEL_COCO) is None


def test_selected_class_names_maps_ids_to_names():
    assert app_module.selected_class_names([0, 1], MODEL_SELF_12) == ["person", "bicycle"]


def test_selected_class_names_none_returns_empty():
    assert app_module.selected_class_names(None, MODEL_SELF_12) == []


# --------------------------------------------------------------------------- #
# detections_from_result
# --------------------------------------------------------------------------- #
class _FakeTensor:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value

    def tolist(self):
        return list(self._value)


class _FakeBoxes:
    def __init__(self, rows):
        self.xyxy = [_FakeTensor(row[0]) for row in rows]
        self.conf = [_FakeTensor(row[1]) for row in rows]
        self.cls = [_FakeTensor(row[2]) for row in rows]


class _FakeResult:
    def __init__(self, rows, names):
        self.boxes = _FakeBoxes(rows) if rows is not None else None
        self.names = names


def test_detections_from_result_formats_values():
    result = _FakeResult(
        rows=[([10.123456, 20.987654, 30.5, 40.5], 0.123456, 0)],
        names={0: "person"},
    )
    detections = app_module.detections_from_result(result)
    assert len(detections) == 1
    item = detections[0]
    assert item["class_id"] == 0
    assert item["class_name"] == "person"
    assert item["confidence"] == pytest.approx(0.1235)   # 保留 4 位
    assert item["box"] == [10.12, 20.99, 30.5, 40.5]     # 保留 2 位


def test_detections_from_result_handles_list_names():
    result = _FakeResult(rows=[([0, 0, 1, 1], 0.9, 1)], names=["person", "bicycle"])
    assert app_module.detections_from_result(result)[0]["class_name"] == "bicycle"


def test_detections_from_result_empty_when_no_boxes():
    assert app_module.detections_from_result(_FakeResult(None, {0: "person"})) == []


def test_detections_from_result_unknown_class_id_falls_back():
    result = _FakeResult(rows=[([0, 0, 1, 1], 0.5, 7)], names={0: "person"})
    assert app_module.detections_from_result(result)[0]["class_name"] == "7"


# --------------------------------------------------------------------------- #
# safe_result_path：结果文件路径防护
# --------------------------------------------------------------------------- #
def test_safe_result_path_returns_existing_file(sandbox):
    target = sandbox / "results" / "demo.jpg"
    target.write_bytes(b"fake-image")
    with app_module.app.test_request_context():
        resolved = app_module.safe_result_path("demo.jpg")
    assert resolved == target


def test_safe_result_path_missing_file_aborts(sandbox):
    with app_module.app.test_request_context():
        with pytest.raises(NotFound):
            app_module.safe_result_path("missing.jpg")


def test_safe_result_path_strips_traversal(sandbox):
    # Path(...).name 会砍掉目录部分，因此只能命中 results 目录内的同名文件
    with app_module.app.test_request_context():
        with pytest.raises(NotFound):
            app_module.safe_result_path("../../app.py")


# --------------------------------------------------------------------------- #
# 数据库初始化与迁移
# --------------------------------------------------------------------------- #
LEGACY_SCHEMA = """
CREATE TABLE detections (
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
    device TEXT NOT NULL
)
"""


def test_init_database_creates_table(sandbox):
    database = sandbox / "fresh.db"
    with sqlite3.connect(database) as connection:
        pass
    original = app_module.DATABASE_PATH
    try:
        app_module.DATABASE_PATH = database
        app_module.init_database()
        with sqlite3.connect(database) as connection:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(detections)")
            }
    finally:
        app_module.DATABASE_PATH = original
    assert {"id", "model", "detection_count"} <= columns


def test_init_database_migrates_legacy_table(sandbox):
    database = sandbox / "legacy.db"
    with sqlite3.connect(database) as connection:
        connection.execute(LEGACY_SCHEMA)
    original = app_module.DATABASE_PATH
    try:
        app_module.DATABASE_PATH = database
        app_module.init_database()
        with sqlite3.connect(database) as connection:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(detections)")
            }
    finally:
        app_module.DATABASE_PATH = original
    assert "model" in columns


def test_init_database_is_idempotent(sandbox):
    database = sandbox / "twice.db"
    original = app_module.DATABASE_PATH
    try:
        app_module.DATABASE_PATH = database
        app_module.init_database()
        app_module.init_database()  # 重复调用不应抛错，也不应重复加列
        with sqlite3.connect(database) as connection:
            columns = [
                row[1] for row in connection.execute("PRAGMA table_info(detections)")
            ]
    finally:
        app_module.DATABASE_PATH = original
    assert columns.count("model") == 1
