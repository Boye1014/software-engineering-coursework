"""训练链脚本的可测单元：标注转换、切分参数、产物收集、指标抽取。

训练本身耗时太长不放进测试，但"数据怎么变成标签""产物怎么收"这些纯逻辑
必须可回归——它们错了，训练出来的模型就是错的，而且很难事后发现。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = REPO_ROOT / "training"


def _load(name: str):
    """按文件路径加载训练脚本，避免训练目录被当成包。"""
    module_name = f"training_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, TRAINING_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prepare_data():
    return _load("prepare_data")


@pytest.fixture(scope="module")
def train_module():
    return _load("train")


def _write_xml(path: Path, body: str, width=100, height=100) -> Path:
    path.write_text(
        f"""<annotation>
  <size><width>{width}</width><height>{height}</height></size>
  {body}
</annotation>""",
        encoding="utf-8",
    )
    return path


def _obj(name: str, xmin, ymin, xmax, ymax, difficult: int = 0) -> str:
    return f"""
  <object>
    <name>{name}</name>
    <difficult>{difficult}</difficult>
    <bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin><xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox>
  </object>"""


# --------------------------------------------------------------------------- #
# VOC XML → YOLO 标签
# --------------------------------------------------------------------------- #
def test_convert_annotation_maps_class_ids(prepare_data, tmp_path):
    from collections import Counter

    xml = _write_xml(
        tmp_path / "a.xml",
        _obj("person", 10, 10, 50, 50) + _obj("bicycle", 0, 0, 100, 100),
    )
    lines, skipped = prepare_data.convert_annotation(xml, Counter())

    assert skipped == 0
    assert len(lines) == 2
    assert lines[0].split()[0] == "0"   # person
    assert lines[1].split()[0] == "1"   # bicycle


def test_convert_annotation_normalizes_coordinates(prepare_data, tmp_path):
    from collections import Counter

    xml = _write_xml(tmp_path / "b.xml", _obj("person", 10, 20, 50, 60), 100, 100)
    line = prepare_data.convert_annotation(xml, Counter())[0][0]
    class_id, cx, cy, w, h = line.split()

    assert class_id == "0"
    assert float(cx) == pytest.approx(0.30, abs=1e-6)
    assert float(cy) == pytest.approx(0.40, abs=1e-6)
    assert float(w) == pytest.approx(0.40, abs=1e-6)
    assert float(h) == pytest.approx(0.40, abs=1e-6)


def test_convert_annotation_skips_difficult(prepare_data, tmp_path):
    from collections import Counter

    xml = _write_xml(tmp_path / "c.xml", _obj("person", 10, 10, 50, 50, difficult=1))
    lines, skipped = prepare_data.convert_annotation(xml, Counter())
    assert lines == []
    assert skipped == 1


def test_convert_annotation_ignores_other_classes(prepare_data, tmp_path):
    from collections import Counter

    xml = _write_xml(tmp_path / "d.xml", _obj("dog", 10, 10, 50, 50) + _obj("train", 0, 0, 90, 90))
    lines, skipped = prepare_data.convert_annotation(xml, Counter())
    assert lines == []
    assert skipped == 0


def test_convert_annotation_drops_degenerate_box(prepare_data, tmp_path):
    from collections import Counter

    xml = _write_xml(tmp_path / "e.xml", _obj("car", 10, 10, 10.5, 10.5))
    lines, _ = prepare_data.convert_annotation(xml, Counter())
    assert lines == [], "宽高不足 1 像素的框应被丢弃"


def test_convert_annotation_clamps_out_of_bounds(prepare_data, tmp_path):
    from collections import Counter

    xml = _write_xml(tmp_path / "f.xml", _obj("car", -20, -20, 200, 200), 100, 100)
    parts = prepare_data.convert_annotation(xml, Counter())[0][0].split()
    cx, cy, w, h = (float(value) for value in parts[1:])

    assert cx == pytest.approx(0.5) and cy == pytest.approx(0.5)
    assert w == pytest.approx(1.0) and h == pytest.approx(1.0)


def test_convert_annotation_handles_broken_xml(prepare_data, tmp_path):
    from collections import Counter

    broken = tmp_path / "broken.xml"
    broken.write_text("<annotation><size>", encoding="utf-8")
    assert prepare_data.convert_annotation(broken, Counter()) == ([], 0)


def test_convert_annotation_without_size_returns_empty(prepare_data, tmp_path):
    from collections import Counter

    xml = tmp_path / "nosize.xml"
    xml.write_text("<annotation><object><name>person</name></object></annotation>", encoding="utf-8")
    assert prepare_data.convert_annotation(xml, Counter()) == ([], 0)


def test_convert_annotation_updates_stats(prepare_data, tmp_path):
    from collections import Counter

    stats = Counter()
    xml = _write_xml(tmp_path / "g.xml", _obj("person", 10, 10, 50, 50) + _obj("person", 0, 0, 20, 20))
    prepare_data.convert_annotation(xml, stats)
    assert stats["person"] == 2


# --------------------------------------------------------------------------- #
# link_or_copy：硬链接优先，失败回落复制
# --------------------------------------------------------------------------- #
def test_link_or_copy_creates_link(prepare_data, tmp_path):
    source = tmp_path / "src.jpg"
    source.write_bytes(b"data")
    target = tmp_path / "dst.jpg"

    prepare_data.link_or_copy(source, target)
    assert target.exists()
    assert target.read_bytes() == b"data"


def test_link_or_copy_is_idempotent(prepare_data, tmp_path):
    source = tmp_path / "src.jpg"
    source.write_bytes(b"data")
    target = tmp_path / "dst.jpg"
    target.write_bytes(b"keep")

    prepare_data.link_or_copy(source, target)
    assert target.read_bytes() == b"keep", "已存在的目标不应被覆盖"


# --------------------------------------------------------------------------- #
# train.py 的辅助函数
# --------------------------------------------------------------------------- #
def test_pick_device_explicit(train_module):
    assert train_module.pick_device("cpu") == "cpu"
    assert train_module.pick_device("0") == "0"


def test_pick_device_auto_returns_valid(train_module):
    assert train_module.pick_device("auto") in {"0", "cpu"}


def test_resolve_data_missing_raises(train_module):
    with pytest.raises(SystemExit):
        train_module.resolve_data("datasets/definitely-not-here/data.yaml")


def test_resolve_data_accepts_existing(tmp_path, train_module):
    yaml = tmp_path / "data.yaml"
    yaml.write_text("path: .\n", encoding="utf-8")
    assert train_module.resolve_data(str(yaml)) == yaml


def test_copy_artifacts_prefers_new_names(tmp_path, train_module, monkeypatch):
    # copy_artifacts 返回相对 REPO_ROOT 的路径，测试时把根指向临时目录
    monkeypatch.setattr(train_module, "REPO_ROOT", tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.png").write_bytes(b"png")
    (run_dir / "BoxPR_curve.png").write_bytes(b"pr")   # 新命名
    target = tmp_path / "out"

    copied = train_module.copy_artifacts(run_dir, target)
    names = {Path(item).name for item in copied}
    assert {"results.png", "BoxPR_curve.png"} <= names
    assert "PR_curve.png" not in names


def test_copy_artifacts_accepts_legacy_name(tmp_path, train_module, monkeypatch):
    monkeypatch.setattr(train_module, "REPO_ROOT", tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "PR_curve.png").write_bytes(b"pr")      # 旧命名
    copied = train_module.copy_artifacts(run_dir, tmp_path / "out")
    assert "PR_curve.png" in {Path(item).name for item in copied}


def test_copy_artifacts_on_empty_dir(tmp_path, train_module, monkeypatch):
    monkeypatch.setattr(train_module, "REPO_ROOT", tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert train_module.copy_artifacts(empty, tmp_path / "out") == []


class _FakeBoxMetrics:
    map = 0.4312
    map50 = 0.5912
    map75 = 0.4530
    mp = 0.8012
    mr = 0.5590
    ap_class_index = [0, 1]
    maps = [0.46, 0.40]
    p = [0.83, 0.71]
    r = [0.58, 0.52]


class _FakeMetrics:
    names = {0: "person", 1: "bicycle"}

    def __init__(self):
        self.box = _FakeBoxMetrics()


def test_extract_metrics_overall(train_module):
    result = train_module.extract_metrics(_FakeMetrics())
    overall = result["overall"]
    assert overall["mAP50"] == pytest.approx(0.5912)
    assert overall["mAP50-95"] == pytest.approx(0.4312)
    assert overall["precision"] == pytest.approx(0.8012)


def test_extract_metrics_per_class(train_module):
    result = train_module.extract_metrics(_FakeMetrics())
    assert set(result["per_class"]) == {"person", "bicycle"}
    assert result["per_class"]["person"]["mAP50-95"] == pytest.approx(0.46)


def test_extract_metrics_survives_missing_attributes(train_module):
    class _Broken:
        names = {}

        class box:  # noqa: N801
            map = 0.1
            map50 = 0.2
            map75 = 0.05
            mp = 0.3
            mr = 0.4
            ap_class_index = []
            maps = []
            p = []
            r = []

    result = train_module.extract_metrics(_Broken())
    assert result["overall"]["mAP50"] == pytest.approx(0.2)
    assert result["per_class"] == {}
