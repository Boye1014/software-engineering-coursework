"""构建 12 类检测数据集：COCO2017-val + VOC2007。

类别与 app.py 界面展示的 12 个核心类别保持一致：

    0 person        1 bicycle      2 car        3 motorcycle
    4 bus           5 truck        6 bird       7 cat
    8 dog           9 bottle      10 laptop    11 cell phone

为什么要把两个数据集合起来：
COCO val2017 里 bicycle / motorcycle / bus / laptop / cell phone 每类只有
300~400 个目标，单独训练这些类别会严重欠拟合；VOC2007 覆盖其中 9 类
（缺 truck / laptop / cell phone），合并后这些小类样本量能翻一倍。

用法：
    python training/prepare_coco12.py
    python training/prepare_coco12.py --limit 300        # 调试用
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "datasets" / "_downloads"
OUTPUT_ROOT = REPO_ROOT / "datasets" / "traffic12"

CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "bird", "cat", "dog", "bottle", "laptop", "cell phone",
]
NAME_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

# COCO 的类别名（部分与我们的命名不同）
COCO_SOURCE_NAMES = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "bird": "bird",
    "cat": "cat",
    "dog": "dog",
    "bottle": "bottle",
    "laptop": "laptop",
    "cell phone": "cell phone",
}

# VOC 的类别名 → 我们的类别名
VOC_SOURCE_NAMES = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorbike": "motorcycle",
    "bus": "bus",
    "bird": "bird",
    "cat": "cat",
    "dog": "dog",
    "bottle": "bottle",
}

ARCHIVES = [
    (
        "val2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",
        "val2017",
    ),
    (
        "annotations_trainval2017.zip",
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "annotations",
    ),
    (
        "VOCtrainval_06-Nov-2007.zip",
        "https://ultralytics.com/assets/VOCtrainval_06-Nov-2007.zip",
        "VOCdevkit",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 COCO+VOC 的 12 类数据集")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.25,
        help="不含目标类的图片保留比例（作为背景负样本）",
    )
    parser.add_argument("--limit", type=int, default=0, help="每个来源最多取 N 张（调试）")
    return parser.parse_args()


def download(name: str, url: str, marker: str) -> Path:
    """断点续传下载并解压，返回解压后的目录。"""
    import requests

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive = CACHE_DIR / name
    target = CACHE_DIR / marker
    if target.exists():
        print(f"  已解压：{target.name}")
        return target

    if not (archive.exists() and zipfile.is_zipfile(archive)):
        part = archive.with_suffix(archive.suffix + ".part")
        attempt = 0
        while attempt < 40:
            attempt += 1
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            try:
                with requests.get(url, headers=headers, stream=True, timeout=30) as response:
                    if response.status_code not in (200, 206):
                        print(f"  HTTP {response.status_code}，重试")
                        time.sleep(5)
                        continue
                    total = int(response.headers.get("content-length") or 0) + have
                    mode = "ab" if response.status_code == 206 and have else "wb"
                    if mode == "wb":
                        have = 0
                    print(f"  下载 {name}（已有 {have / 1048576:.0f}MB）")
                    last = time.time()
                    with open(part, mode) as handle:
                        for chunk in response.iter_content(chunk_size=1 << 20):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            have += len(chunk)
                            if time.time() - last > 15:
                                print(f"    {have / 1048576:.0f}MB / {total / 1048576:.0f}MB")
                                last = time.time()
                    if total and have < total * 0.99:
                        continue
                    part.replace(archive)
                    break
            except Exception as error:  # noqa: BLE001
                print(f"  第 {attempt} 次失败：{type(error).__name__} {error}")
                time.sleep(5)
        else:
            sys.exit(f"{name} 下载失败，请检查网络后重试。")

    print(f"  解压 {name} …")
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(CACHE_DIR)
    return target


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        target.hardlink_to(source)
    except (OSError, AttributeError):
        shutil.copy2(source, target)


def write_sample(
    split: str, stem: str, image_path: Path, lines: list[str], stats: Counter
) -> int:
    image_dir = OUTPUT_ROOT / "images" / split
    label_dir = OUTPUT_ROOT / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    link_or_copy(image_path, image_dir / f"{stem}{image_path.suffix.lower()}")
    (label_dir / f"{stem}.txt").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )
    for line in lines:
        class_id = int(float(line.split()[0]))
        if 0 <= class_id < len(CLASS_NAMES):
            stats[CLASS_NAMES[class_id]] += 1
    return len(lines)


def collect_coco(val_dir: Path, ann_dir: Path, limit: int) -> list[dict]:
    """解析 instances_val2017.json，产出待写入的样本描述。"""
    json_path = ann_dir / "instances_val2017.json"
    if not json_path.exists():
        # 兼容压缩包内自带一层 annotations/ 的解压结果
        json_path = ann_dir / "annotations" / "instances_val2017.json"
    if not json_path.exists():
        sys.exit(f"找不到 COCO 标注文件：{json_path}")
    print(f"  解析 {json_path.name} …")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    # COCO category id -> 我们的类别 id
    category_map: dict[int, int] = {}
    for category in payload["categories"]:
        target_name = COCO_SOURCE_NAMES.get(category["name"])
        if target_name:
            category_map[category["id"]] = NAME_TO_ID[target_name]

    boxes_by_image: dict[int, list[str]] = defaultdict(list)
    for annotation in payload["annotations"]:
        new_id = category_map.get(annotation["category_id"])
        if new_id is None or annotation.get("iscrowd"):
            continue
        x, y, w, h = annotation["bbox"]
        boxes_by_image[annotation["image_id"]].append(
            (new_id, x, y, w, h)  # type: ignore[arg-type]
        )

    size = {item["id"]: (item["width"], item["height"], item["file_name"]) for item in payload["images"]}
    samples = []
    for image_id, raw_boxes in boxes_by_image.items():
        width, height, file_name = size[image_id]
        if width <= 0 or height <= 0:
            continue
        lines = []
        for new_id, x, y, w, h in raw_boxes:
            x0 = max(0.0, min(x, width))
            y0 = max(0.0, min(y, height))
            x1 = max(0.0, min(x + w, width))
            y1 = max(0.0, min(y + h, height))
            if x1 - x0 <= 1 or y1 - y0 <= 1:
                continue
            lines.append(
                f"{new_id} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
                f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}"
            )
        if not lines:
            continue
        image_path = val_dir / file_name
        if not image_path.exists():
            continue
        samples.append(
            {"stem": f"coco_{image_path.stem}", "image": image_path, "lines": lines}
        )
        if limit and len(samples) >= limit:
            break
    return samples


def collect_voc(voc_root: Path, limit: int) -> tuple[list[dict], int, int]:
    """解析 VOC XML，产出待写入的样本描述。"""
    annotations_dir = voc_root / "Annotations"
    images_dir = voc_root / "JPEGImages"
    samples: list[dict] = []
    skipped_difficult = 0
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        size_node = root.find("size")
        if size_node is None:
            continue
        width = float(size_node.findtext("width", "0"))
        height = float(size_node.findtext("height", "0"))
        if width <= 0 or height <= 0:
            continue
        lines = []
        for obj in root.findall("object"):
            source_name = (obj.findtext("name") or "").strip()
            target_name = VOC_SOURCE_NAMES.get(source_name)
            if target_name is None:
                continue
            if (obj.findtext("difficult") or "0").strip() == "1":
                skipped_difficult += 1
                continue
            box = obj.find("bndbox")
            if box is None:
                continue
            try:
                xmin = float(box.findtext("xmin", "0"))
                ymin = float(box.findtext("ymin", "0"))
                xmax = float(box.findtext("xmax", "0"))
                ymax = float(box.findtext("ymax", "0"))
            except ValueError:
                continue
            xmin, xmax = min(xmin, xmax), max(xmin, xmax)
            ymin, ymax = min(ymin, ymax), max(ymin, ymax)
            xmin, ymin = max(0.0, xmin), max(0.0, ymin)
            xmax, ymax = min(width, xmax), min(height, ymax)
            box_w, box_h = xmax - xmin, ymax - ymin
            if box_w <= 1 or box_h <= 1:
                continue
            lines.append(
                f"{NAME_TO_ID[target_name]} {(xmin + xmax) / 2 / width:.6f} "
                f"{(ymin + ymax) / 2 / height:.6f} {box_w / width:.6f} {box_h / height:.6f}"
            )
        image_path = images_dir / f"{xml_path.stem}.jpg"
        if not image_path.exists():
            continue
        samples.append({"stem": f"voc_{xml_path.stem}", "image": image_path, "lines": lines})
        if limit and len(samples) >= limit:
            break
    return samples, skipped_difficult, len(samples)


def main() -> None:
    args = parse_args()

    print("[1/4] 准备数据集压缩包")
    roots: dict[str, Path] = {}
    for name, url, marker in ARCHIVES:
        roots[marker] = download(name, url, marker)

    print("[2/4] 解析标注")
    coco_samples = collect_coco(roots["val2017"], roots["annotations"], args.limit)
    print(f"  COCO：可用样本 {len(coco_samples)} 张")
    voc_samples, skipped, _ = collect_voc(roots["VOCdevkit"] / "VOC2007", args.limit)
    print(f"  VOC ：可用样本 {len(voc_samples)} 张（跳过 difficult 标注 {skipped} 个）")

    print("[3/4] 切分并写入 YOLO 格式")
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    rng = random.Random(args.seed)
    stats: Counter = Counter()
    summary: dict[str, dict[str, int]] = {}

    for source, samples in (("coco", coco_samples), ("voc", voc_samples)):
        # 负样本（没有 12 类目标）按比例保留，其余丢弃，避免算力浪费
        positives = [s for s in samples if s["lines"]]
        negatives = [s for s in samples if not s["lines"]]
        keep_negatives = [
            s for s in negatives if rng.random() < args.negative_ratio
        ]
        chosen = positives + keep_negatives
        rng.shuffle(chosen)
        val_count = max(1, int(len(chosen) * args.val_ratio))
        splits = {"val": chosen[:val_count], "train": chosen[val_count:]}
        for split, items in splits.items():
            for sample in items:
                boxes = write_sample(
                    split, sample["stem"], sample["image"], sample["lines"], stats
                )
                key = f"{source}_{split}"
                summary.setdefault(key, {"images": 0, "boxes": 0})
                summary[key]["images"] += 1
                summary[key]["boxes"] += boxes
        print(
            f"  {source.upper()}：正样本 {len(positives)}，"
            f"负样本保留 {len(keep_negatives)}/{len(negatives)}"
        )

    yaml_path = write_yaml()
    print("[4/4] 完成")
    for key in sorted(summary):
        print(f"  {key:12s} {summary[key]['images']:5d} 张 / {summary[key]['boxes']:6d} 框")
    print(f"  数据配置：{yaml_path.relative_to(REPO_ROOT)}")
    print("类别分布（目标框数）:")
    total = sum(stats.values())
    for name in CLASS_NAMES:
        print(f"  {name:11s} {stats[name]:6d}")
    print(f"  {'合计':11s} {total:6d}")
    if total == 0:
        sys.exit("没有解析到任何目标框。")


def write_yaml() -> Path:
    yaml_path = OUTPUT_ROOT / "data.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_path.write_text(
        "# 由 training/prepare_coco12.py 自动生成，请勿手工修改\n"
        f"path: {OUTPUT_ROOT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    return yaml_path


if __name__ == "__main__":
    main()
