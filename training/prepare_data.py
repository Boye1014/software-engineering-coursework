"""从 PASCAL VOC2007 构建 4 类交通场景子集（YOLO 格式）。

流程：
    下载官方 VOC2007 trainval 压缩包
      -> 解压 JPEGImages / Annotations / ImageSets
      -> 逐张解析 XML 标注，只保留 person / bicycle / car / bus
      -> 归一化为 YOLO 的 (class cx cy w h)
      -> 按固定随机种子切分 train / val
      -> 生成 datasets/traffic/data.yaml

类别 ID 刻意和 COCO 前四类对齐（person=0, bicycle=1, car=2, bus=3），
因此训练出的权重可直接被 app.py 的 CORE_CLASS_IDS 过滤逻辑识别。

用法：
    python training/prepare_data.py
    python training/prepare_data.py --val-ratio 0.1 --limit 200   # 小样本快速验证
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "datasets" / "traffic"
CACHE_DIR = REPO_ROOT / "datasets" / "_downloads"

ARCHIVE_NAME = "VOCtrainval_06-Nov-2007.zip"
DOWNLOAD_URLS = [
    "https://ultralytics.com/assets/VOCtrainval_06-Nov-2007.zip",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtrainval_06-Nov-2007.zip",
]

# 目标类别，顺序即训练后的类别 ID
CLASS_NAMES = ["person", "bicycle", "car", "bus"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 VOC2007 四类 YOLO 数据集")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="仅用前 N 张图片（调试用）")
    return parser.parse_args()


def download_archive() -> Path:
    """断点续传下载 VOC2007，失败自动重试多个镜像。"""
    import requests

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = CACHE_DIR / ARCHIVE_NAME
    if target.exists() and zipfile.is_zipfile(target):
        print(f"[1/4] 已存在可用的压缩包：{target.name} ({target.stat().st_size / 1048576:.1f}MB)")
        return target

    part = target.with_suffix(".zip.part")
    for url in DOWNLOAD_URLS:
        attempt = 0
        while attempt < 30:
            attempt += 1
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            try:
                with requests.get(url, headers=headers, stream=True, timeout=30) as response:
                    if response.status_code not in (200, 206):
                        print(f"  HTTP {response.status_code}，换下一个尝试")
                        break
                    total = int(response.headers.get("content-length") or 0) + have
                    mode = "ab" if response.status_code == 206 and have else "wb"
                    if mode == "wb":
                        have = 0
                    print(f"[1/4] 下载 {url.split('/')[-1]} 已续 {have / 1048576:.0f}MB")
                    last = time.time()
                    with open(part, mode) as handle:
                        for chunk in response.iter_content(chunk_size=1 << 20):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            have += len(chunk)
                            if time.time() - last > 10:
                                print(f"      {have / 1048576:.0f}MB / {total / 1048576:.0f}MB")
                                last = time.time()
                    if total and have < total * 0.99:
                        print("      连接提前结束，继续续传")
                        continue
                    part.replace(target)
                    if not zipfile.is_zipfile(target):
                        raise ValueError("下载完成但压缩包损坏")
                    print(f"[1/4] 下载完成 {target.stat().st_size / 1048576:.1f}MB")
                    return target
            except Exception as error:  # noqa: BLE001
                print(f"  第 {attempt} 次失败：{type(error).__name__} {error}")
                time.sleep(5)
    sys.exit("VOC2007 下载失败，请检查网络后重试。")


def extract_archive(archive: Path) -> Path:
    """解压并返回 VOC2007 根目录。"""
    extract_root = CACHE_DIR / "VOCdevkit"
    voc_root = extract_root / "VOC2007"
    if (voc_root / "Annotations").exists() and (voc_root / "JPEGImages").exists():
        print(f"[2/4] 数据集已解压：{voc_root}")
        return voc_root

    print("[2/4] 解压中…")
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(CACHE_DIR)
    if not voc_root.exists():
        # 兼容 tar 包或不同层级
        candidates = list(CACHE_DIR.rglob("VOC2007"))
        if not candidates:
            sys.exit("压缩包里没有找到 VOC2007 目录")
        voc_root = candidates[0]
    print(f"[2/4] 解压完成：{voc_root}")
    return voc_root


def convert_annotation(xml_path: Path, stats: Counter) -> tuple[list[str], int]:
    """把单张图片的 XML 标注转为 YOLO 行，返回 (标签行, 跳过的框数)。"""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return [], 0

    size = root.find("size")
    width = float(size.findtext("width", "0")) if size is not None else 0.0
    height = float(size.findtext("height", "0")) if size is not None else 0.0
    if width <= 0 or height <= 0:
        return [], 0

    lines: list[str] = []
    skipped = 0
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in CLASS_TO_ID:
            continue
        # difficult 标注意味着目标难以辨认，按惯例不参与训练
        if (obj.findtext("difficult") or "0").strip() == "1":
            skipped += 1
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
        xmin, xmax = max(0.0, min(xmin, xmax)), min(width, max(xmin, xmax))
        ymin, ymax = max(0.0, min(ymin, ymax)), min(height, max(ymin, ymax))
        box_width, box_height = xmax - xmin, ymax - ymin
        if box_width <= 1 or box_height <= 1:
            continue
        center_x = (xmin + xmax) / 2 / width
        center_y = (ymin + ymax) / 2 / height
        norm_w = box_width / width
        norm_h = box_height / height
        class_id = CLASS_TO_ID[name]
        lines.append(
            f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}"
        )
        stats[name] += 1
    return lines, skipped


def link_or_copy(source: Path, target: Path) -> None:
    """同盘优先硬链接，省掉 700MB 冗余拷贝。"""
    if target.exists():
        return
    try:
        target.hardlink_to(source)
    except (OSError, AttributeError):
        shutil.copy2(source, target)


def build_dataset(voc_root: Path, val_ratio: float, seed: int, limit: int) -> None:
    print("[3/4] 解析 XML 标注并转换为 YOLO 格式…")
    annotations = voc_root / "Annotations"
    images_dir = voc_root / "JPEGImages"
    stems = sorted(path.stem for path in annotations.glob("*.xml"))
    if limit:
        stems = stems[:limit]
    if not stems:
        sys.exit(f"在 {annotations} 没有找到标注文件")

    random.Random(seed).shuffle(stems)
    val_count = max(1, int(len(stems) * val_ratio))
    splits = {"val": stems[:val_count], "train": stems[val_count:]}

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    stats: Counter = Counter()
    summary = {}
    skipped_difficult = 0
    missing_image = 0

    for split, split_stems in splits.items():
        output_images = OUTPUT_ROOT / "images" / split
        output_labels = OUTPUT_ROOT / "labels" / split
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        boxes = 0
        for stem in split_stems:
            image_path = images_dir / f"{stem}.jpg"
            if not image_path.exists():
                missing_image += 1
                continue
            lines, skipped = convert_annotation(annotations / f"{stem}.xml", stats)
            skipped_difficult += skipped
            link_or_copy(image_path, output_images / image_path.name)
            (output_labels / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            boxes += len(lines)
        summary[split] = {"images": len(split_stems) - missing_image, "boxes": boxes}
        print(f"  {split:5s}: {summary[split]['images']:5d} 张图片, {boxes:6d} 个目标框")

    write_data_yaml()
    print("[4/4] 数据集构建完成")
    if missing_image:
        print(f"  警告：{missing_image} 张图片缺失并已跳过")
    if skipped_difficult:
        print(f"  跳过 difficult 标注 {skipped_difficult} 个")
    print("类别分布（目标框数）:")
    for name in CLASS_NAMES:
        print(f"  {name:8s} {stats[name]:6d}")
    total = sum(stats.values())
    print(f"  {'合计':8s} {total:6d}")
    if total == 0:
        sys.exit("没有解析到任何目标框，请检查数据集是否完整。")


def write_data_yaml() -> Path:
    yaml_path = OUTPUT_ROOT / "data.yaml"
    names_block = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    yaml_path.write_text(
        "# 由 training/prepare_data.py 自动生成，请勿手工修改\n"
        f"path: {OUTPUT_ROOT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 4\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    print(f"  数据配置：{yaml_path.relative_to(REPO_ROOT)}")
    return yaml_path


def main() -> None:
    args = parse_args()
    archive = download_archive()
    voc_root = extract_archive(archive)
    build_dataset(voc_root, args.val_ratio, args.seed, args.limit)


if __name__ == "__main__":
    main()
