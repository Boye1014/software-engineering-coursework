"""在同一批验证图上公平比较自训模型与官方预训练模型。

做法：验证图的框完全不动，只把标签的类别 ID 从"本项目的 0~11"重映射回
COCO 的 ID 空间，这样官方的 COCO 模型也能在同一批图上评测，mAP 才可比。

用法：
    python training/compare_models.py --name traffic12_yolo11n
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = REPO_ROOT / "datasets" / "traffic12" / "data.yaml"
VAL_ROOT = REPO_ROOT / "datasets" / "traffic12_coco_val"
REPORT_DIR = REPO_ROOT / "training" / "artifacts"

# 本项目的连续 ID -> COCO 官方 ID
COCO_ID_MAP = [0, 1, 2, 3, 5, 7, 14, 15, 16, 39, 63, 67]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自训模型与官方模型同台评测")
    parser.add_argument("--name", default="traffic12_yolo11n", help="自训模型名（对应 models/<name>.pt）")
    parser.add_argument(
        "--baselines",
        nargs="*",
        default=["yolo11n.pt", "yolo11m.pt"],
        help="要对比的官方权重",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--images", type=int, default=20, help="测速用的图片数量")
    return parser.parse_args()


def build_coco_val() -> tuple[Path, Path]:
    """把 12 类验证集的标签重映射成 COCO ID，图片用硬链接。"""
    source_labels = REPO_ROOT / "datasets" / "traffic12" / "labels" / "val"
    source_images = REPO_ROOT / "datasets" / "traffic12" / "images" / "val"
    if not source_labels.is_dir():
        raise SystemExit(f"找不到验证集标签：{source_labels}，请先运行 training/prepare_coco12.py")

    label_dir = VAL_ROOT / "labels" / "val"
    image_dir = VAL_ROOT / "images" / "val"
    if VAL_ROOT.exists():
        shutil.rmtree(VAL_ROOT)
    label_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)

    for label_path in sorted(source_labels.glob("*.txt")):
        lines = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            new_id = int(float(fields[0]))
            if 0 <= new_id < len(COCO_ID_MAP):
                lines.append(f"{COCO_ID_MAP[new_id]} {' '.join(fields[1:])}")
        (label_dir / label_path.name).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        image_path = source_images / f"{label_path.stem}.jpg"
        if image_path.exists():
            target = image_dir / image_path.name
            try:
                target.hardlink_to(image_path)
            except OSError:
                shutil.copy2(image_path, target)

    from ultralytics import YOLO

    names = YOLO("yolo11n.pt").names
    yaml_path = VAL_ROOT / "data.yaml"
    names_block = "\n".join(f"  {index}: {name}" for index, name in sorted(names.items()))
    yaml_path.write_text(
        "# 由 training/compare_models.py 生成：把 12 类标签映射回 COCO ID\n"
        f"path: {VAL_ROOT.as_posix()}\n"
        "train: images/val\n"
        "val: images/val\n"
        f"nc: {len(names)}\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    return yaml_path, image_dir


def measure_latency(weights: str, images: list[Path], imgsz: int, device: str) -> dict[str, float]:
    from ultralytics import YOLO

    model = YOLO(weights)
    result = {}
    for label, dev in (("cpu", "cpu"), ("gpu", device)):
        for image in images[:2]:
            model.predict(str(image), imgsz=imgsz, device=dev, verbose=False)
        timings = []
        for image in images:
            started = time.perf_counter()
            model.predict(str(image), imgsz=imgsz, device=dev, verbose=False)
            timings.append((time.perf_counter() - started) * 1000)
        result[label] = round(min(timings), 1)
    return result


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    if args.device == "auto":
        import torch

        args.device = "0" if torch.cuda.is_available() else "cpu"

    trained_weight = REPO_ROOT / "models" / f"{args.name}.pt"
    if not trained_weight.exists():
        raise SystemExit(f"找不到自训权重：{trained_weight}")

    print("构建 COCO ID 版本的验证集…")
    val_yaml, image_dir = build_coco_val()
    images = sorted(image_dir.glob("*.jpg"))[: args.images]

    rows = []

    print(f"评测自训模型 {trained_weight.name} …")
    metrics = YOLO(str(trained_weight)).val(
        data=str(DATA_YAML), split="val", imgsz=args.imgsz, device=args.device, verbose=False
    )
    latency = measure_latency(str(trained_weight), images, args.imgsz, args.device)
    rows.append(
        {
            "model": f"自训 {args.name}",
            "classes": 12,
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50-95": round(float(metrics.box.map), 4),
            "size_mb": round(trained_weight.stat().st_size / 1048576, 1),
            **{f"latency_{k}_ms": v for k, v in latency.items()},
        }
    )

    for baseline in args.baselines:
        print(f"评测官方模型 {baseline} …")
        metrics = YOLO(baseline).val(
            data=str(val_yaml), split="val", imgsz=args.imgsz, device=args.device, verbose=False
        )
        latency = measure_latency(baseline, images, args.imgsz, args.device)
        weight_path = REPO_ROOT / baseline
        size = weight_path.stat().st_size / 1048576 if weight_path.exists() else 0
        rows.append(
            {
                "model": f"官方 {baseline}",
                "classes": 80,
                "mAP50": round(float(metrics.box.map50), 4),
                "mAP50-95": round(float(metrics.box.map), 4),
                "size_mb": round(size, 1),
                **{f"latency_{k}_ms": v for k, v in latency.items()},
            }
        )

    header = "| 模型 | 类别数 | mAP@0.5 | mAP@0.5:0.95 | 体积 | CPU | GPU |"
    lines = [header, "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['classes']} | {row['mAP50']:.4f} | {row['mAP50-95']:.4f} "
            f"| {row['size_mb']}MB | {row['latency_cpu_ms']}ms | {row['latency_gpu_ms']}ms |"
        )
    table = "\n".join(lines)
    print("\n" + table)

    target = REPORT_DIR / args.name
    target.mkdir(parents=True, exist_ok=True)
    (target / "comparison.md").write_text(table + "\n", encoding="utf-8")
    (target / "comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n已写入 {target.relative_to(REPO_ROOT)}/comparison.md")


if __name__ == "__main__":
    main()
