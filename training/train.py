"""在 VOC2007 四类子集上微调 YOLO11n，并输出可交付的权重与指标。

用法：
    python training/train.py                    # 完整训练（100 epoch）
    python training/train.py --epochs 3         # 先跑通链路再上全量

产物：
    models/yolo11n_traffic.pt              训练得到的最优权重
    models/yolo11n_traffic.metrics.json    训练配置 + 整体/分类别指标
    training/artifacts/<name>/             曲线图、混淆矩阵、results.csv
    runs/train/<name>/                     ultralytics 原始输出目录
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "datasets" / "traffic" / "data.yaml"
MODEL_DIR = REPO_ROOT / "models"
ARTIFACT_ROOT = REPO_ROOT / "training" / "artifacts"
RUN_ROOT = REPO_ROOT / "runs" / "train"

ARTIFACT_FILES = [
    ["results.png"],
    ["confusion_matrix.png"],
    ["confusion_matrix_normalized.png"],
    ["BoxPR_curve.png", "PR_curve.png"],  # ultralytics 8.4 起改名为 BoxPR_curve
    ["BoxF1_curve.png", "F1_curve.png"],
    ["results.csv"],
    ["val_batch0_pred.jpg"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 VOC2007 四类子集上微调 YOLO11n")
    parser.add_argument("--model", default="yolo11n.pt", help="基础权重，默认官方 yolo11n.pt")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="数据配置 yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20, help="mAP 无提升时的早停轮数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto / cpu / 0")
    parser.add_argument("--name", default="traffic_yolo11n", help="本次实验名称")
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    return parser.parse_args()


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def resolve_data(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise SystemExit(
            f"找不到数据配置 {path}\n请先运行：python training/prepare_data.py"
        )
    return path


def extract_metrics(metrics) -> dict:
    """把 ultralytics 的 DetMetrics 抽成纯 Python 结构，避免依赖内部对象。"""
    box = metrics.box
    summary = {
        "mAP50-95": round(float(box.map), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP75": round(float(box.map75), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
    }

    per_class = {}
    try:
        names = metrics.names
        for position, class_id in enumerate(box.ap_class_index):
            class_id = int(class_id)
            per_class[names.get(class_id, str(class_id))] = {
                "mAP50-95": round(float(box.maps[position]), 4),
                "precision": round(float(box.p[position]), 4),
                "recall": round(float(box.r[position]), 4),
            }
    except Exception:  # 结构变化时不影响主流程
        per_class = {}
    return {"overall": summary, "per_class": per_class}


def copy_artifacts(run_dir: Path, target_dir: Path) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for candidates in ARTIFACT_FILES:
        for name in candidates:
            source = run_dir / name
            if source.exists():
                shutil.copy2(source, target_dir / name)
                copied.append(str((target_dir / name).relative_to(REPO_ROOT)))
                break
    return copied


def main() -> None:
    args = parse_args()
    data_path = resolve_data(args.data)
    device = pick_device(args.device)

    from ultralytics import YOLO

    print(f"数据配置：{data_path}")
    print(f"基础权重：{args.model}    设备：{device}")
    print(f"训练参数：epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} lr0={args.lr0}")

    model = YOLO(args.model)
    started = time.perf_counter()
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        device=device,
        lr0=args.lr0,
        project=str(RUN_ROOT),
        name=args.name,
        exist_ok=True,
        plots=True,
        val=True,
        verbose=True,
    )
    elapsed = round(time.perf_counter() - started, 1)

    run_dir = RUN_ROOT / args.name
    best_weight = Path(model.trainer.best)
    if not best_weight.exists():
        raise SystemExit(f"训练结束但没有找到最优权重：{best_weight}")

    MODEL_DIR.mkdir(exist_ok=True)
    deployed_weight = MODEL_DIR / f"{args.name}.pt"
    shutil.copy2(best_weight, deployed_weight)

    print("\n在验证集上复核最优权重…")
    metrics = YOLO(str(deployed_weight)).val(
        data=str(data_path), split="val", device=device, verbose=False
    )
    report = extract_metrics(metrics)

    artifacts = copy_artifacts(run_dir, ARTIFACT_ROOT / args.name)
    epochs_done = int(getattr(model.trainer, "epoch", args.epochs - 1)) + 1
    summary = {
        "experiment": args.name,
        "created_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "model": args.model,
            "data": str(data_path.relative_to(REPO_ROOT)),
            "epochs_planned": args.epochs,
            "epochs_done": epochs_done,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "lr0": args.lr0,
            "seed": args.seed,
            "device": device,
        },
        "train_seconds": elapsed,
        "weight": str(deployed_weight.relative_to(REPO_ROOT)),
        "artifacts": artifacts,
        **report,
    }
    metrics_path = MODEL_DIR / f"{args.name}.metrics.json"
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 46)
    print(f"训练完成，耗时 {elapsed / 60:.1f} 分钟，实际 {epochs_done} epoch")
    print(f"权重      : {deployed_weight.relative_to(REPO_ROOT)}")
    print(f"指标      : {metrics_path.relative_to(REPO_ROOT)}")
    print(f"mAP50     : {report['overall']['mAP50']}")
    print(f"mAP50-95  : {report['overall']['mAP50-95']}")
    for name, values in report["per_class"].items():
        print(f"  {name:8s} mAP50-95={values['mAP50-95']:.3f}  P={values['precision']:.3f}  R={values['recall']:.3f}")
    print("=" * 46)
    print(
        "\n切换到自训模型：\n"
        f"  YOLO_MODEL={deployed_weight.relative_to(REPO_ROOT).as_posix()} python app.py"
    )


if __name__ == "__main__":
    main()
