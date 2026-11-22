"""模型发现、推理设备选择与图片/视频检测流程。"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = Path(os.getenv("YOLO_MODEL", BASE_DIR / "yolo11m.pt"))

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# 课程展示用的核心类别，避免把 COCO 的全部 80 个类别塞进界面。
CORE_CLASS_IDS = [0, 1, 2, 3, 5, 7, 14, 15, 16, 39, 63, 67]
CORE_CLASS_THRESHOLD = 20

MODEL_LABELS = {
    "yolo11m.pt": "YOLO11m · COCO 预训练",
    "yolo11n.pt": "YOLO11n · COCO 轻量预训练",
    "yolov8n.pt": "YOLOv8n · COCO 轻量预训练",
    "yolov8s.pt": "YOLOv8s · COCO 预训练",
}

MODEL_LOCK = threading.Lock()
MODEL_LOAD_LOCK = threading.Lock()
MODELS: dict[str, YOLO] = {}
ACTIVE_DEVICE: str | int | None = None


def choose_device() -> str | int:
    """Prefer CUDA, then Apple MPS, then CPU."""
    requested = os.getenv("YOLO_DEVICE", "auto").strip().lower()
    if requested != "auto":
        return int(requested) if requested.isdigit() else requested
    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def current_device() -> str | int:
    global ACTIVE_DEVICE
    if ACTIVE_DEVICE is None:
        ACTIVE_DEVICE = choose_device()
    return ACTIVE_DEVICE


def available_models() -> list[dict[str, object]]:
    """Enumerate configured and project-local model weights."""
    candidates: list[Path] = [Path(MODEL_PATH)]
    candidates.extend([BASE_DIR / "yolo11m.pt", BASE_DIR / "yolo11n.pt"])
    if MODEL_DIR.is_dir():
        candidates.extend(sorted(MODEL_DIR.glob("*.pt")))

    seen: set[Path] = set()
    catalog: list[dict[str, object]] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            model_id = resolved.relative_to(BASE_DIR).as_posix()
        except ValueError:
            model_id = resolved.as_posix()
        catalog.append(
            {
                "id": model_id,
                "label": MODEL_LABELS.get(model_id, f"{resolved.stem} · 自训权重"),
                "size_mb": round(resolved.stat().st_size / 1048576, 1),
                "path": resolved,
            }
        )
    return catalog


def public_models() -> list[dict[str, object]]:
    return [
        {"id": item["id"], "label": item["label"], "size_mb": item["size_mb"]}
        for item in available_models()
    ]


def resolve_model(model_id: str | None) -> dict[str, object]:
    """Resolve a client model ID against the available model catalog."""
    catalog = available_models()
    if not catalog:
        raise FileNotFoundError(
            "没有找到可用的模型权重。请把 yolo11m.pt 放到项目根目录，或用 training/ 训练一个。"
        )
    if not model_id:
        return catalog[0]
    for item in catalog:
        if item["id"] == model_id:
            return item
    raise ValueError("未知的模型标识，请刷新页面后重试。")


def get_model(model_id: str | None = None) -> YOLO:
    """Load and cache a model on demand."""
    entry = resolve_model(model_id)
    key = str(entry["id"])
    cached = MODELS.get(key)
    if cached is not None:
        return cached
    with MODEL_LOAD_LOCK:
        cached = MODELS.get(key)
        if cached is not None:
            return cached
        model = YOLO(str(entry["path"]))
        MODELS[key] = model
        logging.getLogger("app").info("Loaded %s on device=%s", key, current_device())
        return model


def model_names(model_id: str | None = None) -> list[dict[str, object]]:
    """Return selectable classes, filtering large COCO models to core classes."""
    names = get_model(model_id).names
    if isinstance(names, list):
        items = [{"id": index, "name": name} for index, name in enumerate(names)]
    else:
        items = [{"id": int(index), "name": name} for index, name in sorted(names.items())]
    if len(items) > CORE_CLASS_THRESHOLD:
        return [item for item in items if int(item["id"]) in CORE_CLASS_IDS]
    return items


def clamp_float(value: str | None, minimum: float, maximum: float, default: float) -> float:
    try:
        parsed = float(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def selected_class_ids(values: list[str], model_id: str | None = None) -> list[int] | None:
    valid_ids = {item["id"] for item in model_names(model_id)}
    parsed: list[int] = []
    for value in values:
        try:
            class_id = int(value)
        except (TypeError, ValueError):
            continue
        if class_id in valid_ids:
            parsed.append(class_id)
    unique_ids = sorted(set(parsed))
    return unique_ids or None


def selected_class_names(
    class_ids: list[int] | None, model_id: str | None = None
) -> list[str]:
    names = {int(item["id"]): str(item["name"]) for item in model_names(model_id)}
    return [names[class_id] for class_id in class_ids or [] if class_id in names]


def media_type_for(filename: str) -> str | None:
    extension = Path(filename).suffix.lower()
    if extension in ALLOWED_IMAGE_EXTENSIONS:
        return "image"
    if extension in ALLOWED_VIDEO_EXTENSIONS:
        return "video"
    return None


def run_prediction(
    source: object,
    conf: float,
    iou: float,
    class_ids: list[int] | None,
    model_id: str | None = None,
):
    """Run inference and fall back to CPU if an MPS-only operation fails."""
    global ACTIVE_DEVICE
    model = get_model(model_id)
    options = {
        "source": source,
        "conf": conf,
        "iou": iou,
        "classes": class_ids,
        "device": current_device(),
        "verbose": False,
    }
    try:
        with MODEL_LOCK:
            return model.predict(**options)
    except Exception:
        if current_device() != "mps":
            raise
        logging.getLogger("app").exception("MPS inference failed; falling back to CPU")
        ACTIVE_DEVICE = "cpu"
        options["device"] = "cpu"
        with MODEL_LOCK:
            return model.predict(**options)


def detections_from_result(result) -> list[dict[str, object]]:
    detections: list[dict[str, object]] = []
    if result.boxes is None:
        return detections
    names = result.names
    for box, confidence, class_tensor in zip(
        result.boxes.xyxy,
        result.boxes.conf,
        result.boxes.cls,
    ):
        class_id = int(class_tensor.item())
        if isinstance(names, dict):
            name = names.get(class_id, str(class_id))
        elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            name = names[class_id]
        else:
            name = str(class_id)
        detections.append(
            {
                "class_id": class_id,
                "class_name": str(name),
                "confidence": round(float(confidence.item()), 4),
                "box": [round(float(value), 2) for value in box.tolist()],
            }
        )
    return detections


def process_image(
    input_path: Path,
    output_path: Path,
    conf: float,
    iou: float,
    class_ids: list[int] | None,
    model_id: str | None = None,
):
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError("无法读取图片文件，请确认图片格式正确。")
    results = run_prediction(str(input_path), conf, iou, class_ids, model_id)
    result = results[0]
    annotated = result.plot()
    if not cv2.imwrite(str(output_path), annotated):
        raise OSError("检测结果保存失败。")
    return detections_from_result(result), 1


def _mux_original_audio(input_path: Path, silent_video_path: Path, output_path: Path) -> None:
    """Mux the source audio track onto the annotated video using bundled FFmpeg."""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg = get_ffmpeg_exe()
    except Exception as error:
        raise RuntimeError(
            "保留视频声音需要 FFmpeg，请先运行 python -m pip install -r requirements.txt。"
        ) from error

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(silent_video_path),
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        output_path.unlink(missing_ok=True)
        detail = completed.stderr.strip().splitlines()
        message = detail[-1] if detail else "未知的 FFmpeg 错误"
        raise RuntimeError(f"合并视频音轨失败：{message}")


def process_video(
    input_path: Path,
    output_path: Path,
    conf: float,
    iou: float,
    class_ids: list[int] | None,
    model_id: str | None = None,
    progress_callback=None,
):
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("无法打开视频文件，请换一个常见格式的视频。")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("无法获取视频尺寸。")

    silent_video_path = output_path.with_name(
        f"{output_path.stem}_silent{output_path.suffix}"
    )
    silent_video_path.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(silent_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        silent_video_path.unlink(missing_ok=True)
        raise OSError("无法创建输出视频，可能是当前系统缺少视频编码器。")

    processed_frames = 0
    detection_count = 0
    try:
        try:
            if progress_callback:
                progress_callback(0, total_frames)
            while True:
                success, frame = capture.read()
                if not success:
                    break
                results = run_prediction(frame, conf, iou, class_ids, model_id)
                result = results[0]
                annotated = result.plot()
                if annotated.shape[1] != width or annotated.shape[0] != height:
                    annotated = cv2.resize(annotated, (width, height))
                writer.write(annotated)
                processed_frames += 1
                detection_count += len(detections_from_result(result))
                if progress_callback:
                    progress_callback(processed_frames, total_frames)
        finally:
            capture.release()
            writer.release()
        if processed_frames == 0:
            raise ValueError("视频没有可读取的画面。")
        _mux_original_audio(input_path, silent_video_path, output_path)
    finally:
        silent_video_path.unlink(missing_ok=True)
    return detection_count, processed_frames, total_frames
