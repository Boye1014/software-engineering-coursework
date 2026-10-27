from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask import abort
from flask import redirect
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = Path(os.getenv("YOLO_MODEL", BASE_DIR / "yolo11m.pt"))
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
DATABASE_PATH = BASE_DIR / "detections.db"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# 课程展示用的核心类别，避免把 COCO 的全部 80 个类别塞进界面。
CORE_CLASS_IDS = [0, 1, 2, 3, 5, 7, 14, 15, 16, 39, 63, 67]
# 类别数超过这个阈值的模型（如 COCO 80 类）才做核心类别过滤
CORE_CLASS_THRESHOLD = 20

# 常见权重的友好名称，其余权重统一显示为自训权重。
MODEL_LABELS = {
    "yolo11m.pt": "YOLO11m · COCO 预训练",
    "yolo11n.pt": "YOLO11n · COCO 轻量预训练",
    "yolov8n.pt": "YOLOv8n · COCO 轻量预训练",
    "yolov8s.pt": "YOLOv8s · COCO 预训练",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

MODEL_LOCK = threading.Lock()
MODEL_LOAD_LOCK = threading.Lock()
MODELS: dict[str, YOLO] = {}
ACTIVE_DEVICE: str | int | None = None
JOBS: dict[str, dict[str, object]] = {}
JOBS_LOCK = threading.Lock()


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
    """枚举可切换的权重：环境变量指定的、项目根目录的、models/ 目录的。"""
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
    """把前端传来的标识收敛到白名单内，避免任意路径读取。"""
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
    """按需加载并缓存模型，多个权重可以同时驻留。"""
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
        app.logger.info("Loaded %s on device=%s", key, current_device())
        return model


def init_database() -> None:
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
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
                device TEXT NOT NULL,
                model TEXT NOT NULL DEFAULT ''
            )
            """
        )
        # 旧版本数据库没有 model 字段，这里做一次就地迁移
        columns = {row[1] for row in connection.execute("PRAGMA table_info(detections)")}
        if "model" not in columns:
            connection.execute(
                "ALTER TABLE detections ADD COLUMN model TEXT NOT NULL DEFAULT ''"
            )


# 在导入时建表，保证 `flask run` / gunicorn 等启动方式也能正常读写历史记录
init_database()


def model_names(model_id: str | None = None) -> list[dict[str, object]]:
    """取模型的可选类别。

    完整 COCO 模型有 80 类，界面上只展示 CORE_CLASS_IDS 里的核心类；
    自训模型类别数不多（4 类、12 类等），直接全部展示，避免过滤后对不上。
    """
    names = get_model(model_id).names
    if isinstance(names, list):
        items = [{"id": index, "name": name} for index, name in enumerate(names)]
    else:
        items = [
            {"id": int(index), "name": name}
            for index, name in sorted(names.items())
        ]
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
        app.logger.exception("MPS inference failed; falling back to CPU")
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
        name = names[class_id] if isinstance(names, (list, dict)) else str(class_id)
        if isinstance(names, dict):
            name = names.get(class_id, str(class_id))
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

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise OSError("无法创建输出视频，可能是当前系统缺少视频编码器。")

    processed_frames = 0
    detection_count = 0
    if progress_callback:
        progress_callback(0, total_frames)
    try:
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
    return detection_count, processed_frames, total_frames


def process_video_job(
    job_id: str,
    original_name: str,
    input_path: Path,
    result_path: Path,
    result_filename: str,
    conf: float,
    iou: float,
    class_ids: list[int],
    model_id: str,
    model_label: str,
):
    started = time.perf_counter()

    def report_progress(processed_frames: int, total_frames: int):
        progress = round(processed_frames / total_frames * 100) if total_frames else 0
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job.update(
                    status="processing",
                    progress=min(progress, 100),
                    processed_frames=processed_frames,
                    total_frames=total_frames,
                )

    try:
        detection_count, processed_frames, total_frames = process_video(
            input_path,
            result_path,
            conf,
            iou,
            class_ids,
            model_id,
            progress_callback=report_progress,
        )
        elapsed = round(time.perf_counter() - started, 2)
        selected_names = selected_class_names(class_ids, model_id)
        created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.execute(
                """
                INSERT INTO detections
                (id, original_name, media_type, result_filename, created_at, conf, iou,
                 selected_classes, detection_count, elapsed_seconds, device, model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    original_name,
                    "video",
                    result_filename,
                    created_at,
                    conf,
                    iou,
                    json.dumps(selected_names, ensure_ascii=False),
                    detection_count,
                    elapsed,
                    str(current_device()),
                    model_label,
                ),
            )

        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job.update(
                    status="complete",
                    progress=100,
                    processed_frames=processed_frames,
                    total_frames=total_frames,
                    result_url=f"/media/{result_filename}",
                    download_url=f"/download/{result_filename}",
                    detection_count=detection_count,
                    elapsed_seconds=elapsed,
                    device=str(current_device()),
                    model=model_label,
                    model_id=model_id,
                    selected_classes=selected_names,
                    detections=[],
                )
    except Exception as error:
        result_path.unlink(missing_ok=True)
        app.logger.exception("Video detection failed")
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job.update(status="error", error=str(error))
    finally:
        input_path.unlink(missing_ok=True)


def safe_result_path(filename: str) -> Path:
    safe_name = Path(filename).name
    path = RESULT_DIR / safe_name
    if not path.exists():
        abort(404)
    return path


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify({"error": "文件过大，单个文件不能超过 512 MB。"}), 413


@app.get("/")
def index():
    models = public_models()
    active = models[0] if models else {"id": "", "label": "未找到模型", "size_mb": 0}
    classes = model_names(active["id"]) if models else []
    return render_template(
        "index.html",
        classes=classes,
        models=models,
        active_model=active,
        device=current_device(),
    )


@app.get("/api/classes")
def classes_for_model():
    """返回指定模型的可选类别，供前端切换模型时重建类别筛选区。"""
    try:
        entry = resolve_model(request.args.get("model"))
        items = model_names(str(entry["id"]))
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    return jsonify(
        {
            "model": entry["id"],
            "model_label": entry["label"],
            "classes": items,
            "class_count": len(items),
        }
    )


@app.get("/api/health")
def health():
    models = public_models()
    active = models[0] if models else {"id": "", "label": "未找到模型"}
    return jsonify(
        {
            "status": "ok",
            "model": active["id"],
            "model_label": active["label"],
            "device": str(current_device()),
            "models": models,
        }
    )


@app.get("/api/history")
def history():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM detections ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
    items = []
    for row in rows:
        items.append(
            {
                "id": row["id"],
                "original_name": row["original_name"],
                "media_type": row["media_type"],
                "created_at": row["created_at"],
                "conf": row["conf"],
                "iou": row["iou"],
                "selected_classes": json.loads(row["selected_classes"]),
                "detection_count": row["detection_count"],
                "elapsed_seconds": row["elapsed_seconds"],
                "device": row["device"],
                "model": row["model"] if "model" in row.keys() else "",
                "result_url": f"/media/{row['result_filename']}",
                "download_url": f"/download/{row['result_filename']}",
            }
        )
    return jsonify(items)


@app.delete("/api/history/<record_id>")
def delete_history(record_id: str):
    with sqlite3.connect(DATABASE_PATH) as connection:
        row = connection.execute(
            "SELECT result_filename FROM detections WHERE id = ?", (record_id,)
        ).fetchone()
        if row is None:
            return jsonify({"error": "记录不存在。"}), 404
        result_path = RESULT_DIR / Path(row[0]).name
        result_path.unlink(missing_ok=True)
        connection.execute("DELETE FROM detections WHERE id = ?", (record_id,))
    return jsonify({"ok": True})


@app.get("/api/detect/<job_id>")
def detect_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "检测任务不存在或已过期。"}), 404
        return jsonify(dict(job))


@app.post("/api/detect")
def detect():
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "请选择图片或视频文件。"}), 400

    media_type = media_type_for(uploaded.filename)
    if media_type is None:
        return jsonify({"error": "只支持 JPG、PNG、WEBP、MP4、MOV、AVI、MKV、WEBM 文件。"}), 400

    conf = clamp_float(request.form.get("conf"), 0.01, 0.99, 0.25)
    iou = clamp_float(request.form.get("iou"), 0.01, 0.99, 0.45)
    try:
        model_entry = resolve_model(request.form.get("model"))
        model_id = str(model_entry["id"])
        model_label = str(model_entry["label"])
        class_ids = selected_class_ids(request.form.getlist("classes"), model_id)
        if class_ids is None:
            class_ids = [item["id"] for item in model_names(model_id)]
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    original_name = secure_filename(uploaded.filename) or "upload"
    record_id = uuid.uuid4().hex
    extension = Path(original_name).suffix.lower() or (".jpg" if media_type == "image" else ".mp4")
    input_filename = f"{record_id}{extension}"
    input_path = UPLOAD_DIR / input_filename
    uploaded.save(input_path)

    result_extension = ".jpg" if media_type == "image" else ".mp4"
    result_filename = f"{record_id}_result{result_extension}"
    result_path = RESULT_DIR / result_filename
    started = time.perf_counter()

    # 主动加载一次，把“模型文件损坏”这类错误提前到请求阶段返回
    get_model(model_id)

    if media_type == "video":
        with JOBS_LOCK:
            JOBS[record_id] = {
                "id": record_id,
                "job_id": record_id,
                "media_type": "video",
                "original_name": original_name,
                "status": "queued",
                "progress": 0,
                "processed_frames": 0,
                "total_frames": 0,
                "device": str(current_device()),
                "model": model_label,
                "model_id": model_id,
            }
        worker = threading.Thread(
            target=process_video_job,
            args=(
                record_id,
                original_name,
                input_path,
                result_path,
                result_filename,
                conf,
                iou,
                class_ids,
                model_id,
                model_label,
            ),
            daemon=True,
        )
        worker.start()
        return jsonify(dict(JOBS[record_id])), 202

    try:
        if media_type == "image":
            detections, processed_frames = process_image(
                input_path, result_path, conf, iou, class_ids, model_id
            )
            detection_count = len(detections)
            detection_details = detections
        else:
            detection_count, processed_frames, total_frames = process_video(
                input_path, result_path, conf, iou, class_ids, model_id
            )
            detection_details = []
        elapsed = round(time.perf_counter() - started, 2)
    except Exception as error:
        result_path.unlink(missing_ok=True)
        app.logger.exception("Detection failed")
        return jsonify({"error": str(error)}), 500
    finally:
        input_path.unlink(missing_ok=True)

    selected_names = selected_class_names(class_ids, model_id)
    created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT INTO detections
            (id, original_name, media_type, result_filename, created_at, conf, iou,
             selected_classes, detection_count, elapsed_seconds, device, model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                original_name,
                media_type,
                result_filename,
                created_at,
                conf,
                iou,
                json.dumps(selected_names, ensure_ascii=False),
                detection_count,
                elapsed,
                str(current_device()),
                model_label,
            ),
        )

    response = {
        "id": record_id,
        "media_type": media_type,
        "original_name": original_name,
        "result_url": f"/media/{result_filename}",
        "download_url": f"/download/{result_filename}",
        "detection_count": detection_count,
        "elapsed_seconds": elapsed,
        "device": str(current_device()),
        "model": model_label,
        "model_id": model_id,
        "selected_classes": selected_names,
        "detections": detection_details,
    }
    if media_type == "video":
        response["processed_frames"] = processed_frames
        response["total_frames"] = total_frames
    return jsonify(response)


@app.get("/media/<path:filename>")
def media(filename: str):
    safe_result_path(filename)
    return send_from_directory(RESULT_DIR, Path(filename).name, as_attachment=False)


@app.get("/download/<path:filename>")
def download(filename: str):
    safe_result_path(filename)
    return send_from_directory(
        RESULT_DIR,
        Path(filename).name,
        as_attachment=True,
        download_name=Path(filename).name,
    )


@app.errorhandler(404)
def not_found(_error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "资源不存在。"}), 404
    return redirect("/")


if __name__ == "__main__":
    init_database()
    app.run(host="127.0.0.1", port=5000, debug=True)
