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

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

MODEL_LOCK = threading.Lock()
MODEL = None
ACTIVE_DEVICE: str | int = "cpu"
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
                device TEXT NOT NULL
            )
            """
        )


def get_model() -> YOLO:
    global MODEL, ACTIVE_DEVICE
    if MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"没有找到模型文件：{MODEL_PATH.name}。请将 yolo11m.pt 放到项目根目录。"
            )
        ACTIVE_DEVICE = choose_device()
        MODEL = YOLO(str(MODEL_PATH))
        app.logger.info("Loaded %s on device=%s", MODEL_PATH.name, ACTIVE_DEVICE)
    return MODEL


def model_names() -> list[dict[str, object]]:
    names = get_model().names
    if isinstance(names, list):
        return [
            {"id": index, "name": name}
            for index, name in enumerate(names)
            if index in CORE_CLASS_IDS
        ]
    return [
        {"id": int(index), "name": name}
        for index, name in sorted(names.items())
        if int(index) in CORE_CLASS_IDS
    ]


def clamp_float(value: str | None, minimum: float, maximum: float, default: float) -> float:
    try:
        parsed = float(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def selected_class_ids(values: list[str]) -> list[int] | None:
    valid_ids = {item["id"] for item in model_names()}
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


def selected_class_names(class_ids: list[int] | None) -> list[str]:
    names = {int(item["id"]): str(item["name"]) for item in model_names()}
    return [names[class_id] for class_id in class_ids or [] if class_id in names]


def media_type_for(filename: str) -> str | None:
    extension = Path(filename).suffix.lower()
    if extension in ALLOWED_IMAGE_EXTENSIONS:
        return "image"
    if extension in ALLOWED_VIDEO_EXTENSIONS:
        return "video"
    return None


def run_prediction(source: object, conf: float, iou: float, class_ids: list[int] | None):
    """Run inference and fall back to CPU if an MPS-only operation fails."""
    global ACTIVE_DEVICE
    model = get_model()
    options = {
        "source": source,
        "conf": conf,
        "iou": iou,
        "classes": class_ids,
        "device": ACTIVE_DEVICE,
        "verbose": False,
    }
    try:
        with MODEL_LOCK:
            return model.predict(**options)
    except Exception:
        if ACTIVE_DEVICE != "mps":
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


def process_image(input_path: Path, output_path: Path, conf: float, iou: float, class_ids: list[int] | None):
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError("无法读取图片文件，请确认图片格式正确。")
    results = run_prediction(str(input_path), conf, iou, class_ids)
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
            results = run_prediction(frame, conf, iou, class_ids)
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
            progress_callback=report_progress,
        )
        elapsed = round(time.perf_counter() - started, 2)
        selected_names = selected_class_names(class_ids)
        created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.execute(
                """
                INSERT INTO detections
                (id, original_name, media_type, result_filename, created_at, conf, iou,
                 selected_classes, detection_count, elapsed_seconds, device)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    str(ACTIVE_DEVICE),
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
                    device=str(ACTIVE_DEVICE),
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
    return render_template(
        "index.html",
        classes=model_names(),
        model_name=MODEL_PATH.name,
        device=ACTIVE_DEVICE if MODEL is not None else choose_device(),
    )


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model": MODEL_PATH.name,
            "device": ACTIVE_DEVICE if MODEL is not None else choose_device(),
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
    class_ids = selected_class_ids(request.form.getlist("classes"))
    if class_ids is None:
        class_ids = [item["id"] for item in model_names()]
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
                "device": str(ACTIVE_DEVICE),
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
            ),
            daemon=True,
        )
        worker.start()
        return jsonify(dict(JOBS[record_id])), 202

    try:
        if media_type == "image":
            detections, processed_frames = process_image(
                input_path, result_path, conf, iou, class_ids
            )
            detection_count = len(detections)
            detection_details = detections
        else:
            detection_count, processed_frames, total_frames = process_video(
                input_path, result_path, conf, iou, class_ids
            )
            detection_details = []
        elapsed = round(time.perf_counter() - started, 2)
    except Exception as error:
        result_path.unlink(missing_ok=True)
        app.logger.exception("Detection failed")
        return jsonify({"error": str(error)}), 500
    finally:
        input_path.unlink(missing_ok=True)

    selected_names = selected_class_names(class_ids)
    created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT INTO detections
            (id, original_name, media_type, result_filename, created_at, conf, iou,
             selected_classes, detection_count, elapsed_seconds, device)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                str(ACTIVE_DEVICE),
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
        "device": str(ACTIVE_DEVICE),
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
