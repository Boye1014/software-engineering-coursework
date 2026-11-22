"""Flask routes and single-worker video task queue."""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, TypedDict

from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

import inference
import storage


class RuntimeConfig(TypedDict):
    UPLOAD_DIR: Path
    RESULT_DIR: Path
    DATABASE_PATH: Path


@dataclass(frozen=True)
class VideoTask:
    job_id: str
    original_name: str
    input_path: Path
    result_path: Path
    result_filename: str
    conf: float
    iou: float
    class_ids: list[int]
    model_id: str
    model_label: str
    database_path: Path


JOBS: dict[str, dict[str, object]] = {}
JOBS_LOCK = threading.Lock()

# An unbounded waiting queue feeds exactly one video worker.
_VIDEO_QUEUE: queue.Queue[VideoTask] = queue.Queue()
_ENQUEUE_LOCK = threading.Lock()
_WORKER_LOCK = threading.Lock()
_VIDEO_WORKER: threading.Thread | None = None


def _refresh_queue_positions() -> None:
    with _VIDEO_QUEUE.mutex:
        pending = list(_VIDEO_QUEUE.queue)
    with JOBS_LOCK:
        for position, task in enumerate(pending, start=1):
            job = JOBS.get(task.job_id)
            if job and job.get("status") == "queued":
                job["queue_position"] = position


def _process_video_task(task: VideoTask) -> None:
    started = time.perf_counter()

    def report_progress(processed_frames: int, total_frames: int) -> None:
        progress = round(processed_frames / total_frames * 100) if total_frames else 0
        with JOBS_LOCK:
            job = JOBS.get(task.job_id)
            if job:
                job.update(
                    status="processing",
                    queue_position=0,
                    progress=min(progress, 100),
                    processed_frames=processed_frames,
                    total_frames=total_frames,
                )

    try:
        detection_count, processed_frames, total_frames = inference.process_video(
            task.input_path,
            task.result_path,
            task.conf,
            task.iou,
            task.class_ids,
            task.model_id,
            progress_callback=report_progress,
        )
        elapsed = round(time.perf_counter() - started, 2)
        selected_names = inference.selected_class_names(task.class_ids, task.model_id)
        created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        storage.insert_detection(
            task.database_path,
            record_id=task.job_id,
            original_name=task.original_name,
            media_type="video",
            result_filename=task.result_filename,
            created_at=created_at,
            conf=task.conf,
            iou=task.iou,
            selected_classes=selected_names,
            detection_count=detection_count,
            elapsed_seconds=elapsed,
            device=str(inference.current_device()),
            model=task.model_label,
        )

        with JOBS_LOCK:
            job = JOBS.get(task.job_id)
            if job:
                job.update(
                    status="complete",
                    queue_position=0,
                    progress=100,
                    processed_frames=processed_frames,
                    total_frames=total_frames,
                    result_url=f"/media/{task.result_filename}",
                    download_url=f"/download/{task.result_filename}",
                    detection_count=detection_count,
                    elapsed_seconds=elapsed,
                    device=str(inference.current_device()),
                    model=task.model_label,
                    model_id=task.model_id,
                    selected_classes=selected_names,
                    detections=[],
                )
    except Exception as error:
        task.result_path.unlink(missing_ok=True)
        logging.getLogger("app").exception("Video detection failed")
        with JOBS_LOCK:
            job = JOBS.get(task.job_id)
            if job:
                job.update(status="error", queue_position=0, error=str(error))
    finally:
        task.input_path.unlink(missing_ok=True)


def _video_worker() -> None:
    while True:
        task = _VIDEO_QUEUE.get()
        try:
            with JOBS_LOCK:
                job = JOBS.get(task.job_id)
                if job:
                    job.update(status="processing", queue_position=0)
            _refresh_queue_positions()
            _process_video_task(task)
        except Exception as error:
            logging.getLogger("app").exception("Unexpected video worker failure")
            with JOBS_LOCK:
                job = JOBS.get(task.job_id)
                if job:
                    job.update(status="error", queue_position=0, error=str(error))
            for path in (task.input_path, task.result_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logging.getLogger("app").exception("Could not clean video task file %s", path)
        finally:
            _VIDEO_QUEUE.task_done()
            _refresh_queue_positions()


def _enqueue_video_task(task: VideoTask) -> None:
    global _VIDEO_WORKER
    with _ENQUEUE_LOCK:
        position = _VIDEO_QUEUE.qsize() + 1
        with JOBS_LOCK:
            job = JOBS.get(task.job_id)
            if job:
                job["queue_position"] = position
        _VIDEO_QUEUE.put_nowait(task)
        with _WORKER_LOCK:
            if _VIDEO_WORKER is None or not _VIDEO_WORKER.is_alive():
                _VIDEO_WORKER = threading.Thread(
                    target=_video_worker,
                    name="video-detection-worker",
                    daemon=True,
                )
                _VIDEO_WORKER.start()
    _refresh_queue_positions()


def register_routes(app: Flask, runtime_config: Callable[[], RuntimeConfig]) -> None:
    """Register HTTP endpoints using the app's current storage paths."""

    def safe_result_path(filename: str) -> Path:
        path = storage.result_path(runtime_config()["RESULT_DIR"], filename)
        if path is None:
            abort(404)
        return path

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify({"error": "文件过大，单个文件不能超过 512 MB。"}), 413

    @app.get("/")
    def index():
        models = inference.public_models()
        active = models[0] if models else {"id": "", "label": "未找到模型", "size_mb": 0}
        classes = inference.model_names(active["id"]) if models else []
        return render_template(
            "index.html",
            classes=classes,
            models=models,
            active_model=active,
            device=inference.current_device(),
        )

    @app.get("/api/classes")
    def classes_for_model():
        try:
            entry = inference.resolve_model(request.args.get("model"))
            items = inference.model_names(str(entry["id"]))
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
        models = inference.public_models()
        active = models[0] if models else {"id": "", "label": "未找到模型"}
        return jsonify(
            {
                "status": "ok",
                "model": active["id"],
                "model_label": active["label"],
                "device": str(inference.current_device()),
                "models": models,
            }
        )

    @app.get("/api/history")
    def history():
        config = runtime_config()
        items = storage.list_detections(config["DATABASE_PATH"])
        for item in items:
            filename = item.pop("result_filename")
            item["result_url"] = f"/media/{filename}"
            item["download_url"] = f"/download/{filename}"
        return jsonify(items)

    @app.delete("/api/history/<record_id>")
    def delete_history(record_id: str):
        config = runtime_config()
        deleted = storage.delete_detection(
            config["DATABASE_PATH"], config["RESULT_DIR"], record_id
        )
        if not deleted:
            return jsonify({"error": "记录不存在。"}), 404
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

        media_type = inference.media_type_for(uploaded.filename)
        if media_type is None:
            return jsonify({"error": "只支持 JPG、PNG、WEBP、MP4、MOV、AVI、MKV、WEBM 文件。"}), 400

        conf = inference.clamp_float(request.form.get("conf"), 0.01, 0.99, 0.25)
        iou = inference.clamp_float(request.form.get("iou"), 0.01, 0.99, 0.45)
        try:
            model_entry = inference.resolve_model(request.form.get("model"))
            model_id = str(model_entry["id"])
            model_label = str(model_entry["label"])
            class_ids = inference.selected_class_ids(request.form.getlist("classes"), model_id)
            if class_ids is None:
                class_ids = [item["id"] for item in inference.model_names(model_id)]
        except (FileNotFoundError, ValueError) as error:
            return jsonify({"error": str(error)}), 400

        config = runtime_config()
        original_name = secure_filename(uploaded.filename) or "upload"
        record_id = uuid.uuid4().hex
        extension = Path(original_name).suffix.lower() or (
            ".jpg" if media_type == "image" else ".mp4"
        )
        input_path = config["UPLOAD_DIR"] / f"{record_id}{extension}"
        uploaded.save(input_path)

        result_extension = ".jpg" if media_type == "image" else ".mp4"
        result_filename = f"{record_id}_result{result_extension}"
        result_path = config["RESULT_DIR"] / result_filename
        started = time.perf_counter()

        # Load once at submission time so missing or broken weights fail immediately.
        inference.get_model(model_id)

        if media_type == "video":
            job = {
                "id": record_id,
                "job_id": record_id,
                "media_type": "video",
                "original_name": original_name,
                "status": "queued",
                "queue_position": 0,
                "progress": 0,
                "processed_frames": 0,
                "total_frames": 0,
                "device": str(inference.current_device()),
                "model": model_label,
                "model_id": model_id,
            }
            with JOBS_LOCK:
                JOBS[record_id] = job
            task = VideoTask(
                job_id=record_id,
                original_name=original_name,
                input_path=input_path,
                result_path=result_path,
                result_filename=result_filename,
                conf=conf,
                iou=iou,
                class_ids=class_ids,
                model_id=model_id,
                model_label=model_label,
                database_path=config["DATABASE_PATH"],
            )
            _enqueue_video_task(task)
            with JOBS_LOCK:
                response = dict(JOBS[record_id])
            return jsonify(response), 202

        try:
            detections, _processed_frames = inference.process_image(
                input_path, result_path, conf, iou, class_ids, model_id
            )
            detection_count = len(detections)
            elapsed = round(time.perf_counter() - started, 2)
        except Exception as error:
            result_path.unlink(missing_ok=True)
            app.logger.exception("Detection failed")
            return jsonify({"error": str(error)}), 500
        finally:
            input_path.unlink(missing_ok=True)

        selected_names = inference.selected_class_names(class_ids, model_id)
        created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        storage.insert_detection(
            config["DATABASE_PATH"],
            record_id=record_id,
            original_name=original_name,
            media_type=media_type,
            result_filename=result_filename,
            created_at=created_at,
            conf=conf,
            iou=iou,
            selected_classes=selected_names,
            detection_count=detection_count,
            elapsed_seconds=elapsed,
            device=str(inference.current_device()),
            model=model_label,
        )

        return jsonify(
            {
                "id": record_id,
                "media_type": media_type,
                "original_name": original_name,
                "result_url": f"/media/{result_filename}",
                "download_url": f"/download/{result_filename}",
                "detection_count": detection_count,
                "elapsed_seconds": elapsed,
                "device": str(inference.current_device()),
                "model": model_label,
                "model_id": model_id,
                "selected_classes": selected_names,
                "detections": detections,
            }
        )

    @app.get("/media/<path:filename>")
    def media(filename: str):
        safe_result_path(filename)
        return send_from_directory(
            runtime_config()["RESULT_DIR"], Path(filename).name, as_attachment=False
        )

    @app.get("/download/<path:filename>")
    def download(filename: str):
        safe_result_path(filename)
        safe_name = Path(filename).name
        return send_from_directory(
            runtime_config()["RESULT_DIR"],
            safe_name,
            as_attachment=True,
            download_name=safe_name,
        )

    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith(("/api/", "/media/", "/download/")):
            return jsonify({"error": "资源不存在。"}), 404
        return redirect("/")
