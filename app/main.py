"""
FastAPI application — Employee Activity Detection
All routes: upload, video processing, live WebSocket, activity log, alerts, export.
"""

import base64
import collections
import csv
import io
import logging
import os
import time
import uuid
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse

from .inference import create_detector, get_detector
from .face_recognition import get_face_recognizer

# ── paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
PHOTOS_DIR = MODEL_DIR / "employee_photos"

for d in (STATIC_DIR, UPLOAD_DIR, OUTPUT_DIR, MODEL_DIR, PHOTOS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── in-memory stores ────────────────────────────────────────────────────
activity_log: List[Dict] = []                       # last N detection events
alert_configs: List[Dict] = []                      # [{label, threshold}]
triggered_alerts: List[Dict] = []                   # recent alerts
video_jobs: Dict[str, Dict] = {}                    # job_id → {progress, output_path, status, detections}
MAX_LOG = 5000

# ── performance stats ───────────────────────────────────────────────────
recent_inference_times = collections.deque(maxlen=100)
frames_processed = 0
logger = logging.getLogger(__name__)

# ── app ──────────────────────────────────────────────────────────────────
app = FastAPI(title="ActivityWatch — Employee Activity Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ──────────────────────────────────────────────────────────────
def _append_log(mode: str, label: str, confidence: float, employee_id: str = "unknown", employee_name: str = "Unknown"):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "label": label,
        "confidence": round(confidence, 3),
        "employee_id": employee_id,
        "employee_name": employee_name,
    }
    activity_log.append(entry)
    if len(activity_log) > MAX_LOG:
        del activity_log[: len(activity_log) - MAX_LOG]
    _check_alerts(entry)


def _check_alerts(entry: Dict):
    for cfg in alert_configs:
        if cfg.get("enabled", True) and entry["label"] == cfg["label"]:
            if entry["confidence"] >= cfg["threshold"]:
                alert = {
                    "id": str(uuid.uuid4())[:8],
                    "timestamp": entry["timestamp"],
                    "label": entry["label"],
                    "confidence": entry["confidence"],
                }
                triggered_alerts.append(alert)
                if len(triggered_alerts) > 500:
                    del triggered_alerts[:100]


# ── background video processing ──────────────────────────────────────────
def _process_video(job_id: str, input_path: str):
    """Process an uploaded video frame-by-frame (runs in background thread)."""
    detector = create_detector()
    cap = None
    writer = None
    raw_output_path = str(OUTPUT_DIR / f"{job_id}_raw.mp4")
    output_path = str(OUTPUT_DIR / f"{job_id}.mp4")
    process_started_at = time.time()
    frame_idx = 0
    total_frames = 1
    job_detections: List[Dict] = []

    try:
        logger.info("[video:%s] Opening input video: %s", job_id, input_path)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            video_jobs[job_id]["status"] = "error"
            logger.error("[video:%s] Failed to open input: %s", job_id, input_path)
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "[video:%s] Input opened frames=%s fps=%.3f size=%sx%s",
            job_id,
            total_frames,
            fps,
            w,
            h,
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            video_jobs[job_id]["status"] = "error"
            logger.error("[video:%s] Failed to create VideoWriter: %s", job_id, raw_output_path)
            return
        logger.info("[video:%s] VideoWriter created: %s", job_id, raw_output_path)

        process_every_n_frames = max(1, round(fps / 10))
        last_annotated_frame = None

        global frames_processed
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % process_every_n_frames == 0:
                t0 = time.time()
                annotated, dets = detector.detect_frame(frame)
                elapsed_ms = (time.time() - t0) * 1000.0
                recent_inference_times.append(elapsed_ms)
                frames_processed += 1
                last_annotated_frame = annotated
                for d in dets:
                    _append_log("Video", d["label"], d["confidence"], d.get("employee_id", "unknown"), d.get("employee_name", "Unknown"))
                    job_detections.append(d)

            if last_annotated_frame is not None:
                writer.write(last_annotated_frame)

            frame_idx += 1
            progress = min(round(frame_idx / total_frames * 100, 1), 100.0)
            video_jobs[job_id]["progress"] = max(video_jobs[job_id]["progress"], progress)

            if frame_idx == 1 or frame_idx % 30 == 0 or frame_idx == total_frames:
                elapsed = max(time.time() - process_started_at, 1e-6)
                logger.info(
                    "[video:%s] frame=%s/%s progress=%.1f speed=%.2f fps",
                    job_id,
                    frame_idx,
                    total_frames,
                    video_jobs[job_id]["progress"],
                    frame_idx / elapsed,
                )

        logger.info("[video:%s] Frame processing complete processed_frames=%s", job_id, frame_idx)

        if cap is not None:
            cap.release()
            cap = None
        if writer is not None:
            writer.release()
            writer = None
        logger.info("[video:%s] Released VideoCapture and VideoWriter", job_id)

        if not os.path.exists(raw_output_path):
            video_jobs[job_id]["status"] = "error"
            logger.error("[video:%s] Raw output missing: %s", job_id, raw_output_path)
            return

        raw_size = os.path.getsize(raw_output_path)
        logger.info("[video:%s] Raw output ready path=%s size=%s", job_id, raw_output_path, raw_size)
        if raw_size <= 0:
            video_jobs[job_id]["status"] = "error"
            logger.error("[video:%s] Raw output is empty: %s", job_id, raw_output_path)
            return

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            raw_output_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        logger.info("[video:%s] Running FFmpeg: %s", job_id, " ".join(ffmpeg_cmd))
        ffmpeg_result = subprocess.run(
            ffmpeg_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("[video:%s] FFmpeg stdout: %s", job_id, (ffmpeg_result.stdout or "").strip() or "<empty>")
        logger.info("[video:%s] FFmpeg stderr: %s", job_id, (ffmpeg_result.stderr or "").strip() or "<empty>")

        if not os.path.exists(output_path):
            video_jobs[job_id]["status"] = "error"
            logger.error("[video:%s] Converted output missing: %s", job_id, output_path)
            return

        output_size = os.path.getsize(output_path)
        logger.info("[video:%s] Final output created path=%s size=%s", job_id, output_path, output_size)
        if output_size <= 0:
            video_jobs[job_id]["status"] = "error"
            logger.error("[video:%s] Converted output is empty: %s", job_id, output_path)
            return

        if os.path.exists(raw_output_path):
            os.remove(raw_output_path)
            logger.info("[video:%s] Removed raw output: %s", job_id, raw_output_path)

    except FileNotFoundError:
        video_jobs[job_id]["status"] = "error"
        logger.exception("[video:%s] FFmpeg not found", job_id)
        return
    
    except subprocess.CalledProcessError as e:
        video_jobs[job_id]["status"] = "error"
        logger.error(
            "[video:%s] FFmpeg conversion failed stdout=%s stderr=%s",
            job_id,
            (e.stdout or b"").decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or "<empty>"),
            (e.stderr or b"").decode(errors="ignore") if isinstance(e.stderr, bytes) else (e.stderr or "<empty>"),
        )
        return

    except Exception:
        current_progress = video_jobs.get(job_id, {}).get("progress", 0.0)
        elapsed = max(time.time() - process_started_at, 1e-6)
        logger.exception(
            "[video:%s] Processing failed frame=%s/%s progress=%.1f speed=%.2f fps",
            job_id,
            frame_idx,
            total_frames,
            current_progress,
            frame_idx / elapsed,
        )
        video_jobs[job_id]["status"] = "error"
        return

    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()

    video_jobs[job_id].update(
        {
            "progress": 100.0,
            "status": "done",
            "output_path": output_path,
            "detections": job_detections,
        }
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━  ROUTES  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Upload video ─────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if file.content_type not in (
        "video/mp4",
        "video/avi",
        "video/x-msvideo",
        "video/quicktime",
    ):
        raise HTTPException(400, "Only MP4 / AVI videos are accepted.")

    job_id = str(uuid.uuid4())[:12]
    ext = Path(file.filename or "video.mp4").suffix or ".mp4"
    save_path = UPLOAD_DIR / f"{job_id}{ext}"
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    video_jobs[job_id] = {
        "progress": 0.0,
        "status": "processing",
        "output_path": None,
        "detections": [],
    }
    background_tasks.add_task(_process_video, job_id, str(save_path))
    return {"job_id": job_id}


# ── Video status ─────────────────────────────────────────────────────────
@app.get("/video-status/{job_id}")
async def video_status(job_id: str):
    job = video_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    resp = {"progress": job["progress"], "status": job["status"]}
    if job["status"] == "done":
        resp["output_url"] = f"/video-result/{job_id}"
        # summarise detections
        summary: Dict[str, Dict] = {}
        for d in job.get("detections", []):
            lbl = d["label"]
            if lbl not in summary:
                summary[lbl] = {"count": 0, "total_conf": 0.0}
            summary[lbl]["count"] += 1
            summary[lbl]["total_conf"] += d["confidence"]
        resp["summary"] = [
            {
                "label": lbl,
                "count": v["count"],
                "avg_confidence": round(v["total_conf"] / v["count"], 3),
            }
            for lbl, v in sorted(summary.items(), key=lambda x: -x[1]["count"])
        ]
    return resp


# ── Stream annotated video ───────────────────────────────────────────────
@app.get("/video-result/{job_id}")
async def video_result(job_id: str):
    job = video_jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Video not ready")
    return FileResponse(
        job["output_path"],
        media_type="video/mp4",
        filename=f"annotated_{job_id}.mp4",
    )


# ── WebSocket live feed ──────────────────────────────────────────────────
@app.websocket("/video-stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    detector = get_detector()
    
    last_processed_time = 0.0
    last_result = None
    MIN_FRAME_INTERVAL = 0.08
    global frames_processed
    
    try:
        while True:
            data = await websocket.receive_text()
            t0 = time.time()
            
            if t0 - last_processed_time < MIN_FRAME_INTERVAL and last_result is not None:
                await websocket.send_json(last_result)
                continue

            # decode base64 JPEG
            try:
                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                await websocket.send_json({"error": "Invalid frame"})
                continue

            if frame is None:
                await websocket.send_json({"error": "Could not decode frame"})
                continue

            inf_start = time.time()
            annotated, dets = detector.detect_frame(frame)
            elapsed_ms = (time.time() - inf_start) * 1000.0
            recent_inference_times.append(elapsed_ms)
            frames_processed += 1

            # log detections
            for d in dets:
                _append_log("Live", d["label"], d["confidence"], d.get("employee_id", "unknown"), d.get("employee_name", "Unknown"))

            # encode annotated frame back to base64 JPEG
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64_frame = base64.b64encode(buf.tobytes()).decode("utf-8")

            elapsed = time.time() - t0
            fps_val = round(1.0 / elapsed, 1) if elapsed > 0 else 0

            last_result = {
                "boxes": dets,
                "annotated_frame": b64_frame,
                "fps": fps_val,
            }
            last_processed_time = time.time()

            await websocket.send_json(last_result)
    except WebSocketDisconnect:
        pass


# ── Activity log ─────────────────────────────────────────────────────────
@app.get("/activity-log")
async def get_activity_log():
    return activity_log[-500:]


# ── Export CSV ───────────────────────────────────────────────────────────
@app.get("/export/csv")
async def export_csv():
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=["timestamp", "mode", "label", "confidence", "employee_id", "employee_name"]
    )
    writer.writeheader()
    for row in activity_log:
        writer.writerow(row)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=activity_log.csv"},
    )


# ── Alerts config ───────────────────────────────────────────────────────
@app.post("/alerts/config")
async def set_alert_config(configs: List[Dict]):
    global alert_configs
    alert_configs = configs
    return {"status": "ok", "count": len(configs)}


@app.get("/alerts/config")
async def get_alert_config():
    return alert_configs


# ── Alerts ───────────────────────────────────────────────────────────────
@app.get("/alerts")
async def get_alerts():
    return triggered_alerts[-100:]


# ── Employees ────────────────────────────────────────────────────────────
@app.post("/employees/enroll")
async def enroll_employee(
    employee_id: str = Form(...),
    name: str = Form(...),
    photo: UploadFile = File(...)
):
    content = await photo.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image format")
    
    fr = get_face_recognizer()
    try:
        res = fr.enroll(employee_id, name, img)
    except ValueError as e:
        raise HTTPException(400, detail="No face detected in the uploaded image")
        
    # Save first photo for preview
    if res.get("embedding_count") == 1:
        photo_path = PHOTOS_DIR / f"{employee_id}.jpg"
        cv2.imwrite(str(photo_path), img)
        
    return res

@app.post("/employees/enroll/batch")
async def enroll_employee_batch(
    employee_id: str = Form(...),
    name: str = Form(...),
    photos: List[UploadFile] = File(...)
):
    enrolled = 0
    skipped = 0
    fr = get_face_recognizer()
    
    for photo in photos:
        content = await photo.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            continue
            
        try:
            res = fr.enroll(employee_id, name, img)
            enrolled += 1
            if res.get("embedding_count") == 1:
                photo_path = PHOTOS_DIR / f"{employee_id}.jpg"
                cv2.imwrite(str(photo_path), img)
        except ValueError:
            skipped += 1
            
    return {"enrolled": enrolled, "skipped": skipped, "employee_id": employee_id, "name": name}

@app.get("/employees")
async def get_employees():
    fr = get_face_recognizer()
    return fr.get_all_employees()

@app.get("/employees/{employee_id}")
async def get_employee(employee_id: str):
    fr = get_face_recognizer()
    emp = next((e for e in fr.get_all_employees() if e["employee_id"] == employee_id), None)
    if not emp:
        raise HTTPException(404, "Employee not found")
    return emp

@app.delete("/employees/{employee_id}")
async def delete_employee(employee_id: str):
    fr = get_face_recognizer()
    if fr.delete_employee(employee_id):
        return {"status": "deleted"}
    raise HTTPException(404, "Employee not found")

@app.get("/employees/{employee_id}/preview")
async def preview_employee(employee_id: str):
    photo_path = PHOTOS_DIR / f"{employee_id}.jpg"
    if not photo_path.exists():
        raise HTTPException(404, "Photo not found")
    return FileResponse(str(photo_path), media_type="image/jpeg")


# ── Health / meta ────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# ── Performance Stats ────────────────────────────────────────────────────
@app.get("/perf-stats")
async def get_perf_stats():
    detector = get_detector()
    avg_inference = sum(recent_inference_times) / len(recent_inference_times) if recent_inference_times else 0.0
    avg_fps = 1000.0 / avg_inference if avg_inference > 0 else 0.0
    
    reqs = detector._total_activity_requests
    hits = detector._cache_hits
    hit_rate = hits / reqs if reqs > 0 else 0.0
    
    return {
        "avg_inference_ms": round(avg_inference, 1),
        "avg_fps": round(avg_fps, 1),
        "frames_processed": frames_processed,
        "cache_hit_rate": round(hit_rate, 3)
    }


# ── Serve frontend ───────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>ActivityWatch</h1><p>Frontend not found.</p>")
