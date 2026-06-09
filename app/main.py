"""
FastAPI application — Employee Activity Detection
All routes: upload, video processing, live WebSocket, activity log, alerts, export.
"""

import asyncio
import base64
import csv
import io
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse

from .inference import get_detector

# ── paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

for d in (STATIC_DIR, UPLOAD_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── in-memory stores ────────────────────────────────────────────────────
activity_log: List[Dict] = []                       # last N detection events
alert_configs: List[Dict] = []                      # [{label, threshold}]
triggered_alerts: List[Dict] = []                   # recent alerts
video_jobs: Dict[str, Dict] = {}                    # job_id → {progress, output_path, status, detections}
MAX_LOG = 5000

# ── app ──────────────────────────────────────────────────────────────────
app = FastAPI(title="ActivityWatch — Employee Activity Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ──────────────────────────────────────────────────────────────
def _append_log(mode: str, label: str, confidence: float):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "label": label,
        "confidence": round(confidence, 3),
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
    detector = get_detector()
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        video_jobs[job_id]["status"] = "error"
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = str(OUTPUT_DIR / f"{job_id}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    job_detections: List[Dict] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, dets = detector.detect_frame(frame)
        writer.write(annotated)
        for d in dets:
            _append_log("Video", d["label"], d["confidence"])
            job_detections.append(d)
        frame_idx += 1
        video_jobs[job_id]["progress"] = min(
            round(frame_idx / total_frames * 100, 1), 100.0
        )

    cap.release()
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
    try:
        while True:
            data = await websocket.receive_text()
            t0 = time.time()

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

            annotated, dets = detector.detect_frame(frame)

            # log detections
            for d in dets:
                _append_log("Live", d["label"], d["confidence"])

            # encode annotated frame back to base64 JPEG
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64_frame = base64.b64encode(buf.tobytes()).decode("utf-8")

            elapsed = time.time() - t0
            fps = round(1.0 / elapsed, 1) if elapsed > 0 else 0

            await websocket.send_json(
                {
                    "boxes": dets,
                    "annotated_frame": b64_frame,
                    "fps": fps,
                }
            )
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
        buf, fieldnames=["timestamp", "mode", "label", "confidence"]
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


# ── Health / meta ────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ── Serve frontend ───────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<h1>ActivityWatch</h1><p>Frontend not found.</p>")
