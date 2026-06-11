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
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set

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
activity_log: List[Dict[str, Any]] = []             # raw detection events
event_log: List[Dict[str, Any]] = []                # manager-facing event log
alert_configs: List[Dict[str, Any]] = []            # legacy per-class config
triggered_alerts: List[Dict[str, Any]] = []         # alert history
video_jobs: Dict[str, Dict] = {}                    # job_id → {progress, output_path, status, detections}
MAX_LOG = 5000
MAX_EVENT_LOG = 2000
MAX_ALERTS = 500

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


PRODUCTIVE_ACTIVITIES = {
    "working_at_desk",
    "working_on_laptop",
    "enjoying_team_meeting",
    "applauding_presentation",
}

NON_PRODUCTIVE_ACTIVITIES = {
    "taking_a_nap",
    "on_lunch_break",
    "having_coffee_break",
    "messaging_on_phone",
    "on_phone_call",
}

DEFAULT_ALERT_SETTINGS = {
    "inactivity_threshold": 10.0,
    "nap_threshold": 5.0,
    "unknown_person_threshold": 3.0,
}

ALERT_COOLDOWN_BY_TYPE = {
    "inactivity": 10.0,
    "nap": 5.0,
    "unknown_person": 3.0,
}

ALERT_SEVERITY_RANK = {
    "warning": 1,
    "critical": 2,
}


def _trim_buffer(items: List[Dict[str, Any]], max_size: int) -> None:
    if len(items) > max_size:
        del items[: len(items) - max_size]


def _now_iso() -> str:
    return datetime.now().isoformat()


def _display_name(employee_name: str, fallback: str = "Unknown person") -> str:
    if employee_name and employee_name != "Unknown":
        return employee_name
    return fallback


@dataclass
class AlertSettings:
    inactivity_threshold: float = DEFAULT_ALERT_SETTINGS["inactivity_threshold"]
    nap_threshold: float = DEFAULT_ALERT_SETTINGS["nap_threshold"]
    unknown_person_threshold: float = DEFAULT_ALERT_SETTINGS["unknown_person_threshold"]


@dataclass
class SubjectState:
    source_id: str
    track_id: str
    mode: str
    employee_id: str = "unknown"
    employee_name: str = "Unknown"
    recognized: bool = False
    first_seen_at: float = 0.0
    last_seen_at: float = 0.0
    last_seen_timestamp: str = ""
    last_label: str = ""
    last_confidence: float = 0.0
    last_productive_at: Optional[float] = None
    has_seen_productive: bool = False
    was_productive: bool = False
    inactivity_alert_active: bool = False
    nap_started_at: Optional[float] = None
    nap_alert_active: bool = False
    unknown_started_at: Optional[float] = None
    unknown_alert_active: bool = False


class MonitoringState:
    def __init__(self) -> None:
        self.lock = Lock()
        self.settings = AlertSettings()
        self.total_detections = 0
        self.productive_detections = 0
        self.activity_counts: collections.Counter[str] = collections.Counter()
        self.unique_recognized_employee_ids: Set[str] = set()
        self.employee_productivity: Dict[str, Dict[str, Any]] = {}
        self.subjects: Dict[str, SubjectState] = {}
        self.subjects_by_source: Dict[str, Set[str]] = {}
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.last_alert_times: Dict[str, float] = {}

    def get_alert_settings(self) -> Dict[str, float]:
        return asdict(self.settings)

    def update_alert_settings(self, payload: Dict[str, Any]) -> Dict[str, float]:
        with self.lock:
            for key in DEFAULT_ALERT_SETTINGS:
                value = payload.get(key, getattr(self.settings, key))
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    numeric_value = getattr(self.settings, key)
                setattr(self.settings, key, max(1.0, numeric_value))
            return self.get_alert_settings()

    def _append_event(
        self,
        mode: str,
        event_type: str,
        message: str,
        employee_id: str,
        employee_name: str,
        severity: str = "warning",
        label: str = "",
    ) -> None:
        event_log.append(
            {
                "id": str(uuid.uuid4())[:8],
                "timestamp": _now_iso(),
                "mode": mode,
                "event_type": event_type,
                "message": message,
                "employee_id": employee_id,
                "employee_name": employee_name,
                "severity": severity,
                "label": label or event_type,
            }
        )
        _trim_buffer(event_log, MAX_EVENT_LOG)

    def _resolve_alert(self, alert_key: str) -> None:
        alert = self.active_alerts.pop(alert_key, None)
        if alert:
            alert["active"] = False
            alert["resolved_at"] = _now_iso()

    def _activate_alert(
        self,
        *,
        alert_key: str,
        alert_type: str,
        employee_id: str,
        employee_name: str,
        severity: str,
        message: str,
        observed_at: float,
    ) -> bool:
        if alert_key in self.active_alerts:
            return False
        cooldown = ALERT_COOLDOWN_BY_TYPE[alert_type]
        if observed_at - self.last_alert_times.get(alert_key, float("-inf")) < cooldown:
            return False

        alert = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": _now_iso(),
            "employee_id": employee_id,
            "employee_name": employee_name,
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "active": True,
        }
        self.active_alerts[alert_key] = alert
        self.last_alert_times[alert_key] = observed_at
        triggered_alerts.append(alert)
        _trim_buffer(triggered_alerts, MAX_ALERTS)
        return True

    def _update_employee_productivity(
        self,
        employee_id: str,
        employee_name: str,
        is_productive: bool,
    ) -> None:
        if employee_id == "unknown":
            return
        bucket = self.employee_productivity.setdefault(
            employee_id,
            {
                "employee_id": employee_id,
                "employee_name": employee_name,
                "productive_events": 0,
                "total_events": 0,
            },
        )
        bucket["employee_name"] = employee_name
        bucket["total_events"] += 1
        if is_productive:
            bucket["productive_events"] += 1

    def _subject_key(self, source_id: str, track_id: Any) -> str:
        return f"{source_id}:{track_id}"

    def _upsert_subject(
        self,
        *,
        source_id: str,
        mode: str,
        track_id: str,
        observed_at: float,
        timestamp: str,
    ) -> SubjectState:
        subject_key = self._subject_key(source_id, track_id)
        subject = self.subjects.get(subject_key)
        if subject is None:
            subject = SubjectState(
                source_id=source_id,
                track_id=track_id,
                mode=mode,
                first_seen_at=observed_at,
                last_seen_at=observed_at,
                last_seen_timestamp=timestamp,
            )
            self.subjects[subject_key] = subject
        subject.mode = mode
        subject.last_seen_at = observed_at
        subject.last_seen_timestamp = timestamp
        return subject

    def _clear_subject_alerts(self, subject_key: str, subject: SubjectState) -> None:
        for alert_type in ("inactivity", "nap", "unknown_person"):
            self._resolve_alert(f"{subject_key}:{alert_type}")
        subject.inactivity_alert_active = False
        subject.nap_alert_active = False
        subject.unknown_alert_active = False
        subject.nap_started_at = None
        subject.unknown_started_at = None

    def record_frame(
        self,
        *,
        source_id: str,
        mode: str,
        detections: List[Dict[str, Any]],
        observed_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            frame_time = observed_at if observed_at is not None else time.time()
            timestamp = _now_iso()
            current_subject_keys: Set[str] = set()

            for detection_index, detection in enumerate(detections):
                track_id = str(detection.get("track_id", detection_index))
                subject_key = self._subject_key(source_id, track_id)
                current_subject_keys.add(subject_key)
                subject = self._upsert_subject(
                    source_id=source_id,
                    mode=mode,
                    track_id=track_id,
                    observed_at=frame_time,
                    timestamp=timestamp,
                )

                employee_id = detection.get("employee_id") or "unknown"
                employee_name = detection.get("employee_name") or "Unknown"
                label = detection.get("label") or "unknown_activity"
                confidence = round(float(detection.get("confidence", 0.0)), 3)
                recognized = employee_id != "unknown"
                is_productive = label in PRODUCTIVE_ACTIVITIES

                subject.employee_id = employee_id
                subject.employee_name = employee_name
                subject.recognized = recognized
                subject.last_label = label
                subject.last_confidence = confidence

                raw_entry = {
                    "timestamp": timestamp,
                    "mode": mode,
                    "label": label,
                    "confidence": confidence,
                    "employee_id": employee_id,
                    "employee_name": employee_name,
                    "source_id": source_id,
                    "track_id": track_id,
                    "productive": is_productive,
                }
                activity_log.append(raw_entry)
                _trim_buffer(activity_log, MAX_LOG)

                self.total_detections += 1
                self.activity_counts[label] += 1
                if is_productive:
                    self.productive_detections += 1
                if recognized:
                    self.unique_recognized_employee_ids.add(employee_id)
                self._update_employee_productivity(employee_id, employee_name, is_productive)

                if is_productive:
                    if not subject.has_seen_productive:
                        self._append_event(
                            mode,
                            "started_working",
                            f"{employee_name} started working",
                            employee_id,
                            employee_name,
                            severity="warning",
                            label=label,
                        )
                    elif subject.inactivity_alert_active:
                        self._resolve_alert(f"{subject_key}:inactivity")
                        self._append_event(
                            mode,
                            "resumed_productive_work",
                            f"{employee_name} resumed productive work",
                            employee_id,
                            employee_name,
                            severity="warning",
                            label=label,
                        )
                    subject.last_productive_at = frame_time
                    subject.has_seen_productive = True
                    subject.inactivity_alert_active = False
                elif recognized:
                    reference_time = subject.last_productive_at
                    if reference_time is None:
                        reference_time = subject.first_seen_at
                    inactive_for = frame_time - reference_time
                    if inactive_for >= self.settings.inactivity_threshold:
                        triggered = self._activate_alert(
                            alert_key=f"{subject_key}:inactivity",
                            alert_type="inactivity",
                            employee_id=employee_id,
                            employee_name=employee_name,
                            severity="warning",
                            message=f"{employee_name} inactive for {int(self.settings.inactivity_threshold)} seconds",
                            observed_at=frame_time,
                        )
                        if triggered:
                            self._append_event(
                                mode,
                                "employee_inactive",
                                f"{employee_name} became inactive",
                                employee_id,
                                employee_name,
                                severity="warning",
                                label=label,
                            )
                        subject.inactivity_alert_active = True
                else:
                    self._resolve_alert(f"{subject_key}:inactivity")
                    subject.inactivity_alert_active = False

                if label == "taking_a_nap":
                    if subject.nap_started_at is None:
                        subject.nap_started_at = frame_time
                    nap_for = frame_time - subject.nap_started_at
                    if nap_for >= self.settings.nap_threshold:
                        display_name = _display_name(employee_name)
                        triggered = self._activate_alert(
                            alert_key=f"{subject_key}:nap",
                            alert_type="nap",
                            employee_id=employee_id,
                            employee_name=employee_name,
                            severity="critical",
                            message=f"{display_name} sleeping for {int(self.settings.nap_threshold)} seconds",
                            observed_at=frame_time,
                        )
                        if triggered:
                            self._append_event(
                                mode,
                                "nap_alert",
                                f"Nap alert triggered for {display_name}",
                                employee_id,
                                employee_name,
                                severity="critical",
                                label=label,
                            )
                        subject.nap_alert_active = True
                else:
                    subject.nap_started_at = None
                    self._resolve_alert(f"{subject_key}:nap")
                    subject.nap_alert_active = False

                if not recognized:
                    if subject.unknown_started_at is None:
                        subject.unknown_started_at = frame_time
                    unknown_for = frame_time - subject.unknown_started_at
                    if unknown_for >= self.settings.unknown_person_threshold:
                        triggered = self._activate_alert(
                            alert_key=f"{subject_key}:unknown_person",
                            alert_type="unknown_person",
                            employee_id="unknown",
                            employee_name="Unknown",
                            severity="critical",
                            message="Unknown person detected",
                            observed_at=frame_time,
                        )
                        if triggered:
                            self._append_event(
                                mode,
                                "unknown_person_detected",
                                "Unknown person detected",
                                "unknown",
                                "Unknown",
                                severity="critical",
                                label=label,
                            )
                        subject.unknown_alert_active = True
                else:
                    subject.unknown_started_at = None
                    self._resolve_alert(f"{subject_key}:unknown_person")
                    subject.unknown_alert_active = False

                subject.was_productive = is_productive

            previous_subjects = self.subjects_by_source.get(source_id, set())
            for missing_subject_key in previous_subjects - current_subject_keys:
                missing_subject = self.subjects.pop(missing_subject_key, None)
                if missing_subject is not None:
                    self._clear_subject_alerts(missing_subject_key, missing_subject)

            self.subjects_by_source[source_id] = current_subject_keys

            top_alert = self.get_top_active_alert()
            return {
                "active_alert": top_alert,
                "active_alerts_count": len(self.active_alerts),
                "active_employees": len(self.subjects),
            }

    def get_top_active_alert(self) -> Optional[Dict[str, Any]]:
        active = sorted(
            self.active_alerts.values(),
            key=lambda alert: (
                ALERT_SEVERITY_RANK.get(alert["severity"], 0),
                alert["timestamp"],
            ),
            reverse=True,
        )
        return active[0] if active else None

    def list_recent_events(self, limit: int = 500) -> List[Dict[str, Any]]:
        with self.lock:
            return event_log[-limit:]

    def list_recent_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self.lock:
            return triggered_alerts[-limit:]

    def list_active_alerts(self) -> List[Dict[str, Any]]:
        with self.lock:
            return sorted(
                self.active_alerts.values(),
                key=lambda alert: (
                    ALERT_SEVERITY_RANK.get(alert["severity"], 0),
                    alert["timestamp"],
                ),
                reverse=True,
            )

    def get_dashboard_summary(self) -> Dict[str, Any]:
        with self.lock:
            total_events = self.total_detections
            productive_events = self.productive_detections
            productivity_score = round(
                (productive_events / total_events) * 100, 1
            ) if total_events else 0.0

            employee_breakdown = []
            for employee in self.employee_productivity.values():
                total = employee["total_events"]
                productive = employee["productive_events"]
                employee_breakdown.append(
                    {
                        **employee,
                        "productivity_score": round((productive / total) * 100, 1) if total else 0.0,
                    }
                )

            employee_breakdown.sort(
                key=lambda item: (item["productivity_score"], item["productive_events"]),
                reverse=True,
            )

            return {
                "metrics": {
                    "total_detections": total_events,
                    "active_employees": len(self.subjects),
                    "recognized_employees": len(self.unique_recognized_employee_ids),
                    "productivity_score": productivity_score,
                    "active_alerts": len(self.active_alerts),
                    "productive_events": productive_events,
                    "total_events": total_events,
                },
                "activity_distribution": [
                    {"label": label, "count": count}
                    for label, count in self.activity_counts.most_common()
                ],
                "recent_events": event_log[-10:],
                "active_alert": self.get_top_active_alert(),
                "employee_productivity": employee_breakdown[:6],
            }


monitoring_state = MonitoringState()


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
        source_id = f"video:{job_id}"

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
                monitoring_state.record_frame(
                    source_id=source_id,
                    mode="Video",
                    detections=dets,
                    observed_at=frame_idx / fps,
                )
                for d in dets:
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
        monitoring_state.record_frame(
            source_id=source_id,
            mode="Video",
            detections=[],
            observed_at=frame_idx / fps,
        )

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

            monitoring_snapshot = monitoring_state.record_frame(
                source_id="live",
                mode="Live",
                detections=dets,
            )

            # encode annotated frame back to base64 JPEG
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64_frame = base64.b64encode(buf.tobytes()).decode("utf-8")

            elapsed = time.time() - t0
            fps_val = round(1.0 / elapsed, 1) if elapsed > 0 else 0

            last_result = {
                "boxes": dets,
                "annotated_frame": b64_frame,
                "fps": fps_val,
                "active_alert": monitoring_snapshot["active_alert"],
                "active_alerts_count": monitoring_snapshot["active_alerts_count"],
            }
            last_processed_time = time.time()

            await websocket.send_json(last_result)
    except WebSocketDisconnect:
        monitoring_state.record_frame(
            source_id="live",
            mode="Live",
            detections=[],
        )


# ── Activity log ─────────────────────────────────────────────────────────
@app.get("/activity-log")
async def get_activity_log():
    with monitoring_state.lock:
        return list(activity_log[-500:])


@app.get("/activity-events")
async def get_activity_events():
    return monitoring_state.list_recent_events(limit=500)


@app.get("/dashboard-summary")
async def get_dashboard_summary():
    return monitoring_state.get_dashboard_summary()


# ── Export CSV ───────────────────────────────────────────────────────────
@app.get("/export/csv")
async def export_csv():
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=["timestamp", "mode", "label", "confidence", "employee_id", "employee_name"]
    )
    writer.writeheader()
    with monitoring_state.lock:
        rows = list(activity_log)
    for row in rows:
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


@app.get("/alerts/settings")
async def get_alert_settings():
    return monitoring_state.get_alert_settings()


@app.post("/alerts/settings")
async def set_alert_settings(payload: Dict[str, Any]):
    return monitoring_state.update_alert_settings(payload)


# ── Alerts ───────────────────────────────────────────────────────────────
@app.get("/alerts")
async def get_alerts():
    return monitoring_state.list_recent_alerts(limit=100)


@app.get("/alerts/active")
async def get_active_alerts():
    return monitoring_state.list_active_alerts()


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

@app.post("/employees/validate-photos")
async def validate_employee_photos(photos: List[UploadFile] = File(...)):
    fr = get_face_recognizer()
    results = []

    for photo in photos:
        content = await photo.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            results.append(
                {
                    "filename": photo.filename or "image",
                    "valid_image": False,
                    "face_detected": False,
                    "face_count": 0,
                }
            )
            continue

        inspection = fr.inspect_image(img)
        results.append(
            {
                "filename": photo.filename or "image",
                "valid_image": True,
                **inspection,
            }
        )

    return {"results": results}

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
