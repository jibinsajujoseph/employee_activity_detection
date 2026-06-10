"""
Two-stage inference pipeline:
  Stage 1 — YOLOv8n person detection
  Stage 2 — EfficientNet-B0 activity classification
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from ultralytics import YOLO

from .face_recognition import get_face_recognizer
# ── paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
WEIGHTS_PATH = MODEL_DIR / "efficientnet_b0_employee_activity.pth"
CLASS_MAP_PATH = MODEL_DIR / "class_map.json"

# ── ImageNet normalisation constants ─────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── colour palette for bounding boxes (one per class) ────────────────────
BOX_COLOURS: List[Tuple[int, int, int]] = [
    (108, 99, 255),   # purple  — applauding_presentation
    (34, 197, 94),    # green   — at_team_celebration
    (59, 130, 246),   # blue    — commuting_by_bike
    (249, 115, 22),   # orange  — enjoying_team_meeting
    (236, 72, 153),   # pink    — greeting_colleague
    (168, 85, 247),   # violet  — having_coffee_break
    (239, 68, 68),    # red     — in_heated_discussion
    (20, 184, 166),   # teal    — listening_with_headphones
    (234, 179, 8),    # yellow  — messaging_on_phone
    (132, 204, 22),   # lime    — on_lunch_break
    (14, 165, 233),   # sky     — on_phone_call
    (244, 63, 94),    # rose    — rushing_to_meeting
    (99, 102, 241),   # indigo  — taking_a_nap
    (34, 211, 238),   # cyan    — working_at_desk
    (251, 146, 60),   # amber   — working_on_laptop
]


def _build_classifier_head(num_classes: int) -> nn.Sequential:
    """Custom classification head matching the training architecture."""
    return nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, 512),
        nn.SiLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )


class ActivityDetector:
    """
    End-to-end detector:
      1. YOLOv8n  → person bounding boxes
      2. EfficientNet-B0 → activity classification per crop
    """

    def __init__(self, device: Optional[str] = None):
        # ── device selection ──────────────────────────────────────────
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"[ActivityDetector] Using device: {self.device}")

        # ── face cache ───────────────────────────────────────────────
        self._face_cache = {}
        self._next_track_id = 0

        # ── load class map ────────────────────────────────────────────
        with open(CLASS_MAP_PATH, "r") as f:
            cmap = json.load(f)
        self.idx_to_class: Dict[int, str] = {
            int(k): v for k, v in cmap["idx_to_class"].items()
        }
        self.num_classes = len(self.idx_to_class)
        print(f"[ActivityDetector] Loaded {self.num_classes} activity classes")

        # ── Stage 1: YOLOv8n ─────────────────────────────────────────
        self.yolo = YOLO("yolov8n.pt")
        print("[ActivityDetector] YOLOv8n loaded")

        # ── Stage 2: EfficientNet-B0 + custom head ───────────────────
        self.efficientnet = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

        in_features = self.efficientnet.num_features

        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_classes),
        )

        if WEIGHTS_PATH.exists():
            checkpoint = torch.load(
                str(WEIGHTS_PATH),
                map_location=self.device,
                weights_only=False
            )

            self.efficientnet.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )

            print("[ActivityDetector] EfficientNet-B0 weights loaded successfully")
        else:
            print(
                f"[ActivityDetector] WARNING — weights not found at {WEIGHTS_PATH}; "
                "running with random weights (demo mode)"
            )

        self.efficientnet.to(self.device).eval()

    # ── preprocessing ─────────────────────────────────────────────────
    def _preprocess_crop(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """Resize to 224×224, normalise with ImageNet stats, return tensor."""
        img = cv2.resize(crop_bgr, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1,3,224,224)
        return tensor.to(self.device)

    def reset_cache(self):
        """Reset the face tracking cache (e.g. between videos)."""
        self._face_cache = {}
        self._next_track_id = 0

    def _compute_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        interArea = max(0, x2 - x1) * max(0, y2 - y1)
        if interArea == 0:
            return 0.0
        box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return interArea / float(box1Area + box2Area - interArea)

    # ── per-frame pipeline ────────────────────────────────────────────
    @torch.no_grad()
    def detect_frame(
        self, frame_bgr: np.ndarray, pad: int = 20, recognize_faces: bool = True
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Run the full two-stage pipeline on a single BGR frame.

        Returns:
            annotated_frame (np.ndarray): frame with bounding boxes drawn
            detections (list[dict]): each dict has keys
                x1, y1, x2, y2, label, confidence, employee_id, employee_name, face_similarity
        """
        h, w = frame_bgr.shape[:2]
        annotated = frame_bgr.copy()
        detections: List[Dict] = []

        # Stage 1 — YOLO person detection
        results = self.yolo.predict(
            frame_bgr, classes=[0], conf=0.4, verbose=False
        )

        if not results or len(results[0].boxes) == 0:
            return annotated, detections

        boxes = results[0].boxes
        
        new_face_cache = {}
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # ── Tracking ──────────────────────────────────────────────────
            best_iou = 0
            best_track_id = -1
            for tid, tdata in self._face_cache.items():
                iou = self._compute_iou((x1, y1, x2, y2), tdata["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = tid
            
            if best_iou > 0.4:
                track_id = best_track_id
                tdata = self._face_cache.pop(track_id)
            else:
                track_id = self._next_track_id
                self._next_track_id += 1
                tdata = {
                    "employee_id": "unknown",
                    "employee_name": "Unknown",
                    "similarity": 0.0,
                    "frame_count": 0
                }
            
            tdata["bbox"] = (x1, y1, x2, y2)
            tdata["frame_count"] += 1
            
            # pad crop
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w, x2 + pad)
            cy2 = min(h, y2 + pad)
            crop = frame_bgr[cy1:cy2, cx1:cx2]

            if crop.size == 0:
                new_face_cache[track_id] = tdata
                continue

            # ── Face Recognition ──────────────────────────────────────────
            if recognize_faces and (tdata["frame_count"] == 1 or tdata["frame_count"] % 10 == 0):
                fr = get_face_recognizer()
                res = fr.identify(crop)
                tdata["employee_id"] = res["employee_id"]
                tdata["employee_name"] = res["name"]
                tdata["similarity"] = res["similarity"]
            
            new_face_cache[track_id] = tdata

            # Stage 2 — classify activity
            tensor = self._preprocess_crop(crop)
            logits = self.efficientnet(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = probs.max(0)
            conf_val = round(conf.item(), 3)
            label = self.idx_to_class[idx.item()]

            detections.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "label": label,
                    "confidence": conf_val,
                    "employee_id": tdata["employee_id"],
                    "employee_name": tdata["employee_name"],
                    "face_similarity": tdata["similarity"],
                }
            )

            # ── draw on annotated frame ─────────────────────────────
            class_idx = idx.item() % len(BOX_COLOURS)
            colour = BOX_COLOURS[class_idx]
            
            is_identified = tdata["employee_id"] != "unknown"
            thickness = 3 if is_identified else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness)

            display_label = label.replace("_", " ").title()
            emp_text = tdata["employee_name"]
            act_text = f"{display_label} {conf_val:.0%}"
            
            (tw_emp, th_emp), _ = cv2.getTextSize(emp_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            (tw_act, th_act), _ = cv2.getTextSize(act_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            
            box_w = max(tw_emp, tw_act)
            total_h = th_emp + th_act + 15
            
            cv2.rectangle(
                annotated,
                (x1, y1 - total_h - 10),
                (x1 + box_w + 6, y1),
                colour,
                -1,
            )
            cv2.putText(
                annotated,
                emp_text,
                (x1 + 3, y1 - th_act - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                act_text,
                (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            
        self._face_cache = new_face_cache

        return annotated, detections


# ── module-level singleton (lazy) ────────────────────────────────────────
_detector: Optional[ActivityDetector] = None


def get_detector() -> ActivityDetector:
    """Return (and lazily initialise) the global ActivityDetector."""
    global _detector
    if _detector is None:
        _detector = ActivityDetector()
    return _detector
