import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import insightface
import torch

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
EMBEDDINGS_PATH = MODEL_DIR / "employee_embeddings.json"

class FaceRecognizer:
    def __init__(self):
        print("[FaceRecognizer] Initialising FaceAnalysis (buffalo_l)...")
        # Initialize FaceAnalysis with the buffalo_l model
        self.app = insightface.app.FaceAnalysis(name="buffalo_l")
        
        # Prepare the context based on CUDA availability
        ctx_id = 0 if torch.cuda.is_available() else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        
        self.employee_data: Dict[str, dict] = {}
        self._load_embeddings()

    def _load_embeddings(self):
        if EMBEDDINGS_PATH.exists():
            with open(EMBEDDINGS_PATH, "r") as f:
                self.employee_data = json.load(f)
            print(f"[FaceRecognizer] Loaded {len(self.employee_data)} employees from {EMBEDDINGS_PATH}")
        else:
            self.employee_data = {}
            print("[FaceRecognizer] No existing embeddings found. Starting fresh.")

    def _save_embeddings(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(EMBEDDINGS_PATH, "w") as f:
            json.dump(self.employee_data, f, indent=2)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def enroll(self, employee_id: str, name: str, image_bgr: np.ndarray) -> dict:
        """
        Detect face, extract embedding, and save it.
        Keep max 5 embeddings per employee.
        """
        faces = self.app.get(image_bgr)
        if not faces:
            raise ValueError("No face detected in the uploaded image")
        
        # Assume the largest face is the target if multiple faces are detected
        # Sort by bounding box area
        faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        face = faces[0]
        
        embedding = face.normed_embedding.tolist()

        from datetime import datetime
        now = datetime.now().isoformat()

        if employee_id not in self.employee_data:
            self.employee_data[employee_id] = {
                "name": name,
                "employee_id": employee_id,
                "embeddings": [],
                "enrolled_at": now
            }
        else:
            # Update name just in case it changed
            self.employee_data[employee_id]["name"] = name
            
        embeddings_list = self.employee_data[employee_id]["embeddings"]
        embeddings_list.append(embedding)
        
        # Keep maximum 5 embeddings, removing the oldest if necessary
        if len(embeddings_list) > 5:
            embeddings_list.pop(0)
            
        self._save_embeddings()
        
        return {
            "status": "enrolled",
            "employee_id": employee_id,
            "name": name,
            "embedding_count": len(embeddings_list)
        }

    def identify(self, face_crop_bgr: np.ndarray, threshold: float = 0.65) -> dict:
        """
        Extract embedding from the crop and compare with all stored employees.
        Returns the best match exceeding the threshold.
        """
        if not self.employee_data:
            return {"employee_id": "unknown", "name": "Unknown", "similarity": 0.0}

        faces = self.app.get(face_crop_bgr)
        if not faces:
            return {"employee_id": "unknown", "name": "Unknown", "similarity": 0.0}
            
        # Get embedding of the primary face in the crop
        faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        target_embedding = np.array(faces[0].normed_embedding)

        best_match_id = "unknown"
        best_match_name = "Unknown"
        max_similarity = 0.0

        for emp_id, emp_info in self.employee_data.items():
            for stored_emb_list in emp_info["embeddings"]:
                stored_emb = np.array(stored_emb_list)
                similarity = self._cosine_similarity(target_embedding, stored_emb)
                if similarity > max_similarity:
                    max_similarity = similarity
                    if max_similarity >= threshold:
                        best_match_id = emp_id
                        best_match_name = emp_info["name"]

        if max_similarity >= threshold:
            return {
                "employee_id": best_match_id,
                "name": best_match_name,
                "similarity": float(max_similarity)
            }
        else:
            return {
                "employee_id": "unknown",
                "name": "Unknown",
                "similarity": 0.0
            }

    def get_all_employees(self) -> list:
        employees = []
        for emp_id, emp_info in self.employee_data.items():
            employees.append({
                "employee_id": emp_id,
                "name": emp_info["name"],
                "embedding_count": len(emp_info["embeddings"]),
                "enrolled_at": emp_info.get("enrolled_at", "")
            })
        return employees

    def delete_employee(self, employee_id: str) -> bool:
        if employee_id in self.employee_data:
            del self.employee_data[employee_id]
            self._save_embeddings()
            
            # Optionally try to remove the photo if it exists
            photo_path = MODEL_DIR / "employee_photos" / f"{employee_id}.jpg"
            if photo_path.exists():
                try:
                    photo_path.unlink()
                except Exception:
                    pass
            return True
        return False


_face_recognizer: Optional[FaceRecognizer] = None

def get_face_recognizer() -> FaceRecognizer:
    """Return (and lazily initialise) the global FaceRecognizer."""
    global _face_recognizer
    if _face_recognizer is None:
        _face_recognizer = FaceRecognizer()
    return _face_recognizer
