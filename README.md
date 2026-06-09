# ActivityWatch — Employee Activity Detection

A real-time employee activity detection web application powered by **YOLOv8n** (person detection) and **EfficientNet-B0** (activity classification). Runs locally on `localhost` with a modern dark-themed UI.

## 🏗️ Architecture

```
┌─────────────┐   frames    ┌──────────────────┐   detections   ┌──────────┐
│   Browser    │ ──────────► │  FastAPI Backend  │ ─────────────► │  Frontend │
│  (Webcam /   │ ◄────────── │                  │                │  Display  │
│   Upload)    │  annotated  │  YOLOv8n → Crop  │                └──────────┘
└─────────────┘   frames    │  → EfficientNet  │
                             └──────────────────┘
```

**Two-stage inference pipeline:**
1. **YOLOv8n** — detects persons in the frame (class 0, conf ≥ 0.4)
2. **EfficientNet-B0** — classifies each detected person's activity from 15 categories

## 📋 Activity Classes

| # | Activity | # | Activity |
|---|----------|---|----------|
| 0 | Applauding Presentation | 8 | Messaging on Phone |
| 1 | At Team Celebration | 9 | On Lunch Break |
| 2 | Commuting by Bike | 10 | On Phone Call |
| 3 | Enjoying Team Meeting | 11 | Rushing to Meeting |
| 4 | Greeting Colleague | 12 | Taking a Nap |
| 5 | Having Coffee Break | 13 | Working at Desk |
| 6 | In Heated Discussion | 14 | Working on Laptop |
| 7 | Listening with Headphones | | |

## 🚀 Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place model files

Copy your trained model weights into the `app/models/` directory:

```
app/models/
  ├── efficientnet_b0_employee_activity.pth   # trained EfficientNet-B0 weights
  └── class_map.json                           # class mapping (already included)
```

> **Note:** If the `.pth` file is not present, the app will still run in demo mode with random weights.

### 3. Run the application

```bash
uvicorn app.main:app --reload
```

Open your browser at **http://localhost:8000**

## 🎯 Features

| Feature | Description |
|---------|-------------|
| **Dashboard** | Live stats, activity distribution chart, detection timeline |
| **Upload Video** | Drag-and-drop MP4/AVI, background processing, annotated output |
| **Live Feed** | Real-time webcam detection via WebSocket (~10 FPS) |
| **Activity Log** | Searchable, sortable, paginated table with confidence bars |
| **Alerts** | Per-class toggle + threshold slider, live alert feed |
| **Export** | CSV download of all detections, annotated video download |

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload video for processing |
| `GET` | `/video-status/{job_id}` | Check processing progress |
| `GET` | `/video-result/{job_id}` | Download annotated video |
| `WS` | `/video-stream` | Live WebSocket frame processing |
| `GET` | `/activity-log` | Last 500 detection events |
| `GET` | `/export/csv` | Download full activity log as CSV |
| `POST` | `/alerts/config` | Configure alert thresholds |
| `GET` | `/alerts` | Recent triggered alerts |
| `GET` | `/health` | Backend health check |

## 📁 Project Structure

```
employee_activity_detection/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app with all routes
│   ├── inference.py          # YOLOv8n + EfficientNet-B0 pipeline
│   ├── models/
│   │   ├── efficientnet_b0_employee_activity.pth
│   │   └── class_map.json
│   ├── static/
│   │   └── index.html        # Single-page frontend
│   ├── uploads/               # Temporary uploaded videos
│   └── outputs/               # Annotated output videos
├── requirements.txt
└── README.md
```

## ⚙️ Requirements

- Python 3.9+
- GPU recommended (CUDA) for real-time inference, but CPU works for demo
- Modern browser with WebSocket and `getUserMedia` support
