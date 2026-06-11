# Employee Activity Detection using YOLOv8n and EfficientNet-B0

A real-time computer vision application that detects employees in video streams and classifies their activities using a two-stage deep learning pipeline.

The system combines YOLOv8n for person detection and EfficientNet-B0 for activity recognition, supporting both uploaded video analysis and live webcam inference through a FastAPI backend and browser-based interface.

---

## Key Highlights

- Two-stage AI inference pipeline
- YOLOv8n person detection
- EfficientNet-B0 activity classification
- Employee face recognition using InsightFace
- Multi-person activity recognition and tracking
- Real-time webcam inference via WebSockets
- Video upload and background processing
- Activity logging and productivity monitoring
- Configurable alert system
- Interactive analytics dashboard
- FastAPI backend with browser-based UI

---

## Overview

This project demonstrates an end-to-end AI inference pipeline for human activity recognition in workplace environments.

### Extended Monitoring Pipeline

In addition to activity classification, the system performs employee identification, subject tracking, productivity monitoring, and real-time alert generation.

```text
Input Frame
    │
    ▼
YOLOv8n Person Detection
    │
    ▼
Lightweight Subject Tracking
    │
    ├──────────────► Face Recognition (InsightFace)
    │
    ▼
Person Cropping
    │
    ▼
EfficientNet-B0 Activity Classification
    │
    ▼
Productivity Analysis + Alert Engine
    │
    ▼
Dashboard Metrics + Activity Logs + Alerts
```

### Inference Pipeline

```text
Input Frame
    │
    ▼
YOLOv8n Person Detection
    │
    ▼
Person Cropping
    │
    ▼
EfficientNet-B0 Activity Classification
    │
    ▼
Annotated Output + Activity Logs
```

For each frame:

1. YOLOv8n detects all people present in the scene.
2. Each detected person is cropped from the frame.
3. EfficientNet-B0 classifies the activity being performed.
4. Bounding boxes, labels, and confidence scores are rendered on the output frame.
5. Results are logged and made available through the API.

---

## Activity Classes

The model predicts the following employee activities:

| Class | Activity                  |
| ----- | ------------------------- |
| 0     | Applauding Presentation   |
| 1     | At Team Celebration       |
| 2     | Commuting by Bike         |
| 3     | Enjoying Team Meeting     |
| 4     | Greeting Colleague        |
| 5     | Having Coffee Break       |
| 6     | In Heated Discussion      |
| 7     | Listening with Headphones |
| 8     | Messaging on Phone        |
| 9     | On Lunch Break            |
| 10    | On Phone Call             |
| 11    | Rushing to Meeting        |
| 12    | Taking a Nap              |
| 13    | Working at Desk           |
| 14    | Working on Laptop         |

---

## Features

### Video Upload Processing

- Upload MP4, AVI, or MOV videos
- Background processing using FastAPI tasks
- Annotated output video generation
- Progress tracking through API endpoints

### Live Webcam Inference

- Real-time frame processing via WebSocket
- Multi-person detection support
- Live bounding boxes and activity labels
- FPS reporting

### Employee Recognition

- Employee enrollment using facial embeddings
- Face identification with InsightFace
- Known and unknown person detection
- Persistent employee embedding storage
- Employee management APIs

### Activity Monitoring and Productivity Analytics

- Detection history logging
- Productivity classification based on activity labels
- Employee activity timelines
- Real-time dashboard statistics
- Activity distribution analytics
- Confidence score tracking
- CSV export functionality

### Alert System

- Inactivity detection alerts
- Nap detection alerts
- Unknown person alerts
- Employee missing alerts
- Configurable thresholds and cooldowns
- Alert acknowledgement and resolution tracking

---

## Technology Stack

### Backend

- FastAPI
- Uvicorn
- OpenCV
- WebSockets

### Deep Learning

- YOLOv8n
- EfficientNet-B0
- PyTorch
- TIMM
- InsightFace

### Frontend

- HTML
- JavaScript
- WebSocket API

---

## Project Structure

```text
employee_activity_detection/
├── .venv/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── inference.py
│   ├── face_recognition.py
│   ├── models/
│   │   ├── class_map.json
│   │   ├── efficientnet_b0_employee_activity.pth
│   │   └── employee_embeddings.json
│   ├── outputs/
│   │   └── .gitkeep
│   ├── static/
│   │   └── index.html
│   └── uploads/
│       └── .gitkeep
├── .gitignore
├── README.md
├── requirements.txt

```

---

## Prerequisites

### Python

- Python 3.10 or higher

Verify:

```bash
python --version
```

### FFmpeg

This project uses FFmpeg to convert annotated videos into a browser-compatible MP4 format (H.264).

Verify installation:

```bash
ffmpeg -version
```

#### Windows

Install using Winget:

```powershell
winget install ffmpeg
```

Or download from:
https://ffmpeg.org/download.html

#### macOS

Using Homebrew:

```bash
brew install ffmpeg
```

---

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd employee_activity_detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Model Files

Ensure the following files are available:

```text
app/models/
├── efficientnet_b0_employee_activity.pth
├── class_map.json
└── employee_embeddings.json
```

The employee embeddings file is created automatically if it does not already exist.

### 5. Run the Application

```bash
uvicorn app.main:app --reload
```

Open your browser and navigate to:

```text
http://127.0.0.1:8000
```

---

## API Endpoints

| Method | Endpoint               | Description                   |
| ------ | ---------------------- | ----------------------------- |
| POST   | /upload                | Upload a video for processing |
| GET    | /video-status/{job_id} | Retrieve processing progress  |
| GET    | /video-result/{job_id} | Download annotated video      |
| WS     | /video-stream          | Real-time webcam inference    |
| GET    | /activity-log          | Recent detection events       |
| GET    | /export/csv            | Export activity log as CSV    |
| POST   | /alerts/config         | Configure alert thresholds    |
| GET    | /alerts                | Retrieve recent alerts        |
| GET    | /health                | Application health status     |
| POST   | /employees/enroll      | Enroll a new employee         |
| GET    | /employees             | List enrolled employees       |
| DELETE | /employees/{id}        | Delete an employee            |
| GET    | /dashboard             | Dashboard analytics data      |
| GET    | /alerts/config         | Retrieve alert configuration  |

---

## Sample Workflow

1. Start the FastAPI server.
2. Open the web interface in a browser.
3. Upload a video or start the webcam stream.
4. The backend detects people using YOLOv8n.
5. EfficientNet-B0 classifies activities for each detected person.
6. Results are displayed with labels and confidence scores.
7. Logs and alerts are generated automatically.

---

## Future Improvements

- Advanced multi-object tracking (ByteTrack/DeepSORT)
- Database-backed employee management
- Role-based authentication and authorization
- Docker and Kubernetes deployment
- GPU-optimized inference pipeline
- Distributed video processing
- Advanced reporting and analytics
- Cloud deployment and monitoring
- Mobile dashboard support

---

## License

This project was developed as an AI Engineering portfolio project demonstrating real-time computer vision, employee activity monitoring, face recognition, alerting systems, analytics dashboards, and FastAPI-based deployment.
