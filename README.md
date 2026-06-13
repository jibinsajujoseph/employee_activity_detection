# Employee Activity Detection

A FastAPI-based computer vision application that combines person detection, activity recognition, face recognition, and workplace monitoring features.

The system supports both uploaded video analysis and live webcam inference using a two-stage deep learning pipeline built with YOLOv8n and EfficientNet-B0. In addition to activity recognition, it provides employee identification, productivity analytics, activity logging, and configurable alerting through a browser-based interface.

---

## Key Highlights

- YOLOv8n-based person detection
- EfficientNet-B0 activity recognition
- InsightFace-powered employee identification
- Real-time webcam inference via WebSockets
- Uploaded video processing with annotated outputs
- IoU-based subject tracking across frames
- Employee enrollment and face embedding management
- Productivity analytics and activity monitoring
- Configurable alerting system
- REST API and browser-based dashboard

---

## System Overview

The application processes video streams using a two-stage inference pipeline.

### Monitoring Pipeline

```text
Input Frame
    │
    ▼
YOLOv8n Person Detection
    │
    ▼
IoU-Based Subject Tracking
    │
    ├──────────────► InsightFace Recognition
    │
    ▼
Person Cropping
    │
    ▼
EfficientNet-B0 Activity Classification
    │
    ▼
Activity Logging
    ▼
Productivity Analytics
    ▼
Alert Generation & Dashboard Updates
```

### Inference Flow

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
Annotated Output + Detection Data
```

For each processed frame:

1. YOLOv8n detects people in the scene.
2. Subjects are tracked across frames using bounding-box IoU matching.
3. Face recognition is performed when enabled.
4. Person crops are classified by EfficientNet-B0.
5. Detection results are logged and analysed.
6. Dashboard metrics and alerts are updated.

---

## Activity Classes

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

### Video Processing

- Upload MP4, AVI, or MOV videos
- Background video processing using FastAPI tasks
- Annotated output video generation
- Processing progress tracking
- Detection summaries for completed jobs

### Live Webcam Inference

- Real-time frame processing through WebSockets
- Multi-person detection and activity recognition
- Annotated video stream
- Live alert updates
- FPS reporting

### Employee Recognition

- Employee enrollment from uploaded images
- Multiple face embeddings per employee
- InsightFace-based identification
- Known and unknown person detection
- Employee photo validation
- Employee preview image support
- Employee management APIs

### Activity Monitoring & Analytics

- Detection history logging
- Activity event tracking
- Productivity scoring
- Employee productivity breakdown
- Activity distribution analytics
- Active employee tracking
- Dashboard summary metrics

### Alerting

Supported alert types:

- Employee inactivity
- Nap detection
- Unknown person detection

Features:

- Configurable thresholds
- Alert cooldown handling
- Active and historical alert tracking
- Severity levels
- Event log generation

### Performance Monitoring

- Inference timing statistics
- FPS metrics
- Activity-classification cache statistics
- Processed-frame tracking

---

## Technology Stack

### Computer Vision

- YOLOv8n
- EfficientNet-B0
- InsightFace
- OpenCV

### Backend

- FastAPI
- Uvicorn
- WebSockets

### Deep Learning

- PyTorch
- TIMM
- ONNX Runtime

### Frontend

- HTML
- JavaScript

---

## Project Structure

```text
employee_activity_detection/
├── app/
│   ├── main.py
│   ├── inference.py
│   ├── face_recognition.py
│   ├── models/
│   │   ├── class_map.json
│   │   ├── efficientnet_b0_employee_activity.pth
│   │   ├── employee_embeddings.json
│   │   └── employee_photos/
│   ├── outputs/
│   ├── uploads/
│   └── static/
├── README.md
└── requirements.txt
```

---

## Prerequisites

### Python

- Python 3.10 or higher

```bash
python --version
```

### FFmpeg

Used for converting generated videos into browser-compatible MP4 format.

```bash
ffmpeg -version
```

---

## Setup

### Clone the Repository

```bash
git clone https://github.com/jibinsajujoseph/employee_activity_detection
cd employee_activity_detection
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

### Activate the Environment

macOS / Linux

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

---

## API Endpoints

### Video Processing

| Method | Endpoint               |
| ------ | ---------------------- |
| POST   | /upload                |
| GET    | /video-status/{job_id} |
| GET    | /video-result/{job_id} |
| WS     | /video-stream          |

### Monitoring & Analytics

| Method | Endpoint           |
| ------ | ------------------ |
| GET    | /activity-log      |
| GET    | /activity-events   |
| GET    | /dashboard-summary |
| GET    | /export/csv        |

### Alerts

| Method | Endpoint         |
| ------ | ---------------- |
| GET    | /alerts          |
| GET    | /alerts/active   |
| GET    | /alerts/settings |
| POST   | /alerts/settings |
| GET    | /alerts/config   |
| POST   | /alerts/config   |

### Employee Management

| Method | Endpoint                         |
| ------ | -------------------------------- |
| POST   | /employees/enroll                |
| POST   | /employees/enroll/batch          |
| POST   | /employees/validate-photos       |
| GET    | /employees                       |
| GET    | /employees/{employee_id}         |
| DELETE | /employees/{employee_id}         |
| GET    | /employees/{employee_id}/preview |

### System

| Method | Endpoint    |
| ------ | ----------- |
| GET    | /health     |
| GET    | /perf-stats |

---

## Future Improvements

- ByteTrack or DeepSORT integration
- Database-backed employee management
- Authentication and role-based access control
- Dockerized deployment
- GPU-optimized inference pipeline
- Distributed video processing
- Cloud deployment support
- Mobile-friendly dashboard
- Historical reporting and trend analysis

---

## License

This project was developed as an AI engineering portfolio project demonstrating real-time computer vision, activity recognition, employee identification, monitoring workflows, analytics, and FastAPI-based deployment.
