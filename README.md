# Employee Activity Detection

A real-time computer vision application that detects employees in video streams and classifies their activities using a two-stage deep learning pipeline.

The system combines YOLOv8n for person detection and EfficientNet-B0 for activity recognition, supporting both uploaded video analysis and live webcam inference through a FastAPI backend and browser-based interface.

---

## Overview

This project demonstrates an end-to-end AI inference pipeline for human activity recognition in workplace environments.

### Inference Pipeline

text Input Frame │ ▼ YOLOv8n Person Detection │ ▼ Person Cropping │ ▼ EfficientNet-B0 Activity Classification │ ▼ Annotated Output + Activity Logs

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

### Activity Monitoring

- Detection history logging
- Confidence score tracking
- Activity statistics and summaries
- CSV export functionality

### Alert System

- Activity-specific alert configuration
- Confidence threshold settings
- Triggered alert tracking

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

### Frontend

- HTML
- JavaScript
- WebSocket API

---

## Project Structure

text employee_activity_detection/ │ ├── app/ │ ├── main.py │ ├── inference.py │ ├── models/ │ │ ├── efficientnet_b0_employee_activity.pth │ │ └── class_map.json │ ├── static/ │ │ └── index.html │ ├── uploads/ │ └── outputs/ │ ├── requirements.txt ├── requirements-lock.txt ├── README.md └── yolov8n.pt

---

## Setup

### 1. Clone the Repository

bash git clone <repository-url> cd employee_activity_detection

### 2. Create a Virtual Environment

bash python -m venv .venv

Activate the environment:

macOS/Linux

bash source .venv/bin/activate

Windows

bash .venv\Scripts\activate

### 3. Install Dependencies

bash pip install -r requirements.txt

### 4. Verify Model Files

Ensure the following files are available:

text app/models/ ├── efficientnet_b0_employee_activity.pth └── class_map.json

The application will still start without the trained weights, but predictions will be random.

### 5. Run the Application

bash uvicorn app.main:app --reload

Open:

text http://127.0.0.1:8000

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

- Model quantization and optimization
- Object tracking across frames
- Persistent database storage
- User authentication
- Docker deployment
- GPU acceleration enhancements
- Cloud-based inference deployment
- Analytics dashboard and reporting

---

## License

This project was developed as an AI Engineering portfolio project demonstrating real-time computer vision, deep learning inference pipelines, and FastAPI-based deployment.
