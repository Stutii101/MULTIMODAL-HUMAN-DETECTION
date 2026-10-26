# MULTIMODAL HUMAN DETECTION

AI-based multimodal human fatigue, alertness, and facial expression monitoring system.

**Live Web Application:** [https://multimodal-human-detection-dragon-ball-zee.vercel.app](https://multimodal-human-detection-dragon-ball-zee.vercel.app)  
**Author:** [Stuti Dwivedi](https://github.com/Stutii101)  

---

## Overview

A computer vision project designed for real-time monitoring of facial indicators associated with fatigue, alertness, and emotion. The system combines:
1. **Eye Aspect Ratio (EAR)** using 6 facial landmarks per eye.
2. **PERCLOS (Percentage of Eye Closure)** computed over a 30-second sliding temporal window.
3. **MediaPipe Blendshape Analysis** for smiles, sorrow/distress, and yawning.
4. **Multimodal Fatigue Score Fusion** to synthesize these indicators into a single actionable risk index ($0 - 100$) with safety threshold alerts ($\ge 70$).

---

## Architecture

This project is architected with a hybrid model supporting both **Live Cloud Web Deployment** and **Local Desktop Hardware Execution**:

```
                       ┌────────────────────────────────────────────────────────┐
                       │                     USER / BROWSER                     │
                       └───────────────────────────┬────────────────────────────┘
                                                   │
                        ┌──────────────────────────┴───────────────────────────┐
                        │                                                      │
             [Live Camera / Upload]                                    [REST API Sync]
                        │                                                      │
                        ▼                                                      ▼
       ┌─────────────────────────────────┐                    ┌─────────────────────────────────┐
       │     MediaPipe Tasks Vision      │                    │     Python Serverless API       │
       │       (WebAssembly / Wasm)      │                    │          (Vercel)               │
       ├─────────────────────────────────┤                    ├─────────────────────────────────┤
       │ • 30 FPS client-side processing │                    │ • GET  /api/health              │
       │ • Private: zero frames streamed │                    │ • POST /api/analyze             │
       │ • Exact faceyesmile.py math:    │                    │ • Telemetry logging & validation│
       │   - EAR (362, 385, 387, ...)    │                    └─────────────────────────────────┘
       │   - PERCLOS sliding window      │
       │   - Blendshape expressions      │
       │   - Fatigue score & Alerts      │
       │ • Real-time OpenCV-style HUD    │
       └─────────────────────────────────┘
```

---

## Mathematical Foundations

### 1. Eye Aspect Ratio (EAR)
Calculated using Euclidean distance across 6 landmark points for each eye:
$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$
- **Left Eye Indices:** `[362, 385, 387, 263, 373, 380]`
- **Right Eye Indices:** `[33, 160, 158, 133, 153, 144]`
- **Eye Closure Threshold:** $\text{EAR} < 0.21$

### 2. PERCLOS (Percentage of Eye Closure)
Tracks the proportion of time in a rolling 30-second window during which the eyes are closed ($\text{EAR} < 0.21$):
$$\text{PERCLOS} = \frac{\text{Closed Frames in 30s Window}}{\text{Total Frames in 30s Window}}$$

### 3. Facial Expression & Yawn Detection
Extracted from MediaPipe Face Landmarker blendshape scores:
- **Smile:** $\max(\text{mouthSmileLeft}, \text{mouthSmileRight}) > 0.40$
- **Sorrow / Distress:** $\max(\text{mouthFrownLeft}, \text{mouthFrownRight}) > 0.30 \land \max(\text{browDownLeft}, \text{browDownRight}) > 0.30$
- **Yawn Proxy:** $\text{jawOpen} > 0.40$

### 4. Multimodal Fatigue Fusion Score
$$\text{Score} = \min\Big(100,\; (\text{PERCLOS} \times 60) + (20 \text{ if Yawning else } 0) + (10 \text{ if Sorrow else } 0)\Big)$$
- **0 – 39:** Alert / Normal (Green)
- **40 – 69:** Moderate Fatigue (Amber)
- **70 – 100:** **CRITICAL HIGH FATIGUE ALERT** (Red with visual & audio warnings)

---

## Deployment & Usage

### 1. Live Web Application (Vercel)
Visit the live deployed application:  
👉 **[https://multimodal-human-detection-dragon-ball-zee.vercel.app](https://multimodal-human-detection-dragon-ball-zee.vercel.app)**

Features:
- **Live Camera Mode:** Real-time 30 FPS facial fatigue monitoring running via MediaPipe WebAssembly.
- **Sample Presets:** Instant testing of Alert, Smiling, Yawning, and High Fatigue states without a webcam.
- **Photo Upload:** Upload static photos to analyze face landmarks, EAR, and blendshapes.
- **Serverless API:** Built-in Python API endpoints (`/api/health` and `/api/analyze`).

---

### 2. Local Desktop Script (`faceyesmile.py`)
For running directly on your laptop with OpenCV and webcam GUI windows:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Stutii101/MULTIMODAL-HUMAN-DETECTION.git
   cd MULTIMODAL-HUMAN-DETECTION
   ```

2. **Install local dependencies:**
   ```bash
   pip install -r requirements-local.txt
   ```

3. **Download the MediaPipe model asset (~3.7MB):**
   ```powershell
   Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "face_landmarker.task"
   ```

4. **Run the desktop application:**
   ```bash
   python faceyesmile.py
   ```
   *Press `q` to exit.*

---

## API Documentation

### `GET /api/health`
Returns system health, runtime details, and algorithm thresholds.

**Response:**
```json
{
  "status": "healthy",
  "project": "MULTIMODAL-HUMAN-DETECTION",
  "version": "1.0.0",
  "architecture": "Hybrid Client-Side MediaPipe WASM + Python Serverless API"
}
```

### `POST /api/analyze`
Computes the multimodal fatigue score given facial metrics.

**Payload:**
```json
{
  "ear": 0.15,
  "perclos": 0.75,
  "smile_score": 0.05,
  "frown_score": 0.45,
  "brow_down_score": 0.35,
  "jaw_open_score": 0.65
}
```

**Response:**
```json
{
  "success": true,
  "fatigue_score": 75.0,
  "alert": true,
  "alert_level": "ALERT: HIGH FATIGUE",
  "status_color": "#ef4444"
}
```

---

## License & Credits

Developed by **Stuti Dwivedi**.  
Powered by Google MediaPipe, OpenCV, Python, and Vercel.
