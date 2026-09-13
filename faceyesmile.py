"""
Multimodal Fatigue Monitor - Face Landmarker + Expression Detection
----------------------------------------------------------------------
Uses MediaPipe's Face Landmarker to output 52 "blendshape" scores
(0.0-1.0) per frame - including ready-made smile and frown/sorrow
indicators, plus EAR-based PERCLOS for fatigue and jaw-open for yawning.
No TensorFlow/DeepFace needed.

SETUP (inside your Python venv):
    pip install opencv-python mediapipe numpy

You also need the face_landmarker model file (~3.7MB), placed in the
SAME FOLDER as this script:
    Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "face_landmarker.task"

Run:
    python fatigue_monitor_blendshapes.py
Press 'q' to quit.
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Absolute path - finds the model file next to THIS script file,
# regardless of which folder your terminal is sitting in when you run it.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

# ---- Thresholds (tune after testing on your own faces) ----
EAR_THRESHOLD = 0.21
WINDOW_SECONDS = 30
SMILE_THRESHOLD = 0.4     # blendshape score above this = smiling
FROWN_THRESHOLD = 0.3     # blendshape score above this = frowning/sorrow
BROW_DOWN_THRESHOLD = 0.3 # furrowed brow, often paired with sorrow/distress
JAW_OPEN_THRESHOLD = 0.4  # used as a yawn proxy

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, eye_points, w, h):
    coords = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_points]
    vertical1 = euclidean(coords[1], coords[5])
    vertical2 = euclidean(coords[2], coords[4])
    horizontal = euclidean(coords[0], coords[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)


def get_blendshape_score(blendshapes, name):
    """Look up a specific blendshape score by category name."""
    for shape in blendshapes:
        if shape.category_name == name:
            return shape.score
    return 0.0


def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
        print("Download it with:")
        print('  Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "face_landmarker.task"')
        return

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    eye_closed_history = deque()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    frame_timestamp_ms = 0
    fps_estimate = 30  # rough estimate for VIDEO mode timestamps

    while True:
        ret, frame = cap.read()
        if not ret:
            print("no camera feed")
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms += int(1000 / fps_estimate)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        now = time.time()

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            blendshapes = result.face_blendshapes[0] if result.face_blendshapes else []

            # ---- Fatigue: PERCLOS from EAR ----
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            eye_closed_history.append((now, avg_ear < EAR_THRESHOLD))
            while eye_closed_history and now - eye_closed_history[0][0] > WINDOW_SECONDS:
                eye_closed_history.popleft()
            closed_frames = sum(1 for _, closed in eye_closed_history if closed)
            perclos = closed_frames / len(eye_closed_history) if eye_closed_history else 0

            # ---- Expression scores straight from blendshapes ----
            smile_score = max(
                get_blendshape_score(blendshapes, "mouthSmileLeft"),
                get_blendshape_score(blendshapes, "mouthSmileRight"),
            )
            frown_score = max(
                get_blendshape_score(blendshapes, "mouthFrownLeft"),
                get_blendshape_score(blendshapes, "mouthFrownRight"),
            )
            brow_down_score = max(
                get_blendshape_score(blendshapes, "browDownLeft"),
                get_blendshape_score(blendshapes, "browDownRight"),
            )
            jaw_open_score = get_blendshape_score(blendshapes, "jawOpen")

            is_smiling = smile_score > SMILE_THRESHOLD
            # sorrow = frown + inner brow tension together, more reliable than frown alone
            is_sorrowful = frown_score > FROWN_THRESHOLD and brow_down_score > BROW_DOWN_THRESHOLD
            is_yawning = jaw_open_score > JAW_OPEN_THRESHOLD

            # ---- Fatigue fusion score ----
            score = perclos * 60
            score += 20 if is_yawning else 0
            score += 10 if is_sorrowful else 0
            fatigue_score = min(100, score)

            # ---- Overlays ----
            cv2.putText(frame, f"EAR: {avg_ear:.2f}  PERCLOS: {perclos*100:.0f}%",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, f"Smile: {smile_score:.2f} {'(YES)' if is_smiling else ''}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(frame, f"Sorrow: {frown_score:.2f}/brow {brow_down_score:.2f} {'(YES)' if is_sorrowful else ''}",
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 2)
            cv2.putText(frame, f"Yawning: {'YES' if is_yawning else 'no'}",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            color = (0, 255, 0) if fatigue_score < 40 else \
                    (0, 165, 255) if fatigue_score < 70 else (0, 0, 255)
            cv2.putText(frame, f"FATIGUE SCORE: {fatigue_score:.0f}",
                        (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if fatigue_score >= 70:
                cv2.putText(frame, "ALERT: HIGH FATIGUE!", (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "No face detected", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('Multimodal Fatigue Monitor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()