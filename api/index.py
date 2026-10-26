from http.server import BaseHTTPRequestHandler
import json
import time
from urllib.parse import urlparse

class handler(BaseHTTPRequestHandler):
    """
    Vercel Serverless Function entrypoint for MULTIMODAL-HUMAN-DETECTION.
    Provides /api/health and /api/analyze endpoints mirroring the logic in faceyesmile.py.
    """

    def _set_headers(self, status_code=200, content_type='application/json'):
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip('/')

        if path in ('/api/health', '/api', '/api/status', ''):
            response_data = {
                "status": "healthy",
                "project": "MULTIMODAL-HUMAN-DETECTION",
                "version": "1.0.0",
                "author": "Stuti Dwivedi",
                "architecture": "Hybrid Client-Side MediaPipe WASM + Python Serverless API",
                "endpoints": {
                    "health": "/api/health",
                    "analyze": "/api/analyze"
                },
                "thresholds": {
                    "EAR_THRESHOLD": 0.21,
                    "WINDOW_SECONDS": 30,
                    "SMILE_THRESHOLD": 0.4,
                    "FROWN_THRESHOLD": 0.3,
                    "BROW_DOWN_THRESHOLD": 0.3,
                    "JAW_OPEN_THRESHOLD": 0.4,
                    "FATIGUE_ALERT_THRESHOLD": 70
                },
                "server_time": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(response_data, indent=2).encode('utf-8'))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found", "path": self.path}).encode('utf-8'))

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip('/')

        if path in ('/api/analyze', '/api/calculate-fatigue'):
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode('utf-8')
                payload = json.loads(body) if body else {}

                # Extract metrics or defaults
                ear = float(payload.get('ear', 0.25))
                perclos = float(payload.get('perclos', 0.0))
                smile_score = float(payload.get('smile_score', 0.0))
                frown_score = float(payload.get('frown_score', 0.0))
                brow_down_score = float(payload.get('brow_down_score', 0.0))
                jaw_open_score = float(payload.get('jaw_open_score', 0.0))

                # Exact thresholds from faceyesmile.py
                SMILE_THRESHOLD = 0.4
                FROWN_THRESHOLD = 0.3
                BROW_DOWN_THRESHOLD = 0.3
                JAW_OPEN_THRESHOLD = 0.4

                is_smiling = smile_score > SMILE_THRESHOLD
                is_sorrowful = (frown_score > FROWN_THRESHOLD) and (brow_down_score > BROW_DOWN_THRESHOLD)
                is_yawning = jaw_open_score > JAW_OPEN_THRESHOLD

                # Fatigue fusion score formula from faceyesmile.py:
                # score = perclos * 60 + (20 if is_yawning else 0) + (10 if is_sorrowful else 0)
                score = (perclos * 60.0)
                if is_yawning:
                    score += 20.0
                if is_sorrowful:
                    score += 10.0
                fatigue_score = min(100.0, round(score, 1))

                alert = fatigue_score >= 70.0
                if fatigue_score >= 70.0:
                    alert_level = "ALERT: HIGH FATIGUE"
                    status_color = "#ef4444" # red
                elif fatigue_score >= 40.0:
                    alert_level = "WARNING: MODERATE FATIGUE"
                    status_color = "#f59e0b" # amber
                else:
                    alert_level = "NORMAL / ALERT"
                    status_color = "#10b981" # green

                result = {
                    "success": True,
                    "metrics": {
                        "ear": round(ear, 3),
                        "perclos": round(perclos, 3),
                        "perclos_percentage": round(perclos * 100, 1),
                        "smile_score": round(smile_score, 3),
                        "frown_score": round(frown_score, 3),
                        "brow_down_score": round(brow_down_score, 3),
                        "jaw_open_score": round(jaw_open_score, 3)
                    },
                    "detections": {
                        "is_smiling": is_smiling,
                        "is_sorrowful": is_sorrowful,
                        "is_yawning": is_yawning
                    },
                    "fatigue_score": fatigue_score,
                    "alert": alert,
                    "alert_level": alert_level,
                    "status_color": status_color,
                    "timestamp": time.time()
                }

                self._set_headers(200)
                self.wfile.write(json.dumps(result, indent=2).encode('utf-8'))
            except Exception as e:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": f"Invalid request body: {str(e)}"}).encode('utf-8'))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found", "path": self.path}).encode('utf-8'))
