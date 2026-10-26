import { FilesetResolver, FaceLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// Exact constants and thresholds from faceyesmile.py
const EAR_THRESHOLD = 0.21;
const WINDOW_SECONDS = 30;
const SMILE_THRESHOLD = 0.4;
const FROWN_THRESHOLD = 0.3;
const BROW_DOWN_THRESHOLD = 0.3;
const JAW_OPEN_THRESHOLD = 0.4;
const FATIGUE_ALERT_THRESHOLD = 70;

const LEFT_EYE = [362, 385, 387, 263, 373, 380];
const RIGHT_EYE = [33, 160, 158, 133, 153, 144];

// State variables
let faceLandmarker = null;
let webcamRunning = false;
let videoStream = null;
let lastVideoTime = -1;
let animFrameId = null;
let showMesh = true;
let soundEnabled = false;

// History tracking
let eyeClosedHistory = []; // Array of { time: timestamp_ms, closed: boolean }
let sessionStartTime = null;
let blinkCount = 0;
let yawnCount = 0;
let alertCount = 0;
let lastWasClosed = false;
let lastWasYawning = false;
let lastWasAlert = false;
let fpsCounter = 0;
let fpsLastTime = performance.now();
let currentFps = 0;

// Current telemetry state
let currentMetrics = {
  ear: 0.28,
  perclos: 0.0,
  smile: 0.0,
  frown: 0.0,
  browDown: 0.0,
  jawOpen: 0.0,
  fatigueScore: 0,
  isSmiling: false,
  isSorrowful: false,
  isYawning: false,
  isAlert: false
};

// DOM Elements
const video = document.getElementById("webcam-video");
const canvas = document.getElementById("output-canvas");
const ctx = canvas.getContext("2d");
const placeholder = document.getElementById("viewport-placeholder");
const alertBanner = document.getElementById("alert-banner");
const alertSound = document.getElementById("alert-sound");

const btnToggleCamera = document.getElementById("btn-toggle-camera");
const btnCameraLabel = document.getElementById("btn-camera-label");
const btnToggleMesh = document.getElementById("btn-toggle-mesh");
const meshStateLabel = document.getElementById("mesh-state-label");
const btnToggleSound = document.getElementById("btn-toggle-sound");
const soundStateLabel = document.getElementById("sound-state-label");

const fpsBadge = document.getElementById("fps-badge");
const modelStatusDot = document.getElementById("model-status-dot");
const modelStatusText = document.getElementById("model-status-text");
const apiStatusDot = document.getElementById("api-status-dot");
const apiStatusText = document.getElementById("api-status-text");

// Gauge Elements
const gaugeFill = document.getElementById("gauge-fill-circle");
const fatigueScoreValue = document.getElementById("fatigue-score-value");
const riskBadge = document.getElementById("risk-badge");

// Metrics Card Elements
const earVal = document.getElementById("ear-val");
const earState = document.getElementById("ear-state");
const earBar = document.getElementById("ear-bar");

const perclosVal = document.getElementById("perclos-val");
const perclosState = document.getElementById("perclos-state");
const perclosBar = document.getElementById("perclos-bar");

const smileVal = document.getElementById("smile-val");
const smileState = document.getElementById("smile-state");
const smileBar = document.getElementById("smile-bar");

const yawnVal = document.getElementById("yawn-val");
const yawnState = document.getElementById("yawn-state");
const yawnBar = document.getElementById("yawn-bar");

// HUD Chips
const hudEar = document.getElementById("hud-ear");
const hudPerclos = document.getElementById("hud-perclos");
const hudSmile = document.getElementById("hud-smile");
const hudYawn = document.getElementById("hud-yawn");
const hudSorrow = document.getElementById("hud-sorrow");

// Stats Elements
const statBlinks = document.getElementById("stat-blinks");
const statYawns = document.getElementById("stat-yawns");
const statAlerts = document.getElementById("stat-alerts");
const statTime = document.getElementById("stat-time");

// API Inspection Elements
const btnTestApi = document.getElementById("btn-test-api");
const apiEndpointLabel = document.getElementById("api-endpoint-label");
const apiStatusBadge = document.getElementById("api-status-badge");
const apiJsonOutput = document.getElementById("api-json-output");

// File Upload
const imageUploadInput = document.getElementById("image-upload-input");

// -------------------------------------------------------------
// Mathematical Helper Functions (Identical to faceyesmile.py)
// -------------------------------------------------------------
function euclidean(p1, p2) {
  const dx = p1[0] - p2[0];
  const dy = p1[1] - p2[1];
  return Math.sqrt(dx * dx + dy * dy);
}

function eyeAspectRatio(landmarks, eyePoints, w, h) {
  const coords = eyePoints.map(i => [landmarks[i].x * w, landmarks[i].y * h]);
  const vertical1 = euclidean(coords[1], coords[5]);
  const vertical2 = euclidean(coords[2], coords[4]);
  const horizontal = euclidean(coords[0], coords[3]);
  if (horizontal === 0) return 0.0;
  return (vertical1 + vertical2) / (2.0 * horizontal);
}

function getBlendshapeScore(blendshapes, name) {
  if (!blendshapes || !blendshapes.length) return 0.0;
  for (let i = 0; i < blendshapes.length; i++) {
    const shape = blendshapes[i];
    if (shape.categoryName === name) {
      return shape.score || 0.0;
    }
  }
  return 0.0;
}

// -------------------------------------------------------------
// Initialize MediaPipe FaceLandmarker
// -------------------------------------------------------------
async function initMediaPipe() {
  try {
    modelStatusText.textContent = "Loading MediaPipe WASM...";
    const filesetResolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    modelStatusText.textContent = "Downloading Model (~3.7MB)...";
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU"
      },
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: false,
      runningMode: "VIDEO",
      numFaces: 1
    });

    modelStatusDot.className = "status-dot success";
    modelStatusText.textContent = "Vision Model: Ready";
  } catch (err) {
    console.error("Failed to load MediaPipe model:", err);
    modelStatusDot.className = "status-dot danger";
    modelStatusText.textContent = "Model Load Failed";
  }
}

// -------------------------------------------------------------
// Check Serverless Python API Health
// -------------------------------------------------------------
async function checkApiHealth() {
  try {
    const response = await fetch("/api/health");
    if (response.ok) {
      const data = await response.json();
      apiStatusDot.className = "status-dot success";
      apiStatusText.textContent = "Python API: Connected";
      apiJsonOutput.textContent = JSON.stringify(data, null, 2);
      apiStatusBadge.textContent = "200 OK";
      apiStatusBadge.className = "badge badge-success";
      apiEndpointLabel.textContent = "GET /api/health";
    } else {
      throw new Error(`Status ${response.status}`);
    }
  } catch (err) {
    console.warn("Python API check:", err.message);
    apiStatusDot.className = "status-dot warning";
    apiStatusText.textContent = "Python API: Standalone";
  }
}

// -------------------------------------------------------------
// Webcam Handling
// -------------------------------------------------------------
async function startWebcam() {
  if (!faceLandmarker) {
    alert("Please wait, MediaPipe Face Landmarker model is still loading.");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false
    });

    videoStream = stream;
    video.srcObject = stream;
    await video.play();

    webcamRunning = true;
    placeholder.style.display = "none";
    btnToggleCamera.classList.add("btn-active-stop");
    btnCameraLabel.textContent = "Stop Webcam";

    sessionStartTime = Date.now();
    renderLoop();
  } catch (err) {
    console.error("Error accessing webcam:", err);
    alert(`Could not open webcam: ${err.message}. Please check browser camera permissions.`);
  }
}

function stopWebcam() {
  webcamRunning = false;
  if (videoStream) {
    videoStream.getTracks().forEach(track => track.stop());
    videoStream = null;
  }
  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }
  placeholder.style.display = "flex";
  btnToggleCamera.classList.remove("btn-active-stop");
  btnCameraLabel.textContent = "Start Webcam";
  fpsBadge.textContent = "0 FPS";
  alertBanner.style.display = "none";
}

btnToggleCamera.addEventListener("click", () => {
  if (webcamRunning) {
    stopWebcam();
  } else {
    startWebcam();
  }
});

btnToggleMesh.addEventListener("click", () => {
  showMesh = !showMesh;
  meshStateLabel.textContent = showMesh ? "ON" : "OFF";
});

btnToggleSound.addEventListener("click", () => {
  soundEnabled = !soundEnabled;
  soundStateLabel.textContent = soundEnabled ? "ON" : "OFF";
});

// -------------------------------------------------------------
// Real-Time Frame Processing & Rendering Loop
// -------------------------------------------------------------
function renderLoop() {
  if (!webcamRunning) return;

  const now = performance.now();
  fpsCounter++;
  if (now - fpsLastTime >= 1000) {
    currentFps = fpsCounter;
    fpsCounter = 0;
    fpsLastTime = now;
    fpsBadge.textContent = `${currentFps} FPS`;
  }

  // Ensure canvas dimensions match video
  if (video.videoWidth > 0 && video.videoHeight > 0) {
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const startTimeMs = performance.now();
      const results = faceLandmarker.detectForVideo(video, startTimeMs);
      processAndDrawResults(video, results);
    }
  }

  updateSessionTimer();
  animFrameId = requestAnimationFrame(renderLoop);
}

// -------------------------------------------------------------
// Process Results and Draw OpenCV HUD
// -------------------------------------------------------------
function processAndDrawResults(imageSource, results) {
  const w = canvas.width;
  const h = canvas.height;

  // Clear and draw video frame
  ctx.save();
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(imageSource, 0, 0, w, h);

  const currentTime = Date.now();

  if (results && results.faceLandmarks && results.faceLandmarks.length > 0) {
    const landmarks = results.faceLandmarks[0];
    const blendshapes = (results.faceBlendshapes && results.faceBlendshapes.length > 0) 
      ? results.faceBlendshapes[0].categories 
      : [];

    // 1. Calculate EAR
    const leftEar = eyeAspectRatio(landmarks, LEFT_EYE, w, h);
    const rightEar = eyeAspectRatio(landmarks, RIGHT_EYE, w, h);
    const avgEar = (leftEar + rightEar) / 2.0;

    // Blink counter
    const isClosed = avgEar < EAR_THRESHOLD;
    if (isClosed && !lastWasClosed) {
      blinkCount++;
      statBlinks.textContent = blinkCount;
    }
    lastWasClosed = isClosed;

    // 2. Sliding Window PERCLOS (30s)
    eyeClosedHistory.push({ time: currentTime, closed: isClosed });
    while (eyeClosedHistory.length && (currentTime - eyeClosedHistory[0].time > WINDOW_SECONDS * 1000)) {
      eyeClosedHistory.shift();
    }
    const closedCount = eyeClosedHistory.filter(item => item.closed).length;
    const perclos = eyeClosedHistory.length ? (closedCount / eyeClosedHistory.length) : 0.0;

    // 3. Blendshapes
    const smileScore = Math.max(
      getBlendshapeScore(blendshapes, "mouthSmileLeft"),
      getBlendshapeScore(blendshapes, "mouthSmileRight")
    );
    const frownScore = Math.max(
      getBlendshapeScore(blendshapes, "mouthFrownLeft"),
      getBlendshapeScore(blendshapes, "mouthFrownRight")
    );
    const browDownScore = Math.max(
      getBlendshapeScore(blendshapes, "browDownLeft"),
      getBlendshapeScore(blendshapes, "browDownRight")
    );
    const jawOpenScore = getBlendshapeScore(blendshapes, "jawOpen");

    const isSmiling = smileScore > SMILE_THRESHOLD;
    const isSorrowful = (frownScore > FROWN_THRESHOLD) && (browDownScore > BROW_DOWN_THRESHOLD);
    const isYawning = jawOpenScore > JAW_OPEN_THRESHOLD;

    if (isYawning && !lastWasYawning) {
      yawnCount++;
      statYawns.textContent = yawnCount;
    }
    lastWasYawning = isYawning;

    // 4. Fatigue Score Fusion formula from faceyesmile.py
    let score = perclos * 60;
    if (isYawning) score += 20;
    if (isSorrowful) score += 10;
    const fatigueScore = Math.min(100, Math.round(score));
    const isAlert = fatigueScore >= FATIGUE_ALERT_THRESHOLD;

    if (isAlert && !lastWasAlert) {
      alertCount++;
      statAlerts.textContent = alertCount;
      if (soundEnabled && alertSound) {
        alertSound.currentTime = 0;
        alertSound.play().catch(() => {});
      }
    }
    lastWasAlert = isAlert;

    // Store state
    currentMetrics = {
      ear: avgEar,
      perclos: perclos,
      smile: smileScore,
      frown: frownScore,
      browDown: browDownScore,
      jawOpen: jawOpenScore,
      fatigueScore: fatigueScore,
      isSmiling: isSmiling,
      isSorrowful: isSorrowful,
      isYawning: isYawning,
      isAlert: isAlert
    };

    // 5. Draw Facial Mesh Contours if enabled
    if (showMesh) {
      drawLandmarkMesh(ctx, landmarks, w, h);
    }

    // 6. Draw OpenCV-style HUD Overlays (matching faceyesmile.py exactly)
    drawOpenCvHud(ctx, avgEar, perclos, smileScore, isSmiling, frownScore, browDownScore, isSorrowful, isYawning, fatigueScore);

    // Update UI Elements
    updateDashboardUI(currentMetrics);
  } else {
    // No face detected overlay (from faceyesmile.py line 167)
    ctx.font = "bold 18px sans-serif";
    ctx.fillStyle = "#ef4444";
    ctx.fillText("No face detected", 20, 35);
  }

  ctx.restore();
}

// -------------------------------------------------------------
// OpenCV-Style HUD Overlay Drawing
// -------------------------------------------------------------
function drawOpenCvHud(ctx, avgEar, perclos, smileScore, isSmiling, frownScore, browDownScore, isSorrowful, isYawning, fatigueScore) {
  ctx.save();
  ctx.font = "bold 15px 'JetBrains Mono', monospace";
  ctx.shadowColor = "rgba(0, 0, 0, 0.9)";
  ctx.shadowBlur = 6;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  // Line 1: EAR & PERCLOS
  ctx.fillStyle = "#ffffff";
  ctx.fillText(`EAR: ${avgEar.toFixed(2)}  PERCLOS: ${(perclos * 100).toFixed(0)}%`, 20, 32);

  // Line 2: Smile
  ctx.fillStyle = "#facc15";
  ctx.fillText(`Smile: ${smileScore.toFixed(2)} ${isSmiling ? '(YES)' : ''}`, 20, 58);

  // Line 3: Sorrow
  ctx.fillStyle = "#c7d2fe";
  ctx.fillText(`Sorrow: ${frownScore.toFixed(2)}/brow ${browDownScore.toFixed(2)} ${isSorrowful ? '(YES)' : ''}`, 20, 84);

  // Line 4: Yawning
  ctx.fillStyle = isYawning ? "#ef4444" : "#ffffff";
  ctx.fillText(`Yawning: ${isYawning ? 'YES' : 'no'}`, 20, 110);

  // Line 5: Fatigue Score with dynamic color
  let scoreColor = "#22c55e"; // Green
  if (fatigueScore >= 70) {
    scoreColor = "#ef4444"; // Red
  } else if (fatigueScore >= 40) {
    scoreColor = "#f97316"; // Orange
  }

  ctx.font = "bold 20px 'JetBrains Mono', monospace";
  ctx.fillStyle = scoreColor;
  ctx.fillText(`FATIGUE SCORE: ${fatigueScore}`, 20, 145);

  // Line 6: High Fatigue Alert
  if (fatigueScore >= 70) {
    ctx.font = "bold 22px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#ef4444";
    ctx.fillText("ALERT: HIGH FATIGUE!", 20, 180);
  }

  ctx.restore();
}

// -------------------------------------------------------------
// Draw Landmark Mesh
// -------------------------------------------------------------
function drawLandmarkMesh(ctx, landmarks, w, h) {
  ctx.save();
  ctx.lineWidth = 1.5;

  // Draw eye contours
  ctx.strokeStyle = "#06b6d4";
  drawEyeContour(ctx, landmarks, LEFT_EYE, w, h);
  drawEyeContour(ctx, landmarks, RIGHT_EYE, w, h);

  // Draw mouth contour
  ctx.strokeStyle = "#8b5cf6";
  const mouthIndices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308];
  drawPolygon(ctx, landmarks, mouthIndices, w, h);

  ctx.restore();
}

function drawEyeContour(ctx, landmarks, indices, w, h) {
  drawPolygon(ctx, landmarks, indices, w, h);
}

function drawPolygon(ctx, landmarks, indices, w, h) {
  ctx.beginPath();
  for (let i = 0; i < indices.length; i++) {
    const pt = landmarks[indices[i]];
    const x = pt.x * w;
    const y = pt.y * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.stroke();
}

// -------------------------------------------------------------
// Update Dashboard UI & Cockpit Gauges
// -------------------------------------------------------------
function updateDashboardUI(m) {
  // HUD chips below video
  hudEar.textContent = m.ear.toFixed(2);
  hudPerclos.textContent = `${(m.perclos * 100).toFixed(0)}%`;
  hudSmile.textContent = `${m.smile.toFixed(2)} ${m.isSmiling ? '✓' : ''}`;
  hudYawn.textContent = `${m.jawOpen.toFixed(2)} ${m.isYawning ? '⚠️' : ''}`;
  hudSorrow.textContent = `${m.frown.toFixed(2)}`;

  // Circular Fatigue Gauge
  fatigueScoreValue.textContent = m.fatigueScore;
  const circumference = 2 * Math.PI * 85; // ~534
  const offset = circumference - (m.fatigueScore / 100) * circumference;
  gaugeFill.style.strokeDashoffset = offset;

  if (m.fatigueScore >= 70) {
    gaugeFill.style.stroke = "#ef4444";
    riskBadge.textContent = "CRITICAL FATIGUE";
    riskBadge.className = "badge badge-danger";
    alertBanner.style.display = "block";
  } else if (m.fatigueScore >= 40) {
    gaugeFill.style.stroke = "#f59e0b";
    riskBadge.textContent = "MODERATE FATIGUE";
    riskBadge.className = "badge badge-warning";
    alertBanner.style.display = "none";
  } else {
    gaugeFill.style.stroke = "#10b981";
    riskBadge.textContent = "NORMAL / ALERT";
    riskBadge.className = "badge badge-success";
    alertBanner.style.display = "none";
  }

  // 1. EAR Card
  earVal.textContent = m.ear.toFixed(2);
  earState.textContent = m.ear < EAR_THRESHOLD ? "CLOSED" : "OPEN";
  earState.className = `metric-state ${m.ear < EAR_THRESHOLD ? 'danger' : 'normal'}`;
  earBar.style.width = `${Math.min(100, (m.ear / 0.4) * 100)}%`;

  // 2. PERCLOS Card
  perclosVal.textContent = `${(m.perclos * 100).toFixed(1)}%`;
  perclosState.textContent = m.perclos > 0.4 ? "DROWSY" : "OPTIMAL";
  perclosBar.style.width = `${Math.min(100, m.perclos * 100)}%`;

  // 3. Smile Card
  smileVal.textContent = m.smile.toFixed(2);
  smileState.textContent = m.isSmiling ? "SMILING" : "NEUTRAL";
  smileBar.style.width = `${Math.min(100, (m.smile / 1.0) * 100)}%`;

  // 4. Yawn Card
  yawnVal.textContent = m.jawOpen.toFixed(2);
  yawnState.textContent = m.isYawning ? "YAWNING" : "NORMAL";
  yawnBar.style.width = `${Math.min(100, (m.jawOpen / 1.0) * 100)}%`;
}

function updateSessionTimer() {
  if (!sessionStartTime) return;
  const elapsedSec = Math.floor((Date.now() - sessionStartTime) / 1000);
  const minutes = String(Math.floor(elapsedSec / 60)).padStart(2, "0");
  const seconds = String(elapsedSec % 60).padStart(2, "0");
  statTime.textContent = `${minutes}:${seconds}`;
}

// -------------------------------------------------------------
// Sample Test Preset Scenarios (For testing without webcam)
// -------------------------------------------------------------
function runPresetScenario(ear, perclos, smile, frown, browDown, jawOpen) {
  stopWebcam();
  placeholder.style.display = "none";
  canvas.width = 640;
  canvas.height = 480;

  // Generate stylized avatar face on canvas
  ctx.save();
  ctx.fillStyle = "#0c1222";
  ctx.fillRect(0, 0, 640, 480);

  // Draw face contour outline
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.ellipse(320, 240, 130, 170, 0, 0, 2 * Math.PI);
  ctx.stroke();

  // Draw eyes
  const eyeOpenHeight = ear < 0.2 ? 3 : 16;
  ctx.fillStyle = "#38bdf8";
  ctx.beginPath();
  ctx.ellipse(270, 210, 22, eyeOpenHeight, 0, 0, 2 * Math.PI);
  ctx.ellipse(370, 210, 22, eyeOpenHeight, 0, 0, 2 * Math.PI);
  ctx.fill();

  // Draw mouth
  ctx.beginPath();
  if (jawOpen > 0.4) {
    // Open mouth / yawning
    ctx.ellipse(320, 310, 30, 35, 0, 0, 2 * Math.PI);
    ctx.fillStyle = "#ef4444";
    ctx.fill();
    ctx.stroke();
  } else if (smile > 0.4) {
    // Smiling arc
    ctx.arc(320, 290, 40, 0.1 * Math.PI, 0.9 * Math.PI, false);
    ctx.strokeStyle = "#facc15";
    ctx.lineWidth = 4;
    ctx.stroke();
  } else {
    // Neutral mouth
    ctx.moveTo(290, 310);
    ctx.lineTo(350, 310);
    ctx.stroke();
  }

  // Draw OpenCV HUD directly
  const isSmiling = smile > SMILE_THRESHOLD;
  const isSorrowful = frown > FROWN_THRESHOLD && browDown > BROW_DOWN_THRESHOLD;
  const isYawning = jawOpen > JAW_OPEN_THRESHOLD;

  let score = perclos * 60;
  if (isYawning) score += 20;
  if (isSorrowful) score += 10;
  const fatigueScore = Math.min(100, Math.round(score));

  drawOpenCvHud(ctx, ear, perclos, smile, isSmiling, frown, browDown, isSorrowful, isYawning, fatigueScore);
  ctx.restore();

  currentMetrics = {
    ear, perclos, smile, frown, browDown, jawOpen, fatigueScore,
    isSmiling, isSorrowful, isYawning, isAlert: fatigueScore >= 70
  };

  updateDashboardUI(currentMetrics);
}

document.getElementById("btn-sample-alert").addEventListener("click", () => {
  runPresetScenario(0.32, 0.05, 0.1, 0.05, 0.05, 0.0);
});

document.getElementById("btn-sample-smile").addEventListener("click", () => {
  runPresetScenario(0.29, 0.02, 0.85, 0.05, 0.05, 0.05);
});

document.getElementById("btn-sample-yawn").addEventListener("click", () => {
  runPresetScenario(0.18, 0.25, 0.05, 0.1, 0.1, 0.8);
});

document.getElementById("btn-sample-drowsy").addEventListener("click", () => {
  runPresetScenario(0.12, 0.85, 0.0, 0.6, 0.5, 0.7);
});

// -------------------------------------------------------------
// Image File Upload Testing
// -------------------------------------------------------------
imageUploadInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const img = new Image();
  img.src = URL.createObjectURL(file);
  img.onload = async () => {
    stopWebcam();
    placeholder.style.display = "none";
    canvas.width = img.width;
    canvas.height = img.height;

    if (faceLandmarker) {
      const results = faceLandmarker.detect(img);
      processAndDrawResults(img, results);
    } else {
      ctx.drawImage(img, 0, 0);
    }
  };
});

// -------------------------------------------------------------
// Test /api/analyze Serverless Endpoint
// -------------------------------------------------------------
btnTestApi.addEventListener("click", async () => {
  btnTestApi.textContent = "Calling API...";
  apiEndpointLabel.textContent = "POST /api/analyze";

  const payload = {
    ear: currentMetrics.ear,
    perclos: currentMetrics.perclos,
    smile_score: currentMetrics.smile,
    frown_score: currentMetrics.frown,
    brow_down_score: currentMetrics.browDown,
    jaw_open_score: currentMetrics.jawOpen
  };

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    apiStatusBadge.textContent = `${res.status} OK`;
    apiStatusBadge.className = res.ok ? "badge badge-success" : "badge badge-danger";
    apiJsonOutput.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    apiStatusBadge.textContent = "Error";
    apiStatusBadge.className = "badge badge-danger";
    apiJsonOutput.textContent = JSON.stringify({ error: err.message }, null, 2);
  } finally {
    btnTestApi.textContent = "Test /api/analyze";
  }
});

// Auto-run on load
initMediaPipe();
checkApiHealth();
