"""
app.py

Flask API + simple web UI that accepts an uploaded video file and returns
the predicted action, using the same CNN+LSTM pipeline as training.

Uses a proper open-set rejection mechanism: compares the video's internal
embedding against each class's centroid (learned from training data). If
it's too far from every centroid, it's flagged as "not a trained action" -
this generalizes to genuinely unfamiliar inputs, not just a fixed list of
negative examples.

Run locally with:
    python app.py
"""

import os
import json
import base64
import tempfile

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask import Flask, request, jsonify

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "action_recognition_model.keras")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "models", "labels.json")
OPEN_SET_PATH = os.path.join(os.path.dirname(__file__), "models", "open_set_profile.json")
SEQUENCE_LENGTH = 20
FRAME_SIZE = 128

app = Flask(__name__)

print("Loading trained LSTM model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH) as f:
    LABELS = json.load(f)
with open(OPEN_SET_PATH) as f:
    OPEN_SET_PROFILE = json.load(f)
    for class_name in OPEN_SET_PROFILE:
        OPEN_SET_PROFILE[class_name]["centroid"] = np.array(
            OPEN_SET_PROFILE[class_name]["centroid"], dtype=np.float32
        )

print("Building embedding sub-model for open-set rejection...")
inp = tf.keras.Input(shape=(SEQUENCE_LENGTH, 1280))
x = inp
embedding_output = None
for layer in model.layers:
    x = layer(x)
    if layer.name == "embedding_layer":
        embedding_output = x
embedding_model = tf.keras.Model(inputs=inp, outputs=embedding_output)

print("Loading frozen MobileNetV2 feature extractor...")
feature_extractor = MobileNetV2(
    input_shape=(FRAME_SIZE, FRAME_SIZE, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
)
feature_extractor.trainable = False


def sample_frame_indices(total_frames, target_len):
    if total_frames <= 0:
        return []
    if total_frames <= target_len:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, target_len).astype(int).tolist()


def extract_sequence(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(sample_frame_indices(total_frames, SEQUENCE_LENGTH))

    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        idx += 1
    cap.release()

    if not frames:
        return None

    batch = np.array(frames, dtype=np.float32)
    batch = preprocess_input(batch)
    features = feature_extractor.predict(batch, verbose=0)

    if len(features) < SEQUENCE_LENGTH:
        pad = np.tile(features[-1], (SEQUENCE_LENGTH - len(features), 1))
        features = np.vstack([features, pad])

    return features


def check_open_set(sequence, predicted_label):
    """Returns (is_known, distance, threshold) by comparing this video's
    embedding against the predicted class's centroid."""
    embedding = embedding_model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
    profile = OPEN_SET_PROFILE[predicted_label]
    distance = float(np.linalg.norm(embedding - profile["centroid"]))
    threshold = profile["threshold"]
    return distance <= threshold, distance, threshold


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ActionSense</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --bg: #06110d;
        --panel: rgba(7, 25, 18, 0.88);
        --panel-solid: #081a13;
        --panel-hover: rgba(11, 45, 31, 0.94);
        --border: rgba(69, 196, 128, 0.32);
        --border-soft: rgba(69, 196, 128, 0.18);
        --text: #d9f5e4;
        --text-dim: #8fb6a0;
        --accent: #35f27b;
        --accent-soft: #a9f2c1;
        --accent-dark: #08784b;
        --deep: #04100b;
        --danger: #ff8f9a;
    }
    * { box-sizing: border-box; }
    body {
        margin: 0;
        background: #3E745B;
        color: var(--text);
        font-family: 'Space Grotesk', sans-serif;
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 32px 24px;
    }
    .page {
        width: 100%;
        max-width: 1080px;
        padding: 34px;
        border-radius: 24px;
        background: linear-gradient(145deg, rgba(8, 27, 19, 0.92), rgba(4, 16, 11, 0.82));
        border: 1px solid rgba(81, 222, 139, 0.18);
        box-shadow:
            0 28px 90px rgba(0, 0, 0, 0.5),
            0 0 70px rgba(33, 215, 109, 0.10);
        backdrop-filter: blur(18px);
    }
    .head { margin-bottom: 32px; padding: 4px 2px; }
    .title {
        font-size: 54px;
        font-weight: 700;
        margin: 0 0 8px 0;
        color: var(--accent-soft);
        text-shadow: 0 0 30px rgba(53, 242, 123, 0.16);
        letter-spacing: -0.01em;
    }
    .subtitle {
        color: var(--accent-soft);
        font-size: 19px;
        margin: 0;
    }
    .subtitle::first-line { font-weight: 600; }

    .layout {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 0.92fr);
        gap: 28px;
        align-items: stretch;
    }
    .right {
        border-left: 1px solid var(--border-soft);
        padding-left: 28px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 32px;
    }

    /* --- Viewfinder upload zone --- */
    .viewfinder {
        position: relative;
        aspect-ratio: 16 / 10;
        border-radius: 18px;
        background:
            linear-gradient(145deg, rgba(10, 35, 24, 0.96), rgba(4, 18, 12, 0.94));
        border: 1px solid var(--border);
        box-shadow:
            inset 0 1px 0 rgba(170, 255, 203, 0.05),
            0 18px 50px rgba(0, 0, 0, 0.32),
            0 0 34px rgba(31, 219, 111, 0.08);
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 24px;
        overflow: hidden;
        transition: background 0.2s ease, box-shadow 0.25s ease;
    }
    .viewfinder:hover {
        background: linear-gradient(145deg, rgba(12, 48, 32, 0.98), rgba(5, 22, 15, 0.96));
        border-color: rgba(53, 242, 123, 0.48);
        box-shadow:
            inset 0 1px 0 rgba(170, 255, 203, 0.07),
            0 18px 55px rgba(0, 0, 0, 0.34),
            0 0 42px rgba(31, 219, 111, 0.14);
    }
    .viewfinder.active { box-shadow: 0 0 0 1px rgba(236, 141, 171, 0.45), 0 0 28px rgba(236, 141, 171, 0.16); }
    .corner {
        position: absolute;
        width: 22px;
        height: 22px;
        border-color: var(--border);
        transition: border-color 0.25s ease;
        z-index: 2;
    }
    .viewfinder.active .corner { border-color: var(--accent); }
    .corner.tl { top: 14px; left: 14px; border-top: 2px solid; border-left: 2px solid; }
    .corner.tr { top: 14px; right: 14px; border-top: 2px solid; border-right: 2px solid; }
    .corner.bl { bottom: 14px; left: 14px; border-bottom: 2px solid; border-left: 2px solid; }
    .corner.br { bottom: 14px; right: 14px; border-bottom: 2px solid; border-right: 2px solid; }

    .vf-label { color: var(--text-dim); font-size: 14px; line-height: 1.5; z-index: 1; position: relative; }
    .vf-label.hidden { display: none; }
    .vf-label .filename { color: var(--text); font-weight: 500; word-break: break-all; }
    .vf-label .vf-note { font-size: 12px; margin-top: 8px; }

    .preview-media {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: none;
    }
    .preview-media.show { display: block; }

    .scanline {
        position: absolute;
        left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent), transparent);
        box-shadow: 0 0 12px rgba(236, 141, 171, 0.8);
        top: 0;
        opacity: 0;
        pointer-events: none;
        z-index: 2;
    }
    .viewfinder.scanning .scanline {
        opacity: 1;
        animation: sweep 1.1s linear infinite;
    }
    @keyframes sweep {
        0% { top: 6%; }
        100% { top: 94%; }
    }

    input[type="file"] { display: none; }

    .analyze-btn {
        width: 100%;
        margin-top: 20px;
        padding: 14px;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 15px;
        font-weight: 600;
        color: var(--deep);
        background: linear-gradient(180deg, #45f88b, #18c969);
        border: none;
        border-radius: 12px;
        cursor: pointer;
        transition: opacity 0.2s ease;
    }
    .analyze-btn:disabled { background: var(--accent-dark); color: var(--text-dim); cursor: not-allowed; }
    .analyze-btn:not(:disabled):hover {
        background: linear-gradient(180deg, #68ffa1, #20d777);
        color: #03200f;
        box-shadow: 0 0 28px rgba(53, 242, 123, 0.22);
    }

    .error-msg { color: var(--accent); font-size: 14px; margin-top: 16px; display: none; }
    .error-msg.show { display: block; }

    /* --- Result readout --- */
    .result-status { font-size: 14px; color: var(--text-dim); margin-bottom: 10px; }
    .result-action { font-size: 30px; font-weight: 700; color: var(--accent-soft); }
    .result-action.no-match { color: var(--accent); }
    .result-readout {
        font-family: 'JetBrains Mono', monospace;
        font-size: 48px;
        font-weight: 500;
        color: var(--accent);
        margin-top: 4px;
        letter-spacing: -0.02em;
    }
    .result-hint { color: var(--text-dim); font-size: 14px; margin-top: 10px; line-height: 1.5; }
    .idle { color: var(--text-dim); font-size: 15px; line-height: 1.6; }

    /* --- Supported actions legend --- */
    .legend-title { font-size: 13px; color: var(--text-dim); margin-bottom: 12px; }
    .legend-tags { display: flex; flex-wrap: wrap; gap: 8px; }
    .legend-tag {
        font-size: 13px;
        padding: 6px 12px;
        border: 1px solid var(--border);
        border-radius: 999px;
        color: var(--accent-soft);
        background: rgba(17, 55, 38, 0.65);
        border-color: rgba(69, 196, 128, 0.28);
        box-shadow: inset 0 1px 0 rgba(180, 255, 207, 0.04);
    }

    @media (max-width: 860px) {
        body { align-items: flex-start; padding: 18px 12px; }
        .page { padding: 24px; border-radius: 18px; }
        .title { font-size: 40px; }
        .layout { grid-template-columns: minmax(0, 1fr); gap: 32px; }
        .right {
            border-left: none;
            padding-left: 0;
            border-top: 1px solid var(--border-soft);
            padding-top: 28px;
        }
    }
</style>
</head>
<body>
<div class="page">
    <div class="head">
        <h1 class="title">ActionSense</h1>
        <p class="subtitle">Upload a video and it tells you what action it sees.</p>
    </div>

    <div class="layout">
        <div class="left">
            <div id="viewfinder" class="viewfinder">
                <div class="corner tl"></div>
                <div class="corner tr"></div>
                <div class="corner bl"></div>
                <div class="corner br"></div>
                <div class="scanline"></div>
                <video id="previewVideo" class="preview-media" muted loop playsinline></video>
                <img id="previewImg" class="preview-media" alt="">
                <div class="vf-label" id="vfLabel">Click or drop a video here</div>
                <input type="file" id="fileInput" accept="video/*">
            </div>
            <button id="analyzeBtn" class="analyze-btn" disabled>Analyze video</button>
            <div class="error-msg" id="errorMsg"></div>
        </div>

        <div class="right">
            <div>
                <div class="result-status" id="resultStatus">Result</div>
                <div id="resultBody">
                    <div class="idle">Choose a video, then press Analyze video. The detected action shows up here.</div>
                </div>
            </div>

            <div>
                <div class="legend-title">Trained to recognize six actions</div>
                <div class="legend-tags">
                    <span class="legend-tag">Biking</span>
                    <span class="legend-tag">Horse racing</span>
                    <span class="legend-tag">Jumping jacks</span>
                    <span class="legend-tag">Playing guitar</span>
                    <span class="legend-tag">Push-ups</span>
                    <span class="legend-tag">Walking a dog</span>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    var viewfinder = document.getElementById('viewfinder');
    var fileInput = document.getElementById('fileInput');
    var vfLabel = document.getElementById('vfLabel');
    var previewVideo = document.getElementById('previewVideo');
    var previewImg = document.getElementById('previewImg');
    var frameTimer = null;
    var previewToken = 0;
    var analyzeBtn = document.getElementById('analyzeBtn');
    var errorMsg = document.getElementById('errorMsg');
    var resultStatus = document.getElementById('resultStatus');
    var resultBody = document.getElementById('resultBody');
    var selectedFile = null;
    var currentObjectUrl = null;

    viewfinder.addEventListener('click', function () { fileInput.click(); });
    viewfinder.addEventListener('dragover', function (e) { e.preventDefault(); viewfinder.classList.add('active'); });
    viewfinder.addEventListener('dragleave', function () {
        if (!selectedFile) viewfinder.classList.remove('active');
    });
    viewfinder.addEventListener('drop', function (e) {
        e.preventDefault();
        if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', function () {
        if (fileInput.files.length) setFile(fileInput.files[0]);
    });

    function setIdle() {
        resultStatus.textContent = 'Result';
        resultBody.innerHTML = '<div class="idle">Choose a video, then press Analyze video. The detected action shows up here.</div>';
    }

    function stopFramePlayback() {
        if (frameTimer) {
            clearInterval(frameTimer);
            frameTimer = null;
        }
        previewImg.classList.remove('show');
        previewImg.removeAttribute('src');
    }

    function resetPreview() {
        previewToken += 1;
        stopFramePlayback();
        previewVideo.pause();
        previewVideo.removeAttribute('src');
        previewVideo.load();
        previewVideo.classList.remove('show');
        if (currentObjectUrl) {
            URL.revokeObjectURL(currentObjectUrl);
            currentObjectUrl = null;
        }
    }

    function showLabel(file, note) {
        vfLabel.textContent = '';
        var sel = document.createElement('div');
        sel.textContent = 'Selected';
        var nm = document.createElement('div');
        nm.className = 'filename';
        nm.textContent = file.name;
        vfLabel.appendChild(sel);
        vfLabel.appendChild(nm);
        if (note) {
            var n = document.createElement('div');
            n.className = 'vf-note';
            n.textContent = note;
            vfLabel.appendChild(n);
        }
        vfLabel.classList.remove('hidden');
    }

    function canPlayNatively(file) {
        var name = file.name.toLowerCase();
        if (name.slice(-4) === '.avi') return false;
        if (file.type && previewVideo.canPlayType(file.type) === '') return false;
        return true;
    }

    // For formats the browser cannot play (such as AVI), the server sends
    // back a set of preview frames and the page plays them as a loop.
    function loadFramePreview(file) {
        var token = previewToken;
        showLabel(file, 'Preparing preview...');
        var formData = new FormData();
        formData.append('video', file);
        fetch('/preview', { method: 'POST', body: formData })
            .then(function (res) {
                return res.json().then(function (data) { return { ok: res.ok, data: data }; });
            })
            .then(function (result) {
                if (token !== previewToken) return;
                if (!result.ok || !result.data.frames || !result.data.frames.length) {
                    showLabel(file, 'Preview is not available for this format');
                    return;
                }
                var frames = result.data.frames;
                var fps = Math.min(Math.max(result.data.fps || 12, 4), 30);
                var i = 0;
                previewImg.src = frames[0];
                previewImg.classList.add('show');
                vfLabel.classList.add('hidden');
                frameTimer = setInterval(function () {
                    i = (i + 1) % frames.length;
                    previewImg.src = frames[i];
                }, 1000 / fps);
            })
            .catch(function () {
                if (token !== previewToken) return;
                showLabel(file, 'Preview is not available for this format');
            });
    }

    previewVideo.addEventListener('error', function () {
        if (!selectedFile || !previewVideo.getAttribute('src')) return;
        var file = selectedFile;
        resetPreview();
        loadFramePreview(file);
    });

    function setFile(file) {
        selectedFile = file;
        viewfinder.classList.add('active');
        resetPreview();

        if (canPlayNatively(file)) {
            currentObjectUrl = URL.createObjectURL(file);
            previewVideo.src = currentObjectUrl;
            previewVideo.classList.add('show');
            vfLabel.classList.add('hidden');
            previewVideo.play().catch(function () {});
        } else {
            loadFramePreview(file);
        }

        analyzeBtn.disabled = false;
        errorMsg.classList.remove('show');
        setIdle();
    }

    function animateCount(el, target) {
        var duration = 600;
        var start = performance.now();
        function tick(now) {
            var progress = Math.min((now - start) / duration, 1);
            el.textContent = (target * progress).toFixed(1) + '%';
            if (progress < 1) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    }

    analyzeBtn.addEventListener('click', function () {
        if (!selectedFile) return;
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
        viewfinder.classList.add('scanning');
        errorMsg.classList.remove('show');
        resultStatus.textContent = 'Analyzing';
        resultBody.innerHTML = '<div class="idle">Scanning the video frames...</div>';

        var formData = new FormData();
        formData.append('video', selectedFile);

        fetch('/predict', { method: 'POST', body: formData })
            .then(function (res) {
                return res.json().then(function (data) { return { ok: res.ok, data: data }; });
            })
            .then(function (result) {
                viewfinder.classList.remove('scanning');
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = 'Analyze video';

                if (!result.ok) {
                    setIdle();
                    errorMsg.textContent = result.data.error || 'Something went wrong.';
                    errorMsg.classList.add('show');
                    return;
                }

                var data = result.data;
                if (!data.is_known_action) {
                    resultStatus.textContent = 'Result';
                    resultBody.innerHTML = '<div class="result-action no-match">No matching action found</div>' +
                        '<div class="result-hint">This video does not match any of the six trained actions listed below.</div>';
                } else {
                    resultStatus.textContent = 'Detected';
                    resultBody.innerHTML = '<div class="result-action">' + data.predicted_action + '</div>' +
                        '<div class="result-readout" id="confReadout">0.0%</div>';
                    animateCount(document.getElementById('confReadout'), data.confidence * 100);
                }
            })
            .catch(function (err) {
                viewfinder.classList.remove('scanning');
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = 'Analyze video';
                setIdle();
                errorMsg.textContent = 'Request failed: ' + err.message;
                errorMsg.classList.add('show');
            });
    });
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return HTML_PAGE


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/preview", methods=["POST"])
def preview():
    """Return a small set of JPEG frames so the page can show a preview of
    formats the browser cannot play itself (for example AVI)."""
    if "video" not in request.files:
        return jsonify({"error": "Send a video file under the 'video' form field."}), 400

    video_file = request.files["video"]
    suffix = os.path.splitext(video_file.filename or "")[1] or ".avi"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        video_file.save(tmp.name)
        tmp_path = tmp.name

    frames = []
    fps = 12.0
    try:
        cap = cv2.VideoCapture(tmp_path)
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, -(-total // 120)) if total > 0 else 2
        fps = source_fps / step

        idx = 0
        while len(frames) < 150:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                h, w = frame.shape[:2]
                if w > 480:
                    scale = 480.0 / w
                    frame = cv2.resize(frame, (480, int(h * scale)))
                encoded, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if encoded:
                    frames.append("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii"))
            idx += 1
        cap.release()
    finally:
        os.remove(tmp_path)

    if not frames:
        return jsonify({"error": "Could not read any frames for the preview."}), 400

    return jsonify({"fps": fps, "frames": frames})


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "Send a video file under the 'video' form field."}), 400

    video_file = request.files["video"]
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp:
        video_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        sequence = extract_sequence(tmp_path)
    finally:
        os.remove(tmp_path)

    if sequence is None:
        return jsonify({"error": "Could not read any frames from the video."}), 400

    probs = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    predicted_label = LABELS[pred_idx]

    is_known, distance, threshold = check_open_set(sequence, predicted_label)

    return jsonify({
        "predicted_action": predicted_label,
        "confidence": float(probs[pred_idx]),
        "all_scores": {LABELS[i]: float(p) for i, p in enumerate(probs)},
        "is_known_action": is_known,
        "distance": distance,
        "threshold": threshold,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
