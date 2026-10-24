"""
predict_video.py

Takes a path to a video file and predicts which action it shows, using
the same CNN+LSTM pipeline as training. This is an easier way to test
the model than performing actions live in front of a webcam.

Run:
    python src/predict_video.py path\to\some_video.avi
"""

import sys
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "models/action_recognition_model.keras"
LABELS_PATH = "models/labels.json"
SEQUENCE_LENGTH = 20
FRAME_SIZE = 128


def sample_frame_indices(total_frames, target_len):
    if total_frames <= 0:
        return []
    if total_frames <= target_len:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, target_len).astype(int).tolist()


def extract_sequence(video_path, feature_extractor):
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict_video.py path\\to\\video.avi")
        sys.exit(1)

    video_path = sys.argv[1]

    print("Loading trained LSTM model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        labels = json.load(f)

    print("Loading frozen MobileNetV2 feature extractor...")
    feature_extractor = MobileNetV2(
        input_shape=(FRAME_SIZE, FRAME_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    feature_extractor.trainable = False

    print(f"Processing '{video_path}'...")
    sequence = extract_sequence(video_path, feature_extractor)
    if sequence is None:
        print("Could not read any frames from that video.")
        sys.exit(1)

    probs = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    print("\n--- Prediction ---")
    print(f"Predicted action: {labels[pred_idx]}")
    print(f"Confidence: {probs[pred_idx]:.2%}")
    print("\nAll scores:")
    for label, p in sorted(zip(labels, probs), key=lambda x: -x[1]):
        print(f"  {label:20s} {p:.2%}")


if __name__ == "__main__":
    main()