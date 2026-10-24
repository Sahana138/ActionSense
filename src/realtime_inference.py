"""
realtime_inference.py

Opens the webcam, buffers the last SEQUENCE_LENGTH frames, runs each
through the same frozen MobileNetV2 used during training, and feeds the
resulting feature sequence to the trained LSTM to predict the current
action live.

Run:
    python src/realtime_inference.py
"""

import json
import collections
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "models/action_recognition_model.keras"
LABELS_PATH = "models/labels.json"
SEQUENCE_LENGTH = 20
FRAME_SIZE = 128
PREDICT_EVERY_N_FRAMES = 5  # re-run prediction every N new frames, not every single one


def main():
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

    frame_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
    frame_counter = 0
    current_label = "..."
    current_confidence = 0.0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check the camera index/permissions.")

    print("Webcam started. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Preprocess this frame into a CNN feature vector
        resized = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        batch = preprocess_input(np.expand_dims(rgb.astype(np.float32), axis=0))
        feature_vec = feature_extractor.predict(batch, verbose=0)[0]  # (1280,)
        frame_buffer.append(feature_vec)

        frame_counter += 1

        if len(frame_buffer) == SEQUENCE_LENGTH and frame_counter % PREDICT_EVERY_N_FRAMES == 0:
            seq = np.expand_dims(np.array(frame_buffer), axis=0)  # (1, 20, 1280)
            probs = model.predict(seq, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            current_label = labels[pred_idx]
            current_confidence = probs[pred_idx]

        text = f"{current_label} ({current_confidence:.2f})"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)

        cv2.imshow("ActionSense - Real-Time Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()