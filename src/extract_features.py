"""
extract_features.py

Walks through data/raw_videos/<action_name>/*.avi, samples a fixed number
of frames per video, runs each frame through a FROZEN pretrained MobileNetV2
(transfer learning - we do NOT train this CNN), and saves the resulting
per-frame feature sequence as a .npy file under
data_processed/features/<action_name>/<video_name>.npy

This satisfies the proposal's "Feature Engineering" step (CNN learns
discriminative features) and "Optimization" step (transfer learning),
while staying cheap enough to run on CPU only - the CNN's weights are
frozen, so this is just a forward pass, never backpropagation.

Run:
    python src/extract_features.py
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

RAW_VIDEOS_DIR = "data/raw_videos"
OUTPUT_DIR = "data_processed/features"
SEQUENCE_LENGTH = 20     # frames sampled per video (kept low for CPU speed)
FRAME_SIZE = 128         # resize each frame to 128x128 before the CNN

print("Loading frozen MobileNetV2 (transfer learning - weights are not trained)...")
feature_extractor = MobileNetV2(
    input_shape=(FRAME_SIZE, FRAME_SIZE, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",          # global average pooling -> one 1280-dim vector per frame
)
feature_extractor.trainable = False  # freeze - this is the "transfer learning" part
FEATURE_DIM = feature_extractor.output_shape[-1]  # 1280 for MobileNetV2


def sample_frame_indices(total_frames, target_len):
    if total_frames <= 0:
        return []
    if total_frames <= target_len:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, target_len).astype(int).tolist()


def read_sampled_frames(video_path, target_len):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(sample_frame_indices(total_frames, target_len))

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
    return frames


def pad_sequence(feature_seq, target_len):
    current_len = len(feature_seq)
    if current_len == 0:
        return np.zeros((target_len, FEATURE_DIM), dtype=np.float32)
    if current_len >= target_len:
        return np.array(feature_seq[:target_len])
    pad_count = target_len - current_len
    padding = np.tile(feature_seq[-1], (pad_count, 1))
    return np.vstack([feature_seq, padding])


def process_video(video_path):
    frames = read_sampled_frames(video_path, SEQUENCE_LENGTH)
    if not frames:
        return np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)

    batch = np.array(frames, dtype=np.float32)
    batch = preprocess_input(batch)  # MobileNetV2-specific normalization
    features = feature_extractor.predict(batch, verbose=0)  # (num_frames, 1280)
    return pad_sequence(features, SEQUENCE_LENGTH)


def main():
    if not os.path.isdir(RAW_VIDEOS_DIR):
        raise FileNotFoundError(f"'{RAW_VIDEOS_DIR}' not found.")

    action_classes = sorted(
        d for d in os.listdir(RAW_VIDEOS_DIR)
        if os.path.isdir(os.path.join(RAW_VIDEOS_DIR, d))
    )
    if not action_classes:
        raise FileNotFoundError(f"No action sub-folders found inside '{RAW_VIDEOS_DIR}'.")

    print(f"Found {len(action_classes)} action classes: {action_classes}")

    for action in action_classes:
        src_dir = os.path.join(RAW_VIDEOS_DIR, action)
        dst_dir = os.path.join(OUTPUT_DIR, action)
        os.makedirs(dst_dir, exist_ok=True)

        video_files = [f for f in os.listdir(src_dir)
                       if f.lower().endswith((".mp4", ".avi", ".mov"))]

        for video_file in tqdm(video_files, desc=f"Processing '{action}'"):
            video_path = os.path.join(src_dir, video_file)
            out_path = os.path.join(dst_dir, os.path.splitext(video_file)[0] + ".npy")
            if os.path.exists(out_path):
                continue  # skip already-processed videos
            sequence = process_video(video_path)
            np.save(out_path, sequence)

    print("Done. CNN feature sequences saved under", OUTPUT_DIR)


if __name__ == "__main__":
    main()