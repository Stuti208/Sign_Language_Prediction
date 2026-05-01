"""
DYNAMIC/data_prep.py
Extracts 30-frame keypoint sequences from ISL gesture videos.
Outputs: DATA/Train_Sequences/{action}/{action}_NNN.npy  shape=(30, 147)

Feature layout (147 total):
  [0:21]   pose  — 7 landmarks × 3  (indices 0,11,12,13,14,15,16)
  [21:84]  left hand  — 21 landmarks × 3
  [84:147] right hand — 21 landmarks × 3
"""

import os
import urllib.request
import pathlib
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

# ─────────────────────────── Paths ───────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))

# Folder containing raw videos: Video_Dataset/{action}/{video}.mp4
INPUT_VIDEOS_PATH = r"C:\Users\aagam\Documents\Major Project 2\Sign Language Predictio Model\DYNAMIC\dataset2\Video_Dataset\Video_Dataset"

# Where processed sequences are saved
OUTPUT_PATH = os.path.join(BASE_DIR, 'DATA', 'Train_Sequences')

# MediaPipe .task model files
HAND_MODEL_PATH = os.path.join(BASE_DIR, 'hand_landmarker.task')
POSE_MODEL_PATH = os.path.join(BASE_DIR, 'pose_landmarker_lite.task')
HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

# ─────────────────────────── Config ──────────────────────────
SEQUENCE_LENGTH = 30
POSE_INDICES    = [0, 11, 12, 13, 14, 15, 16]   # 7 pose keypoints

# ─────────────────────────── Download models ─────────────────
def _download_if_needed(url, path):
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)
        print(f"  Saved: {path}")

_download_if_needed(HAND_MODEL_URL, HAND_MODEL_PATH)
_download_if_needed(POSE_MODEL_URL, POSE_MODEL_PATH)

# ─────────────────────────── Build detectors ─────────────────
_hand_opts = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
_pose_opts = mp_vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
hand_detector = mp_vision.HandLandmarker.create_from_options(_hand_opts)
pose_detector = mp_vision.PoseLandmarker.create_from_options(_pose_opts)

# ─────────────────────────── Keypoint extraction ─────────────
def extract_keypoints(hand_result, pose_result):
    """Returns a (147,) feature vector matching the training format."""
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)

    for i, handedness_list in enumerate(hand_result.handedness):
        label  = handedness_list[0].category_name   # 'Left' or 'Right' (user POV)
        coords = np.array([[lm.x, lm.y, lm.z]
                           for lm in hand_result.hand_landmarks[i]]).flatten()
        if label == 'Left':
            lh = coords
        else:
            rh = coords

    pose_coords = [0.0] * (len(POSE_INDICES) * 3)
    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks[0]
        for j, idx in enumerate(POSE_INDICES):
            lm = landmarks[idx]
            pose_coords[j*3 : j*3+3] = [lm.x, lm.y, lm.z]

    return np.concatenate([pose_coords, lh, rh])   # shape (147,)

# ─────────────────────────── Main processing ─────────────────
def process_video_dataset():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    data_root    = pathlib.Path(INPUT_VIDEOS_PATH)
    action_paths = sorted([p for p in data_root.iterdir() if p.is_dir()])

    print(f"Found {len(action_paths)} sign folders to process.")

    total_saved  = 0
    total_failed = 0

    for action_path in tqdm(action_paths, desc="Signs"):
        action       = action_path.name
        video_files  = sorted(action_path.glob('*.mp4'))
        save_dir     = os.path.join(OUTPUT_PATH, action)
        os.makedirs(save_dir, exist_ok=True)

        for seq_idx, video_file in enumerate(video_files):
            cap            = cv2.VideoCapture(str(video_file))
            frame_sequence = []

            if not cap.isOpened():
                total_failed += 1
                continue

            while len(frame_sequence) < SEQUENCE_LENGTH:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                hand_result = hand_detector.detect(mp_image)
                pose_result = pose_detector.detect(mp_image)
                keypoints   = extract_keypoints(hand_result, pose_result)
                frame_sequence.append(keypoints)

            cap.release()

            if len(frame_sequence) == SEQUENCE_LENGTH:
                filename = f"{action}_{seq_idx:03d}.npy"
                np.save(os.path.join(save_dir, filename),
                        np.array(frame_sequence, dtype=np.float32))
                total_saved += 1
            else:
                total_failed += 1

    print(f"\nDone. Saved: {total_saved} sequences | Skipped (too short): {total_failed}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == '__main__':
    process_video_dataset()
