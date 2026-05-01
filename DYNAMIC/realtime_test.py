"""
DYNAMIC/realtime_test.py
Real-time ISL dynamic gesture recognition using the trained pure LSTM model.
Press 'q' to quit, 'r' to reset the sequence buffer.
"""

import os, time, urllib.request, numpy as np, cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from collections import deque, Counter
from tensorflow.keras.models import load_model

# ─────────────────────────── Paths ───────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'MODELS')
MODEL_FILE = os.path.join(MODELS_DIR, 'best_model.keras')
ACTIONS_FILE = os.path.join(MODELS_DIR, 'actions_list.txt')

HAND_MODEL_PATH = os.path.join(BASE_DIR, 'hand_landmarker.task')
POSE_MODEL_PATH = os.path.join(BASE_DIR, 'pose_landmarker_lite.task')
HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

# ─────────────────────────── Config ──────────────────────────
SEQUENCE_LENGTH      = 30
POSE_INDICES         = [0, 11, 12, 13, 14, 15, 16]
CONFIDENCE_THRESHOLD = 0.5
PREDICT_CADENCE      = 5
HISTORY_LENGTH       = 10
MOTION_THRESHOLD     = 0.008  # min motion in hand keypoints to trigger prediction
PREDICTION_HOLD_SEC  = 2.5    # keep last prediction visible for this long after hands leave

# ─────────────────────────── Download models ─────────────────
def _download_if_needed(url, path):
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)
        print("  Done.")

_download_if_needed(HAND_MODEL_URL, HAND_MODEL_PATH)
_download_if_needed(POSE_MODEL_URL, POSE_MODEL_PATH)

# ─────────────────────────── MediaPipe detectors ─────────────
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
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    for i, handedness_list in enumerate(hand_result.handedness):
        label  = handedness_list[0].category_name
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
    return np.concatenate([pose_coords, lh, rh])

# ─────────────────────────── Load LSTM model ─────────────────
print("Loading model ...")
if not os.path.exists(MODEL_FILE):
    print(f"ERROR: Model not found at {MODEL_FILE}")
    print("Run model_train.py first.")
    exit(1)
if not os.path.exists(ACTIONS_FILE):
    print(f"ERROR: actions_list.txt not found at {ACTIONS_FILE}")
    exit(1)

model = load_model(MODEL_FILE, compile=False)
with open(ACTIONS_FILE) as f:
    ACTIONS = [l.strip() for l in f if l.strip()]

print(f"Model loaded. Predicting {len(ACTIONS)} signs.")

# ─────────────────────────── Drawing helpers ─────────────────
def draw_landmarks(image, hand_result, pose_result):
    h, w = image.shape[:2]

    # Draw hand landmarks
    for hand_lms in hand_result.hand_landmarks:
        for lm in hand_lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)

    # Draw selected pose landmarks
    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks[0]
        for idx in POSE_INDICES:
            lm = landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, (255, 128, 0), -1)

def draw_hud(image, sign, confidence, buf_fill, hands_detected, motion=0.0):
    h, w = image.shape[:2]
    bar_max = 200

    # Background panel
    cv2.rectangle(image, (0, 0), (w, 90), (20, 20, 20), -1)

    # Buffer progress bar
    filled = int(bar_max * buf_fill / SEQUENCE_LENGTH)
    cv2.rectangle(image, (10, 10), (10 + bar_max, 30), (60, 60, 60), -1)
    bar_color = (0, 200, 100) if buf_fill == SEQUENCE_LENGTH else (0, 150, 255)
    cv2.rectangle(image, (10, 10), (10 + filled, 30), bar_color, -1)
    cv2.putText(image, f"Buffer {buf_fill}/{SEQUENCE_LENGTH}", (220, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Hands indicator
    hand_color = (0, 255, 100) if hands_detected else (80, 80, 80)
    cv2.circle(image, (w - 30, 20), 10, hand_color, -1)

    # Prediction
    conf_pct = f"{confidence*100:.0f}%"
    pred_color = (0, 255, 100) if confidence >= CONFIDENCE_THRESHOLD else (80, 180, 255)
    cv2.putText(image, f"{sign}  {conf_pct}", (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, pred_color, 2)

    # Motion score + controls hint
    motion_color = (0, 220, 80) if motion >= MOTION_THRESHOLD else (80, 80, 200)
    cv2.putText(image, f"motion:{motion:.4f} thr:{MOTION_THRESHOLD}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, motion_color, 1)
    cv2.putText(image, "q=quit  r=reset", (w - 160, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

# ─────────────────────────── Main loop ───────────────────────
def run():
    sequence_buffer    = deque(maxlen=SEQUENCE_LENGTH)
    prediction_history = deque(maxlen=HISTORY_LENGTH)
    frame_counter      = 0
    display_sign       = "WAITING..."
    display_conf       = 0.0
    last_pred_time     = 0.0   # timestamp of last confident prediction

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Webcam started. Press 'q' to quit, 'r' to reset buffer.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        hand_result = hand_detector.detect(mp_image)
        pose_result = pose_detector.detect(mp_image)

        hands     = len(hand_result.hand_landmarks) > 0
        keypoints = extract_keypoints(hand_result, pose_result)
        sequence_buffer.append(keypoints)
        frame_counter += 1

        # Always compute motion for HUD display
        if len(sequence_buffer) == SEQUENCE_LENGTH:
            _arr   = np.array(sequence_buffer)
            motion = float(np.mean(np.std(_arr[:, 21:], axis=0)))
        else:
            motion = 0.0

        if not hands:
            # Clear history immediately so next sign starts fresh
            prediction_history.clear()
            # Hold display for PREDICTION_HOLD_SEC then reset to WAITING
            if time.time() - last_pred_time > PREDICTION_HOLD_SEC:
                display_sign = "WAITING..."
                display_conf = 0.0

        # Predict only when hands visible, buffer full, cadence reached, motion detected
        if (hands
                and len(sequence_buffer) == SEQUENCE_LENGTH
                and frame_counter % PREDICT_CADENCE == 0):

            seq_array    = np.array(sequence_buffer)
            hand_feats   = seq_array[:, 21:]
            motion_score = float(np.mean(np.std(hand_feats, axis=0)))

            if motion_score >= MOTION_THRESHOLD:
                seq_in = np.expand_dims(seq_array, axis=0)
                probs  = model(seq_in, training=False).numpy()[0]
                idx    = int(np.argmax(probs))
                conf   = float(probs[idx])
                if conf >= CONFIDENCE_THRESHOLD:
                    prediction_history.append(ACTIONS[idx])
                    last_pred_time = time.time()
                else:
                    prediction_history.append('__uncertain__')

        # Majority vote over recent history
        if prediction_history:
            counts = Counter(prediction_history)
            top    = counts.most_common(1)[0][0]
            if top != '__uncertain__':
                display_sign = top
                display_conf = counts[top] / len(prediction_history)
        draw_landmarks(frame, hand_result, pose_result)
        draw_hud(frame, display_sign, display_conf,
                 len(sequence_buffer), hands, motion)

        cv2.imshow('ISL Dynamic Gesture — LSTM', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            sequence_buffer.clear()
            prediction_history.clear()
            frame_counter  = 0
            display_sign   = "WAITING..."
            display_conf   = 0.0
            print("Buffer reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
