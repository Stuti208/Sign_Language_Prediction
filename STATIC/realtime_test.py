"""
STATIC/realtime_test.py
Real-time ISL static gesture recognition using EfficientNetB0.

How to use:
  - Stand ~60-80 cm from camera so your UPPER BODY fills the frame.
  - Hold your hand gesture at CHEST level in the centre of the frame.
  - Keep a plain / neutral background behind you if possible.
  - Press 'q' to quit.
"""

import os, time
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ─────────────────────────── Paths ───────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, 'MODELS')
MODEL_PATH  = os.path.join(MODELS_DIR, 'best_model.keras')
LABELS_PATH = os.path.join(MODELS_DIR, 'class_names.txt')

# ─────────────────────────── Config ──────────────────────────
IMG_SIZE             = 128
CONFIDENCE_THRESHOLD = 0.60
PREDICTION_HOLD_SEC  = 2.0

# ─────────────────────────── Load model ──────────────────────
print("Loading model ...")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    exit(1)

model = load_model(MODEL_PATH, compile=False)

with open(LABELS_PATH) as f:
    CLASS_NAMES = [l.strip() for l in f if l.strip()]

print(f"Model loaded — {len(CLASS_NAMES)} classes.")

# ─────────────────────────── Predict ─────────────────────────
def predict(frame):
    img  = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img  = preprocess_input(img.astype(np.float32))
    prob = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    idx  = int(np.argmax(prob))
    top3 = np.argsort(prob)[::-1][:3]
    top3_str = '  '.join([f"{CLASS_NAMES[i]}:{prob[i]*100:.0f}%" for i in top3])
    return CLASS_NAMES[idx], float(prob[idx]), top3_str

# ─────────────────────────── Main loop ───────────────────────
def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    ret, frame = cap.read()
    if not ret:
        return
    fh, fw = frame.shape[:2]

    # Guide box — centre 60% of frame (matches training image framing)
    bw = int(fw * 0.60)
    bh = int(fh * 0.80)
    bx = (fw - bw) // 2
    by = (fh - bh) // 2

    display_sign   = None
    display_conf   = 0.0
    last_pred_time = 0.0

    print("Webcam ready.")
    print("Stand so your upper body fills the guide box, hold gesture at chest level.")
    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        sign, conf, top3 = predict(frame)

        if conf >= CONFIDENCE_THRESHOLD:
            display_sign   = sign
            display_conf   = conf
            last_pred_time = time.time()
        elif time.time() - last_pred_time > PREDICTION_HOLD_SEC:
            display_sign = None
            display_conf = 0.0

        # ── Guide box ──
        box_color = (0, 255, 100) if display_sign else (0, 180, 255)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), box_color, 2)
        cv2.putText(frame, "Keep upper body + hand inside", (bx, by - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        # ── Crosshair at chest level (centre of box) ──
        cx = bx + bw // 2
        cy = by + bh // 2
        cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (100, 100, 255), 1)
        cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (100, 100, 255), 1)
        cv2.putText(frame, "hand here", (cx + 18, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)

        # ── Top bar ──
        cv2.rectangle(frame, (0, 0), (fw, 70), (20, 20, 20), -1)
        if display_sign and display_conf >= CONFIDENCE_THRESHOLD:
            cv2.putText(frame, f"{display_sign}   {display_conf*100:.0f}%",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 100), 2)
        else:
            cv2.putText(frame, "WAITING...", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (80, 80, 200), 2)

        # ── Top-3 ──
        cv2.putText(frame, top3, (10, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(frame, "q=quit", (fw - 80, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        cv2.imshow('ISL Static Gesture — EfficientNetB0', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
