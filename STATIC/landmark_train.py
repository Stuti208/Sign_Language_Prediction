"""
STATIC/landmark_train.py
Extract MediaPipe hand landmarks from ISL training images → train MLP.

Speed optimisations:
  - Max 200 images per class  (7 200 total, ~2-3 min extraction)
  - ThreadPoolExecutor for parallel extraction
  - Landmarks cached to .npz — re-runs skip extraction entirely
Target : < 10 min total on CPU
"""

import os, time, urllib.request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import cv2

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ─────────────────────────── Paths ───────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, 'isl_dataset')
MODELS_DIR  = os.path.join(BASE_DIR, 'MODELS')
DYNAMIC_DIR = os.path.join(os.path.dirname(BASE_DIR), 'DYNAMIC')
CACHE_PATH  = os.path.join(MODELS_DIR, 'landmarks_cache.npz')
os.makedirs(MODELS_DIR, exist_ok=True)

HAND_MODEL_PATH = os.path.join(DYNAMIC_DIR, 'hand_landmarker.task')
HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# ─────────────────────────── Config ──────────────────────────
MAX_PER_CLASS = 200    # images per class — enough for landmarks, much faster
NUM_WORKERS   = 4      # parallel threads for extraction
FEATURE_DIM   = 63
SEED          = 42
VAL_SPLIT     = 0.15
BATCH_SIZE    = 64
MAX_EPOCHS    = 80

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─────────────────────────── Download landmarker ─────────────
if not os.path.exists(HAND_MODEL_PATH):
    print("Downloading hand_landmarker.task ...")
    urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    print("  Done.")

# ─────────────────────────── Per-thread detector factory ─────
_thread_local = {}

def get_detector():
    tid = id(mp_python.BaseOptions)  # simple key; one detector per thread is fine
    import threading
    key = threading.get_ident()
    if key not in _thread_local:
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        _thread_local[key] = mp_vision.HandLandmarker.create_from_options(opts)
    return _thread_local[key]

# ─────────────────────────── Single-image extraction ─────────
def process_image(args):
    fpath, label = args
    try:
        img = cv2.imread(fpath)
        if img is None:
            return None
        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = get_detector().detect(mp_img)
        if not result.hand_landmarks:
            return None
        lms    = result.hand_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
        coords -= coords[0]
        scale   = np.linalg.norm(coords[9]) + 1e-6
        coords /= scale
        return (coords.flatten(), label)
    except Exception:
        return None

# ─────────────────────────── 1. Load / extract features ──────
class_dirs  = sorted([d for d in os.listdir(DATA_DIR)
                       if os.path.isdir(os.path.join(DATA_DIR, d))])
CLASS_NAMES = class_dirs
NUM_CLASSES = len(CLASS_NAMES)
label_map   = {c: i for i, c in enumerate(CLASS_NAMES)}

if os.path.exists(CACHE_PATH):
    print(f"\n[1/3] Loading cached landmarks from {CACHE_PATH} ...")
    data    = np.load(CACHE_PATH)
    X, y    = data['X'], data['y']
    print(f"  Loaded {len(X)} samples from cache.")
else:
    print(f"\n[1/3] Extracting landmarks (max {MAX_PER_CLASS}/class, {NUM_WORKERS} threads) ...")
    rng   = np.random.default_rng(SEED)
    tasks = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(DATA_DIR, cls)
        files   = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
        if len(files) > MAX_PER_CLASS:
            files = rng.choice(files, MAX_PER_CLASS, replace=False).tolist()
        for fname in files:
            tasks.append((os.path.join(cls_dir, fname), label_map[cls]))

    print(f"  Total images to process: {len(tasks)}")
    t0      = time.time()
    X_list, y_list = [], []
    done    = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(process_image, t): t for t in tasks}
        for fut in as_completed(futures):
            done += 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                rate    = done / elapsed
                eta     = (len(tasks) - done) / rate
                print(f"  {done}/{len(tasks)}  |  {rate:.0f} img/s  |  ETA {eta:.0f}s")
            res = fut.result()
            if res is not None:
                X_list.append(res[0])
                y_list.append(res[1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    np.savez_compressed(CACHE_PATH, X=X, y=y)
    elapsed = time.time() - t0
    print(f"\n  Extraction done in {elapsed:.0f}s")
    print(f"  Extracted : {len(X)} / {len(tasks)} images")
    print(f"  Cache saved → {CACHE_PATH}")

# Per-class breakdown
print("\n  Per-class sample counts:")
for i, cls in enumerate(CLASS_NAMES):
    cnt = int(np.sum(y == i))
    print(f"    {cls:>3}: {cnt}")

# ─────────────────────────── 2. Split ────────────────────────
print("\n[2/3] Splitting train / val ...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VAL_SPLIT, stratify=y, random_state=SEED)
print(f"  Train : {len(X_train)} | Val : {len(X_val)}")

classes_path = os.path.join(MODELS_DIR, 'landmark_classes.txt')
with open(classes_path, 'w') as f:
    f.write('\n'.join(CLASS_NAMES))

# ─────────────────────────── 3. Train MLP ────────────────────
print("\n[3/3] Training MLP ...")

model = Sequential([
    Input(shape=(FEATURE_DIM,)),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(NUM_CLASSES, activation='softmax'),
], name='ISL_Static_MLP')

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()

best_path = os.path.join(MODELS_DIR, 'landmark_model.keras')
callbacks = [
    ModelCheckpoint(best_path, monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=15,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=7, min_lr=1e-6, verbose=1),
]

t0 = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)
train_min = (time.time() - t0) / 60
best_val  = max(history.history['val_accuracy'])
best_ep   = history.history['val_accuracy'].index(best_val) + 1

print(f"\n{'='*55}")
print(f"  Best val accuracy : {best_val*100:.2f}%  (epoch {best_ep})")
print(f"  Training time     : {train_min:.1f} min")
print(f"  Model saved       → {best_path}")
print(f"{'='*55}")

# ── Classification report ──
y_pred  = np.argmax(model.predict(X_val, verbose=0), axis=1)
report  = classification_report(y_val, y_pred, target_names=CLASS_NAMES, zero_division=0)
print(report)
report_path = os.path.join(MODELS_DIR, 'landmark_report.txt')
with open(report_path, 'w') as f:
    f.write(f"Best val accuracy : {best_val*100:.2f}%  (epoch {best_ep})\n")
    f.write(f"Training time     : {train_min:.1f} min\n\n")
    f.write(report)
print(f"Report → {report_path}")

# ── Plots ──
ep = range(1, len(history.history['accuracy']) + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ep, history.history['accuracy'],     label='Train')
axes[0].plot(ep, history.history['val_accuracy'], label='Val')
axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(ep, history.history['loss'],     label='Train')
axes[1].plot(ep, history.history['val_loss'], label='Val')
axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'landmark_curves.png'), dpi=120)
plt.close()

cm = confusion_matrix(y_val, y_pred)
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix — Validation Set')
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'landmark_confusion.png'), dpi=100)
plt.close()
print(f"Plots → {MODELS_DIR}")
print("\nTraining complete.")
