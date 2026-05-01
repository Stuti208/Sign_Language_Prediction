"""
DYNAMIC/model_train.py
Pure LSTM model for ISL dynamic gesture recognition — 31 classes.

Architecture  : LSTM(128) → LSTM(64) + BatchNorm + Dropout
Input shape   : (30, 147)  — 30 frames × 147 keypoint features
Dataset       : 31 classes × 60 sequences = 1860 sequences
Augmentation  : 3× (noise-low, noise-high, spatial jitter) → 7440 total
Target        : highest accuracy, no over/under-fitting, ≤30 min on CPU
"""

import os, time, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                      BatchNormalization, Input)
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                         ReduceLROnPlateau)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# ─────────────────────────── Paths ───────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'DATA', 'Train_Sequences')
MODELS_DIR = os.path.join(BASE_DIR, 'MODELS')
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────── Hyperparameters ─────────────────
SEQUENCE_LENGTH = 30
FEATURE_DIM     = 147
BATCH_SIZE      = 32
MAX_EPOCHS      = 120
VAL_SPLIT       = 0.15
SEED            = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────── 1. Load data ────────────────────
print("\n[1/5] Loading sequences ...")
t0 = time.time()

if not os.path.isdir(DATA_DIR):
    print(f"ERROR: DATA_DIR not found:\n  {DATA_DIR}")
    print("Run data_prep.py first.")
    exit(1)

action_dirs = sorted([d for d in os.listdir(DATA_DIR)
                       if os.path.isdir(os.path.join(DATA_DIR, d))])
if not action_dirs:
    print(f"ERROR: No class folders found in {DATA_DIR}")
    exit(1)

ACTIONS     = action_dirs
NUM_CLASSES = len(ACTIONS)
label_to_idx = {a: i for i, a in enumerate(ACTIONS)}
print(f"  Classes  : {NUM_CLASSES}  →  {ACTIONS}")

sequences, labels = [], []
for action in tqdm(ACTIONS, desc="  Reading"):
    folder = os.path.join(DATA_DIR, action)
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.npy'):
            continue
        seq = np.load(os.path.join(folder, fname))
        if seq.shape == (SEQUENCE_LENGTH, FEATURE_DIM):
            sequences.append(seq.astype(np.float32))
            labels.append(label_to_idx[action])
        else:
            print(f"  SKIP {fname}: shape {seq.shape}")

X      = np.array(sequences, dtype=np.float32)   # (N, 30, 147)
y_int  = np.array(labels,    dtype=np.int32)
print(f"  Loaded   : {len(X)} sequences in {time.time()-t0:.1f}s")

# ─────────────────────────── 2. Augmentation ─────────────────
print("\n[2/5] Augmenting (3× copies: low-noise, high-noise, spatial-jitter) ...")
rng = np.random.default_rng(SEED)

# Copy 1 — low Gaussian noise (fine-grain variation)
X_n1 = np.clip(X + rng.normal(0, 0.005, X.shape).astype(np.float32), 0.0, 1.0)

# Copy 2 — higher Gaussian noise (coarser variation)
X_n2 = np.clip(X + rng.normal(0, 0.012, X.shape).astype(np.float32), 0.0, 1.0)

# Copy 3 — spatial jitter: same offset per sequence (simulates body shift)
offsets = rng.normal(0, 0.008, (len(X), 1, FEATURE_DIM)).astype(np.float32)
X_sj    = np.clip(X + offsets, 0.0, 1.0)

X     = np.concatenate([X, X_n1, X_n2, X_sj], axis=0)
y_int = np.concatenate([y_int] * 4,            axis=0)

# Shuffle
perm  = rng.permutation(len(X))
X, y_int = X[perm], y_int[perm]
print(f"  After augmentation : {len(X)} sequences")

# ─────────────────────────── 3. Split & encode ───────────────
print("\n[3/5] Splitting train/val ...")
X_train, X_val, y_train_int, y_val_int = train_test_split(
    X, y_int, test_size=VAL_SPLIT, stratify=y_int, random_state=SEED)

y_train = to_categorical(y_train_int, NUM_CLASSES)
y_val   = to_categorical(y_val_int,   NUM_CLASSES)
print(f"  Train : {len(X_train)} | Val : {len(X_val)}")

# Class weights — handles any per-class imbalance after augmentation
cw_vals = compute_class_weight('balanced', classes=np.unique(y_train_int),
                                y=y_train_int)
class_weights = dict(enumerate(cw_vals))

# Save actions list
actions_path = os.path.join(MODELS_DIR, 'actions_list.txt')
with open(actions_path, 'w') as f:
    f.write('\n'.join(ACTIONS))
print(f"  actions_list.txt → {actions_path}")

# ─────────────────────────── 4. Build model ──────────────────
print("\n[4/5] Building pure LSTM model ...")

model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM)),

    # ── Layer 1 ──────────────────────────────────────────────
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.4),

    # ── Layer 2 ──────────────────────────────────────────────
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.4),

    # ── Classifier head ──────────────────────────────────────
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(NUM_CLASSES, activation='softmax'),
], name='ISL_LSTM_31')

model.compile(
    optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()
print(f"\n  Parameters: {model.count_params():,}")

# ─────────────────────────── 5. Train ────────────────────────
print("\n[5/5] Training ...")

best_path  = os.path.join(MODELS_DIR, 'best_model.keras')
final_path = os.path.join(MODELS_DIR, 'final_lstm_model.keras')

callbacks = [
    ModelCheckpoint(
        filepath=best_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=25,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1,
    ),
]

t_start = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)
train_min = (time.time() - t_start) / 60

model.save(final_path)
print(f"\n  best_model.keras  → {best_path}")
print(f"  final_lstm_model  → {final_path}")
print(f"  Training time     : {train_min:.1f} min")

# ─────────────────────────── Results ─────────────────────────
best_val_acc = max(history.history['val_accuracy'])
best_epoch   = history.history['val_accuracy'].index(best_val_acc) + 1
epochs_run   = len(history.history['loss'])

print(f"\n{'='*55}")
print(f"  Best val accuracy : {best_val_acc*100:.2f}%  (epoch {best_epoch}/{epochs_run})")
print(f"  Training time     : {train_min:.1f} min")
print(f"{'='*55}")

# Classification report
y_pred_prob = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
y_pred      = np.argmax(y_pred_prob, axis=1)
report = classification_report(y_val_int, y_pred,
                                target_names=ACTIONS, zero_division=0)
print("\nClassification Report (validation):\n")
print(report)

report_path = os.path.join(MODELS_DIR, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write(f"Classes           : {NUM_CLASSES}\n")
    f.write(f"Best val accuracy : {best_val_acc*100:.2f}%  (epoch {best_epoch})\n")
    f.write(f"Training time     : {train_min:.1f} min\n\n")
    f.write(report)
print(f"Report → {report_path}")

# ── Training curves ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ep = range(1, epochs_run + 1)
axes[0].plot(ep, history.history['accuracy'],     label='Train')
axes[0].plot(ep, history.history['val_accuracy'], label='Val')
axes[0].axvline(best_epoch, color='red', linestyle='--',
                label=f'Best (ep {best_epoch})')
axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(ep, history.history['loss'],     label='Train')
axes[1].plot(ep, history.history['val_loss'], label='Val')
axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
curves_path = os.path.join(MODELS_DIR, 'training_curves.png')
plt.savefig(curves_path, dpi=120); plt.close()
print(f"Training curves  → {curves_path}")

# ── Confusion matrix ──
cm   = confusion_matrix(y_val_int, y_pred)
size = max(10, NUM_CLASSES // 2)
fig, ax = plt.subplots(figsize=(size, size - 2))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=ACTIONS, yticklabels=ACTIONS, ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix — Validation Set')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0,  fontsize=8)
plt.tight_layout()
cm_path = os.path.join(MODELS_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=100); plt.close()
print(f"Confusion matrix → {cm_path}")

print("\nPhase 1 training complete.")
