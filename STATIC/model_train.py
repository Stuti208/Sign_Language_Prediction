"""
STATIC/model_train.py
CNN + Transfer Learning for ISL static gesture recognition.

Dataset      : isl_dataset/  — 36 classes (0-9, A-Z), 1000 images each
Base model   : EfficientNetB0 (ImageNet weights, frozen → then fine-tuned)
Input size   : 128 × 128 × 3
Training     : Two-phase — head only, then fine-tune top 30 layers
Target       : highest accuracy, ≤ 30 min on CPU
"""

import os, time, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                         ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam

# ─────────────────────────── Paths ───────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'isl_dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'MODELS')
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────── Config ──────────────────────────
IMG_SIZE    = 128
BATCH_SIZE  = 32
SEED        = 42
VAL_SPLIT   = 0.20

# Phase 1 — train classifier head only (base frozen)
PHASE1_EPOCHS = 15
PHASE1_LR     = 1e-3

# Phase 2 — unfreeze top layers and fine-tune
PHASE2_EPOCHS    = 20
PHASE2_LR        = 1e-4
UNFREEZE_FROM    = -30   # unfreeze last 30 layers of EfficientNetB0

tf.random.set_seed(SEED)

# ─────────────────────────── 1. Data pipeline ────────────────
print("\n[1/4] Building data pipeline ...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset='training',
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset='validation',
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

CLASS_NAMES = train_ds.class_names
NUM_CLASSES = len(CLASS_NAMES)
print(f"  Classes  : {NUM_CLASSES}  →  {CLASS_NAMES}")

# Save class labels
labels_path = os.path.join(MODELS_DIR, 'class_names.txt')
with open(labels_path, 'w') as f:
    f.write('\n'.join(CLASS_NAMES))
print(f"  Labels saved → {labels_path}")

# Augmentation (training only)
augment = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.15),
], name='augmentation')

def preprocess_train(image, label):
    image = augment(image, training=True)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label

def preprocess_val(image, label):
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label

AUTOTUNE  = tf.data.AUTOTUNE
train_ds  = train_ds.map(preprocess_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds    = val_ds.map(preprocess_val,   num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
print("  Pipeline ready.")

# ─────────────────────────── 2. Build model ──────────────────
print("\n[2/4] Building EfficientNetB0 model ...")

base = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base.trainable = False   # freeze for Phase 1

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs, outputs, name='ISL_EfficientNetB0')
print(f"  Total params     : {model.count_params():,}")
print(f"  Trainable params : {sum(np.prod(v.shape) for v in model.trainable_variables):,}  (head only)")

# ─────────────────────────── 3. Phase 1 — train head ─────────
print(f"\n[3/4] Phase 1 — training classifier head ({PHASE1_EPOCHS} epochs max) ...")

best_path  = os.path.join(MODELS_DIR, 'best_model.keras')
final_path = os.path.join(MODELS_DIR, 'final_static_model.keras')

model.compile(
    optimizer=Adam(PHASE1_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

cb_p1 = [
    ModelCheckpoint(best_path, monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=5,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=3, min_lr=1e-6, verbose=1),
]

t0 = time.time()
hist1 = model.fit(train_ds, validation_data=val_ds,
                  epochs=PHASE1_EPOCHS, callbacks=cb_p1, verbose=1)
p1_min = (time.time() - t0) / 60
print(f"  Phase 1 done in {p1_min:.1f} min  |  best val acc: {max(hist1.history['val_accuracy'])*100:.2f}%")

# ─────────────────────────── 4. Phase 2 — fine-tune ──────────
print(f"\n[4/4] Phase 2 — fine-tuning top layers ({PHASE2_EPOCHS} epochs max) ...")

# Unfreeze last N layers of base
base.trainable = True
for layer in base.layers[:UNFREEZE_FROM]:
    layer.trainable = False

trainable_now = sum(np.prod(v.shape) for v in model.trainable_variables)
print(f"  Trainable params after unfreeze: {trainable_now:,}")

model.compile(
    optimizer=Adam(PHASE2_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

cb_p2 = [
    ModelCheckpoint(best_path, monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=7,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=4, min_lr=1e-7, verbose=1),
]

t0 = time.time()
hist2 = model.fit(train_ds, validation_data=val_ds,
                  epochs=PHASE2_EPOCHS, callbacks=cb_p2, verbose=1)
p2_min = (time.time() - t0) / 60
print(f"  Phase 2 done in {p2_min:.1f} min  |  best val acc: {max(hist2.history['val_accuracy'])*100:.2f}%")

model.save(final_path)
total_min = p1_min + p2_min
print(f"\n  best_model.keras   → {best_path}")
print(f"  final_static_model → {final_path}")
print(f"  Total training     : {total_min:.1f} min")

# ─────────────────────────── Results ─────────────────────────
all_val_acc = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
best_val    = max(all_val_acc)
print(f"\n{'='*55}")
print(f"  Best val accuracy : {best_val*100:.2f}%")
print(f"  Total time        : {total_min:.1f} min")
print(f"{'='*55}")

# Classification report
print("\nGenerating classification report ...")
y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

report = classification_report(y_true, y_pred,
                                target_names=CLASS_NAMES, zero_division=0)
print(report)
report_path = os.path.join(MODELS_DIR, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write(f"Best val accuracy : {best_val*100:.2f}%\n")
    f.write(f"Total time        : {total_min:.1f} min\n\n")
    f.write(report)
print(f"Report → {report_path}")

# Training curves
ep1 = range(1, len(hist1.history['accuracy']) + 1)
ep2 = range(len(ep1) + 1, len(ep1) + len(hist2.history['accuracy']) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(ep1, hist1.history['accuracy'],     'b-',  label='Train P1')
axes[0].plot(ep1, hist1.history['val_accuracy'], 'b--', label='Val P1')
axes[0].plot(ep2, hist2.history['accuracy'],     'g-',  label='Train P2')
axes[0].plot(ep2, hist2.history['val_accuracy'], 'g--', label='Val P2')
axes[0].axvline(len(ep1), color='gray', linestyle=':', label='Phase boundary')
axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(ep1, hist1.history['loss'],     'b-',  label='Train P1')
axes[1].plot(ep1, hist1.history['val_loss'], 'b--', label='Val P1')
axes[1].plot(ep2, hist2.history['loss'],     'g-',  label='Train P2')
axes[1].plot(ep2, hist2.history['val_loss'], 'g--', label='Val P2')
axes[1].axvline(len(ep1), color='gray', linestyle=':')
axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'training_curves.png'), dpi=120)
plt.close()

# Confusion matrix
cm   = confusion_matrix(y_true, y_pred)
size = max(12, NUM_CLASSES // 2)
fig, ax = plt.subplots(figsize=(size, size - 2))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix — Validation Set')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0,  fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'confusion_matrix.png'), dpi=100)
plt.close()
print(f"Plots saved → {MODELS_DIR}")

print("\nPhase 2 (static model) training complete.")
