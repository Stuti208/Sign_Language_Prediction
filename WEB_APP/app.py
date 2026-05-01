"""
WEB_APP/app.py
Flask backend for ISL Sign Language Recognition Web Application.
Handles both static (landmarks → MLP) and dynamic (sequence → LSTM) predictions.

Install:  pip install flask
Run:      python app.py
Open:     http://localhost:5000
"""

import os
import numpy as np
from flask import Flask, render_template, request, jsonify
import tensorflow as tf

# ✅ Fix for Keras model loading compatibility issues
from keras.layers import Dense, BatchNormalization

# Patch Dense
original_dense_init = Dense.__init__

def patched_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    original_dense_init(self, *args, **kwargs)

Dense.__init__ = patched_dense_init

# Patch BatchNormalization
original_bn_init = BatchNormalization.__init__

def patched_bn_init(self, *args, **kwargs):
    kwargs.pop("renorm", None)
    kwargs.pop("renorm_clipping", None)
    kwargs.pop("renorm_momentum", None)
    original_bn_init(self, *args, **kwargs)

BatchNormalization.__init__ = patched_bn_init

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

# ── Model paths ──────────────────────────────────────────────
STATIC_MODEL_PATH   = os.path.join(PARENT_DIR, 'STATIC',  'MODELS', 'landmark_model.keras')
STATIC_LABELS_PATH  = os.path.join(PARENT_DIR, 'STATIC',  'MODELS', 'landmark_classes.txt')
DYNAMIC_MODEL_PATH  = os.path.join(PARENT_DIR, 'DYNAMIC', 'MODELS', 'best_model.keras')
DYNAMIC_LABELS_PATH = os.path.join(PARENT_DIR, 'DYNAMIC', 'MODELS', 'actions_list.txt')

# ── Load models ──────────────────────────────────────────────
print("Loading models...")

static_model = tf.keras.models.load_model(STATIC_MODEL_PATH, compile=False)
with open(STATIC_LABELS_PATH) as f:
    STATIC_LABELS = [l.strip() for l in f if l.strip()]

dynamic_model = tf.keras.models.load_model(DYNAMIC_MODEL_PATH, compile=False)
with open(DYNAMIC_LABELS_PATH) as f:
    DYNAMIC_LABELS = [l.strip() for l in f if l.strip()]

# Warm-up (avoids first-call delay)
static_model(np.zeros((1, 63), dtype=np.float32), training=False)
dynamic_model(np.zeros((1, 30, 147), dtype=np.float32), training=False)

print(f"Ready!  Static: {len(STATIC_LABELS)} classes  |  Dynamic: {len(DYNAMIC_LABELS)} classes")

# ── Routes ───────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           static_count=len(STATIC_LABELS),
                           dynamic_count=len(DYNAMIC_LABELS))


@app.route('/api/predict/static', methods=['POST'])
def predict_static():
    data      = request.get_json()
    landmarks = np.array(data['landmarks'], dtype=np.float32).reshape(1, 63)
    prob      = static_model(landmarks, training=False).numpy()[0]
    idx       = int(np.argmax(prob))
    conf      = float(prob[idx])
    top3      = [{'label': STATIC_LABELS[i], 'confidence': round(float(prob[i]), 4)}
                 for i in np.argsort(prob)[::-1][:3]]
    return jsonify({'label': STATIC_LABELS[idx], 'confidence': conf,
                    'top3': top3, 'mode': 'static'})


@app.route('/api/predict/dynamic', methods=['POST'])
def predict_dynamic():
    data     = request.get_json()
    sequence = np.array(data['sequence'], dtype=np.float32).reshape(1, 30, 147)
    prob     = dynamic_model(sequence, training=False).numpy()[0]
    idx      = int(np.argmax(prob))
    conf     = float(prob[idx])
    top3     = [{'label': DYNAMIC_LABELS[i], 'confidence': round(float(prob[i]), 4)}
                for i in np.argsort(prob)[::-1][:3]]
    return jsonify({'label': DYNAMIC_LABELS[idx], 'confidence': conf,
                    'top3': top3, 'mode': 'dynamic'})


if __name__ == '__main__':
    print("\n  Open your browser at:  http://localhost:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
