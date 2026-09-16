# Sign Language Interpretation Using Deep Learning

A real-time Indian Sign Language (ISL) recognition system built on a dual-pathway deep learning architecture — separating static fingerspelling from dynamic word-level gestures for optimized accuracy and speed.

> Research paper under publication — *"From Motion to Meaning: Decoding Sign Language Gestures Using Deep Learning"*

---

## The Problem It Solves

Indian Sign Language has no widely available real-time recognition tool. Existing systems either handle only static gestures or struggle with real-world conditions like lighting changes and occlusions. This system addresses both — static fingerspelling and dynamic word signs — in a single integrated web application.

---

## How It Works

The system uses a **dual-pathway approach**:

- **Static Pathway** — recognizes 36 ISL fingerspelling classes (A-Z, 0-9) using CNN/ViT architectures
- **Dynamic Pathway** — recognizes 61 ISL word signs using a BiLSTM network on skeletal keypoint sequences

Instead of processing raw pixels, both pathways use **MediaPipe Holistic** to extract a 258-dimensional skeletal keypoint vector per frame — making the system invariant to lighting, background, and skin tone.

---

## My Contributions

- Built the **BiLSTM dynamic recognition model** from scratch
- Designed the **sequence preprocessing pipeline** — MediaPipe keypoint extraction, frame standardization to 40 frames, frame skipping for efficiency
- **Integrated both static and dynamic models** into a Flask web application for real-time interpretation

---

## Model Performance

### Static Pathway

| Model | Accuracy | Parameters | Inference Speed |
|---|---|---|---|
| Custom CNN | 95.44% | 0.69M | 8.20 FPS |
| MobileNetV2 | 98.10% | 2.30M | 6.90 FPS |
| ViT-Huge | 100.00% | 706.48M | 0.16 FPS |

### Dynamic Pathway

| Model | Test Accuracy | F1-Score | Parameters |
|---|---|---|---|
| BiLSTM (proposed) | 89.2% | 0.889 | 0.263M |
| CNN-LSTM (baseline) | 74.24% | 0.738 | 0.347M |

The BiLSTM outperformed the CNN-LSTM hybrid by 14.9% — compressing the 258-feature keypoint vector through TimeDistributed layers degraded subtle motion dynamics.

---

## Architecture Overview

```
Input (Video/Webcam)
    │
    ├── Static frames → MediaPipe ROI → CNN / ViT
    │                                      → 36 fingerspelling classes
    │
    └── Video sequence → MediaPipe Holistic → 258-keypoint vector (40 frames)
                                                → BiLSTM
                                                → 61 ISL word signs
                                                
Both outputs → Flask Web App → Real-time display
```

---

## Why BiLSTM Over CNN-LSTM Hybrid

The BiLSTM processes the raw 258-keypoint vector directly in both forward and backward directions — capturing subtle co-articulation patterns between frames. The CNN-LSTM hybrid introduced TimeDistributed Dense layers to compress features before the LSTM, which lost critical motion dynamics and dropped accuracy by 14.9%.

---

## Tech Stack

| Component | Technology |
|---|---|
| Dynamic Model | BiLSTM, TensorFlow, Keras |
| Feature Extraction | MediaPipe Holistic |
| Static Models | Custom CNN, MobileNetV2, ViT |
| Web App | Flask |
| Frontend | HTML, CSS |

---

## How to Run Locally

**Prerequisites:** Python 3.8+, pip

```bash
# Clone the repo
git clone https://github.com/Stuti208/sign-language-interpretation.git
cd sign-language-interpretation

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Research Publication

> *"From Motion to Meaning: Decoding Sign Language Gestures Using Deep Learning"*
> Stuti Jain, Srishti Agarwal, Subhanshi Agarwal, Ayushi Agarwal
> ABES Engineering College, Ghaziabad — Under Publication

---

## Author

**Stuti Jain**
[LinkedIn](https://www.linkedin.com/in/stuti-jain-754b20244/) · [GitHub](https://github.com/Stuti208)
[LinkedIn](https://www.linkedin.com/in/srishtiagl/)) · [GitHub](https://github.com/srishti-cmd)

