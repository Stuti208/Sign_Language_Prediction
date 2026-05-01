import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque, Counter # Counter added for stability averaging

# --- 1. Configuration (MUST MATCH training setup) ---
MODELS_PATH = os.path.join('..', 'MODELS') 
MODEL_FILE = os.path.join(MODELS_PATH, 'final_cnn_lstm_model.keras') # Use the final saved model
ACTIONS_FILE = os.path.join(MODELS_PATH, 'actions_list.txt')

SEQUENCE_LENGTH = 30 # Window size for CNN-LSTM input
POSE_INDICES = [0, 11, 12, 13, 14, 15, 16] # Keypoint indices used in data prep
KEYPOINT_DIM = 147 # Feature dimension (verified working size)

CONFIDENCE_THRESHOLD = 0.5 # Minimum confidence needed for a raw prediction
PREDICT_CADENCE = 5 # Run the full model prediction only every N frames (Speed Optimization)
HISTORY_LENGTH = 10 # Number of recent frames to use for stability check (Stability Optimization)

# --- 2. Load Model and Actions ---
try:
    model = load_model(MODEL_FILE)
    
    with open(ACTIONS_FILE, 'r') as f:
        ACTIONS = [line.strip() for line in f]
    
    print(f"Model loaded. Ready to predict {len(ACTIONS)} signs.")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure model_train.py has completed and saved 'final_cnn_lstm_model.keras' in the 3_MODELS folder.")
    exit()

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils 

# --- 3. Helper Functions (Feature Extraction) ---
def mediapipe_detection(image, model):
    """Processes an image for MediaPipe detection."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB); image.flags.writeable = False
    results = model.process(image); image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

def extract_keypoints(results):
    """Extracts keypoints in the (147 feature) format used for training."""
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3)
    
    pose_coords = []
    if results.pose_landmarks:
        for i in POSE_INDICES:
            lm = results.pose_landmarks.landmark[i]
            pose_coords.extend([lm.x, lm.y, lm.z])
    else:
        pose_coords = np.zeros(len(POSE_INDICES) * 3)

    return np.concatenate([pose_coords, lh, rh])

def draw_landmarks(image, results):
    """Draws keypoints on the image."""
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

def visualize_prediction(image, prediction, confidence):
    """Displays the stable prediction on the frame."""
    text = f'{prediction} ({confidence*100:.2f}%)'
    # Use a larger font/color if the prediction is stable/certain
    color = (0, 255, 0) if prediction != 'UNCERTAIN' and prediction != 'WAITING...' else (0, 165, 255)
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

# --- 4. Real-Time Prediction Loop ---
def run_realtime_prediction():
    # Sliding window and history buffer initialization
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH) 
    prediction_history = deque(maxlen=HISTORY_LENGTH)
    
    # Initial state variables
    predicted_sign = "WAITING..."
    confidence = 0.0
    frame_counter = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1) # Mirror view
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)

            # 1. Feature Extraction and Sequence Buffer Update
            keypoints = extract_keypoints(results)
            sequence_buffer.append(keypoints)
            frame_counter += 1

            # 2. Prediction (Only runs if buffer is full AND cadence is met)
            if len(sequence_buffer) == SEQUENCE_LENGTH and frame_counter % PREDICT_CADENCE == 0:
                
                input_sequence = np.expand_dims(np.array(sequence_buffer), axis=0) # Shape (1, 30, 147)
                res = model.predict(input_sequence, verbose=0)[0]
                
                prediction_index = np.argmax(res)
                confidence = res[prediction_index]
                
                # Check threshold
                if confidence > CONFIDENCE_THRESHOLD: 
                    # Add confident prediction to history
                    prediction_history.append(ACTIONS[prediction_index])
                else:
                    prediction_history.append("UNCERTAIN")

            # 3. Stabilization and Display
            if prediction_history:
                # Use the mode (most frequent item) for stable output
                stable_prediction = Counter(prediction_history).most_common(1)[0][0]
                
                # Only update displayed sign if the history has a stable prediction
                if stable_prediction != "UNCERTAIN":
                    predicted_sign = stable_prediction
                else:
                    predicted_sign = "UNCERTAIN" # Or just keep the last stable sign
            
            # Use the latest confidence for display, but the stable sign
            visualize_prediction(image, predicted_sign, confidence)
            cv2.imshow('CNN-LSTM Real-Time Sign Prediction', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_realtime_prediction()