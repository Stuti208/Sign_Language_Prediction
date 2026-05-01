import cv2
import mediapipe as mp
import numpy as np
import os
import pathlib 
from tqdm import tqdm 

# --- 1. Configuration (CRITICAL: REPLACE PLACEHOLDERS WITH ABSOLUTE PATHS) ---
# Replace this with the ABSOLUTE path to the 'Video_Dataset/Video_Dataset' folder
INPUT_VIDEOS_PATH = r"C:\Users\aagam\Documents\Major Project 2\Sign Language Predictio Model\dataset2\Video_Dataset\Video_Dataset" 

# Replace this with the ABSOLUTE path to your '2_DATA_PROCESSED' output folder
OUTPUT_PATH = r"C:\Users\aagam\Documents\Major Project 2\Sign Language Predictio Model\DATA_PROCESSED" 

SEQUENCE_LENGTH = 30  # Number of frames for the LSTM input
KEYPOINT_DIM = 138    # Features per frame
POSE_INDICES = [0, 11, 12, 13, 14, 15, 16] # Selected Pose landmarks


# --- 2. Initialize MediaPipe & Helper Functions ---
mp_holistic = mp.solutions.holistic 

def mediapipe_detection(image, model):
    """Processes an image and returns landmark results."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB); image.flags.writeable = False
    results = model.process(image); image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    """Flattens and extracts relevant hand and pose keypoints."""
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

# Ensure the output directory exists
os.makedirs(os.path.join(OUTPUT_PATH, 'Train_Sequences'), exist_ok=True)


# --- 3. Main Processing Logic ---

def process_video_dataset():
    
    data_root = pathlib.Path(INPUT_VIDEOS_PATH)
    
    # Reliably find all sign word sub-directories (Bear, Break, Budget, etc.)
    ACTIONS_PATHS = [p for p in data_root.iterdir() if p.is_dir()]
    ACTIONS = [p.name for p in ACTIONS_PATHS]
    
    # --- IMMEDIATE FEEDBACK PRINT STATEMENT ---
    print(f"--- Found {len(ACTIONS)} sign word folders to process. ---")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # Loop through each sign folder (e.g., 'Bear', 'Break')
        for action_path in tqdm(ACTIONS_PATHS, desc="Processing Signs"):
            action = action_path.name 
            
            # Find all video files (.mp4) inside the current folder
            video_files = sorted([str(p) for p in action_path.glob('*.mp4')])
            
            # Loop through each video instance for the current sign
            for sequence_count, video_file in enumerate(video_files):
                
                cap = cv2.VideoCapture(video_file)
                frame_sequence = []
                
                if not cap.isOpened():
                    # print(f"Warning: Could not open video file: {video_file}")
                    continue
                
                # --- Frame Sampling/Extraction ---
                while len(frame_sequence) < SEQUENCE_LENGTH:
                    ret, frame = cap.read()
                    
                    if not ret or frame is None:
                        break 
                    
                    # Process frame and extract keypoints
                    image, results = mediapipe_detection(frame, holistic)
                    keypoints = extract_keypoints(results)
                    frame_sequence.append(keypoints)
                
                cap.release()
                
                # --- Save Processed Sequence ---
                if len(frame_sequence) == SEQUENCE_LENGTH:
                    save_dir = os.path.join(OUTPUT_PATH, 'Train_Sequences', action)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # Generate a unique filename
                    npy_filename = f"{action.replace(' ', '_')}_{sequence_count:03d}.npy" 
                    
                    np.save(os.path.join(save_dir, npy_filename), np.array(frame_sequence))
                
                # Optional: If video is too short, the sequence is discarded. 
                # If you need to pad short videos, add logic here.
                    
    print("\nData Preparation Complete! Keypoint sequences are ready for CNN-LSTM training.")
    cv2.destroyAllWindows()

if __name__ == '__main__': 
    process_video_dataset()