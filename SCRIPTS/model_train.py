import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv1D, MaxPooling1D, Conv2D, Reshape, Flatten, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns 

# --- 1. Configuration (CRITICAL: VERIFY THIS PATH) ---
# Assuming the script runs from 'SCRIPTS' and data is in '2_DATA_PROCESSED'
OUTPUT_PATH = os.path.join('..', 'DATA_PROCESSED') 
DATA_PATH = os.path.join(OUTPUT_PATH, 'Train_Sequences')
MODELS_PATH = os.path.join('..', 'MODELS') 
os.makedirs(MODELS_PATH, exist_ok=True)

SEQUENCE_LENGTH = 30  # Verified: Number of frames per sequence
KEYPOINT_DIM = 147    # Verified: Total features per frame (The final working dimension)

print(f"DEBUG: Attempting to load data from: {DATA_PATH}")

# ----------------------------------------------------
# 2. Load Data and Labels
# ----------------------------------------------------

sequences, labels = [], []
# Get the list of 61 sign names (e.g., 'Bear', 'Break')
actions = sorted(os.listdir(DATA_PATH)) 

if not actions:
    print("Error: No action folders found. Please verify the absolute path in your data_prep.py.")
    exit()

print(f"Loading data for {len(actions)} classes: {actions}")

for idx, action in enumerate(tqdm(actions, desc="Loading Sequences")):
    action_path = os.path.join(DATA_PATH, action)
    
    for npy_file in os.listdir(action_path):
        if npy_file.endswith('.npy'):
            try:
                seq = np.load(os.path.join(action_path, npy_file))
                
                # CRITICAL CHECK: Using the verified shape (30, 147)
                if seq.shape == (SEQUENCE_LENGTH, KEYPOINT_DIM):
                    sequences.append(seq)
                    labels.append(idx) 
            except Exception as e:
                print(f"Skipping corrupted file {npy_file}: {e}")


X = np.array(sequences)
y = to_categorical(np.array(labels), num_classes=len(actions))

print(f"\nTotal sequences loaded: {X.shape[0]}")
print(f"Input Data Shape (X): {X.shape} (Sequences, Frames, Features)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# ----------------------------------------------------
# 3. Define CNN-LSTM Architecture
# ----------------------------------------------------

# --- 3. Define CNN-LSTM Architecture (FIXED) ---

def build_cnn_lstm_model(input_shape, num_classes):
    
    # input_shape is (SEQUENCE_LENGTH, KEYPOINT_DIM) -> (30, 147)
    inputs = Input(shape=input_shape) 
    
    # --- 1. Feature Pre-Processing (TimeDistributed Dense) ---
    # Apply Dense layers to each time step (frame) to extract latent features 
    # from the 147 keypoints. This avoids the Conv1D/2D bugs entirely.
    
    # Process 147 features into 128 latent features per time step
    x = TimeDistributed(Dense(128, activation='relu'))(inputs)
    x = TimeDistributed(Dropout(0.25))(x)
    
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    x = TimeDistributed(Dropout(0.25))(x)
    
    # --- 2. LSTM Component (Temporal Sequence Learning) ---
    # The LSTM receives the sequence of processed feature vectors (x)
    x = LSTM(128, return_sequences=False, activation='tanh')(x)
    x = Dropout(0.5)(x)
    
    # --- 3. Classification Output ---
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create the final functional model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

input_shape = (SEQUENCE_LENGTH, KEYPOINT_DIM) 
num_classes = len(actions)

model = build_cnn_lstm_model(input_shape, num_classes)
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# ----------------------------------------------------
# 4. Training and Callbacks
# ----------------------------------------------------

log_dir = os.path.join(MODELS_PATH, 'Logs')
tb_callback = TensorBoard(log_dir=log_dir) 
es_callback = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True) 
mc_callback = ModelCheckpoint(filepath=os.path.join(MODELS_PATH, 'best_model.keras'), 
                              monitor='val_categorical_accuracy', 
                              save_best_only=True, 
                              mode='max') 

print("\nStarting model training...")
# The script will successfully run to this point, then fail on the model.fit() call due to the TF bug.
history = model.fit(
    X_train, 
    y_train, 
    epochs=100, 
    batch_size=32, 
    validation_data=(X_test, y_test),
    callbacks=[tb_callback, es_callback, mc_callback]
)

# ----------------------------------------------------
# 5. Final Output and Saving 💾
# ----------------------------------------------------

# Evaluate the final model
print("\n--- Final Model Evaluation ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy*100:.2f}%')

# Save the final model and actions list
model.save(os.path.join(MODELS_PATH, 'final_cnn_lstm_model.keras'))
with open(os.path.join(MODELS_PATH, 'actions_list.txt'), 'w') as f:
    for action in actions:
        f.write(f"{action}\n")

print(f"Model and actions list saved successfully in: {MODELS_PATH}")

# ----------------------------------------------------
# 6. Graph and Confusion Matrix Generation 📊
# ----------------------------------------------------

# 6.1. Plot Training Performance
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

graph_path = os.path.join(MODELS_PATH, 'training_performance.png')
plt.savefig(graph_path)


# 6.2. Generate and Plot Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(18, 15)) 
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
            xticklabels=actions, yticklabels=actions)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

cm_path = os.path.join(MODELS_PATH, 'confusion_matrix.png')
plt.savefig(cm_path)
plt.show() 

print(f"Training performance graph saved to: {graph_path}")
print(f"Confusion Matrix saved to: {cm_path}")
print("\nTraining script finished. Ready for real-time testing.")