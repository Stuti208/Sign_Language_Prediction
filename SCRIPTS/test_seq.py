import numpy as np
import os

# --- YOU MUST REPLACE THIS PATH with the actual path to ONE saved .npy file ---
TEST_FILE_PATH = r"C:\Users\aagam\Documents\Major Project 2\Sign Language Predictio Model\DATA_PROCESSED\Train_Sequences\Bear\Bear_000.npy" 

try:
    test_seq = np.load(TEST_FILE_PATH) 
    print(f"File loaded successfully. Actual shape is: {test_seq.shape}")
except Exception as e:
    print(f"Error loading test file. Check the path: {e}")