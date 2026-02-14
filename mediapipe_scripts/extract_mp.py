import cv2
import numpy as np
import os
import tensorflow as tf

# Standard direct imports
import mediapipe.solutions.hands as mp_hands
import mediapipe.solutions.drawing_utils as mp_draw

# Initialize hands
hands = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ... (rest of your script below)

# --- ABSOLUTE PATH LOGIC ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'sequences')

actions = np.array(['Naam'])  # Update with your actual action names

print("Starting Skeletal Feature Extraction...")

for action in actions:
    for seq in range(30):
        seq_path = os.path.join(DATA_PATH, action, str(seq))
        if not os.path.exists(seq_path): continue
        
        sequence_data = []
        for f_num in range(60):
            img = cv2.imread(os.path.join(seq_path, f"{f_num}.jpg"))
            # 21 landmarks * 3 coordinates (x,y,z) * 2 hands = 126 values
            feat = np.zeros(126) 
            
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                
                if results.multi_hand_landmarks:
                    for i, hand_lms in enumerate(results.multi_hand_landmarks):
                        if i < 2:
                            # Extract x, y, z for all 21 points and flatten into a list
                            lms = np.array([[res.x, res.y, res.z] for res in hand_lms.landmark]).flatten()
                            feat[i*63:(i+1)*63] = lms
            
            sequence_data.append(feat)
        
        # Save as _mp.npy to keep it separate from YOLO data
        np.save(os.path.join(DATA_PATH, action, f"{seq}_mp.npy"), np.array(sequence_data))
    print(f"Extraction complete for: {action}")

print("\nSuccess! You can now run 3_train_mp.py")