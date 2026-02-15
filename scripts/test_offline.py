import cv2
import numpy as np
import os
from ultralytics import YOLO
import tensorflow as tf

# --- PATH SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
YOLO_PATH = os.path.join(ROOT_DIR, 'models', 'yolov8_hand.pt')
# Use .keras or .h5 based on your file name
LSTM_PATH = os.path.join(ROOT_DIR, 'models', 'slr_lstm_model.h5')

# --- CONFIGURATION ---
# Pick a folder you want to test (Change this to test different videos)
TEST_FOLDER = os.path.join(ROOT_DIR, 'data', 'sequences', 'K', '9')
actions = np.array(['Naam']) # Should match your training list

# 1. Load Models
model_yolo = YOLO(YOLO_PATH)
model_lstm = tf.keras.models.load_model(LSTM_PATH)

print(f"--- TESTING FOLDER: {TEST_FOLDER} ---")

sequence = []

# 2. Process all 60 frames in that folder
for i in range(60):
    img_path = os.path.join(TEST_FOLDER, f"{i}.jpg")
    frame = cv2.imread(img_path)
    
    if frame is None:
        print(f"Error: Frame {i} not found.")
        continue

    # Resize to match training
    frame_resized = cv2.resize(frame, (640, 640))
    
    # YOLO Detection (Very sensitive for testing)
    results = model_yolo(frame_resized, conf=0.05, verbose=False)
    feat = np.zeros(10)
    
    if len(results[0].boxes) > 0:
        # Get coordinates for the first hand found
        box = results[0].boxes.data.cpu().numpy()[0][:5]
        feat[:5] = box
        
        # Show detection result for debugging
        annotated = results[0].plot()
        cv2.imshow("Reviewing Frames", annotated)
        cv2.waitKey(30) # Play back like a video

    # Normalize and add to sequence
    sequence.append(feat / 640.0)

# 3. Final Prediction
if len(sequence) == 60:
    res = model_lstm.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
    prediction = actions[np.argmax(res)]
    confidence = np.max(res)
    
    print("\n" + "="*30)
    print(f"RESULT: {prediction}")
    print(f"CONFIDENCE: {confidence:.2%}")
    print("="*30)

cv2.destroyAllWindows()