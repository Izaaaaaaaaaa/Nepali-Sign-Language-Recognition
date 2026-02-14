import cv2
import numpy as np
import os
from ultralytics import YOLO
import tensorflow as tf

# --- PATH LOGIC ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
YOLO_PATH = os.path.join(ROOT_DIR, 'models', 'yolov8_hand.pt')
LSTM_PATH = os.path.join(ROOT_DIR, 'models', 'slr_lstm_model.h5')

# 1. Load Trained Models
model_yolo = YOLO(YOLO_PATH)
model_lstm = tf.keras.models.load_model(LSTM_PATH)

# 2. Configuration
# Match these exactly to your train_lstm.py actions
actions = np.array(['Naam']) # DOWNSCALED

sequence = []
sentence = "Analyzing..."

# 3. Start Webcam
cap = cv2.VideoCapture(0)

print("Starting Standardized System...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # FIX 1: Resize to SQUARE 640x640 (Matches YOLO training exactly)
    frame_resized = cv2.resize(frame, (640, 640))

    # FIX 2: Set confidence to 0.05 (Super sensitive)
    results = model_yolo(frame_resized, conf=0.05, imgsz=640, verbose=False)
    
    annotated_frame = results[0].plot() 

    feat = np.zeros((2, 5))
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes.data.cpu().numpy()
        for i, box in enumerate(boxes):
            if i < 2:
                feat[i] = box[:5]
        # DEBUG PRINT: Uncomment this to see if numbers are moving
        # print(f"Coordinates: {feat[0][0]}, {feat[0][1]}") 

    # --- LSTM LOGIC ---
    # We use frame_resized width (640) for normalization
    sequence.append(feat.flatten() / 640.0)
    sequence = sequence[-60:]
    
    if len(sequence) == 60:
        res = model_lstm.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        # Lowered recognition threshold to 0.3 for the demo
        if np.max(res) > 0.3:
            sentence = actions[np.argmax(res)].replace('_', ' ')

    # UI DISPLAY
    cv2.rectangle(annotated_frame, (0,0), (640, 50), (245, 117, 16), -1)
    cv2.putText(annotated_frame, f"SIGN: {sentence}", (15, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Show the feed (this will be a 640x640 square window)
    cv2.imshow('Nepali SLR System - Stability Mode', annotated_frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()