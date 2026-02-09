import cv2
import numpy as np
import os
from ultralytics import YOLO
import tensorflow as tf

# --- ABSOLUTE PATH LOGIC ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
YOLO_PATH = os.path.join(ROOT_DIR, 'models', 'yolov8_hand.pt')
LSTM_PATH = os.path.join(ROOT_DIR, 'models', 'slr_lstm_model.h5')

# 1. Load Trained Models
model_yolo = YOLO(YOLO_PATH)
model_lstm = tf.keras.models.load_model(LSTM_PATH)

# 2. Configuration
actions = np.array(['Namaskar', 'Mero', 'Naam', 'Malai', 'Bhok', 'Lagyo', 'Timro', 'Naam', 'K', 'Ho', 
                   'Timi', 'Kati', 'Barsa', 'Malai', 'Gadit', 'Mann', 'Parcha', 'Chiya', 'Khana', 'Jaam', 'Bajey', 'Aunu', 'Bhayo'])

sequence = []
sentence = "Analyzing..."
cap = cv2.VideoCapture(0)

print("Nepali SLR Live. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # STEP 1: YOLO Detection
    results = model_yolo(frame, verbose=False)
    feat = np.zeros((2, 5))
    boxes = results[0].boxes.data.cpu().numpy()
    for i, box in enumerate(boxes):
        if i < 2: feat[i] = box[:5]
    
    # STEP 2: Sequence Handling (Normalization applied)
    sequence.append(feat.flatten() / 640.0) 
    sequence = sequence[-60:]
    
    # STEP 3: Prediction
    if len(sequence) == 60:
        res = model_lstm.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        if res[np.argmax(res)] > 0.8: # Threshold check
            # Convert 'Mero_Naam' to 'Mero Naam' for display
            sentence = actions[np.argmax(res)].replace('_', ' ')

    # STEP 4: Visualizing
    # Blue background bar
    cv2.rectangle(frame, (0,0), (640, 50), (245, 117, 16), -1)
    # Detected text
    cv2.putText(frame, sentence, (15, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Nepali SLR System', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()