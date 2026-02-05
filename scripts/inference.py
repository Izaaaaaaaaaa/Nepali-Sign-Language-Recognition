import cv2
import numpy as np
import os
from ultralytics import YOLO
import tensorflow as tf

# --- PATH LOGIC ---
# This ensures the script finds the model even if you move the folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'slr_lstm_model.h5')

# Load Models
model_yolo = YOLO('yolov8n.pt') # Change to 'yolov8_hand.pt' after custom training
model_lstm = tf.keras.models.load_model(MODEL_PATH)

actions = np.array(['Namaste', 'Kasto_Cha', 'Mero_Naam', 'Sahayog', 'Dhanyabaad', 
                   'Ma_Nepali', 'Khana_Khayau', 'Sanchai_Chu', 'Birami_Chu', 'Ghar_Kata'])

sequence = []
sentence = "Analyzing..."
threshold = 0.8 # 80% confidence required to display a result

cap = cv2.VideoCapture(0)

print("System is running. Press 'q' on the window to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 1. YOLO Detection (Find hands/person)
    results = model_yolo(frame, verbose=False)
    frame_features = np.zeros((2, 5))
    boxes = results[0].boxes.data.cpu().numpy()
    
    for i, box in enumerate(boxes):
        if i < 2: 
            frame_features[i] = box[:5]
    
    # 2. Sequence Handling (Buffer for 60 frames)
    sequence.append(frame_features.flatten())
    sequence = sequence[-60:] # Always keep the last 60 frames
    
    # 3. Prediction Logic
    if len(sequence) == 60:
        # Pass the 60-frame sequence to the LSTM
        res = model_lstm.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        
        # Check if model is confident enough
        if res[np.argmax(res)] > threshold:
            sentence = actions[np.argmax(res)]

    # 4. Display Result on Screen
    # Rectangle for background
    cv2.rectangle(frame, (0,0), (640, 50), (245, 117, 16), -1)
    cv2.putText(frame, f'SENTENCE: {sentence}', (15, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Nepali SLR Live System', frame)
    
    # Exit gracefully
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()