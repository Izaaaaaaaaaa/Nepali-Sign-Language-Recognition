import cv2
import numpy as np
import os
from ultralytics import YOLO
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
YOLO_PATH = os.path.join(ROOT_DIR, 'models', 'yolov8_hand.pt')
LSTM_PATH = os.path.join(ROOT_DIR, 'models', 'slr_lstm_model.h5')

model_yolo = YOLO(YOLO_PATH)
model_lstm = tf.keras.models.load_model(LSTM_PATH)
actions = np.array(['Naam']) # DOWNSCALED

sequence = []
sentence = "Analyzing..."
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    results = model_yolo(frame, verbose=False)
    feat = np.zeros((2, 5))
    boxes = results[0].boxes.data.cpu().numpy()
    for i, box in enumerate(boxes):
        if i < 2: feat[i] = box[:5]
    
    sequence.append(feat.flatten() / 640.0)
    sequence = sequence[-60:]
    
    if len(sequence) == 60:
        res = model_lstm.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        # For sigmoid/1-class, if probability > 0.5, it's a match
        if res > 0.8:
            sentence = "Naam"

    cv2.rectangle(frame, (0,0), (640, 50), (245, 117, 16), -1)
    cv2.putText(frame, sentence, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Downscaled Demo', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()