import cv2
import numpy as np
from ultralytics import YOLO
from keras import load_model

model_yolo = YOLO('yolov8n.pt')
model_lstm = load_model('../models/slr_lstm_model.h5')
actions = np.array(['Namaste', 'Kasto_Cha', 'Mero_Naam', 'Sahayog', 'Dhanyabaad', 
                   'Ma_Nepali', 'Khana_Khayau', 'Sanchai_Chu', 'Birami_Chu', 'Ghar_Kata'])

sequence = []
sentence = "Waiting..."
threshold = 0.8 # Only show result if 80% confident

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # 1. YOLO Detection
    results = model_yolo(frame, verbose=False)
    frame_features = np.zeros((2, 5))
    boxes = results[0].boxes.data.cpu().numpy()
    for i, box in enumerate(boxes):
        if i < 2: frame_features[i] = box[:5]
    
    # 2. Sequence Handling
    sequence.append(frame_features.flatten())
    sequence = sequence[-30:] # Always take the last 30 frames
    
    # 3. Prediction
    if len(sequence) == 30:
        res = model_lstm.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        if res[np.argmax(res)] > threshold:
            sentence = actions[np.argmax(res)]

    # 4. Display
    cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
    cv2.putText(frame, f'Detected: {sentence}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Nepali Sign Language System', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()