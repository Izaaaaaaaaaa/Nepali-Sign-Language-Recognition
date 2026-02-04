import cv2
import numpy as np
import os
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (yolov8n.pt is fastest)
model = YOLO('yolov8n.pt') 

DATA_PATH = '../data/sequences'
actions = np.array(['Namaste', 'Kasto_Cha', 'Mero_Naam', 'Sahayog', 'Dhanyabaad', 
                   'Ma_Nepali', 'Khana_Khayau', 'Sanchai_Chu', 'Birami_Chu', 'Ghar_Kata'])

for action in actions:
    for sequence in range(30):
        window_features = []
        for frame_num in range(30):
            img_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.jpg")
            img = cv2.imread(img_path)
            
            # Run YOLOv8 detection
            results = model(img, verbose=False)
            
            # Extract coordinates of detected objects (focusing on person/hands)
            # We will store the [x1, y1, x2, y2, conf] of the first two detections
            frame_data = np.zeros((2, 5)) # Support 2 hands
            for i, r in enumerate(results[0].boxes.data.tolist()):
                if i < 2: # Take first two detections
                    frame_data[i] = r[:5]
            
            window_features.append(frame_data.flatten())
        
        # Save coordinates as a numpy file for LSTM
        np.save(os.path.join(DATA_PATH, action, f"{sequence}.npy"), window_features)