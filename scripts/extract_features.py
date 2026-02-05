import cv2
import numpy as np
import os
from ultralytics import YOLO

# --- PATH LOGIC ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'sequences')

model = YOLO('yolov8n.pt') # Change to 'yolov8_hand.pt' after you train custom YOLO

actions = np.array(['Namaste', 'Kasto_Cha', 'Mero_Naam', 'Sahayog', 'Dhanyabaad', 
                   'Ma_Nepali', 'Khana_Khayau', 'Sanchai_Chu', 'Birami_Chu', 'Ghar_Kata'])

print("Extracting features from 60-frame sequences...")

for action in actions:
    for sequence in range(30):
        video_feature_data = [] 
        for frame_num in range(60):
            img_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.jpg")
            img = cv2.imread(img_path)
            
            if img is None:
                video_feature_data.append(np.zeros(10))
                continue

            results = model(img, verbose=False)
            frame_features = np.zeros((2, 5)) 
            boxes = results[0].boxes.data.cpu().numpy()
            for i, box in enumerate(boxes):
                if i < 2: frame_features[i] = box[:5]
            
            video_feature_data.append(frame_features.flatten())
        
        np.save(os.path.join(DATA_PATH, action, f"{sequence}.npy"), np.array(video_feature_data))
    print(f"Processed: {action}")