import cv2
import numpy as np
import os
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'sequences')
YOLO_MODEL = os.path.join(ROOT_DIR, 'models', 'yolov8_hand.pt')

model = YOLO(YOLO_MODEL)
actions = np.array(['Namaskar', 'Mero', 'Naam', 'Malai', 'Bhok', 'Lagyo', 'Timro', 'Naam', 'K', 'Ho', 
                   'Timi', 'Kati', 'Barsa', 'Malai', 'Gadit', 'Mann', 'Parcha', 'Chiya', 'Khana', 'Jaam', 'Bajey', 'Aunu', 'Bhayo'])

for action in actions:
    for seq in range(30):
        video_data = []
        for f_num in range(60):
            img = cv2.imread(os.path.join(DATA_PATH, action, str(seq), f"{f_num}.jpg"))
            feat = np.zeros((2, 5))
            if img is not None:
                results = model(img, verbose=False)
                boxes = results[0].boxes.data.cpu().numpy()
                for i, box in enumerate(boxes):
                    if i < 2: feat[i] = box[:5]
            video_data.append(feat.flatten())
        np.save(os.path.join(DATA_PATH, action, f"{seq}.npy"), np.array(video_data))
    print(f"Features Extracted: {action}")