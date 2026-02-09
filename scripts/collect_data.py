import cv2
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'sequences')

# DOWNSCALED ACTION
actions = np.array(['Naam'])
no_sequences = 30 
sequence_length = 60 

for action in actions: 
    for seq in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(seq)), exist_ok=True)

cap = cv2.VideoCapture(0)
try:
    for action in actions:
        for seq in range(30):
            for frame_num in range(60):
                ret, frame = cap.read()
                if frame_num == 0:
                    cv2.putText(frame, f'STARTING {action}', (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.imshow('Collection', frame)
                    cv2.waitKey(2000)
                cv2.imshow('Collection', frame)
                cv2.imwrite(os.path.join(DATA_PATH, action, str(seq), f"{frame_num}.jpg"), frame)
                if cv2.waitKey(10) & 0xFF == ord('q'): raise KeyboardInterrupt
finally:
    cap.release()
    cv2.destroyAllWindows()