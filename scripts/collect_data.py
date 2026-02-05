import cv2
import numpy as np
import os

# --- PATH LOGIC ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'sequences')

actions = np.array(['Namaste', 'Kasto_Cha', 'Mero_Naam', 'Sahayog', 'Dhanyabaad', 
                   'Ma_Nepali', 'Khana_Khayau', 'Sanchai_Chu', 'Birami_Chu', 'Ghar_Kata'])
no_sequences = 30 
sequence_length = 60 # Updated to 60 frames (2 seconds)

# Create folders
for action in actions: 
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

print(f"Data will be saved in: {DATA_PATH}")
print("Press 'q' to stop early.")

try:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret: break

                if frame_num == 0: 
                    cv2.putText(frame, 'GET READY...', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, f'Action: {action} | Video: {sequence}', (120,250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)
                    cv2.waitKey(2000) 
                else: 
                    cv2.putText(frame, f'Recording {action} | Video {sequence} | Frame {frame_num}', (15,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)

                img_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.jpg")
                cv2.imwrite(img_path, frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    raise KeyboardInterrupt
finally:
    cap.release()
    cv2.destroyAllWindows()