import cv2
import numpy as np
import os

# Configuration
DATA_PATH = os.path.join('../data/sequences') 
# 10 Nepali Sentences
actions = np.array(['Namaste', 'Kasto_Cha', 'Mero_Naam', 'Sahayog', 'Dhanyabaad', 
                   'Ma_Nepali', 'Khana_Khayau', 'Sanchai_Chu', 'Birami_Chu', 'Ghar_Kata'])
no_sequences = 30 # 30 videos per sentence
sequence_length = 30 # 30 frames per video

for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)
with cap:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()

                if frame_num == 0: 
                    cv2.putText(frame, f'STARTING COLLECTION for {action}', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)
                    cv2.waitKey(2000) # Pause to prepare
                
                cv2.putText(frame, f'Collecting frames for {action} Video {sequence}', (15,12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                
                # Save the raw image
                cv2.imwrite(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.jpg"), frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
cap.release()
cv2.destroyAllWindows()