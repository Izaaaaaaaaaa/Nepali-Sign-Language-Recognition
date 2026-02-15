import cv2
import numpy as np
import os
from ultralytics import YOLO
import tensorflow as tf

# --- PATH SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
YOLO_PATH = os.path.join(ROOT_DIR, 'models', 'yolov8_hand.pt')
LSTM_PATH = os.path.join(ROOT_DIR, 'models', 'slr_lstm_model.keras')

# --- CONFIGURATION ---
# Change the action and folder number here to test different recordings
TEST_FOLDER = os.path.join(ROOT_DIR, 'data', 'sequences', 'K', '27')
actions = np.array(['Timro','Naam','K','ho']) # FULL SCALE

# 1. Load Models
print("Loading Models...")
model_yolo = YOLO(YOLO_PATH)
model_lstm = tf.keras.models.load_model(LSTM_PATH)

sequence_buffer = []

print(f"Analyzing: {TEST_FOLDER}")

# 2. Sequential Processing with Visual Masking
for i in range(60):
    img_path = os.path.join(TEST_FOLDER, f"{i}.jpg")
    frame = cv2.imread(img_path)
    
    feat = np.zeros(10)
    
    if frame is not None:
        # --- THE BLACK BOX MASK ---
        # Draw a solid black rectangle to hide the action name text
        # (x1, y1), (x2, y2), color, thickness -1 (fill)
        cv2.rectangle(frame, (0, 0), (450, 45), (0, 0, 0), -1) 

        frame_resized = cv2.resize(frame, (640, 640))
        results = model_yolo(frame_resized, conf=0.1, verbose=False)
        
        # Draw boxes on the masked frame
        annotated_frame = results[0].plot()
        
        if len(results[0].boxes) > 0:
            box = results[0].boxes.data.cpu().numpy()[0][:5]
            feat[:5] = box
            
        cv2.putText(annotated_frame, f"Processing Frame: {i}", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Offline Test - Masked Feed", annotated_frame)
        
        if cv2.waitKey(20) & 0xFF == ord('q'): break
    
    sequence_buffer.append(feat / 640.0)

# 3. Final Prediction Logic
if len(sequence_buffer) == 60:
    seq_np = np.array(sequence_buffer)
    # Robust Centering: Find first non-zero frame
    first_valid = next((f for f in seq_np if np.any(f != 0)), seq_np[0])
    relative_seq = seq_np - first_valid

    res = model_lstm.predict(np.expand_dims(relative_seq, axis=0), verbose=0)[0]
    prediction = actions[np.argmax(res)].replace('_', ' ')
    
    # Show Final Result on screen
    result_img = cv2.imread(os.path.join(TEST_FOLDER, "59.jpg"))
    cv2.rectangle(result_img, (0, 0), (450, 45), (0, 0, 0), -1) # Mask last frame too
    cv2.rectangle(result_img, (0, 250), (640, 330), (0, 255, 0), -1)
    cv2.putText(result_img, f"DETECTED: {prediction}", (50, 305), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3)
    cv2.imshow("Offline Test - Masked Feed", result_img)
    print(f"Result: {prediction}")
    cv2.waitKey(4000)

cv2.destroyAllWindows()