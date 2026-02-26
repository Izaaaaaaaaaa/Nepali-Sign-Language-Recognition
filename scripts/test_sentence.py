import cv2
import numpy as np
import os
from ultralytics import YOLO
import tensorflow as tf

# --- PATH SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
SENTENCE_ROOT = os.path.join(ROOT_DIR, 'sentence')
YOLO_PATH = os.path.join(ROOT_DIR, 'models', 'yolov8_hand.pt')
LSTM_PATH = os.path.join(ROOT_DIR, 'models', 'slr_lstm_model.keras')

# 1. Load Models
model_yolo = YOLO(YOLO_PATH)
model_lstm = tf.keras.models.load_model(LSTM_PATH)

actions = np.array(['Timro','Naam','K','ho']) # FULL SCALE

subfolders = ['4', '5', '6', '7']
final_sentence_list = []

print(f"--- Sentence Construction Mode (Masked) ---")

for folder in subfolders:
    path = os.path.join(SENTENCE_ROOT, folder)
    if not os.path.exists(path): continue

    word_sequence = []
    last_frame = None

    for i in range(60):
        img = cv2.imread(os.path.join(path, f"{i}.jpg"))
        feat = np.zeros(10)
        if img is not None:
            # --- THE BLACK BOX MASK ---
            cv2.rectangle(img, (0, 0), (450, 45), (0, 0, 0), -1) 

            res = model_yolo(cv2.resize(img, (640, 640)), conf=0.1, verbose=False)
            annotated = res[0].plot()
            if len(res[0].boxes) > 0:
                feat[:5] = res[0].boxes.data.cpu().numpy()[0][:5]
            
            last_frame = annotated.copy()
            cv2.putText(annotated, f"Analyzing Segment {folder}...", (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Sentence Demo", annotated)
            if cv2.waitKey(20) & 0xFF == ord('q'): break
        
        word_sequence.append(feat / 640.0)

    # Prediction
    if len(word_sequence) == 60:
        seq_np = np.array(word_sequence)
        first_valid = next((f for f in seq_np if np.any(f != 0)), seq_np[0])
        relative_seq = seq_np - first_valid

        pred_res = model_lstm.predict(np.expand_dims(relative_seq, axis=0), verbose=0)[0]
        word = actions[np.argmax(pred_res)].replace('_', ' ')
        final_sentence_list.append(word)

        # Flash intermediate result
        if last_frame is not None:
            cv2.rectangle(last_frame, (0, 250), (640, 350), (0, 255, 0), -1)
            cv2.putText(last_frame, f"WORD FOUND: {word}", (50, 315), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
            cv2.imshow("Sentence Demo", last_frame)
            cv2.waitKey(1200)

# Final Reveal
summary = np.zeros((300, 1000, 3), dtype=np.uint8)
cv2.putText(summary, "FINAL TRANSLATION:", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(summary, " ".join(final_sentence_list), (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
cv2.imshow("Final Result", summary)
print("SENTENCE: " + " ".join(final_sentence_list))
cv2.waitKey(0)
cv2.destroyAllWindows()