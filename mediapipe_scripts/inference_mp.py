import cv2
import numpy as np
import os
import tensorflow as tf

# MediaPipe imports - Compatible with latest versions
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Set to False for faster webcam tracking
    max_num_hands=2, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- ABSOLUTE PATH LOGIC ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'slr_lstm_mp.keras')

# Load the trained LSTM brain (The one trained on skeletal data)
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    exit()

model_lstm = tf.keras.models.load_model(MODEL_PATH)

# Configuration
actions = np.array(['Naam'])  # Update with actual action labels used during training

sequence = []
sentence = "Watching..."
threshold = 0.8  # 80% confidence required

cap = cv2.VideoCapture(0)
print("MediaPipe SLR System Live. Press 'q' on the window to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break
    
    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # MediaPipe needs RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    # Feature vector for 2 hands (21 landmarks * 3 coords * 2 hands = 126)
    feat = np.zeros(126)
    
    if results.multi_hand_landmarks:
        for i, hand_lms in enumerate(results.multi_hand_landmarks):
            # VISUALIZATION: Draw the skeleton on the frame
            mp_draw.draw_landmarks(
                frame, 
                hand_lms, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            if i < 2:
                # Extract x, y, z for all 21 points
                lms = np.array([[res.x, res.y, res.z] for res in hand_lms.landmark]).flatten()
                feat[i*63:(i+1)*63] = lms

    # Sequence Handling
    # MediaPipe landmarks are already normalized (0 to 1), no division needed
    sequence.append(feat)
    sequence = sequence[-60:]  # Buffer the last 60 frames
    
    # Prediction
    if len(sequence) == 60:
        res = model_lstm.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        
        if np.max(res) > threshold:
            sentence = actions[np.argmax(res)].replace('_', ' ')

    # UI Overlay
    cv2.rectangle(frame, (0, 0), (640, 50), (245, 117, 16), -1)
    cv2.putText(frame, f"SKELETAL-SLR: {sentence}", (15, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Nepali SLR - Stable MediaPipe Engine', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
hands.close()