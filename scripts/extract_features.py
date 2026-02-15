import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

# --- ABSOLUTE PATH RESOLUTION ---
# This ensures the script works regardless of where the terminal is opened
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'sequences')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# Create the models folder if it's not there
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Configuration (Names must match your folder names exactly)
actions = np.array(['Timro','Naam','K','ho']) # FULL SCALE

# If you are on the "downscaled" branch for testing, uncomment the line below:
# actions = np.array(['Naam']) 

label_map = {label:num for num, label in enumerate(actions)}

# 2. Data Loading (Robust/Fault-Tolerant Logic)
sequences, labels = [], []
print("--- LOADING DATA ---")

for action in actions:
    count = 0
    for seq_num in range(30): # Looks for files 0.npy through 29.npy
        file_path = os.path.join(DATA_PATH, action, f"{seq_num}.npy")
        
        # SAFETY CHECK: If you deleted folder 0, this 'if' will skip it safely
        if os.path.exists(file_path):
            try:
                res = np.load(file_path)
                
                # --- NORMALIZATION ---
                # Scaling coordinate values from pixels (0-640) to 0-1 range
                sequences.append(res / 640.0)
                labels.append(label_map[action])
                count += 1
            except Exception as e:
                print(f"Warning: Could not load {file_path}. Error: {e}")
        
    print(f"Loaded {count} sequences for: {action}")

# Verify we have data before continuing
if len(sequences) == 0:
    print("FATAL ERROR: No .npy files found. Did you run extract_features.py?")
    exit()

X = np.array(sequences)

# 3. Model Logic (Auto-Detects 1 vs Multiple Classes)
if len(actions) > 1:
    y = tf.keras.utils.to_categorical(labels).astype(int)
    output_activation = 'softmax'
    loss_fn = 'categorical_crossentropy'
else:
    y = np.array(labels)
    output_activation = 'sigmoid'
    loss_fn = 'binary_crossentropy'

# Split: 90% to learn, 10% to test accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# 4. Build LSTM Neural Network (Keras 3.0 Compatible)
model = tf.keras.Sequential([
    # Input layer: 60 frames, 10 coordinates per frame
    tf.keras.layers.Input(shape=(60, 10)), 
    
    tf.keras.layers.LSTM(64, return_sequences=True, activation='relu'),
    tf.keras.layers.Dropout(0.2), # Prevents memorizing the background
    tf.keras.layers.LSTM(128, return_sequences=False, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(actions), activation=output_activation)
])

model.compile(optimizer='Adam', loss=loss_fn, metrics=['accuracy'])

# 5. Training
print(f"\n--- TRAINING BRAIN ON {len(X)} VIDEOS ---")
model.fit(X_train, y_train, epochs=200, batch_size=32)

# 6. Save the Result
model_save_path = os.path.join(MODELS_DIR, 'slr_lstm_model.keras')
model.save(model_save_path)

print(f"\nSUCCESS: Brain saved to {model_save_path}")