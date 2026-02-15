import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

# --- PATH LOGIC ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'sequences')
SAVE_PATH = os.path.join(ROOT_DIR, 'models', 'slr_lstm_model.keras')

# 1. Configuration
actions = np.array(['Timro','Naam','K','ho']) # FULL SCALE
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:
    for seq_num in range(30):
        file_path = os.path.join(DATA_PATH, action, f"{seq_num}.npy")
        if os.path.exists(file_path):
            res = np.load(file_path)
            res_norm = res / 640.0 
            
            # Find first non-zero frame
            first_valid = next((f for f in res_norm if np.any(f != 0)), res_norm[0])
            rel_res = res_norm - first_valid

            # Add Original and 4 Augmentations
            sequences.append(rel_res)
            labels.append(label_map[action])
            
            for _ in range(4): # 4 more random variations
                noise = np.random.normal(0, 0.003, rel_res.shape) # Smaller noise for stability
                sequences.append(rel_res + noise)
                labels.append(label_map[action])

X = np.array(sequences)
y = tf.keras.utils.to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)

# 2. THE STABILIZED MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(60, 10)),
    # FIX 1: Switched to 'tanh' activation (Standard for LSTMs, much more stable than relu)
    tf.keras.layers.LSTM(64, return_sequences=True, activation='tanh'),
    tf.keras.layers.BatchNormalization(), # Keeps numbers in a healthy range
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(128, return_sequences=False, activation='tanh'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(actions), activation='softmax')
])

# FIX 2: Added Gradient Clipping (clipnorm=1.0) 
# This prevents the accuracy from 'crashing' by limiting how large a mathematical change can be.
# FIX 3: Lowered Learning Rate to 0.0001
opt = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# FIX 4: Added Early Stopping with "Restore Best Weights"
# If the model starts to crash, it will stop and save the best version it found.
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=30, restore_best_weights=True
)

print(f"Training on {len(X)} stabilized sequences...")
model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

model.save(SAVE_PATH)
print("SUCCESS: Stabilized Model Saved.")