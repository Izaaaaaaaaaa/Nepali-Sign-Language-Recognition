import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

# --- PATH LOGIC ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'sequences')
SAVE_PATH = os.path.join(ROOT_DIR, 'models', 'slr_lstm_model.h5')

# 1. Configuration
actions = np.array(['Namaskar', 'Mero', 'Naam', 'Malai', 'Bhok', 'Lagyo', 'Timro', 'Naam', 'K', 'Ho', 
                   'Timi', 'Kati', 'Barsa', 'Malai', 'Gadit', 'Mann', 'Parcha', 'Chiya', 'Khana', 'Jaam', 'Bajey', 'Aunu', 'Bhayo'])
label_map = {label:num for num, label in enumerate(actions)}

# 2. Load and Normalize Data
sequences, labels = [], []
print("Loading .npy sequences...")

for action in actions:
    for seq in range(30):
        path = os.path.join(DATA_PATH, action, f"{seq}.npy")
        if os.path.exists(path):
            # Normalization: scaling coordinates (0-640) to (0-1)
            res = np.load(path) / 640.0
            sequences.append(res)
            labels.append(label_map[action])

X = np.array(sequences)
y = tf.keras.utils.to_categorical(labels).astype(int)

# Split into 90% training / 10% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# 3. Build LSTM Model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(60, 10)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(128, return_sequences=False, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(actions), activation='softmax')
])

# 4. Compile and Train
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(f"Training started for {len(actions)} Nepali sentences...")
model.fit(X_train, y_train, epochs=200, batch_size=32)

# 5. Save
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
model.save(SAVE_PATH)
print(f"SUCCESS: Model saved to {SAVE_PATH}")