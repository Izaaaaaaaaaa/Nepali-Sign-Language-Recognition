import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

# --- PATH LOGIC ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'sequences')
SAVE_PATH = os.path.join(ROOT_DIR, 'models', 'slr_lstm_mp.keras')

actions = np.array(['Namaskar_Mero_Naam', 'Malai_Bhok_Lagyo', 'Timro_Naam_K_Ho', 
                   'Timi_Kati_Barsa', 'Malai_Gadit_Mann_Parcha', 'Chiya_Khana_Jaam', 'Bajey_Aunu_Bhayo'])
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
print("Loading Skeletal sequences...")

for action in actions:
    for seq in range(30):
        p = os.path.join(DATA_PATH, action, f"{seq}_mp.npy")
        if os.path.exists(p):
            # MediaPipe data is already 0-1, so no extra normalization needed
            sequences.append(np.load(p))
            labels.append(label_map[action])

X = np.array(sequences)
y = tf.keras.utils.to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Model Architecture optimized for 126 features
model = tf.keras.Sequential([
    tf.keras.Input(shape=(60, 126)),
    tf.keras.layers.LSTM(64, return_sequences=True, activation='relu'),
    tf.keras.layers.LSTM(128, return_sequences=False, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(actions), activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training the skeletal LSTM...")
model.fit(X_train, y_train, epochs=150, batch_size=32)

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
model.save(SAVE_PATH)
print(f"SUCCESS: Model saved to {SAVE_PATH}")