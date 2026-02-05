import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Aliases to fix VS Code yellow line warnings
to_categorical = tf.keras.utils.to_categorical
Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout

# --- ABSOLUTE PATH LOGIC ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'sequences')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Configuration
actions = np.array(['Namaste', 'Kasto_Cha', 'Mero_Naam', 'Sahayog', 'Dhanyabaad', 
                   'Ma_Nepali', 'Khana_Khayau', 'Sanchai_Chu', 'Birami_Chu', 'Ghar_Kata'])

label_map = {label:num for num, label in enumerate(actions)}

# 2. Load Data from .npy files
sequences, labels = [], []
print("Loading data from .npy sequences...")

for action in actions:
    for sequence in range(30):
        # Join path correctly to find the numpy files
        file_path = os.path.join(DATA_PATH, action, f"{sequence}.npy")
        if os.path.exists(file_path):
            res = np.load(file_path)
            sequences.append(res)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split into Training (90%) and Testing (10%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# 3. Build LSTM Neural Network Architecture
model = Sequential([
    # Input shape: 60 frames, 10 coordinates per frame
    LSTM(64, return_sequences=True, activation='relu', input_shape=(60, 10)),
    Dropout(0.2),
    LSTM(128, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(actions.shape[0], activation='softmax') # 10 output classes
])

# 4. Compile and Train
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training brain on 60-frame sequences...")
model.fit(X_train, y_train, epochs=150, batch_size=32)

# 5. Save the final model
model_save_path = os.path.join(MODELS_DIR, 'slr_lstm_model.h5')
model.save(model_save_path)
print(f"Success: Model saved to {model_save_path}")