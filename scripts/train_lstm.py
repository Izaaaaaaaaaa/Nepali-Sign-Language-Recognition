import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'sequences')
SAVE_PATH = os.path.join(ROOT_DIR, 'models', 'slr_lstm_model.h5')

actions = np.array(['Naam']) # DOWNSCALED
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for seq in range(5):
        p = os.path.join(DATA_PATH, action, f"{seq}.npy")
        if os.path.exists(p):
            sequences.append(np.load(p) / 640.0) 
            labels.append(label_map[action])

X = np.array(sequences)
y = np.array(labels) # Just use 0s for single class

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(60, 10)),
    tf.keras.layers.LSTM(128, return_sequences=False, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(actions), activation='sigmoid') # DOWNSCALED
])

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=100) # Accuracy will hit 1.0 quickly
model.save(SAVE_PATH)