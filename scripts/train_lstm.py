import numpy as np
import os
import tensorflow as tf # Added explicit import
from sklearn.model_selection import train_test_split
from keras import to_categorical
from keras import Sequential
from keras import LSTM, Dense
from keras import EarlyStopping

# Ensure your GPU doesn't run out of memory (Optional but recommended)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

DATA_PATH = '../data/sequences'
actions = np.array(['Namaste', 'Kasto_Cha', 'Mero_Naam', 'Sahayog', 'Dhanyabaad', 
                   'Ma_Nepali', 'Khana_Khayau', 'Sanchai_Chu', 'Birami_Chu', 'Ghar_Kata'])

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(30):
        # Path logic updated to find the .npy files
        res = np.load(os.path.join(DATA_PATH, action, f"{sequence}.npy"))
        sequences.append(res)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# LSTM Architecture
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 10)))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Using categorical_crossentropy because we have 10 classes
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping helps prevent overfitting
early_stop = EarlyStopping(monitor='loss', patience=20)

print("Training started...")
model.fit(X_train, y_train, epochs=200, batch_size=32, callbacks=[early_stop])

os.makedirs('../models', exist_ok=True)
model.save('../models/slr_lstm_model.h5')
print("Success: Model saved to ../models/slr_lstm_model.h5")