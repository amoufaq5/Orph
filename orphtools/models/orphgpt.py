# orphtools/models/orphgpt.py

import tensorflow as tf
import os

class OrphGPT:
    def __init__(self, vocab_size, num_classes):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, 128),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train, epochs=10, batch_size=32,
                       validation_data=(X_val, y_val))

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "orphgpt_model.h5"))
