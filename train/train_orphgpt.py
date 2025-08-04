# train_orphgpt.py

import numpy as np
import pickle
import json
from orphtools.models.orph_gpt import OrphGPT

# Load preprocessed data
X_train = np.load("orphtools/artifacts/X_train.npy")
X_val = np.load("orphtools/artifacts/X_val.npy")
X_test = np.load("orphtools/artifacts/X_test.npy")

y_train = np.load("orphtools/artifacts/y_train.npy")
y_val = np.load("orphtools/artifacts/y_val.npy")
y_test = np.load("orphtools/artifacts/y_test.npy")

with open("orphtools/artifacts/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("orphtools/artifacts/label_index.json", "r") as f:
    label_index = json.load(f)

# Define model
vocab_size = min(len(tokenizer.word_index) + 1, 5000)
num_classes = len(label_index)

model = OrphGPT(vocab_size=vocab_size, num_classes=num_classes)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc * 100:.2f}%")

# Save model
model.save("orphtools/artifacts/orphgpt_model.h5")
