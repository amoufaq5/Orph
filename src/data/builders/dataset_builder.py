# orphtools/preprocessing/dataset_builder.py

import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from pathlib import Path
import pickle

def build_dataset(merged_path, max_vocab=5000, max_len=100, test_size=0.1, val_size=0.1):
    with open(merged_path, "r") as f:
        data = json.load(f)

    texts = [entry["text"] for entry in data]
    labels = [entry["label"] for entry in data]

    # Tokenize texts
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    # Encode labels
    label_index = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    index_label = {idx: label for label, idx in label_index.items()}
    encoded_labels = [label_index[lbl] for lbl in labels]
    onehot_labels = to_categorical(encoded_labels)

    # Split dataset
    X_temp, X_test, y_temp, y_test = train_test_split(padded, onehot_labels, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)

    # Save everything
    Path("orphtools/artifacts").mkdir(parents=True, exist_ok=True)
    np.save("orphtools/artifacts/X_train.npy", X_train)
    np.save("orphtools/artifacts/X_val.npy", X_val)
    np.save("orphtools/artifacts/X_test.npy", X_test)
    np.save("orphtools/artifacts/y_train.npy", y_train)
    np.save("orphtools/artifacts/y_val.npy", y_val)
    np.save("orphtools/artifacts/y_test.npy", y_test)

    with open("orphtools/artifacts/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    with open("orphtools/artifacts/label_index.json", "w") as f:
        json.dump(label_index, f, indent=2)

    print("✅ Dataset built and saved.")

    return {
        "tokenizer": tokenizer,
        "label_index": label_index,
        "vocab_size": min(len(tokenizer.word_index) + 1, max_vocab),
        "num_classes": len(label_index)
    }
