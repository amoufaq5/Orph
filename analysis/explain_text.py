# orphtools/analysis/explain_text.py

import numpy as np
import tensorflow as tf
import pickle
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load artifacts
model = load_model("orphtools/artifacts/orphgpt_model.h5")
with open("orphtools/artifacts/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("orphtools/artifacts/label_index.json", "r") as f:
    label_index = json.load(f)

index_label = {v: k for k, v in label_index.items()}
MAX_LEN = 300

def explain_prediction(text):
    tokens = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(tokens, maxlen=MAX_LEN)

    # Get gradients of the predicted class w.r.t. the input
    input_tensor = tf.convert_to_tensor(padded)
    input_tensor = tf.Variable(input_tensor)

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        preds = model(input_tensor)
        pred_class = tf.argmax(preds[0])

        loss = preds[:, pred_class]

    grads = tape.gradient(loss, input_tensor).numpy()[0]
    inputs = padded[0]
    token_importance = {}

    for idx, token_id in enumerate(inputs):
        if token_id != 0:
            word = tokenizer.index_word.get(token_id, f"<{token_id}>")
            token_importance[word] = float(grads[idx])

    sorted_tokens = sorted(token_importance.items(), key=lambda x: abs(x[1]), reverse=True)

    return {
        "predicted_class": index_label[int(pred_class)],
        "token_scores": sorted_tokens[:20]  # top 20 tokens
    }
