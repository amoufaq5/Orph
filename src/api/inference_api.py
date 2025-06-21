# inference_api.py
from flask import Flask, request, jsonify
import tensorflow as tf
from transformers import AutoTokenizer
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load tokenizer and model
MODEL_PATH = "checkpoints/orph_multimodal"
model = tf.keras.models.load_model(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Helper functions
def process_text(text):
    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        return_tensors="tf",
        max_length=256
    )
    return tokens["input_ids"], tokens["token_type_ids"], tf.zeros_like(tokens["input_ids"])

def process_image(image_file):
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224)) / 255.0
    return tf.convert_to_tensor([img], dtype=tf.float32)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text")
    image = request.files.get("image")

    if not text or not image:
        return jsonify({"error": "Text and image are required."}), 400

    input_ids, token_type_ids, modality_ids = process_text(text)
    image_tensor = process_image(image)

    prediction = model([input_ids, token_type_ids, modality_ids, image_tensor])
    prob = float(prediction.numpy()[0][0])

    return jsonify({"probability": prob, "class": int(prob > 0.5)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
