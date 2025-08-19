from flask import Flask, request, jsonify
from src.utils.config_loader import load_config
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = Flask(__name__)

# Load config
config = load_config("conf/config.yaml")
API_KEY = os.getenv("ORPH_API_KEY", config.api.key)

# Load model
model_path = "out/text_gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

@app.route("/api/chat", methods=["POST"])
def chat():
    # API key check
    key = request.headers.get("X-API-Key")
    if key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    message = data.get("message", "")

    inputs = tokenizer(message, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=100)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
