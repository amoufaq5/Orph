# chatbot_api.py

"""
Flask API for Orph medical chatbot using fine-tuned BioMedLM
"""

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "models/biomed_finetuned"

app = Flask(__name__)

# Load model and tokenizer
print("🧠 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("input", "")

    prompt = f"Symptoms: {user_input}\nDiagnosis and Advice:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded.split("Diagnosis and Advice:")[-1].strip()

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
