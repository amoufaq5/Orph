# chatbot_api.py with Grad-CAM image return

from flask import Flask, request, jsonify, send_file
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import uuid
from diagnostic_image_model import DiagnosticModel, load_image, predict, grad_cam

MODEL_PATH = "models/biomed_finetuned"
IMAGE_MODEL_PATH = "models/xray_model.pt"
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)

# Load BioMed LLM
print("🧠 Loading BioMed model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
llm_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm_model.to(device)

# Load image model
print("🩻 Loading image diagnostic model...")
image_model = DiagnosticModel()
image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
image_model.to(device)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form.get("input", "")
    uploaded_file = request.files.get("file")

    if uploaded_file:
        filename = f"{uuid.uuid4().hex}_{uploaded_file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        outputpath = os.path.join(OUTPUT_DIR, f"gradcam_{filename}.jpg")
        uploaded_file.save(filepath)

        try:
            image_tensor = load_image(filepath)
            diagnosis, prob = predict(image_model, image_tensor)
            grad_cam(image_model, image_tensor, output_path=outputpath)
            return jsonify({
                "response": f"Image diagnosis: {diagnosis} (confidence: {max(prob):.2f})",
                "heatmap": f"/gradcam/{os.path.basename(outputpath)}"
            })
        except Exception as e:
            return jsonify({"response": f"Error processing image: {str(e)}"})

    if user_input:
        prompt = f"Symptoms: {user_input}\nDiagnosis and Advice:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = llm_model.generate(
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

    return jsonify({"response": "No valid input or file received."})

@app.route("/gradcam/<filename>")
def serve_gradcam(filename):
    return send_file(os.path.join(OUTPUT_DIR, filename), mimetype='image/jpeg')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
