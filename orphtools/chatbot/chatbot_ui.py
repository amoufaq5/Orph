# orphtools/chatbot/chatbot_ui.py
from flask import Flask, request, jsonify
from functools import wraps
import jwt
import datetime
import os
import json
import hashlib

from orphtools.chatbot.chatbot_engine import ChatbotEngine
from orphtools.models.diagnosis_engine import DiagnosisEngine
from orphtools.logic.clarification_loop import ClarificationLoop
from orphtools.logic.referral_generator import ReferralGenerator

SECRET_KEY = "your-secret-key"  # Change this in production!
USER_DB_PATH = "logs/users.json"
SESSION_LOG_PATH = "logs/sessions_log.json"
os.makedirs("logs", exist_ok=True)

# Load or create user database
def load_users():
    if os.path.exists(USER_DB_PATH):
        with open(USER_DB_PATH, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_DB_PATH, "w") as f:
        json.dump(users, f, indent=2)

users = load_users()

# JWT Authentication Decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"error": "Token is missing"}), 401
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            request.user = data["user"]
        except:
            return jsonify({"error": "Token is invalid or expired"}), 403
        return f(*args, **kwargs)
    return decorated

# Register route
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    if username in users:
        return jsonify({"error": "Username already exists"}), 409

    password_hash = hashlib.sha256(password.encode()).hexdigest()
    users[username] = password_hash
    save_users(users)
    return jsonify({"message": "User registered successfully"}), 201

# Login route
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    password_hash = hashlib.sha256(password.encode()).hexdigest()

    if username in users and users[username] == password_hash:
        token = jwt.encode({
            "user": username,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=6)
        }, SECRET_KEY, algorithm="HS256")
        return jsonify({"token": token})
    return jsonify({"error": "Invalid credentials"}), 401

# Log session
def log_session(username, message, response):
    session = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "user": username,
        "input": message,
        "response": response
    }
    if os.path.exists(SESSION_LOG_PATH):
        with open(SESSION_LOG_PATH, "r") as f:
            sessions = json.load(f)
    else:
        sessions = []
    sessions.append(session)
    with open(SESSION_LOG_PATH, "w") as f:
        json.dump(sessions, f, indent=2)

# Initialize core components
engine = DiagnosisEngine("dmis-lab/biobert-base-cased-v1.1", threshold=0.75)
clarifier = ClarificationLoop()
referrer = ReferralGenerator()
chatbot = ChatbotEngine(engine, clarifier, referrer)

@app.route("/upload", methods=["POST"])
@token_required
def upload():
    if 'file' not in request.files or 'text' not in request.form:
        return jsonify({"error": "Missing file or text input"}), 400

    file = request.files['file']
    symptom_text = request.form['text']
    meta_json = request.form.get('meta', '{}')  # optional structured fields
    try:
        meta = json.loads(meta_json)
    except:
        meta = {}

    # Save file
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    # Import and run vectorizers
    from orphtools.models.visual_diagnosis import VisualDiagnosis
    from orphtools.models.diagnosis_engine import DiagnosisEngine
    from orphtools.logic.asmethod_parser import ASMethodParser
    from orphtools.models.fusion_engine import FusionEngine

    visual = VisualDiagnosis()
    image_vec = visual.extract_features(filepath).unsqueeze(0)  # shape: [1, 2048]

    text_model = DiagnosisEngine("dmis-lab/biobert-base-cased-v1.1", threshold=0.0)
    tokenizer = text_model.tokenizer
    model = text_model.model

    inputs = tokenizer(symptom_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        text_vec = outputs.hidden_states[-1][:, 0, :] if hasattr(outputs, 'hidden_states') else outputs.logits
    text_vec = text_vec.squeeze(0)

    parser = ASMethodParser()
    struct_vec = torch.tensor(parser.parse(meta)).unsqueeze(0)

    # Match dimension sizes
    image_vec = image_vec[:, :512]
    text_vec = text_vec.unsqueeze(0)

    # Run fusion
    fusion = FusionEngine(text_dim=768, image_dim=512, structured_dim=8, output_dim=5)
    fusion.eval()
    with torch.no_grad():
        pred = fusion(text_vec, image_vec, struct_vec)
        probs = torch.softmax(pred, dim=-1).cpu().numpy().flatten()

    result = [{"disease": f"Disease_{i}", "score": float(score)} for i, score in enumerate(probs) if score > 0.1]

    return jsonify({
        "message": "Image and text processed",
        "predictions": result
    })


# Run the server
if __name__ == "__main__":
    app.run(debug=True)
