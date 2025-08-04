from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from functools import wraps
from pymongo import MongoClient
import jwt
import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'secretkey'

# MongoDB setup
mongo = MongoClient("mongodb://localhost:27017")
db = mongo["orph"]
chat_history = db["chat_history"]

users = {"admin": "admin"}  # dummy in-memory store

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 403
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            request.user = data['username']
        except:
            return jsonify({'message': 'Token is invalid!'}), 403
        return f(*args, **kwargs)
    return decorated

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    users[data['username']] = data['password']
    return jsonify({'message': 'User registered'})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    if users.get(data['username']) == data['password']:
        token = jwt.encode({'username': data['username'], 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=12)}, app.config['SECRET_KEY'], algorithm='HS256')
        return jsonify({'token': token})
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/chat', methods=['POST'])
@token_required
def chat():
    text = request.json['message']
    return jsonify({'response': [{'disease': 'Flu', 'score': 0.85}, {'disease': 'Cold', 'score': 0.5}]})

@app.route('/upload', methods=['POST'])
@token_required
def upload():
    file = request.files['file']
    text = request.form.get('text')
    meta = request.form.get('meta')
    return jsonify({'predictions': [{'disease': 'Pneumonia', 'score': 0.9}]})

@app.route('/generate_report', methods=['POST'])
@token_required
def generate_report():
    data = request.json
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.drawString(100, 750, "Diagnosis Report")
    p.drawString(100, 730, f"Patient Age: {data['meta'].get('age', '')}")
    p.drawString(100, 710, f"Duration: {data['meta'].get('duration_days', '')} days")
    p.drawString(100, 690, f"Danger Symptoms: {', '.join(data['meta'].get('danger_symptoms', []))}")
    p.drawString(100, 670, f"Input: {data['text']}")
    y = 650
    for item in data['predictions']:
        p.drawString(100, y, f"Prediction: {item['disease']} (Score: {item['score']})")
        y -= 20
    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="report.pdf", mimetype='application/pdf')

# POST new message to history
@app.route("/history", methods=["POST"])
@token_required
def save_history():
    data = request.json
    entry = {
        "username": request.user,
        "input": data.get("input", ""),
        "output": data.get("output", {})
    }
    chat_history.insert_one(entry)
    return jsonify({"status": "saved"})

# GET all history for logged-in user
@app.route("/history", methods=["GET"])
@token_required
def get_history():
    entries = list(chat_history.find({"username": request.user}, {"_id": 0}))
    return jsonify(entries)

if __name__ == '__main__':
    app.run(debug=True)
