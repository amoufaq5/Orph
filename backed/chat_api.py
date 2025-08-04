# chat_api.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import jwt
import datetime
from functools import wraps
from orphtools.analysis.explain_text import explain_prediction

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'your-secret-key'

# Simulated user DB (replace with real DB)
users = {
    'admin': 'password123'
}

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing'}), 403
        try:
            jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        except:
            return jsonify({'error': 'Token is invalid or expired'}), 403
        return f(*args, **kwargs)
    return decorated

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username, password = data.get('username'), data.get('password')
    if username in users and users[username] == password:
        token = jwt.encode({'user': username, 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=12)},
                           app.config['SECRET_KEY'], algorithm="HS256")
        return jsonify({'token': token})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/chat', methods=['POST'])
@token_required
def chat():
    data = request.json
    text = data.get('message', '')

    try:
        result = explain_prediction(text)
        return jsonify({
            "prediction": result["predicted_class"],
            "explanation": result["token_scores"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
