# orphtools/chatbot/chatbot_ui.py
from flask import Flask, request, jsonify
from orphtools.chatbot.chatbot_engine import ChatbotEngine

# Dummy imports — these should be real initialized components
from orphtools.models.diagnosis_engine import DiagnosisEngine
from orphtools.logic.clarification_loop import ClarificationLoop
from orphtools.logic.referral_generator import ReferralGenerator

# Initialize core components
engine = DiagnosisEngine("dmis-lab/biobert-base-cased-v1.1", threshold=0.75)
clarifier = ClarificationLoop()
referrer = ReferralGenerator()
chatbot = ChatbotEngine(engine, clarifier, referrer)

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message")
    if not message:
        return jsonify({"error": "Missing 'message' field"}), 400

    response = chatbot.process_input(message)
    return jsonify({"response": response})


# Run the server
if __name__ == "__main__":
    app.run(debug=True)
