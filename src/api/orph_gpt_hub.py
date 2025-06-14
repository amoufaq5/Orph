from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Map GPT agent routes to internal services (or eventually microservices)
GPT_AGENTS = {
    "cardio": "http://localhost:8080/predict_cardio",
    "research": "http://localhost:8081/predict_research",  # to be built later
    "image": "http://localhost:8082/predict_image"         # to be built later
}

@app.route("/orph", methods=["POST"])
def route_orph():
    try:
        # Expected payload: { "agent": "cardio", "input": {...} }
        data = request.json
        agent = data.get("agent")
        user_input = data.get("input")

        if agent not in GPT_AGENTS:
            return jsonify({"error": f"Unknown agent '{agent}'"}), 400

        # Forward to correct agent API
        target_url = GPT_AGENTS[agent]
        response = requests.post(target_url, json=user_input)

        return jsonify({
            "agent": agent,
            "response": response.json()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=9000, debug=True)
