from __future__ import annotations
import os
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.utils.config import Config
from src.utils.logging import init_logger
from src.chat.infer_text import TextInference

CONFIDENCE_THRESHOLD = float(os.getenv("ORPH_CONF_THRESH", "0.75"))
API_KEY_ENV = "ORPH_API_KEY"

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": os.getenv("ORPH_CORS", "*")}})
logger = init_logger("server")

_TEXT: TextInference | None = None
_CFG: Config | None = None


def cfg() -> Config:
    global _CFG
    if _CFG is None:
        _CFG = Config.load(os.getenv("ORPH_CONFIG", "conf/config.yaml"))
    return _CFG


def text_model() -> TextInference:
    global _TEXT
    if _TEXT is None:
        ckpt = cfg().get("train_text.output_dir", "outputs/checkpoints/text")
        _TEXT = TextInference(ckpt_dir=ckpt)
    return _TEXT


def require_api_key(req) -> tuple[bool, dict | None]:
    sent = req.headers.get("X-API-Key")
    want = os.getenv(API_KEY_ENV, "devkey")
    if not sent or sent != want:
        return False, {"error": "Unauthorized"}
    return True, None


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def chat():
    ok, err = require_api_key(request)
    if not ok:
        return jsonify(err), 401
    payload = request.get_json(force=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400

    res = text_model().self_consistent_answer(message, n=int(os.getenv("ORPH_SELFCONS", "5")))
    answer, conf = res["answer"], float(res["confidence"])
    follow = None if conf >= CONFIDENCE_THRESHOLD else "Please provide more details (ASMETHOD)."

    return jsonify({
        "answer": answer,
        "confidence": conf,
        "follow_up": follow,
        "samples": res["samples"],
    })


@app.post("/api/predict-image")
def predict_image():
    ok, err = require_api_key(request)
    if not ok:
        return jsonify(err), 401
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files['file']
    try:
        _ = Image.open(BytesIO(f.read())).convert('RGB')
    except Exception as e:
        logger.exception("Failed to read image")
        return jsonify({"error": f"Invalid image: {e}"}), 400
    # Placeholder: wire VisionInference when you finalize vision ckpt
    return jsonify({"ok": True, "note": "Vision model not wired yet in this route."})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port
