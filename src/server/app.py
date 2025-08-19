from __future__ import annotations
import os
from typing import Any, Dict
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Local packages
# Middleware: X-API-Key gate (reads env ORPH_API_KEY, or api.key from config)
from src.server.middleware import require_api_key
# Unified I/O adapters (normalize inputs, build responses, consistent errors)
from src.adapters import (
    normalize_chat_request, make_chat_response, make_error,
    normalize_clarify_request, make_clarify_response,
    normalize_referral_request, make_referral_response,
    generate_referral, clarify, parse_asmethod,  # optional adapters if present
)
# Text inference wrapper (loads whichever checkpoint dir you point it to)
from src.chat.infer_text import TextInference

# --- App
app = Flask(__name__)
CORS(app)

# --- Config / env
MODEL_DIR = os.getenv("MODEL_DIR", os.getenv("ORPH_MODEL_DIR", "out/text_gpt2"))  # your saved checkpoint
PORT = int(os.getenv("PORT", "8000"))

# Lazy model (init on first call so the process starts fast)
_text_infer: TextInference | None = None
def _get_text_infer() -> TextInference:
    global _text_infer
    if _text_infer is None:
        _text_infer = TextInference(ckpt_dir=MODEL_DIR)
    return _text_infer


# --------------------
# Health
# --------------------
@app.get("/api/health")
def health() -> Any:
    return jsonify({"status": "ok", "model_dir": MODEL_DIR})


# --------------------
# Chat
# --------------------
@app.post("/api/chat")
@require_api_key
def api_chat() -> Any:
    try:
        req = normalize_chat_request(request.get_json(force=True) or {})
    except ValueError as e:
        return jsonify(make_error(str(e))), 400

    ti = _get_text_infer()
    # self-consistency sampling (n) and max tokens are handled inside TextInference
    result: Dict[str, Any] = ti.self_consistent_answer(req["message"], n=5)

    # (Optional) downstream logic: referral/clarification can also be invoked here
    # ref = generate_referral({"asmethod": req.get("asmethod", {}), "prediction": result}) if 'generate_referral' in globals() else None

    return jsonify(make_chat_response(
        answer=result.get("answer", ""),
        confidence=float(result.get("confidence", 0.0)),
        follow_up=result.get("follow_up"),
        samples=result.get("samples"),
        referral=None,   # or `ref` if you enable it above
    ))


# --------------------
# Clarify
# --------------------
@app.post("/api/clarify")
@require_api_key
def api_clarify() -> Any:
    try:
        req = normalize_clarify_request(request.get_json(force=True) or {})
    except ValueError as e:
        return jsonify(make_error(str(e))), 400

    if "clarify" not in globals():
        # Fallback if adapter not present yet
        needed = bool(req["confidence"] < 0.75)
        question = "Could you share duration, severity, and danger symptoms?" if needed else None
        return jsonify(make_clarify_response(needed, question))

    out = clarify({"confidence": req["confidence"], "asmethod": req["asmethod"]})
    return jsonify(make_clarify_response(out.get("needed", False), out.get("question")))


# --------------------
# Referral
# --------------------
@app.post("/api/referral")
@require_api_key
def api_referral() -> Any:
    try:
        req = normalize_referral_request(request.get_json(force=True) or {})
    except ValueError as e:
        return jsonify(make_error(str(e))), 400

    if "generate_referral" not in globals():
        # conservative default: refer when confidence low or any danger tags present
        pred = req["prediction"]
        low_conf = float(pred.get("confidence", 0.0)) < 0.5
        tags = pred.get("tags", [])
        refer = bool(low_conf or ("danger" in tags))
        reason = "Low confidence or danger indicators" if refer else "No referral criteria met"
        return jsonify(make_referral_response(refer, reason, tags))

    out = generate_referral({"asmethod": req["asmethod"], "prediction": req["prediction"]})
    return jsonify(make_referral_response(out.get("refer", False), out.get("reason"), out.get("tags", [])))


# --------------------
# ASMETHOD parse (free text -> structured)
# --------------------
@app.post("/api/asmethod/parse")
@require_api_key
def api_asmethod_parse() -> Any:
    payload = request.get_json(force=True) or {}
    text = str(payload.get("text", "")).strip()
    if not text:
        return jsonify(make_error("Missing 'text'")), 400

    if "parse_asmethod" not in globals():
        # Simple placeholder parser
        filled = {"age": "", "self": "", "medication": "", "extra_meds": "", "time": "", "history": "", "other_symptoms": "", "danger_symptoms": ""}
        missing = [k for k, v in filled.items() if not v]
        return jsonify({"filled": filled, "missing": missing})

    res = parse_asmethod(text)
    return jsonify({"filled": res.get("filled", {}), "missing": res.get("missing", [])})


# --------------------
# Entrypoint
# --------------------
if __name__ == "__main__":
    # Use `flask run` or run directly
    app.run(host="0.0.0.0", port=PORT)
