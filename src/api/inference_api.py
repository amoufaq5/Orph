import os, time
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# ---------- config ----------
MODEL_DIR = os.getenv("ORPH_INFER_MODEL_DIR", "outputs/user_models/demo/latest")
TOKENIZER_DIR = os.getenv("ORPH_INFER_TOKENIZER_DIR", "outputs/tokenizer/orph_bpe_32k")
MAX_NEW_TOKENS = int(os.getenv("ORPH_INFER_MAX_NEW_TOKENS", "256"))
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# ---------- load ----------
app = Flask(__name__)
CORS(app)

_tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
if _tok.pad_token is None:
    if "<pad>" in _tok.get_vocab():
        _tok.pad_token = "<pad>"
    else:
        _tok.add_special_tokens({"pad_token": "<pad>"})

def _load_model(model_dir: str):
    p = Path(model_dir)
    if not p.exists():
        raise RuntimeError(f"Model path not found: {p.resolve()}")
    m = AutoModelForCausalLM.from_pretrained(model_dir)
    m.resize_token_embeddings(len(_tok))
    m.to(DEVICE)
    m.eval()
    return m

_model = _load_model(MODEL_DIR)

# ---------- helpers ----------
SYSTEM_PREFIX = "<s>\n"
SYSTEM_SUFFIX = "\n</s>"

def build_prompt(instruction: str, input_text: str = "", mode: str = "sft"):
    if mode == "cot":
        # If you fine-tuned with CoT format, you can bias towards reasoning
        return (
            f"{SYSTEM_PREFIX}Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nReasoning:\n"
        )
    # default SFT-style
    return (
        f"{SYSTEM_PREFIX}Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nResponse:\n"
    )

def gen_once(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.7, top_p: float = 0.9):
    enc = _tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = _model.generate(
            **enc,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            eos_token_id=_tok.eos_token_id
        )
    text = _tok.decode(out[0], skip_special_tokens=False)
    # Return only the completion after the prompt
    return text[len(prompt):].split("</s>", 1)[0].strip()

# ---------- routes ----------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "device": DEVICE, "model_dir": MODEL_DIR})

@app.route("/reload", methods=["POST"])
def reload_model():
    """Reload model/tokenizer without restarting the container."""
    global _model, _tok
    body = request.get_json(force=True, silent=True) or {}
    model_dir = body.get("model_dir", MODEL_DIR)
    tokenizer_dir = body.get("tokenizer_dir", TOKENIZER_DIR)

    # reload tokenizer if changed
    if tokenizer_dir != TOKENIZER_DIR:
        _tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
        if _tok.pad_token is None:
            if "<pad>" in _tok.get_vocab():
                _tok.pad_token = "<pad>"
            else:
                _tok.add_special_tokens({"pad_token": "<pad>"})

    # reload model
    _model = _load_model(model_dir)
    return jsonify({"ok": True, "model_dir": model_dir, "tokenizer_dir": tokenizer_dir})

@app.route("/generate", methods=["POST"])
def generate():
    """
    JSON body:
    {
      "instruction": "Advise a patient with mild tension headache.",
      "input": "Age 28, no red flags.",
      "mode": "sft" | "cot",
      "max_new_tokens": 256,
      "temperature": 0.7,
      "top_p": 0.9
    }
    """
    data = request.get_json(force=True)
    instruction = (data.get("instruction") or "").strip()
    _input = (data.get("input") or "").strip()
    mode = (data.get("mode") or "sft").lower()

    if not instruction:
        return jsonify({"error": "instruction required"}), 400

    prompt = build_prompt(instruction, _input, mode)
    out = gen_once(
        prompt,
        max_new_tokens=int(data.get("max_new_tokens", MAX_NEW_TOKENS)),
        temperature=float(data.get("temperature", 0.7)),
        top_p=float(data.get("top_p", 0.9))
    )
    return jsonify({"completion": out})

@app.route("/stream", methods=["POST"])
def stream():
    """
    Basic token streaming using TextIteratorStreamer.
    """
    data = request.get_json(force=True)
    instruction = (data.get("instruction") or "").strip()
    _input = (data.get("input") or "").strip()
    mode = (data.get("mode") or "sft").lower()
    if not instruction:
        return jsonify({"error": "instruction required"}), 400

    prompt = build_prompt(instruction, _input, mode)
    enc = _tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    streamer = TextIteratorStreamer(_tok, skip_prompt=True, skip_special_tokens=False)
    import threading
    def _worker():
        with torch.no_grad():
            _model.generate(
                **enc,
                do_sample=True,
                top_p=float(data.get("top_p", 0.9)),
                temperature=float(data.get("temperature", 0.7)),
                max_new_tokens=int(data.get("max_new_tokens", MAX_NEW_TOKENS)),
                eos_token_id=_tok.eos_token_id,
                streamer=streamer
            )
    t = threading.Thread(target=_worker, daemon=True); t.start()

    def event_stream():
        buf = ""
        for token_text in streamer:
            buf += token_text
            # emit small chunks, strip after </s>
            if "</s>" in buf:
                chunk, _ = buf.split("</s>", 1)
                yield chunk
                break
            if len(buf) > 40:
                yield buf
                buf = ""
        if buf:
            yield buf

    # Flask streaming response (text/plain SSE-like without headers)
    from flask import Response
    return Response(event_stream(), mimetype="text/plain")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6060, debug=True)