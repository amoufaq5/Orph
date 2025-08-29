from flask import Flask, request, jsonify, send_file, abort
from pathlib import Path
import json, shutil, io, zipfile

from .storage import save_dataset
from .jobs import submit_job, job_status, job_logs, job_model_dir

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    user_id = request.form.get("user_id", "default")
    f = request.files.get("file")
    if not f: abort(400, "No file uploaded")
    path = save_dataset(f, user_id)
    return jsonify({"ok": True, "path": path})

@app.route("/start_finetune", methods=["POST"])
def start_finetune():
    """
    JSON body:
    {
      "user_id": "abc",
      "sft_path": "data/clean/text_supervised.jsonl",
      "cot_path": "data/clean/text_cot.jsonl",
      "curriculum": "sft:0.7,cot:0.3",
      "base_model": "gpt2",
      "max_length": 512,
      "epochs": 1,
      "simulate": true
    }
    """
    cfg = request.get_json(force=True)
    user_id = cfg.get("user_id", "default")
    job_id = submit_job(user_id, cfg)
    return jsonify({"ok": True, "job_id": job_id})

@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    st = job_status(job_id)
    return jsonify(st)

@app.route("/logs/<job_id>", methods=["GET"])
def logs(job_id):
    log_text = job_logs(job_id)
    return jsonify({"job_id": job_id, "logs": log_text})

@app.route("/download_model/<job_id>", methods=["GET"])
def download_model(job_id):
    model_dir = job_model_dir(job_id)
    if not model_dir or not Path(model_dir).exists():
        abort(404, "Model not found")
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in Path(model_dir).rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(model_dir).as_posix())
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name=f"{job_id}_model.zip", mimetype="application/zip")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055, debug=True)
