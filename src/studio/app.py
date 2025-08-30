from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
from pathlib import Path
import io, zipfile, json

# storage / jobs
from .storage import save_dataset, list_models, get_by_job, report_paths
from .jobs import (
    submit_job, submit_scrape_job, submit_build_job,
    job_status, job_logs, job_model_dir
)

app = Flask(__name__)
CORS(app)  # allow local React dev / cross-origin

# -----------------------------
# Health & templates
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/templates", methods=["GET"])
def templates():
    presets = [
        {
            "id": "doctor",
            "label": "DoctorGPT",
            "base_model": "gpt2",
            "curriculum": "sft:0.85,cot:0.15",
            "max_length": 1024,
            "epochs": 1,
            "notes": "General clinician; heavier SFT, lighter CoT"
        },
        {
            "id": "symptom",
            "label": "SymptomGPT",
            "base_model": "gpt2",
            "curriculum": "sft:0.6,cot:0.4",
            "max_length": 768,
            "epochs": 1,
            "notes": "Triage-heavy; more CoT for reasoning"
        },
        {
            "id": "pharma",
            "label": "PharmaGPT",
            "base_model": "gpt2",
            "curriculum": "sft:0.9,cot:0.1",
            "max_length": 768,
            "epochs": 1,
            "notes": "Drug info & counseling; SFT-biased"
        },
        {
            "id": "research",
            "label": "ResearchGPT",
            "base_model": "gpt2",
            "curriculum": "sft:0.5,cot:0.5",
            "max_length": 1024,
            "epochs": 1,
            "notes": "PubMed-focused; balanced SFT/CoT"
        }
    ]
    return jsonify({"templates": presets})

# -----------------------------
# Datasets
# -----------------------------
@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    user_id = request.form.get("user_id", "default")
    f = request.files.get("file")
    if not f:
        abort(400, "No file uploaded")
    path = save_dataset(f, user_id)
    return jsonify({"ok": True, "path": path})

# -----------------------------
# Jobs: train / scrape / build
# -----------------------------
@app.route("/start_finetune", methods=["POST"])
def start_finetune():
    """
    JSON:
    {
      "user_id": "demo",
      "sft_path": "data/clean/text_supervised.jsonl",
      "cot_path": "data/clean/text_cot.jsonl",
      "curriculum": "sft:0.7,cot:0.3",
      "base_model": "gpt2",
      "max_length": 512,
      "epochs": 1,
      "simulate": true
    }
    """
    cfg = request.get_json(force=True) or {}
    user_id = cfg.get("user_id", "default")
    job_id = submit_job(user_id, cfg)
    return jsonify({"ok": True, "job_id": job_id})

@app.route("/start_scrape", methods=["POST"])
def start_scrape():
    """
    JSON:
    {
      "user_id": "demo",
      "sources": ["all"] | ["pubmed_fetch.py", ...],
      "env": {"NCBI_EMAIL":"you@example.com","NCBI_API_KEY":"...","WIKI_LANG":"en"}
    }
    """
    cfg = request.get_json(force=True) or {}
    user_id = cfg.get("user_id", "default")
    job_id = submit_scrape_job(user_id, cfg)
    return jsonify({"ok": True, "job_id": job_id})

@app.route("/build_dataset", methods=["POST"])
def build_dataset_route():
    """
    JSON:
    {
      "user_id": "demo",
      "seed": 42,
      "train_ratio": 0.94, "val_ratio": 0.03, "test_ratio": 0.03
    }
    """
    cfg = request.get_json(force=True) or {}
    user_id = cfg.get("user_id", "default")
    # (Optional guard) ratios must sum to ~1.0
    tr = float(cfg.get("train_ratio", 0.94))
    vr = float(cfg.get("val_ratio", 0.03))
    te = float(cfg.get("test_ratio", 0.03))
    if abs((tr + vr + te) - 1.0) > 1e-6:
        abort(400, "train/val/test ratios must sum to 1.0")
    job_id = submit_build_job(user_id, cfg)
    return jsonify({"ok": True, "job_id": job_id})

# -----------------------------
# Status & logs (shared)
# -----------------------------
@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    return jsonify(job_status(job_id))

@app.route("/logs/<job_id>", methods=["GET"])
def logs(job_id):
    return jsonify({"job_id": job_id, "logs": job_logs(job_id)})

# -----------------------------
# Registry, reports, downloads
# -----------------------------
@app.route("/list_models", methods=["GET"])
def list_models_route():
    return jsonify({"models": list_models()})

@app.route("/download_model/<job_id>", methods=["GET"])
def download_model(job_id):
    # prefer live job dir; fall back to registry
    model_dir = job_model_dir(job_id)
    if not model_dir:
        item = get_by_job(job_id)
        model_dir = item.get("model_dir", "") if item else ""
    if not model_dir or not Path(model_dir).exists():
        abort(404, "Model not found")

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in Path(model_dir).rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(model_dir).as_posix())
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name=f"{job_id}.zip", mimetype="application/zip")

@app.route("/report/<job_id>", methods=["GET"])
def report_json(job_id):
    item = get_by_job(job_id) or {}
    model_dir = item.get("model_dir", "")
    if not model_dir or not Path(model_dir).exists():
        abort(404, "Model directory not found")
    j, _ = report_paths(model_dir)
    if not j:
        abort(404, "report.json not found")
    return send_file(str(j), mimetype="application/json", as_attachment=False)

@app.route("/report_pdf/<job_id>", methods=["GET"])
def report_pdf(job_id):
    item = get_by_job(job_id) or {}
    model_dir = item.get("model_dir", "")
    if not model_dir or not Path(model_dir).exists():
        abort(404, "Model directory not found")
    _, pdf = report_paths(model_dir)
    if not pdf:
        abort(404, "report.pdf not found")
    return send_file(str(pdf), mimetype="application/pdf", as_attachment=True, download_name=f"{job_id}_report.pdf")

@app.route("/scrape_report/<job_id>", methods=["GET"])
def get_scrape_report(job_id):
    item = get_by_job(job_id) or {}
    model_dir = item.get("model_dir", "")
    if not model_dir or not Path(model_dir).exists():
        abort(404, "Scrape job directory not found")
    rp = Path(model_dir) / "scrape_report.json"
    if not rp.exists():
        abort(404, "scrape_report.json not found (job still running?)")
    return send_file(str(rp), mimetype="application/json", as_attachment=False)

@app.route("/build_report/<job_id>", methods=["GET"])
def get_build_report(job_id):
    item = get_by_job(job_id) or {}
    model_dir = item.get("model_dir", "")
    if not model_dir or not Path(model_dir).exists():
        abort(404, "Build job directory not found")
    rp = Path(model_dir) / "build_report.json"
    if not rp.exists():
        abort(404, "build_report.json not found (job still running?)")
    return send_file(str(rp), mimetype="application/json", as_attachment=False)

# -----------------------------
# Promotions / Tags
# -----------------------------
@app.route("/promote_model", methods=["POST"])
def promote_model():
    """
    JSON: { "job_id": "...", "tag": "prod" }
    """
    from .storage import add_tag
    data = request.get_json(force=True) or {}
    job_id = data.get("job_id")
    tag = (data.get("tag") or "prod").strip().lower()
    if not job_id:
        abort(400, "job_id required")
    add_tag(job_id, tag)
    return jsonify({"ok": True, "job_id": job_id, "tag": tag})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055, debug=True)
