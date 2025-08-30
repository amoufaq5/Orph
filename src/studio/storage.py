from pathlib import Path
from datetime import datetime
import json
from werkzeug.utils import secure_filename

from .config import DATASETS_DIR, MODELS_DIR, LOGS_DIR

# -----------------------------
# Datasets / Paths
# -----------------------------
def save_dataset(file_storage, user_id: str) -> str:
    """Save uploaded dataset under data/user_datasets/<user_id>/<fname>"""
    fname = secure_filename(file_storage.filename or "dataset.jsonl")
    user_dir = DATASETS_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / fname
    file_storage.save(path)
    return str(path)

def model_dir(user_id: str, job_id: str) -> Path:
    p = MODELS_DIR / user_id / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def logs_path(user_id: str, job_id: str) -> Path:
    p = LOGS_DIR / user_id
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{job_id}.log"

def report_paths(model_dir_path: str):
    p = Path(model_dir_path)
    j = p / "report.json"
    pdf = p / "report.pdf"
    return (j if j.exists() else None), (pdf if pdf.exists() else None)

# -----------------------------
# Registry (models/jobs)
# -----------------------------
REGISTRY = MODELS_DIR / "registry.json"

def _load_registry():
    if REGISTRY.exists():
        try:
            return json.loads(REGISTRY.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"models": []}

def _save_registry(reg):
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")

def upsert_registry_entry(job_id: str, user_id: str, status: str, model_dir_path: str, cfg: dict):
    reg = _load_registry()
    items = reg.get("models", [])
    now = datetime.utcnow().isoformat() + "Z"
    for it in items:
        if it.get("job_id") == job_id:
            it.update({"status": status, "model_dir": model_dir_path, "updated_at": now})
            _save_registry(reg); return
    items.append({
        "job_id": job_id,
        "user_id": user_id,
        "status": status,
        "model_dir": model_dir_path,
        "config": cfg or {},
        "tags": [],
        "created_at": now,
        "updated_at": now,
    })
    reg["models"] = items
    _save_registry(reg)

def set_status(job_id: str, status: str):
    reg = _load_registry()
    now = datetime.utcnow().isoformat() + "Z"
    for it in reg.get("models", []):
        if it.get("job_id") == job_id:
            it["status"] = status
            it["updated_at"] = now
            _save_registry(reg); return

def set_model_dir(job_id: str, path: str):
    reg = _load_registry()
    now = datetime.utcnow().isoformat() + "Z"
    for it in reg.get("models", []):
        if it.get("job_id") == job_id:
            it["model_dir"] = path
            it["updated_at"] = now
            _save_registry(reg); return

def add_tag(job_id: str, tag: str):
    reg = _load_registry()
    now = datetime.utcnow().isoformat() + "Z"
    for it in reg.get("models", []):
        if it.get("job_id") == job_id:
            tags = set(it.get("tags", []))
            tags.add(tag)
            it["tags"] = sorted(tags)
            it["updated_at"] = now
            _save_registry(reg); return

def list_models():
    return _load_registry().get("models", [])

def get_by_job(job_id: str):
    for it in list_models():
        if it.get("job_id") == job_id:
            return it
    return None

# ---- dataset build artifacts ----
def build_job_dir(user_id: str, job_id: str) -> Path:
    p = MODELS_DIR / user_id / f"build_{job_id}"
    p.mkdir(parents=True, exist_ok=True)
    return p

