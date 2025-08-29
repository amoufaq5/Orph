from pathlib import Path
from werkzeug.utils import secure_filename
from .config import DATASETS_DIR, MODELS_DIR, LOGS_DIR

for d in [DATASETS_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def save_dataset(file_storage, user_id: str) -> str:
    fname = secure_filename(file_storage.filename)
    user_dir = DATASETS_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / fname
    file_storage.save(path)
    return str(path)

def model_path(user_id: str, job_id: str) -> str:
    p = MODELS_DIR / user_id / job_id
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

def log_path(user_id: str, job_id: str) -> str:
    p = LOGS_DIR / user_id / f"{job_id}.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)
