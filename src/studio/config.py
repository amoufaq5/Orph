import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]    # project root
DATASETS_DIR = ROOT / "data" / "user_datasets"
MODELS_DIR   = ROOT / "outputs" / "user_models"
LOGS_DIR     = ROOT / "outputs" / "studio_logs"

# later: env vars for cloud creds, queue URLs, etc.
CLOUD_BACKEND = os.getenv("ORPH_CLOUD_BACKEND", "local")  # "local" | "runpod" | "gcp" | "aws"
