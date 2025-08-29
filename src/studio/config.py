from pathlib import Path
import os

# Project root = two levels up from this file (…/src/studio/config.py -> project root)
ROOT = Path(__file__).resolve().parents[2]

# Storage
DATASETS_DIR = ROOT / "data" / "user_datasets"
MODELS_DIR   = ROOT / "outputs" / "user_models"
LOGS_DIR     = ROOT / "outputs" / "studio_logs"
TOKENIZER_DIR= ROOT / "outputs" / "tokenizer" / "orph_bpe_32k"

# Cloud backend placeholder (switch later to runpod/gcp/aws)
CLOUD_BACKEND = os.getenv("ORPH_CLOUD_BACKEND", "local")

# Ensure folders exist
for d in [DATASETS_DIR, MODELS_DIR, LOGS_DIR, TOKENIZER_DIR]:
    d.mkdir(parents=True, exist_ok=True)