#!/usr/bin/env bash
# Orph end-to-end pipeline runner: Kaggle -> API scrapes -> clean/build -> tokenizer -> train (OrphGPT)
# Works on 1x or multi-GPU. Reads configs from conf/.

set -euo pipefail

# ----------------- Paths (persisted on RunPod volume) -----------------
RAW_DIR="${RAW_DIR:-/workspace/data/raw}"
KAGGLE_DIR="${KAGGLE_DIR:-/workspace/data/raw/kaggle}"
CLEAN_DIR="${CLEAN_DIR:-/workspace/data/clean}"
CKPT_DIR="${CKPT_DIR:-/workspace/data/checkpoints}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/workspace/data/tokenizer}"
LOG_DIR="${LOG_DIR:-/workspace/data/logs}"

mkdir -p "$RAW_DIR" "$KAGGLE_DIR" "$CLEAN_DIR" "$CKPT_DIR" "$TOKENIZER_DIR" "$LOG_DIR"

# ----------------- Modes -----------------
# ALL | KAGGLE_ONLY | SCRAPE_ONLY | CLEAN_ONLY | BUILD_ONLY | TOKENIZER_ONLY | PRETRAIN_ONLY | TRAIN_ONLY
MODE="${MODE:-ALL}"

# ----------------- Config (H100 defaults) -----------------
# You can point to conf/train_h100_2gpu.yaml when using >1 GPU
TRAIN_CONFIG="${TRAIN_CONFIG:-conf/train_h100.yaml}"

# ----------------- Kaggle controls -----------------
KAGGLE_CATALOG="${KAGGLE_CATALOG:-conf/kaggle_catalog.yaml}"
KAGGLE_ENABLE_HEAVY="${KAGGLE_ENABLE_HEAVY:-false}"     # true to pull heavy categories
KAGGLE_ONLY_KEYS="${KAGGLE_ONLY_KEYS:-}"                # e.g. "pubmed_qa medmcqa"
KAGGLE_MAX_ITEMS="${KAGGLE_MAX_ITEMS:-0}"               # 0 = unlimited
KAGGLE_SLEEP="${KAGGLE_SLEEP:-0.0}"

# ----------------- Helpers -----------------
log() { echo "[orph] $*"; }
exists() { command -v "$1" >/dev/null 2>&1; }

gpu_count() {
  if exists nvidia-smi; then
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' '
  else
    echo 0
  fi
}

# prefer these training entrypoints in order:
TRAIN_ENTRY=""
if [[ -f src/training/train_from_scratch.py ]]; then
  TRAIN_ENTRY="src/training/train_from_scratch.py"
elif [[ -f src/training/train_text.py ]]; then
  TRAIN_ENTRY="src/training/train_text.py"
elif [[ -f scripts/make_pretrain_from_supervised.py ]]; then
  TRAIN_ENTRY="scripts/make_pretrain_from_supervised.py"
fi

# ----------------- Stages -----------------
kaggle_stage() {
  log "KAGGLE stage"
  if [[ ! -f src/data_prep/scrapers/kaggle_fetch.py ]]; then
    log "kaggle_fetch.py not found -> skipping"
    return 0
  fi

  local args=( --out "$KAGGLE_DIR" --catalog "$KAGGLE_CATALOG" )
  [[ "$KAGGLE_ENABLE_HEAVY" == "true" ]] && args+=( --enable-heavy )
  [[ -n "$KAGGLE_ONLY_KEYS" ]] && args+=( --only $KAGGLE_ONLY_KEYS )
  [[ "$KAGGLE_MAX_ITEMS" != "0" ]] && args+=( --max-items "$KAGGLE_MAX_ITEMS" )
  [[ "$KAGGLE_SLEEP/1" != "0.0/1" ]] && args+=( --sleep "$KAGGLE_SLEEP" )

  python src/data_prep/scrapers/kaggle_fetch.py "${args[@]}" 2>&1 | tee -a "$LOG_DIR/kaggle_fetch.log" || true
}

scrape_api_stage() {
  log "API SCRAPE stage"
  [[ -f src/data_prep/scrapers/pubmed_fetch.py ]] && \
    python src/data_prep/scrapers/pubmed_fetch.py --out "$RAW_DIR" 2>&1 | tee -a "$LOG_DIR/pubmed_fetch.log" || true

  [[ -f src/data_prep/scrapers/openfda_labels_fetch.py ]] && \
    python src/data_prep/scrapers/openfda_labels_fetch.py --out "$RAW_DIR" 2>&1 | tee -a "$LOG_DIR/openfda_labels_fetch.log" || true

  [[ -f src/data_prep/scrapers/medlineplus_fetch.py ]] && \
    python src/data_prep/scrapers/medlineplus_fetch.py --out "$RAW_DIR" 2>&1 | tee -a "$LOG_DIR/medlineplus_fetch.log" || true

  [[ -f src/data_prep/scrapers/clinicaltrials_fetch.py ]] && \
    python src/data_prep/scrapers/clinicaltrials_fetch.py --out "$RAW_DIR" 2>&1 | tee -a "$LOG_DIR/clinicaltrials_fetch.log" || true

  [[ -f src/data_prep/scrape_all.py ]] && \
    python src/data_prep/scrape_all.py --out "$RAW_DIR" 2>&1 | tee -a "$LOG_DIR/scrape_all.log" || true
}

clean_stage() {
  log "CLEAN stage"
  [[ -f src/data/cleaners/text_cleaner.py ]] && \
    python src/data/cleaners/text_cleaner.py --in "$RAW_DIR" --out "$RAW_DIR" 2>&1 | tee -a "$LOG_DIR/text_cleaner.log" || true

  [[ -f src/data/cleaners/symptom_tagger.py ]] && \
    python src/data/cleaners/symptom_tagger.py --in "$RAW_DIR" --out "$RAW_DIR" 2>&1 | tee -a "$LOG_DIR/symptom_tagger.log" || true

  [[ -f src/data_prep/clean_pubmedqa.py ]] && \
    python src/data_prep/clean_pubmedqa.py --in "$RAW_DIR" --out "$CLEAN_DIR" 2>&1 | tee -a "$LOG_DIR/clean_pubmedqa.log" || true

  [[ -f src/data_prep/clean_pubmed_rct.py ]] && \
    python src/data_prep/clean_pubmed_rct.py --in "$RAW_DIR" --out "$CLEAN_DIR" 2>&1 | tee -a "$LOG_DIR/clean_pubmed_rct.log" || true

  [[ -f src/data_prep/clean_med_mcq.py ]] && \
    python src/data_prep/clean_med_mcq.py --in "$RAW_DIR" --out "$CLEAN_DIR" 2>&1 | tee -a "$LOG_DIR/clean_med_mcq.log" || true

  [[ -f src/data_prep/clean_drugs_reviews.py ]] && \
    python src/data_prep/clean_drugs_reviews.py --in "$RAW_DIR" --out "$CLEAN_DIR" 2>&1 | tee -a "$LOG_DIR/clean_drugs_reviews.log" || true

  [[ -f src/data_prep/clean_transcriptions.py ]] && \
    python src/data_prep/clean_transcriptions.py --in "$RAW_DIR" --out "$CLEAN_DIR" 2>&1 | tee -a "$LOG_DIR/clean_transcriptions.log" || true
}

build_stage() {
  log "BUILD/MERGE stage"
  if [[ -f src/data/builders/dataset_builder.py ]]; then
    python src/data/builders/dataset_builder.py --in "$RAW_DIR" --clean "$CLEAN_DIR" --out "$CLEAN_DIR"
  elif [[ -f src/data/dataset_merger.py ]]; then
    python src/data/dataset_merger.py --in "$RAW_DIR" --out "$CLEAN_DIR"
  elif [[ -f src/data_prep/build_dataset.py ]]; then
    python src/data_prep/build_dataset.py --in "$RAW_DIR" --out "$CLEAN_DIR"
  else
    log "no dataset builder found; skipping merge"
  fi

  [[ -f src/data/builders/text_supervised_builder.py ]] && \
    python src/data/builders/text_supervised_builder.py --in "$CLEAN_DIR" --out "$CLEAN_DIR/supervised.jsonl" || true

  [[ -f src/data/builders/text_pretrain_builder.py ]] && \
    python src/data/builders/text_pretrain_builder.py --in "$CLEAN_DIR" --out "$CLEAN_DIR/pretrain.txt" || true

  [[ -f src/data/pipelines/build_synthetic_supervised.py ]] && \
    python src/data/pipelines/build_synthetic_supervised.py --out "$RAW_DIR/synthetic.jsonl" || true
}

tokenizer_stage() {
  log "TOKENIZER stage"
  if [[ -f scripts/build_tokenizer.py ]]; then
    python scripts/build_tokenizer.py --data "$CLEAN_DIR" --out "$TOKENIZER_DIR"
  elif [[ -f src/training/train_tokenizer.py ]]; then
    python src/training/train_tokenizer.py --data "$CLEAN_DIR" --out "$TOKENIZER_DIR"
  else
    log "no tokenizer entry found; skipping"
  fi
}

train_stage() {
  log "TRAIN (OrphGPT) stage"
  if [[ -z "$TRAIN_ENTRY" ]]; then
    log "no training entrypoint found; expected one of:
         src/training/train_from_scratch.py | src/training/train_text.py | scripts/make_pretrain_from_supervised.py"
    return 0
  fi

  local cmd=( python "$TRAIN_ENTRY" )
  if [[ -f "$TRAIN_CONFIG" ]]; then
    cmd+=( --config "$TRAIN_CONFIG" )
  else
    # fallback: pass minimal dirs if your script doesn't support --config
    cmd+=( --data_dir "$CLEAN_DIR" --ckpt_dir "$CKPT_DIR" )
  fi

  local gpus
  gpus=$(gpu_count)
  if [[ "$gpus" -ge 2 ]]; then
    log "Detected $gpus GPUs -> using torchrun"
    cmd=( torchrun --nproc_per_node="$gpus" "$TRAIN_ENTRY" )
    [[ -f "$TRAIN_CONFIG" ]] && cmd+=( --config "$TRAIN_CONFIG" ) || cmd+=( --data_dir "$CLEAN_DIR" --ckpt_dir "$CKPT_DIR" )
  fi

  echo "[cmd] ${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee -a "$LOG_DIR/train.log"
}

# ----------------- Driver -----------------
case "$MODE" in
  KAGGLE_ONLY)     kaggle_stage ;;
  SCRAPE_ONLY)     scrape_api_stage ;;
  CLEAN_ONLY)      clean_stage ;;
  BUILD_ONLY)      build_stage ;;
  TOKENIZER_ONLY)  tokenizer_stage ;;
  PRETRAIN_ONLY|TRAIN_ONLY)  train_stage ;;
  ALL)             kaggle_stage; scrape_api_stage; clean_stage; build_stage; tokenizer_stage; train_stage ;;
  *)               echo "Unknown MODE=${MODE}"; exit 1 ;;
esac

log "Pipeline complete."
