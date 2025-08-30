#!/usr/bin/env bash
set -euo pipefail

# -------- directories (persistent volume is /workspace/data on RunPod) ----------
RAW_DIR="${RAW_DIR:-/workspace/data/raw}"
CLEAN_DIR="${CLEAN_DIR:-/workspace/data/clean}"
CKPT_DIR="${CKPT_DIR:-/workspace/data/checkpoints}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/workspace/data/tokenizer}"
LOG_DIR="${LOG_DIR:-/workspace/data/logs}"
MODE="${MODE:-ALL}"   # ALL | SCRAPE_ONLY | CLEAN_ONLY | BUILD_ONLY | TOKENIZER_ONLY | PRETRAIN_ONLY | TRAIN_ONLY

mkdir -p "$RAW_DIR" "$CLEAN_DIR" "$CKPT_DIR" "$TOKENIZER_DIR" "$LOG_DIR"

log(){ echo "[orph] $*"; }

# -------- 0) SCRAPE -----------------------------------------------------------
scrape() {
  log "SCRAPE phase"

  # Individual scrapers (run only if files exist)
  [[ -f src/data_prep/scrapers/pubmed_fetch.py ]] && \
    python src/data_prep/scrapers/pubmed_fetch.py --out "$RAW_DIR" 2>&1 | tee -a "$LOG_DIR/pubmed_fetch.log" || true

  [[ -f src/data_prep/scrapers/openfda_labels_fetch.py ]] && \
    python src/data_prep/scrapers/openfda_labels_fetch.py --out "$RAW_DIR" 2>&1 | tee -a "$LOG_DIR/openfda_labels_fetch.log" || true

  [[ -f src/data_prep/scrapers/medlineplus_fetch.py ]] && \
    python src/data_prep/scrapers/medlineplus_fetch.py --out "$RAW_DIR" 2>&1 | tee -a "$LOG_DIR/medlineplus_fetch.log" || true

  [[ -f src/data_prep/scrapers/clinicaltrials_fetch.py ]] && \
    python src/data_prep/scrapers/clinicaltrials_fetch.py --out "$RAW_DIR" 2>&1 | tee -a "$LOG_DIR/clinicaltrials_fetch.log" || true

  # Umbrella runner if you use it
  [[ -f src/data_prep/scrape_all.py ]] && \
    python src/data_prep/scrape_all.py --out "$RAW_DIR" 2>&1 | tee -a "$LOG_DIR/scrape_all.log" || true
}

# -------- 1) CLEAN ------------------------------------------------------------
clean() {
  log "CLEAN phase"

  # Generic cleaners you have
  [[ -f src/data/cleaners/text_cleaner.py ]] && \
    python src/data/cleaners/text_cleaner.py --in "$RAW_DIR" --out "$RAW_DIR" 2>&1 | tee -a "$LOG_DIR/text_cleaner.log" || true

  [[ -f src/data/cleaners/symptom_tagger.py ]] && \
    python src/data/cleaners/symptom_tagger.py --in "$RAW_DIR" --out "$RAW_DIR" 2>&1 | tee -a "$LOG_DIR/symptom_tagger.log" || true

  # Dataset-specific cleaners (run if present)
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

# -------- 2) BUILD / MERGE ----------------------------------------------------
build_dataset() {
  log "BUILD/MERGE phase"

  if [[ -f src/data/builders/dataset_builder.py ]]; then
    python src/data/builders/dataset_builder.py --in "$RAW_DIR" --clean "$CLEAN_DIR" --out "$CLEAN_DIR"
  elif [[ -f src/data/dataset_merger.py ]]; then
    python src/data/dataset_merger.py --in "$RAW_DIR" --out "$CLEAN_DIR"
  elif [[ -f src/data_prep/build_dataset.py ]]; then
    python src/data_prep/build_dataset.py --in "$RAW_DIR" --out "$CLEAN_DIR"
  else
    log "no dataset builder found; skipping merge"
  fi

  # Optional: build task-specific corpora if present
  [[ -f src/data/builders/text_supervised_builder.py ]] && \
    python src/data/builders/text_supervised_builder.py --in "$CLEAN_DIR" --out "$CLEAN_DIR/supervised.jsonl" || true

  [[ -f src/data/builders/text_pretrain_builder.py ]] && \
    python src/data/builders/text_pretrain_builder.py --in "$CLEAN_DIR" --out "$CLEAN_DIR/pretrain.txt" || true

  # Optional: synthetic
  [[ -f src/data/pipelines/build_synthetic_supervised.py ]] && \
    python src/data/pipelines/build_synthetic_supervised.py --out "$RAW_DIR/synthetic.jsonl" || true
}

# -------- 3) TOKENIZER --------------------------------------------------------
train_tokenizer() {
  log "TOKENIZER phase"
  if [[ -f scripts/build_tokenizer.py ]]; then
    python scripts/build_tokenizer.py --data "$CLEAN_DIR" --out "$TOKENIZER_DIR"
  elif [[ -f src/training/train_tokenizer.py ]]; then
    python src/training/train_tokenizer.py --data "$CLEAN_DIR" --out "$TOKENIZER_DIR"
  else
    log "no tokenizer script found; skipping"
  fi
}

# -------- 4) PRETRAIN / TRAIN -------------------------------------------------
pretrain() {
  log "PRETRAIN/TRAIN phase"
  if [[ -f scripts/make_pretrain_from_supervised.py ]]; then
    python scripts/make_pretrain_from_supervised.py --data "$CLEAN_DIR" --out "$CKPT_DIR"
  elif [[ -f src/training/train_from_scratch.py ]]; then
    python src/training/train_from_scratch.py --data_dir "$CLEAN_DIR" --ckpt_dir "$CKPT_DIR"
  elif [[ -f src/training/train_text.py ]]; then
    python src/training/train_text.py --data_dir "$CLEAN_DIR" --ckpt_dir "$CKPT_DIR"
  else
    log "no pretrain/train entry found; skipping"
  fi
}

# -------- driver --------------------------------------------------------------
case "$MODE" in
  SCRAPE_ONLY)     scrape ;;
  CLEAN_ONLY)      clean ;;
  BUILD_ONLY)      build_dataset ;;
  TOKENIZER_ONLY)  train_tokenizer ;;
  PRETRAIN_ONLY)   pretrain ;;
  TRAIN_ONLY)      pretrain ;;
  ALL)             scrape; clean; build_dataset; train_tokenizer; pretrain ;;
  *)               echo "Unknown MODE=$MODE"; exit 1 ;;
esac

log "done."
