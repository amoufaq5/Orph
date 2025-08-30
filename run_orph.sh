#!/usr/bin/env bash
set -euo pipefail

# Persistent dirs on RunPod volume:
RAW_DIR="${RAW_DIR:-/workspace/data/raw}"
KAGGLE_DIR="${KAGGLE_DIR:-/workspace/data/raw/kaggle}"
CLEAN_DIR="${CLEAN_DIR:-/workspace/data/clean}"
CKPT_DIR="${CKPT_DIR:-/workspace/data/checkpoints}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/workspace/data/tokenizer}"
LOG_DIR="${LOG_DIR:-/workspace/data/logs}"

# Modes
MODE="${MODE:-ALL}"   # ALL | KAGGLE_ONLY | SCRAPE_ONLY | CLEAN_ONLY | BUILD_ONLY | TOKENIZER_ONLY | PRETRAIN_ONLY | TRAIN_ONLY

# Kaggle flags (safe defaults)
KAGGLE_CATALOG="${KAGGLE_CATALOG:-config/kaggle_catalog.yaml}"
KAGGLE_ENABLE_HEAVY="${KAGGLE_ENABLE_HEAVY:-false}"    # true to include heavy sections
KAGGLE_ONLY_KEYS="${KAGGLE_ONLY_KEYS:-}"               # space-separated keys to restrict (e.g. "pubmed_qa medmcqa")
KAGGLE_MAX_ITEMS="${KAGGLE_MAX_ITEMS:-0}"              # 0 = unlimited
KAGGLE_SLEEP="${KAGGLE_SLEEP:-0.0}"

mkdir -p "$RAW_DIR" "$KAGGLE_DIR" "$CLEAN_DIR" "$CKPT_DIR" "$TOKENIZER_DIR" "$LOG_DIR"

log(){ echo "[orph]" "$@"; }

kaggle_stage() {
  log "KAGGLE stage"
  local args=( --out "$KAGGLE_DIR" --catalog "$KAGGLE_CATALOG" )
  if [[ "$KAGGLE_ENABLE_HEAVY" == "true" ]]; then args+=( --enable-heavy ); fi
  if [[ -n "$KAGGLE_ONLY_KEYS" ]]; then args+=( --only $KAGGLE_ONLY_KEYS ); fi
  if [[ "$KAGGLE_MAX_ITEMS" != "0" ]]; then args+=( --max-items "$KAGGLE_MAX_ITEMS" ); fi
  if [[ "$KAGGLE_SLEEP" != "0.0" ]]; then args+=( --sleep "$KAGGLE_SLEEP" ); fi

  if [[ -f src/data_prep/scrapers/kaggle_fetch.py ]]; then
    python src/data_prep/scrapers/kaggle_fetch.py "${args[@]}" 2>&1 | tee -a "$LOG_DIR/kaggle_fetch.log" || true
  else
    log "kaggle_fetch.py not found; skipping"
  fi
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
    log "no tokenizer script found; skipping"
  fi
}

train_stage() {
  log "PRETRAIN/TRAIN stage"
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

case "$MODE" in
  KAGGLE_ONLY)     kaggle_stage ;;
  SCRAPE_ONLY)     scrape_api_stage ;;
  CLEAN_ONLY)      clean_stage ;;
  BUILD_ONLY)      build_stage ;;
  TOKENIZER_ONLY)  tokenizer_stage ;;
  PRETRAIN_ONLY)   train_stage ;;
  TRAIN_ONLY)      train_stage ;;
  ALL)             kaggle_stage; scrape_api_stage; clean_stage; build_stage; tokenizer_stage; train_stage ;;
  *)               echo "Unknown MODE=$MODE"; exit 1 ;;
esac

log "done."
