#!/usr/bin/env bash
set -euo pipefail

# TEXT SOURCES
python -m src.data_prep.scrapers.pubmed --out data/raw/pubmed --term "randomized controlled trial[pt] OR review[pt]" --mindate 2018
python -m src.data_prep.scrapers.clinicaltrials --out data/raw/clinicaltrials --expr "(asthma OR diabetes OR hypertension)"
python -m src.data_prep.scrapers.openfda_labels --out data/raw/openfda/labels --max_docs 2000
python -m src.data_prep.scrapers.dailymed --out data/raw/dailymed --max_docs 500

# MERGE
python -m src.data_prep.merge.dataset_merger \
  --inputs data/raw/pubmed data/raw/clinicaltrials data/raw/openfda/labels data/raw/dailymed \
  --out data/cleaned/text_corpus.jsonl

# RAG
python -m src.rag.index_builder "data/cleaned/text_corpus.jsonl" -- out_dir data/artifacts/rag

# TOKENIZER
python -m src.tokenizer.train_tokenizer --jsonl data/cleaned/text_corpus.jsonl --out_dir data/artifacts/tokenizer --vocab_size 48000
