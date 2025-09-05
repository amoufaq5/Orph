#!/usr/bin/env bash
set -e
python -m src.data_prep.scrapers.pubmed --out data/raw/pubmed
python -m src.data_prep.scrapers.pmc_oa --out data/raw/pmc_oa
python -m src.data_prep.scrapers.dailymed --out data/raw/dailymed
python -m src.data_prep.scrapers.openfda_labels --out data/raw/openfda/labels
python -m src.data_prep.scrapers.clinicaltrials --out data/raw/clinicaltrials

# Merge to one corpus (text-only for RAG bootstrap)
python -m src.data_prep.merge.dataset_merger \
  --inputs data/raw/pubmed data/raw/pmc_oa data/raw/dailymed data/raw/openfda/labels data/raw/clinicaltrials \
  --out data/cleaned/text_corpus.jsonl

# Build RAG index
python -m src.rag.index_builder \
  "$(ls data/cleaned/text_corpus.jsonl)" \
  -- out_dir data/artifacts/rag
