import os, json, glob
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.io import read_jsonl, ensure_dir
from src.utils.logger import get_logger
log = get_logger("rag")

def build_index(corpus_paths, out_dir, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    ensure_dir(out_dir)
    enc = SentenceTransformer(model_name)
    texts, metas = [], []
    for p in corpus_paths:
        for r in read_jsonl(p):
            text = r.get("text") or ""
            if not text: continue
            texts.append(text[:2000])
            metas.append({"source": r["meta"].get("source","unknown"), "id": r.get("id"), "license": r["meta"].get("license","unknown")})
    log.info(f"Encoding {len(texts)} passages...")
    X = enc.encode(texts, batch_size=256, convert_to_numpy=True, show_progress_bar=True)
    idx = faiss.IndexFlatIP(X.shape[1]); faiss.normalize_L2(X); idx.add(X)
    faiss.write_index(idx, os.path.join(out_dir, "faiss.index"))
    with open(os.path.join(out_dir, "metas.json"), "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "metas": metas}, f)
    log.info(f"Index saved → {out_dir}")
