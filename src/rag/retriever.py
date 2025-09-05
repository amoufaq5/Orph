import os, json
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index_dir, model_name="sentence-transformers/all-MiniLM-L6-v2", top_k=5):
        self.top_k = top_k
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        with open(os.path.join(index_dir, "metas.json"), "r", encoding="utf-8") as f:
            store = json.load(f)
        self.texts = store["texts"]; self.metas = store["metas"]

    def search(self, query: str):
        q = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q)
        D, I = self.index.search(q, self.top_k)
        hits = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx == -1: continue
            hits.append({"score": float(score), "text": self.texts[idx], "meta": self.metas[idx]})
        return hits
