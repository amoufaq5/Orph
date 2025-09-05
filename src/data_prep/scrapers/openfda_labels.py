# src/data_prep/scrapers/openfda_labels.py
from __future__ import annotations
import os, sys, time, json
from typing import Iterator, Dict, Any

try:
    from .base import Scraper, req_json
except ImportError:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if ROOT not in sys.path: sys.path.insert(0, ROOT)
    from src.data_prep.scrapers.base import Scraper, req_json

OPENFDA_API = "https://api.fda.gov/drug/label.json"
API_KEY = os.getenv("OPENFDA_API_KEY")

class OpenFDALabelsScraper(Scraper):
    def __init__(self, out_dir: str, query: str = None, limit: int = 100, max_records: int = 100000):
        super().__init__(out_dir)
        self.query = query
        self.limit = max(1, min(limit, 100))  # OpenFDA max 100
        self.max_records = max_records

    def stream(self) -> Iterator[Dict[str, Any]]:
        skip = 0
        total = None
        while True:
            p = {"limit": self.limit, "skip": skip}
            if self.query: p["search"] = self.query
            if API_KEY: p["api_key"] = API_KEY
            js = req_json(OPENFDA_API, p, min_sleep=0.25)
            meta = js.get("meta", {}).get("results", {})
            if total is None:
                total = meta.get("total", 0)
                print(f"[openfda] total={total} (cap={self.max_records})")
            results = js.get("results", [])
            if not results: break
            for r in results:
                out = {
                    "id": r.get("id"),
                    "effective_time": r.get("effective_time"),
                    "spl_set_id": r.get("spl_set_id"),
                    "openfda": r.get("openfda", {}),
                    "sections": {k: v for k, v in r.items() if isinstance(v, list) and k not in ("openfda",)},
                    "source": "openfda_label",
                }
                yield out
            skip += self.limit
            if skip >= min(total or 0, self.max_records): break
            time.sleep(0.25)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--query", default=None, help='OpenFDA search, e.g. openfda.generic_name:"ibuprofen"')
    parser.add_argument("--max_records", type=int, default=50000)
    args = parser.parse_args()

    path = os.path.join(args.out, "openfda_labels.jsonl")
    os.makedirs(args.out, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in OpenFDALabelsScraper(args.out, args.query, limit=100, max_records=args.max_records).stream():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
