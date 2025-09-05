# src/data_prep/scrapers/openfda_labels.py
from __future__ import annotations
import os, argparse
from typing import Iterator, Dict, Optional

from .base import Scraper, HttpClient, RateLimiter, mk_id
from src.utils.logger import get_logger
log = get_logger("openfda")

API = "https://api.fda.gov/drug/label.json"
API_KEY = os.getenv("OPENFDA_API_KEY")

class OpenFDALabelsScraper(Scraper):
    name = "openfda_labels"

    def __init__(self, out_dir: str, limit_per_page: int, shard_size: int, max_docs: Optional[int]):
        headers = {}
        if API_KEY: headers["X-Api-Key"] = API_KEY
        client = HttpClient(timeout=60, headers=headers)
        super().__init__(out_dir, client=client, shard_size=shard_size, max_docs=max_docs)
        self.limit = max(1, min(100, limit_per_page))
        # OpenFDA rate guidance ~ 240 req/min with key; be safe:
        self.rl = RateLimiter(calls_per_sec=4.0 if API_KEY else 2.0, burst=2)

    def stream(self) -> Iterator[Dict]:
        skip, seen = 0, 0
        while True:
            if self.max_docs and seen >= self.max_docs:
                return
            self.rl.sleep()
            params = {"limit": self.limit, "skip": skip}
            if API_KEY: params["api_key"] = API_KEY
            js = self.client.json(API, params=params)
            results = js.get("results", [])
            if not results:
                break
            for d in results:
                if self.max_docs and seen >= self.max_docs:
                    return
                yield self._to_row(d)
                seen += 1
            skip += self.limit

    def _to_row(self, doc: Dict) -> Dict:
        of = doc.get("openfda", {})
        product = (of.get("brand_name") or of.get("generic_name") or ["unknown"])[0]
        bw = " ".join(doc.get("boxed_warning", [])[:1])
        adverse = " ".join(doc.get("adverse_reactions", [])[:1])
        indic = " ".join(doc.get("indications_and_usage", [])[:1])
        text = f"Label for {product}. Indications: {indic}. Boxed warning: {bw}. Adverse reactions: {adverse}"
        return {
            "id": mk_id("openfda"),
            "modality": ["text"],
            "task": "summarize",
            "text": text,
            "answer": None,
            "rationale": None,
            "labels": {},
            "meta": {"source":"openfda","license":"public-domain","spl_set_id": doc.get("spl_set_id","")},
            "split":"train"
        }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit_per_page", type=int, default=100)
    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--max_docs", type=int, default=None)
    args = ap.parse_args()
    OpenFDALabelsScraper(args.out, args.limit_per_page, args.shard_size, args.max_docs).run()
