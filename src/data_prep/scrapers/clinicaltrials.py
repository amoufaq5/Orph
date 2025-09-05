# src/data_prep/scrapers/clinicaltrials.py
from __future__ import annotations
import argparse
from typing import Iterator, Dict, Optional

from .base import Scraper, HttpClient, RateLimiter, mk_id
from src.utils.logger import get_logger
log = get_logger("clinicaltrials")

API = "https://clinicaltrials.gov/api/v2"

class ClinicalTrialsScraper(Scraper):
    name = "clinicaltrials"

    def __init__(self, out_dir: str, expr: str, page_size: int, shard_size: int, max_docs: Optional[int]):
        client = HttpClient(base_url=API, timeout=60)
        super().__init__(out_dir, client=client, shard_size=shard_size, max_docs=max_docs)
        self.expr = expr
        # API is OK with ~ 5–10 qps; stay conservative
        self.rl = RateLimiter(calls_per_sec=4.0, burst=2)
        self.page_size = min(max(1, page_size), 100)

    def _page(self, token: str | None) -> Dict:
        self.rl.sleep()
        params = {"format": "json", "query.term": self.expr, "pageSize": self.page_size}
        if token: params["pageToken"] = token
        return self.client.json("studies", params=params)

    def stream(self) -> Iterator[Dict]:
        token = None
        seen = 0
        while True:
            js = self._page(token)
            studies = js.get("studies", [])
            for st in studies:
                if self.max_docs and seen >= self.max_docs:
                    return
                yield from self._to_rows(st)
                seen += 1
            token = js.get("nextPageToken")
            if not token:
                break

    def _to_rows(self, st: Dict) -> Iterator[Dict]:
        prot = st.get("protocolSection", {})
        ident = prot.get("identificationModule", {})
        conds = prot.get("conditionsModule", {}).get("conditions", [])
        status = prot.get("statusModule", {}).get("overallStatus", "")
        desc = prot.get("descriptionModule", {}).get("briefSummary", "")
        nct = ident.get("nctId","")
        title = ident.get("briefTitle","")
        text = f"{title} ({nct}). Conditions={', '.join(conds)}. Status={status}. Summary: {desc}"
        yield {
            "id": mk_id("ct"),
            "modality": ["text"],
            "task": "summarize",
            "text": text,
            "answer": None,
            "rationale": None,
            "labels": {},
            "meta": {"source": "clinicaltrials", "license": "public-domain", "nct": nct},
            "split": "train"
        }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--expr", default="(asthma OR diabetes OR hypertension)")
    ap.add_argument("--page_size", type=int, default=100)
    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--max_docs", type=int, default=None)
    args = ap.parse_args()
    ClinicalTrialsScraper(args.out, args.expr, args.page_size, args.shard_size, args.max_docs).run()
