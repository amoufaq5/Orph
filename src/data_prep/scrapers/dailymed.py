# src/data_prep/scrapers/dailymed.py
from __future__ import annotations
import argparse
from typing import Iterator, Dict, Optional

from .base import Scraper, HttpClient, RateLimiter, mk_id
from src.utils.logger import get_logger
log = get_logger("dailymed")

INDEX_API = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
SPL_API   = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.json"

class DailyMedScraper(Scraper):
    name = "dailymed"

    def __init__(self, out_dir: str, page_size: int, shard_size: int, max_docs: Optional[int]):
        client = HttpClient(timeout=60)
        super().__init__(out_dir, client=client, shard_size=shard_size, max_docs=max_docs)
        self.ps = max(1, min(100, page_size))
        self.rl = RateLimiter(calls_per_sec=3.0, burst=2)

    def _list_page(self, page_idx: int) -> Dict:
        self.rl.sleep()
        return self.client.json(INDEX_API, params={"pagesize": self.ps, "page": page_idx})

    def _get_spl(self, setid: str) -> Dict:
        self.rl.sleep()
        return self.client.json(SPL_API.format(setid=setid))

    def stream(self) -> Iterator[Dict]:
        page = 1
        seen = 0
        while True:
            js = self._list_page(page)
            items = js.get("data", [])
            if not items:
                break
            for it in items:
                if self.max_docs and seen >= self.max_docs:
                    return
                setid = it.get("setid")
                if not setid:
                    continue
                try:
                    spl = self._get_spl(setid)
                    yield self._to_row(spl)
                    seen += 1
                except Exception as e:
                    log.warning(f"[dailymed] setid={setid} failed: {e}")
                    continue
            page += 1

    def _to_row(self, spl: Dict) -> Dict:
        data = spl.get("data", {})
        title = data.get("title", "")
        sections = {sec.get("code",""): sec.get("text","") for sec in data.get("sections", [])}
        indications = sections.get("34067-9","") or sections.get("34067-9 Indications & Usage","") or ""
        ddix = sections.get("34073-7","") or ""
        txt = f"{title}. Indications: {indications[:600]}. Interactions: {ddix[:600]}"
        return {
            "id": mk_id("dailymed"),
            "modality": ["text"],
            "task": "ddi",
            "text": txt,
            "answer": None,
            "rationale": None,
            "labels": {},
            "meta": {"source":"dailymed","license":"public-domain"},
            "split":"train"
        }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--page_size", type=int, default=100)
    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--max_docs", type=int, default=None)
    args = ap.parse_args()
    DailyMedScraper(args.out, args.page_size, args.shard_size, args.max_docs).run()
