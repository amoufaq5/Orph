# src/data_prep/scrapers/dailymed.py
from __future__ import annotations
import os, sys, time, json
from typing import Iterator, Dict, Any

try:
    from .base import Scraper, req_json
except ImportError:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if ROOT not in sys.path: sys.path.insert(0, ROOT)
    from src.data_prep.scrapers.base import Scraper, req_json

# DailyMed SPLs listing; docs: https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm
BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"

class DailyMedScraper(Scraper):
    def __init__(self, out_dir: str, page_size: int = 100, max_pages: int = 500):
        super().__init__(out_dir)
        self.page_size = max(1, min(page_size, 1000))
        self.max_pages = max_pages

    def stream(self) -> Iterator[Dict[str, Any]]:
        page = 1
        while page <= self.max_pages:
            p = {"page": page, "pagesize": self.page_size}
            js = req_json(BASE, p, min_sleep=0.25)
            data = js.get("data", [])
            if not data:
                break
            for item in data:
                yield {
                    "setid": item.get("setid"),
                    "title": item.get("title"),
                    "effective_time": item.get("effective_time"),
                    "spl_version": item.get("spl_version"),
                    "source": "dailymed_spl",
                }
            page += 1
            time.sleep(0.25)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--page_size", type=int, default=100)
    parser.add_argument("--max_pages", type=int, default=500)
    args = parser.parse_args()

    path = os.path.join(args.out, "dailymed_spls.jsonl")
    os.makedirs(args.out, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in DailyMedScraper(args.out, args.page_size, args.max_pages).stream():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
