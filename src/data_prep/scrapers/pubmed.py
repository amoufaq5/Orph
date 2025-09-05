from typing import Iterator, Dict
from .base import Scraper, mk_id
from src.utils.logger import get_logger
log = get_logger("pubmed")

class PubMedScraper(Scraper):
    name = "pubmed"

    def stream(self) -> Iterator[Dict]:
        # TODO: replace with real Entrez or E-utilities fetcher (respect rate limits & TOS)
        # Demo rows only
        demo = [
            {"title":"Asthma management updates","abstract":"... beta-agonists ...","doi":"10.demo/1","date":"2023-03-01"},
            {"title":"Diabetes complications review","abstract":"... HbA1c ...","doi":"10.demo/2","date":"2024-07-21"},
        ]
        for d in demo:
            yield {
              "id": mk_id("pubmed"),
              "modality": ["text"],
              "task": "summarize",
              "text": f"{d['title']} — {d['abstract']}",
              "answer": None,
              "rationale": None,
              "labels": {},
              "meta": {"source":"pubmed","pubdate": d["date"], "doi": d["doi"], "license":"public-domain"},
              "split": "train"
            }

if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    PubMedScraper(args.out).run()
