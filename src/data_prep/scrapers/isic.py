import os
from typing import Iterator, Dict
from .base import Scraper, mk_id

class ISICScraper(Scraper):
    name = "isic"
    def stream(self) -> Iterator[Dict]:
        # TODO: download metadata CSV; for demo emit stubs; images saved under data/images/isic/
        meta = {"image_path":"data/images/isic/ham10000_0001.jpg","diagnosis":"melanocytic_nevus"}
        yield {
          "id": mk_id("isic"),
          "modality": ["image"],
          "task": "classification",
          "text": None,
          "image_path": meta["image_path"],
          "answer": meta["diagnosis"],
          "rationale": None,
          "labels": {"icd10":[], "snomed":[], "rxnorm":[], "meddra":[]},
          "meta": {"source":"isic","license":"CC-BY"},
          "split":"train"
        }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ISICScraper(ap.parse_args().out).run()
