from typing import Iterator, Dict
from .base import Scraper, mk_id

class CheXpertScraper(Scraper):
    name = "chexpert"
    def stream(self) -> Iterator[Dict]:
        # TODO: read CheXpert train.csv; emit rows for findings
        row = {"image":"data/images/chexpert/p1.png","labels":{"Cardiomegaly":1,"Edema":0}}
        yield {
          "id": mk_id("chexpert"),
          "modality": ["image"],
          "task": "classification",
          "text": None,
          "image_path": row["image"],
          "answer": "Cardiomegaly" if row["labels"]["Cardiomegaly"]==1 else "No cardiomegaly",
          "rationale": None,
          "labels": {},
          "meta": {"source":"chexpert","license":"custom"},
          "split":"train"
        }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    CheXpertScraper(ap.parse_args().out).run()
