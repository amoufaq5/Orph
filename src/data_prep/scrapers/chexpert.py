import csv, argparse, os
from typing import Iterator, Dict
from .base import Scraper, mk_id

class CheXpertScraper(Scraper):
    name = "chexpert"
    def __init__(self, out_dir: str, images_root: str, train_csv: str):
        super().__init__(out_dir); self.images_root=images_root; self.train_csv=train_csv

    def stream(self) -> Iterator[Dict]:
        with open(self.train_csv, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                img_rel = row["Path"]
                image_path = os.path.join(self.images_root, img_rel)
                # Example: turn Cardiomegaly=1 into label
                label = "Cardiomegaly" if row.get("Cardiomegaly","") == "1.0" else "No cardiomegaly"
                yield {
                    "id": mk_id("chexpert"),
                    "modality": ["image"],
                    "task": "classification",
                    "text": None,
                    "image_path": image_path.replace("\\","/"),
                    "answer": label,
                    "rationale": None,
                    "labels": {},
                    "meta": {"source":"chexpert","license":"custom"},
                    "split":"train"
                }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--train_csv", required=True)
    args = ap.parse_args()
    CheXpertScraper(args.out, args.images_root, args.train_csv).run()
