import os, argparse, pandas as pd
from typing import Iterator, Dict
from .base import Scraper, mk_id
from src.utils.logger import get_logger
log = get_logger("ham10000")

"""
Ingests HAM10000 (Kaggle) or ISIC CSV-style dumps.
Expected files:
  - images under: data/images/isic/ (or your chosen root)
  - metadata CSV: HAM10000_metadata.csv with columns:
      image_id, dx (diagnosis), dx_type, age, sex, localization
"""

class HAM10000Scraper(Scraper):
    name = "isic_ham10000"

    def __init__(self, out_dir: str, images_root: str, meta_csv: str):
        super().__init__(out_dir)
        self.images_root = images_root
        self.meta_csv = meta_csv

    def stream(self) -> Iterator[Dict]:
        df = pd.read_csv(self.meta_csv)
        total = 0
        for _, r in df.iterrows():
            img_file = f"{r['image_id']}.jpg"
            image_path = os.path.join(self.images_root, img_file)
            if not os.path.exists(image_path): 
                continue
            dx = str(r.get("dx","unknown"))
            meta = {
                "source": "isic_ham10000",
                "license": "CC-BY",
                "age": int(r["age"]) if not pd.isna(r["age"]) else None,
                "sex": r.get("sex", None),
                "site": r.get("localization", None)
            }
            yield {
                "id": mk_id("isic"),
                "modality": ["image"],
                "task": "classification",
                "text": None,
                "image_path": image_path.replace("\\","/"),
                "answer": dx,
                "rationale": None,
                "labels": {"icd10": [], "snomed": []},
                "meta": meta,
                "split": "train"
            }
            total += 1
        log.info(f"[HAM10000] emitted {total} rows.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--images_root", default="data/images/isic")
    ap.add_argument("--meta_csv", default="data/images/isic/HAM10000_metadata.csv")
    args = ap.parse_args()
    HAM10000Scraper(args.out, args.images_root, args.meta_csv).run()
