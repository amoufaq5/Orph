import os, argparse, csv, subprocess, json, shutil
from typing import Iterator, Dict
from .base import Scraper, mk_id
from src.utils.logger import get_logger
log = get_logger("isic")

"""
Preferred path uses the official ISIC CLI for reliable filtered download:
  pip install isic-cli
  isic image download --search 'diagnosis_3:"Melanoma Invasive"' data/images/isic/

Why CLI? It's the supported way for filtered pulls and faster bulk access.
Docs/examples: CLI repo.  (See conf: search query examples.)

Fallback HTTP mode is left as a stub if CLI is unavailable.
"""

def have_isic_cli() -> bool:
    return shutil.which("isic") is not None or os.path.exists(os.getenv("ISIC_CLI",""))

def run_isic_cli(download_dir: str, search: str | None, limit: int | None):
    cmd = ["isic", "image", "download", download_dir]
    if search: cmd += ["--search", search]
    if limit:  cmd += ["--limit", str(limit)]
    log.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

class ISICScraper(Scraper):
    name = "isic_api"

    def __init__(self, out_dir: str, images_dir: str, search: str | None, limit: int | None):
        super().__init__(out_dir)
        self.images_dir = images_dir
        self.search = search
        self.limit = limit

    def stream(self) -> Iterator[Dict]:
        if not have_isic_cli():
            log.warning("isic-cli not found; please `pip install isic-cli`. Falling back would require custom HTTP code.")
            return iter([])

        os.makedirs(self.images_dir, exist_ok=True)
        run_isic_cli(self.images_dir, self.search, self.limit)

        # The CLI writes metadata CSV(s) inside the images directory.
        # We scan for *.csv and emit schema v2 rows for each image path.
        img_dir = self.images_dir
        meta_csvs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".csv")]
        if not meta_csvs:
            log.warning("No metadata CSV produced by CLI; emitting rows by walking image files only.")
            for root, _, files in os.walk(img_dir):
                for fn in files:
                    if fn.lower().endswith((".jpg",".jpeg",".png")):
                        yield {
                          "id": mk_id("isic"),
                          "modality": ["image"],
                          "task": "classification",
                          "text": None,
                          "image_path": os.path.join(root, fn).replace("\\","/"),
                          "answer": None,
                          "rationale": None,
                          "labels": {"icd10": [], "snomed": []},
                          "meta": {"source": "isic_cli", "license": "CC-BY"},
                          "split": "train"
                        }
            return

        for mcsv in meta_csvs:
            with open(mcsv, newline="", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    # Common cols exposed by CLI: "isic_id", "file_name", "diagnosis_3", ...
                    fn = r.get("file_name") or r.get("image") or r.get("name")
                    if not fn: continue
                    img_path = os.path.join(img_dir, fn)
                    dx = r.get("diagnosis_3") or r.get("diagnosis") or None
                    yield {
                        "id": mk_id("isic"),
                        "modality": ["image"],
                        "task": "classification",
                        "text": None,
                        "image_path": img_path.replace("\\","/"),
                        "answer": dx,
                        "rationale": None,
                        "labels": {"icd10": [], "snomed": []},
                        "meta": {"source":"isic_cli","license":"CC-BY","isic_id": r.get("isic_id")},
                        "split":"train"
                    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--images_dir", default="data/images/isic")
    ap.add_argument("--search", default=None, help="ISIC CLI search string")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    ISICScraper(args.out, args.images_dir, args.search, args.limit).run()
