import os, time, argparse, requests
from typing import Iterator, Dict
from .base import Scraper, mk_id
from src.utils.logger import get_logger
log = get_logger("openfda")

API = "https://api.fda.gov/drug/label.json"
API_KEY = os.getenv("OPENFDA_API_KEY")  # optional
LIMIT = 100

def page(skip: int):
    params = {"limit": LIMIT, "skip": skip}
    if API_KEY: params["api_key"]=API_KEY
    r = requests.get(API, params=params, timeout=45)
    if r.status_code == 429:
        time.sleep(2.0)
        r = requests.get(API, params=params, timeout=45)
    r.raise_for_status()
    time.sleep(0.12)
    return r.json()

def to_row(doc: dict) -> Dict:
    openfda = doc.get("openfda", {})
    product = (openfda.get("brand_name") or openfda.get("generic_name") or ["unknown"])[0]
    boxed = " ".join(doc.get("boxed_warning", [])[:1])
    adverse = " ".join(doc.get("adverse_reactions", [])[:1])
    indications = " ".join(doc.get("indications_and_usage", [])[:1])
    text = f"Label for {product}. Indications: {indications}. Boxed warning: {boxed}. Adverse reactions: {adverse}"
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

class OpenFDALabelsScraper(Scraper):
    name = "openfda_labels"
    def __init__(self, out_dir: str, max_docs: int):
        super().__init__(out_dir); self.max_docs = max_docs

    def stream(self) -> Iterator[Dict]:
        skip, seen = 0, 0
        while True:
            js = page(skip)
            results = js.get("results", [])
            if not results: break
            for d in results:
                yield to_row(d)
                seen += 1
                if self.max_docs and seen >= self.max_docs:
                    return
            skip += LIMIT

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_docs", type=int, default=2000)
    args = ap.parse_args()
    OpenFDALabelsScraper(args.out, args.max_docs).run()
