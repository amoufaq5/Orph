import argparse, requests, time
from typing import Iterator, Dict
from .base import Scraper, mk_id

INDEX_API = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
SPL_API   = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.json"

def list_spls(skip=0, limit=100):
    r = requests.get(INDEX_API, params={"pagesize":limit, "page": skip//limit + 1}, timeout=45)
    r.raise_for_status(); time.sleep(0.2)
    js = r.json()
    items = js.get("data", [])
    return items

def get_spl(setid: str) -> dict:
    r = requests.get(SPL_API.format(setid=setid), timeout=45)
    r.raise_for_status(); time.sleep(0.2)
    return r.json()

def to_row(spl: dict) -> Dict:
    data = spl.get("data", {})
    title = data.get("title","")
    sections = {sec.get("code",""): sec.get("text","") for sec in data.get("sections", [])}
    indications = sections.get("34067-9","") or sections.get("34067-9 Indications & Usage","")
    ddix = sections.get("34073-7","")  # Interactions if present
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

class DailyMedScraper(Scraper):
    name = "dailymed"
    def __init__(self, out_dir: str, max_docs: int):
        super().__init__(out_dir); self.max_docs = max_docs

    def stream(self) -> Iterator[Dict]:
        skip, seen = 0, 0
        while True:
            items = list_spls(skip)
            if not items: break
            for it in items:
                setid = it.get("setid")
                if not setid: continue
                try:
                    spl = get_spl(setid)
                    yield to_row(spl)
                    seen += 1
                    if self.max_docs and seen >= self.max_docs: return
                except Exception:
                    continue
            skip += 100

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_docs", type=int, default=1000)
    args = ap.parse_args()
    DailyMedScraper(args.out, args.max_docs).run()
