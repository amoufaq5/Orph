import argparse, requests, time
from typing import Iterator, Dict, List
from .base import Scraper, mk_id
from src.utils.logger import get_logger
log = get_logger("clinicaltrials")

API = "https://clinicaltrials.gov/api/v2/studies"
FIELDS = ["NCTId","BriefTitle","Condition","InterventionName","OverallStatus","StudyType","StartDate","CompletionDate","ResultsFirstPostDate","BriefSummary"]

def fetch_chunk(expr: str, page_token: str|None=None) -> dict:
    params = {"format":"json","query.term":expr,"pageSize":100}
    if page_token: params["pageToken"]=page_token
    r = requests.get(API, params=params, timeout=45)
    r.raise_for_status()
    time.sleep(0.2)
    return r.json()

def to_rows(st: dict) -> Iterator[Dict]:
    id_ = st.get("protocolSection",{}).get("identificationModule",{}).get("nctId","")
    title = st.get("protocolSection",{}).get("identificationModule",{}).get("briefTitle","")
    conds = st.get("protocolSection",{}).get("conditionsModule",{}).get("conditions",[])
    summ = st.get("protocolSection",{}).get("descriptionModule",{}).get("briefSummary","")
    status = st.get("protocolSection",{}).get("statusModule",{}).get("overallStatus","")
    txt = f"{title} ({id_}). Conditions={', '.join(conds)}. Status={status}. Summary: {summ}"
    yield {
        "id": mk_id("ct"),
        "modality": ["text"],
        "task": "summarize",
        "text": txt,
        "answer": None,
        "rationale": None,
        "labels": {},
        "meta": {"source":"clinicaltrials","license":"public-domain","nct":id_},
        "split":"train"
    }

class ClinicalTrialsScraper(Scraper):
    name = "clinicaltrials"
    def __init__(self, out_dir: str, expr: str):
        super().__init__(out_dir); self.expr = expr

    def stream(self) -> Iterator[Dict]:
        token = None
        total = 0
        while True:
            js = fetch_chunk(self.expr, token)
            studies = js.get("studies", [])
            for s in studies:
                total += 1
                yield from to_rows(s)
            token = js.get("nextPageToken")
            if not token: break
        log.info(f"[clinicaltrials] studies processed: {total}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--expr", default="(asthma OR diabetes OR hypertension)")
    args = ap.parse_args()
    ClinicalTrialsScraper(args.out, args.expr).run()
