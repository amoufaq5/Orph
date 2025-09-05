# src/data_prep/scrapers/clinicaltrials.py
from __future__ import annotations
import os, sys, time, json
from typing import Iterator, Dict, Any

try:
    from .base import Scraper, req_json
except ImportError:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if ROOT not in sys.path: sys.path.insert(0, ROOT)
    from src.data_prep.scrapers.base import Scraper, req_json

CT_BASE = "https://clinicaltrials.gov/api/query/study_fields"
FIELDS = "NCTId,Condition,BriefTitle,LocationCountry,OverallStatus,StartDate,CompletionDate,Phase,EnrollmentCount"

def fetch_page(expr: str, min_rank: int, max_rank: int) -> Dict[str, Any]:
    p = {"expr": expr, "fields": FIELDS, "min_rnk": min_rank, "max_rnk": max_rank, "fmt": "json"}
    return req_json(CT_BASE, p, min_sleep=0.25)

class ClinicalTrialsScraper(Scraper):
    def __init__(self, out_dir: str, expr: str, page_size: int = 100):
        super().__init__(out_dir)
        self.expr = expr
        self.page_size = max(20, min(page_size, 1000))

    def stream(self) -> Iterator[Dict[str, Any]]:
        js = fetch_page(self.expr, 1, 1)
        n = int(js["StudyFieldsResponse"]["NStudiesFound"])
        print(f"[ct] found {n} for expr='{self.expr}'")

        min_rnk = 1
        while min_rnk <= n:
            max_rnk = min(min_rnk + self.page_size - 1, n)
            js = fetch_page(self.expr, min_rnk, max_rnk)
            studies = js["StudyFieldsResponse"]["StudyFields"]
            for s in studies:
                yield {
                    "nct_id": (s.get("NCTId") or [""])[0],
                    "title": (s.get("BriefTitle") or [""])[0],
                    "condition": s.get("Condition", []),
                    "status": (s.get("OverallStatus") or [""])[0],
                    "phase": (s.get("Phase") or [""])[0],
                    "enrollment": (s.get("EnrollmentCount") or [""])[0],
                    "country": (s.get("LocationCountry") or [""])[0],
                    "start_date": (s.get("StartDate") or [""])[0],
                    "completion_date": (s.get("CompletionDate") or [""])[0],
                    "source": "clinicaltrials.gov",
                }
            time.sleep(0.25)
            min_rnk = max_rnk + 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--expr", required=True)
    parser.add_argument("--page_size", type=int, default=100)
    args = parser.parse_args()

    out_path = os.path.join(args.out, "clinicaltrials.jsonl")
    os.makedirs(args.out, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for row in ClinicalTrialsScraper(args.out, args.expr, args.page_size).stream():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
