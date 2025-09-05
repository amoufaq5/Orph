# src/data_prep/scrapers/clinicaltrials.py
from __future__ import annotations
import os, sys, time, json
from typing import Iterator, Dict, Any, List, Optional

try:
    from .base import Scraper, req_json
except ImportError:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if ROOT not in sys.path: sys.path.insert(0, ROOT)
    from src.data_prep.scrapers.base import Scraper, req_json

CT_V1 = "https://clinicaltrials.gov/api/query/study_fields"
CT_V2 = "https://clinicaltrials.gov/api/v2/studies"

FIELDS_V1 = "NCTId,Condition,BriefTitle,LocationCountry,OverallStatus,StartDate,CompletionDate,Phase,EnrollmentCount"

def fetch_v1(expr: str, min_rank: int, max_rank: int) -> Dict[str, Any]:
    p = {"expr": expr, "fields": FIELDS_V1, "min_rnk": min_rank, "max_rnk": max_rank, "fmt": "json"}
    return req_json(CT_V1, p, min_sleep=0.25)

def try_v1_total(expr: str) -> Optional[int]:
    try:
        js = fetch_v1(expr, 1, 1)
        return int(js["StudyFieldsResponse"]["NStudiesFound"])
    except Exception:
        return None

def iter_v1(expr: str, page_size: int) -> Iterator[Dict[str, Any]]:
    n = try_v1_total(expr)
    if n is None:
        return
    print(f"[ct] (v1) found {n} for expr='{expr}'")
    min_rnk = 1
    while min_rnk <= n:
        max_rnk = min(min_rnk + page_size - 1, n)
        js = fetch_v1(expr, min_rnk, max_rnk)
        for s in js["StudyFieldsResponse"]["StudyFields"]:
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
                "source": "clinicaltrials.gov_v1",
            }
        time.sleep(0.25)
        min_rnk = max_rnk + 1

def fetch_v2(query: str, page_size: int, page_token: Optional[str]) -> Dict[str, Any]:
    p = {"query.term": query, "pageSize": page_size}
    if page_token: p["pageToken"] = page_token
    return req_json(CT_V2, p, min_sleep=0.25)

def iter_v2(expr: str, page_size: int) -> Iterator[Dict[str, Any]]:
    print(f"[ct] using v2 fallback for expr='{expr}'")
    token: Optional[str] = None
    while True:
        js = fetch_v2(expr, page_size, token)
        studies: List[Dict[str, Any]] = js.get("studies", [])
        if not studies:
            break
        for s in studies:
            prot = s.get("protocolSection", {}) or {}
            ident = prot.get("identificationModule", {}) or {}
            status = prot.get("statusModule", {}) or {}
            conds = prot.get("conditionsModule", {}) or {}
            design = prot.get("designModule", {}) or {}
            contacts = prot.get("contactsLocationsModule", {}) or {}
            locs = contacts.get("locations") or []
            country = locs[0].get("country") if locs else None
            start_date = (status.get("startDateStruct") or {}).get("date")
            completion_date = (status.get("completionDateStruct") or {}).get("date")
            phases = design.get("phases") or [None]
            yield {
                "nct_id": ident.get("nctId"),
                "title": ident.get("briefTitle"),
                "condition": conds.get("conditions") or [],
                "status": status.get("overallStatus"),
                "phase": phases[0],
                "enrollment": (design.get("enrollmentInfo") or {}).get("count"),
                "country": country,
                "start_date": start_date,
                "completion_date": completion_date,
                "source": "clinicaltrials.gov_v2",
            }
        token = js.get("nextPageToken")
        if not token:
            break
        time.sleep(0.25)

class ClinicalTrialsScraper(Scraper):
    def __init__(self, out_dir: str, expr: str, page_size: int = 100):
        super().__init__(out_dir)
        self.expr = expr
        self.page_size = max(20, min(page_size, 1000))

    def stream(self) -> Iterator[Dict[str, Any]]:
        yielded = False
        for row in iter_v1(self.expr, self.page_size):
            yielded = True
            yield row
        if not yielded:
            for row in iter_v2(self.expr, self.page_size):
                yield row

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--expr", required=True)
    parser.add_argument("--page_size", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "clinicaltrials.jsonl")
    with open(out_path, "a", encoding="utf-8") as f:
        for row in ClinicalTrialsScraper(args.out, args.expr, args.page_size).stream():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
