"""
ClinicalTrials.gov API v1 study_fields endpoint.
Docs: https://clinicaltrials.gov/api/gui/ref/syntax
"""
from pathlib import Path
from ._common import RAW_OUT, write_jsonl, http_get, backoff_sleep

OUT = RAW_OUT / "clinicaltrials.jsonl"
BASE = "https://clinicaltrials.gov/api/query/study_fields"

FIELDS = [
    "NCTId","BriefTitle","Condition","InterventionName",
    "BriefSummary","OverallStatus","StudyType","Phase","EnrollmentCount",
    "LocationCity","LocationCountry","StartDate","CompletionDate"
]

def fetch_batch(expr: str, min_rnk: int, max_rnk: int, fmt="json"):
    params = {
        "expr": expr,
        "fields": ",".join(FIELDS),
        "min_rnk": min_rnk,
        "max_rnk": max_rnk,
        "fmt": fmt
    }
    r = http_get(BASE, params=params)
    return r.json()["StudyFieldsResponse"]

def main():
    expr = "AREA[Condition]Neoplasm OR AREA[Condition]Diabetes OR AREA[Condition]Cardiovascular"
    size = 1000
    start = 1
    acc = []
    while True:
        end = start + size - 1
        data = fetch_batch(expr, start, end)
        n = data["NStudiesFound"]
        items = data["StudyFields"]
        for it in items:
            row = { "source":"clinicaltrials" }
            for f in FIELDS:
                v = it.get(f, [])
                row[f] = v[0] if isinstance(v, list) else v
            acc.append(row)
        print(f"Fetched {len(items)} / total {n} (rank {start}-{end})")
        if end >= n or not items:
            break
        start = end + 1
        backoff_sleep(1)
    write_jsonl(OUT, acc)

if __name__ == "__main__":
    main()
