"""
openFDA drug label API (SPL) -> JSONL
Docs: https://open.fda.gov/apis/drug/label/
"""
from pathlib import Path
from ._common import RAW_OUT, write_jsonl, http_get, backoff_sleep

OUT = RAW_OUT / "openfda_drug_labels.jsonl"
BASE = "https://api.fda.gov/drug/label.json"
# openFDA rate limit default ~240 req/min; use 'limit' paging
QUERY = 'effective_time:[20000101+TO+*] AND (_exists_:indications_and_usage OR _exists_:dosage_and_administration)'
LIMIT = 100

def main():
    skip = 0
    rows = []
    while True:
        params = {"search": QUERY, "limit": LIMIT, "skip": skip}
        r = http_get(BASE, params=params)
        data = r.json()
        results = data.get("results", [])
        if not results:
            break
        for d in results:
            rows.append({
                "source":"openfda_label",
                "id": d.get("id"),
                "set_id": d.get("set_id"),
                "effective_time": d.get("effective_time"),
                "product_type": (d.get("openfda",{}) or {}).get("product_type"),
                "generic_name": (d.get("openfda",{}) or {}).get("generic_name"),
                "brand_name": (d.get("openfda",{}) or {}).get("brand_name"),
                "route": (d.get("openfda",{}) or {}).get("route"),
                "indications_and_usage": d.get("indications_and_usage"),
                "dosage_and_administration": d.get("dosage_and_administration"),
                "warnings": d.get("warnings"),
                "contraindications": d.get("contraindications"),
                "adverse_reactions": d.get("adverse_reactions"),
                "information_for_patients": d.get("information_for_patients"),
            })
        print(f"Fetched {len(results)} (skip={skip})")
        skip += LIMIT
        backoff_sleep(0)
    write_jsonl(OUT, rows)

if __name__ == "__main__":
    main()
