from __future__ import annotations
import os, sys, time, json
from typing import Iterator, Dict, Any

try:
    from .base import Scraper, req_json
except ImportError:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if ROOT not in sys.path: sys.path.insert(0, ROOT)
    from src.data_prep.scrapers.base import Scraper, req_json

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_KEY = os.getenv("NCBI_API_KEY")
CONTACT = os.getenv("SCRAPER_EMAIL", "noreply@example.com")

def esearch(term: str, mindate: str|None=None, maxdate: str|None=None):
    p = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": 0,         # only need count + history
        "usehistory": "y",
        "sort": "date",
    }
    if API_KEY: p["api_key"] = API_KEY
    if mindate: p["mindate"] = mindate
    if maxdate: p["maxdate"] = maxdate
    js = req_json(f"{EUTILS}/esearch.fcgi", p, headers={"User-Agent": f"Orph/1.0 ({CONTACT})"})
    es = js["esearchresult"]
    return int(es["count"]), es.get("webenv"), es.get("querykey")

def esummary_history(webenv: str, qk: str, retstart: int, retmax: int) -> Dict[str, Any]:
    p = {
        "db": "pubmed",
        "retmode": "json",
        "retstart": retstart,
        "retmax": retmax,
        "WebEnv": webenv,
        "query_key": qk,
    }
    if API_KEY: p["api_key"] = API_KEY
    return req_json(f"{EUTILS}/esummary.fcgi", p, headers={"User-Agent": f"Orph/1.0 ({CONTACT})"})

class PubMedScraper(Scraper):
    def __init__(self, out_dir: str, term: str, mindate: str|None, maxdate: str|None, pagesize: int = 500):
        super().__init__(out_dir)
        self.term, self.mindate, self.maxdate = term, mindate, maxdate
        self.pagesize = max(1, min(pagesize, 1000))  # ESummary happy up to ~1k

    def stream(self) -> Iterator[Dict[str, Any]]:
        count, webenv, qk = esearch(self.term, self.mindate, self.maxdate)
        print(f"[pubmed] found {count} ids for term='{self.term}'")
        got = 0
        while got < count:
            meta = esummary_history(webenv, qk, retstart=got, retmax=min(self.pagesize, count - got))
            uids = meta.get("result", {}).get("uids", [])
            for uid in uids:
                rec = meta["result"].get(uid)
                if not rec:
                    continue
                yield {
                    "pmid": uid,
                    "title": rec.get("title"),
                    "pubdate": rec.get("pubdate"),
                    "authors": [a.get("name") for a in rec.get("authors", [])],
                    "journal": rec.get("fulljournalname"),
                    "source": "pubmed_esummary",
                }
            got += len(uids)
            time.sleep(0.34)  # polite pacing

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--term", required=True)
    parser.add_argument("--mindate", default=None)
    parser.add_argument("--maxdate", default=None)
    parser.add_argument("--pagesize", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "pubmed_esummary.jsonl")
    with open(out_path, "a", encoding="utf-8") as f:
        for row in PubMedScraper(args.out, args.term, args.mindate, args.maxdate, args.pagesize).stream():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
