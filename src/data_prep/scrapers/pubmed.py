# src/data_prep/scrapers/pubmed.py
from __future__ import annotations
import os, sys, time, json
from typing import Iterator, Dict, Any, List

# import guard to allow -m and direct
try:
    from .base import Scraper, req_json
except ImportError:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if ROOT not in sys.path: sys.path.insert(0, ROOT)
    from src.data_prep.scrapers.base import Scraper, req_json

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_KEY = os.getenv("NCBI_API_KEY")
CONTACT = os.getenv("SCRAPER_EMAIL", "noreply@example.com")

def esearch(term: str, mindate: str | None = None, maxdate: str | None = None,
            retstart: int = 0, retmax: int = 0):
    p = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retstart": retstart,
        "retmax": retmax,    # 0 => only count + history
        "usehistory": "y",
        "sort": "date"
    }
    if API_KEY: p["api_key"] = API_KEY
    if mindate: p["mindate"] = mindate
    if maxdate: p["maxdate"] = maxdate
    js = req_json(f"{EUTILS}/esearch.fcgi", p, headers={"User-Agent": f"Orph/1.0 ({CONTACT})"})
    es = js["esearchresult"]
    return es.get("idlist", []), int(es["count"]), es.get("webenv"), es.get("querykey")

def iter_all_ids(term: str, mindate: str | None = None, maxdate: str | None = None, step: int = 10000):
    _, count, webenv, qk = esearch(term, mindate, maxdate, retmax=0)
    print(f"[pubmed] found {count} ids for term='{term}'")
    got = 0
    while got < count:
        p = {
            "db": "pubmed",
            "retmode": "json",
            "retstart": got,
            "retmax": min(step, count - got),
            "usehistory": "y",
            "query_key": qk,
            "WebEnv": webenv,
        }
        if API_KEY: p["api_key"] = API_KEY
        js = req_json(f"{EUTILS}/esearch.fcgi", p, headers={"User-Agent": f"Orph/1.0 ({CONTACT})"})
        ids = js["esearchresult"].get("idlist", [])
        if not ids: break
        yield ids
        got += len(ids)

def esummary(pmids: List[str]) -> Dict[str, Any]:
    p = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
        "retmax": len(pmids),
    }
    if API_KEY: p["api_key"] = API_KEY
    return req_json(f"{EUTILS}/esummary.fcgi", p, headers={"User-Agent": f"Orph/1.0 ({CONTACT})"})

class PubMedScraper(Scraper):
    def __init__(self, out_dir: str, term: str, mindate: str | None, maxdate: str | None, chunk: int = 500):
        super().__init__(out_dir)
        self.term, self.mindate, self.maxdate = term, mindate, maxdate
        self.chunk = max(1, min(chunk, 1000))

    def stream(self) -> Iterator[Dict[str, Any]]:
        for id_batch in iter_all_ids(self.term, self.mindate, self.maxdate, step=10000):
            for i in range(0, len(id_batch), self.chunk):
                pmids = id_batch[i:i+self.chunk]
                meta = esummary(pmids)
                uids = meta.get("result", {}).get("uids", [])
                for uid in uids:
                    rec = meta["result"].get(uid)
                    if not rec: continue
                    yield {
                        "pmid": uid,
                        "title": rec.get("title"),
                        "pubdate": rec.get("pubdate"),
                        "authors": [a.get("name") for a in rec.get("authors", [])],
                        "journal": rec.get("fulljournalname"),
                        "source": "pubmed_esummary",
                    }
                time.sleep(0.34)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--term", required=True)
    parser.add_argument("--mindate", default=None)
    parser.add_argument("--maxdate", default=None)
    parser.add_argument("--chunk", type=int, default=500)
    args = parser.parse_args()

    out_path = os.path.join(args.out, "pubmed_esummary.jsonl")
    os.makedirs(args.out, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for row in PubMedScraper(args.out, args.term, args.mindate, args.maxdate, args.chunk).stream():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
