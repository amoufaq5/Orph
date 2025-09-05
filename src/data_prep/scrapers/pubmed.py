# src/data_prep/scrapers/pubmed.py
from __future__ import annotations
import os, sys, time, json
from typing import Iterator, Dict, Any, Optional

# Import guard (works with -m and direct path)
try:
    from .base import Scraper, req_json
except ImportError:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if ROOT not in sys.path: sys.path.insert(0, ROOT)
    from src.data_prep.scrapers.base import Scraper, req_json

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_KEY = os.getenv("NCBI_API_KEY")
CONTACT = os.getenv("SCRAPER_EMAIL", "noreply@example.com")
_DEF_HEADERS = {"User-Agent": f"Orph/1.0 ({CONTACT})"}

def esearch(term: str, mindate: Optional[str] = None, maxdate: Optional[str] = None):
    """Get count + History server (WebEnv/query_key)."""
    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": 0,           # count + history only
        "usehistory": "y",
        "sort": "date",
        "tool": "Orph",
        "email": CONTACT,
    }
    if API_KEY: params["api_key"] = API_KEY
    if mindate: params["mindate"] = mindate
    if maxdate: params["maxdate"] = maxdate
    js = req_json(f"{EUTILS}/esearch.fcgi", params, headers=_DEF_HEADERS)
    es = js["esearchresult"]
    return int(es["count"]), es.get("webenv"), es.get("querykey")

def esummary_history(webenv: str, qk: str, retstart: int, retmax: int) -> Dict[str, Any]:
    """Page ESummary via History (avoids long id= URLs → no 414)."""
    params = {
        "db": "pubmed",
        "version": "2.0",      # cleaner JSON
        "retmode": "json",
        "retstart": retstart,
        "retmax": retmax,
        "WebEnv": webenv,
        "query_key": qk,
        "tool": "Orph",
        "email": CONTACT,
    }
    if API_KEY: params["api_key"] = API_KEY
    return req_json(f"{EUTILS}/esummary.fcgi", params, headers=_DEF_HEADERS)

class PubMedScraper(Scraper):
    def __init__(self, out_dir: str, term: str, mindate: Optional[str], maxdate: Optional[str], pagesize: int = 200):
        super().__init__(out_dir)
        self.term, self.mindate, self.maxdate = term, mindate, maxdate
        # Keep pages modest to reduce malformed payload risk
        self.pagesize = max(50, min(pagesize, 500))

    def stream(self) -> Iterator[Dict[str, Any]]:
        count, webenv, qk = esearch(self.term, self.mindate, self.maxdate)
        print(f"[pubmed] found {count} ids for term='{self.term}'")
        got = 0
        while got < count:
            batch = esummary_history(webenv, qk, retstart=got, retmax=min(self.pagesize, count - got))
            res = batch.get("result", {})
            uids = res.get("uids", [])
            for uid in uids:
                rec = res.get(uid)
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
    # keep --term optional to match some older runners; default keeps prior behavior
    parser.add_argument("--term", default='randomized controlled trial[pt] OR review[pt]')
    parser.add_argument("--mindate", default=None)
    parser.add_argument("--maxdate", default=None)

    # Primary flag
    parser.add_argument("--pagesize", type=int, default=200, help="Records per page (50–500)")
    # Back-compat aliases
    parser.add_argument("--chunk", type=int, dest="pagesize", help="Alias for --pagesize")
    parser.add_argument("--efetch_chunk", type=int, dest="pagesize", help="Alias for --pagesize")

    # Parse but IGNORE unknown old flags like --shard_size, --max_docs
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[pubmed] ignoring unknown args from older runner: {unknown}")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "pubmed_esummary.jsonl")
    with open(out_path, "a", encoding="utf-8") as f:
        for row in PubMedScraper(args.out, args.term, args.mindate, args.maxdate, args.pagesize).stream():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
