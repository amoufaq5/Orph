# src/data_prep/scrapers/pubmed.py
import os, time
from typing import Iterator, Dict, Any, List
from .base import Scraper, req_json, DEFAULT_USER_AGENT

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_KEY = os.getenv("NCBI_API_KEY")
CONTACT = os.getenv("SCRAPER_EMAIL", "noreply@example.com")

def esearch(term: str, mindate: str | None = None, maxdate: str | None = None,
            retstart: int = 0, retmax: int = 0) -> tuple[list[str], int, str | None, str | None]:
    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retstart": retstart,
        "retmax": retmax,       # use 0 when you only need count + history
        "usehistory": "y",
        "sort": "date",
    }
    if API_KEY: params["api_key"] = API_KEY
    if mindate: params["mindate"] = mindate
    if maxdate: params["maxdate"] = maxdate

    js = req_json(f"{EUTILS}/esearch.fcgi", params, headers={"User-Agent": f"Orph/1.0 ({CONTACT})"})
    es = js["esearchresult"]
    ids = es.get("idlist", [])
    count = int(es["count"])
    return ids, count, es.get("webenv"), es.get("querykey")

def iter_all_ids(term: str, mindate: str | None = None, maxdate: str | None = None,
                 step: int = 10000) -> Iterator[List[str]]:
    _, count, webenv, qk = esearch(term, mindate, maxdate, retmax=0)
    got = 0
    while got < count:
        params = {
            "db": "pubmed",
            "retmode": "json",
            "retstart": got,
            "retmax": min(step, count - got),
            "usehistory": "y",
            "query_key": qk,
            "WebEnv": webenv,
        }
        if API_KEY: params["api_key"] = API_KEY
        js = req_json(f"{EUTILS}/esearch.fcgi", params, headers={"User-Agent": f"Orph/1.0 ({CONTACT})"})
        batch = js["esearchresult"].get("idlist", [])
        if not batch:
            break
        yield batch
        got += len(batch)

def efetch(pmids: List[str]) -> Dict[str, Any]:
    # Use JSON where possible; EFetch for PubMed primarily returns XML,
    # but we can request JSON via 'retmode=json' on ESummary for metadata,
    # then EFetch for details as XML if you need full MEDLINE (handle separately).
    # Here we’ll use ESummary (JSON) for robustness.
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
        "retmax": len(pmids),
    }
    if API_KEY: params["api_key"] = API_KEY
    return req_json(f"{EUTILS}/esummary.fcgi", params, headers={"User-Agent": f"Orph/1.0 ({CONTACT})"})

class PubMedScraper(Scraper):
    def __init__(self, out_dir: str, term: str, mindate: str | None, maxdate: str | None, chunk: int = 500):
        super().__init__(out_dir)
        self.term = term
        self.mindate = mindate
        self.maxdate = maxdate
        self.chunk = max(1, min(chunk, 1000))

    def stream(self) -> Iterator[Dict[str, Any]]:
        # Log once: count
        _, count, _, _ = esearch(self.term, self.mindate, self.maxdate, retmax=0)
        print(f"[pubmed] found {count} ids for term='{self.term}'")

        for id_batch in iter_all_ids(self.term, self.mindate, self.maxdate, step=10000):
            # Fetch in manageable chunks for ESummary
            for i in range(0, len(id_batch), self.chunk):
                pmids = id_batch[i:i+self.chunk]
                meta = efetch(pmids)
                # Write or yield records; adapt to your writer
                uids = meta.get("result", {}).get("uids", [])
                for uid in uids:
                    rec = meta["result"].get(uid)
                    if not rec:
                        continue
                    # Example normalization (adapt to your schema)
                    row = {
                        "pmid": uid,
                        "title": rec.get("title"),
                        "pubdate": rec.get("pubdate"),
                        "authors": [a.get("name") for a in rec.get("authors", [])],
                        "journal": rec.get("fulljournalname"),
                        "source": "pubmed_esummary",
                    }
                    # Your writer should dump rows to JSONL here.
                    # For compatibility with your existing pipeline, you might have a write_jsonl(out_path, row).
                    yield row
                # polite pacing between ESummary calls
                time.sleep(0.34)

if __name__ == "__main__":
    import argparse, json, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--term", required=True)
    parser.add_argument("--mindate", default=None)
    parser.add_argument("--maxdate", default=None)
    parser.add_argument("--chunk", type=int, default=500)
    args = parser.parse_args()

    scr = PubMedScraper(args.out, args.term, args.mindate, args.maxdate, args.chunk)
    out_path = os.path.join(args.out, "pubmed_esummary.jsonl")
    os.makedirs(args.out, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for row in scr.stream():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
