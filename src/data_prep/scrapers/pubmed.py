# src/data_prep/scrapers/pubmed.py
from __future__ import annotations
import os, argparse, xml.etree.ElementTree as ET
from typing import Iterator, Dict, List, Optional

from .base import Scraper, HttpClient, RateLimiter, mk_id
from src.utils.logger import get_logger
log = get_logger("pubmed")

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_KEY = os.getenv("NCBI_API_KEY")  # optional → higher rate limits

def _rate_limiter() -> RateLimiter:
    # NCBI guideline: ~3 req/s without key, ~10 req/s with key
    cps = 10.0 if API_KEY else 3.0
    return RateLimiter(calls_per_sec=cps, burst=3)

def esearch_ids(client: HttpClient, rl: RateLimiter, term: str, mindate: Optional[str], maxdate: Optional[str], chunk: int = 10000) -> List[str]:
    q = term
    if mindate or maxdate:
        q += f" AND ({mindate or '1800'}:{maxdate or '3000'}[dp])"
    params = {"db": "pubmed", "term": q, "retmode": "json", "retmax": 0}
    if API_KEY: params["api_key"] = API_KEY
    rl.sleep()
    js = client.json(f"{EUTILS}/esearch.fcgi", params=params)
    count = int(js["esearchresult"]["count"])
    log.info(f"[pubmed] matched {count} PMIDs for: {term}")
    ids = []
    for start in range(0, count, chunk):
        rl.sleep()
        p = {"db":"pubmed","term":q,"retmode":"json","retstart":start,"retmax":chunk}
        if API_KEY: p["api_key"] = API_KEY
        page = client.json(f"{EUTILS}/esearch.fcgi", params=p)
        ids.extend(page["esearchresult"]["idlist"])
    return ids

def efetch_xml(client: HttpClient, rl: RateLimiter, pmid_chunk: List[str]) -> str:
    rl.sleep()
    p = {"db":"pubmed","retmode":"xml","id":",".join(pmid_chunk)}
    if API_KEY: p["api_key"] = API_KEY
    return client.text(f"{EUTILS}/efetch.fcgi", params=p)

def parse_pubmed_xml(xml_text: str) -> Iterator[Dict]:
    root = ET.fromstring(xml_text)
    for art in root.findall(".//PubmedArticle"):
        title = (art.findtext(".//ArticleTitle") or "").strip()
        abs_parts = [a.text or "" for a in art.findall(".//Abstract/AbstractText")]
        abstract = " ".join(x.strip() for x in abs_parts if x).strip()
        # year can also be MedlineDate like "2018 Jan"
        year = (art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate") or "").strip()
        doi = ""
        for idn in art.findall(".//ArticleIdList/ArticleId"):
            if idn.attrib.get("IdType") == "doi":
                doi = (idn.text or "").strip()
        if not (title or abstract):
            continue
        yield {
            "id": mk_id("pubmed"),
            "modality": ["text"],
            "task": "summarize",
            "text": f"{title} — {abstract}",
            "answer": None,
            "rationale": None,
            "labels": {},
            "meta": {"source":"pubmed","pubdate": year, "doi": doi, "license":"public-domain"},
            "split": "train"
        }

class PubMedScraper(Scraper):
    name = "pubmed"

    def __init__(self, out_dir: str, term: str, mindate: Optional[str], maxdate: Optional[str], chunk: int, shard_size: int, max_docs: Optional[int]):
        rl = _rate_limiter()
        client = HttpClient(timeout=60)
        super().__init__(out_dir, client=client, shard_size=shard_size, max_docs=max_docs)
        self.term, self.mindate, self.maxdate, self.efetch_chunk = term, mindate, maxdate, chunk
        self.rl = rl

    def stream(self) -> Iterator[Dict]:
        ids = esearch_ids(self.client, self.rl, self.term, self.mindate, self.maxdate)
        for i in range(0, len(ids), self.efetch_chunk):
            xml = efetch_xml(self.client, self.rl, ids[i:i+self.efetch_chunk])
            yield from parse_pubmed_xml(xml)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--term", default="(clinical trial[pt] OR review[pt])")
    ap.add_argument("--mindate", default=None)
    ap.add_argument("--maxdate", default=None)
    ap.add_argument("--efetch_chunk", type=int, default=200)
    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--max_docs", type=int, default=None)
    args = ap.parse_args()
    PubMedScraper(args.out, args.term, args.mindate, args.maxdate, args.efetch_chunk, args.shard_size, args.max_docs).run()
