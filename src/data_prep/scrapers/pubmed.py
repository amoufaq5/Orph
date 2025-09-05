import os, time, math, argparse, xml.etree.ElementTree as ET
import requests
from typing import Iterator, Dict, List
from .base import Scraper, mk_id
from src.utils.logger import get_logger
log = get_logger("pubmed")

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_KEY = os.getenv("5fa3b3391d1cb5dd412e9092373d68385c08")  # optional; improves limits

def _req(url, params, sleep=0.34, tries=5):
    # ~3 req/s without key; be polite
    for t in range(tries):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            if sleep: time.sleep(sleep)
            return r
        time.sleep(1.5 * (t+1))
    r.raise_for_status()

def esearch(term: str, mindate=None, maxdate=None, retmax=10000) -> List[str]:
    params = {
        "db":"pubmed","term":term,"retmax":0,"retmode":"json"
    }
    if API_KEY: params["api_key"]=API_KEY
    if mindate or maxdate:
        params["term"] += f" AND ({mindate or '1800'}:{maxdate or '3000'}[dp])"
    j = _req(f"{EUTILS}/esearch.fcgi", params).json()
    count = int(j["esearchresult"]["count"])
    log.info(f"[pubmed] found {count} ids for term='{term}'")
    ids = []
    for start in range(0, count, retmax):
        p = {"db":"pubmed","retmode":"json","retstart":start,"retmax":retmax}
        if API_KEY: p["api_key"]=API_KEY
        p["term"] = params["term"]
        js = _req(f"{EUTILS}/esearch.fcgi", p).json()
        ids.extend(js["esearchresult"]["idlist"])
    return ids

def efetch_xml(pmid_chunk: List[str]) -> str:
    p = {"db":"pubmed","retmode":"xml","id":",".join(pmid_chunk)}
    if API_KEY: p["api_key"]=API_KEY
    return _req(f"{EUTILS}/efetch.fcgi", p).text

def parse_pubmed_xml(xml_text: str) -> Iterator[Dict]:
    root = ET.fromstring(xml_text)
    for art in root.findall(".//PubmedArticle"):
        # Title
        title = (art.findtext(".//ArticleTitle") or "").strip()
        # Abstract (may be multiple AbstractText nodes)
        abs_parts = [a.text or "" for a in art.findall(".//Abstract/AbstractText")]
        abstract = " ".join(x.strip() for x in abs_parts if x).strip()
        # Date
        year = art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate") or ""
        # DOI
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
            "meta": {"source":"pubmed","pubdate":year,"doi":doi,"license":"public-domain"},
            "split":"train"
        }

class PubMedScraper(Scraper):
    name = "pubmed"
    def __init__(self, out_dir: str, term: str, mindate: str|None, maxdate: str|None, chunk: int):
        super().__init__(out_dir); self.term=term; self.mindate=mindate; self.maxdate=maxdate; self.chunk=chunk

    def stream(self) -> Iterator[Dict]:
        ids = esearch(self.term, self.mindate, self.maxdate)
        for i in range(0, len(ids), self.chunk):
            xml = efetch_xml(ids[i:i+self.chunk])
            for row in parse_pubmed_xml(xml):
                yield row

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--term", default="(clinical trial[pt] OR review[pt])")
    ap.add_argument("--mindate", default=None)  # e.g., 2015
    ap.add_argument("--maxdate", default=None)
    ap.add_argument("--chunk", type=int, default=200)
    args = ap.parse_args()
    PubMedScraper(args.out, args.term, args.mindate, args.maxdate, args.chunk).run()
