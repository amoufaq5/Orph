"""
PubMed E-utilities (esearch + efetch) to pull abstracts.
Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
Note: Abstract copyrights vary. Prefer PMC Open Access when possible.
"""
import os, math, time
from pathlib import Path
from xml.etree import ElementTree as ET
from ._common import RAW_OUT, write_jsonl, http_get, backoff_sleep

EMAIL = os.getenv("NCBI_EMAIL", "abdulrahman.moufak@gmail.com")          # set to be polite: your@email
API_KEY = os.getenv("NCBI_API_KEY", "7e91b19d102e4b119461ffc2b05cb25a4708")      # optional NCBI key for higher rate
TERM = '(english[Language]) AND (humans[MeSH Terms]) AND (review[Publication Type])'
# For PMC Open Access subset, you can target pmc oa ids via separate APIs later.

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

def esearch(term, retmax=10000):
    params = {
        "db": "pubmed", "term": term, "retmode":"json", "retmax": retmax,
        "email": EMAIL, "api_key": API_KEY
    }
    r = http_get(ESEARCH, params=params)
    j = r.json()
    ids = j["esearchresult"].get("idlist", [])
    return ids

def efetch(ids):
    params = {
        "db": "pubmed", "retmode": "xml",
        "id": ",".join(ids),
        "email": EMAIL, "api_key": API_KEY
    }
    r = http_get(EFETCH, params=params, headers={"Accept":"application/xml"})
    return r.content

def parse_pubmed_xml(xml_bytes):
    root = ET.fromstring(xml_bytes)
    for art in root.findall(".//PubmedArticle"):
        pmid = (art.findtext(".//PMID") or "").strip()
        title = (art.findtext(".//ArticleTitle") or "").strip()
        abst = " ".join([p.text or "" for p in art.findall(".//AbstractText")]).strip()
        journal = (art.findtext(".//Journal/Title") or "").strip()
        year = (art.findtext(".//JournalIssue/PubDate/Year") or "").strip()
        if abst:
            yield {
                "source":"pubmed",
                "pmid": pmid, "title": title, "journal": journal, "year": year,
                "abstract": abst
            }

def main():
    ids = esearch(TERM, retmax=2000)  # adjust as needed
    print(f"Found IDs: {len(ids)}")
    rows = []
    B = 200
    for i in range(0, len(ids), B):
        batch = ids[i:i+B]
        xml = efetch(batch)
        for row in parse_pubmed_xml(xml):
            rows.append(row)
        print(f"Fetched {i+B}/{len(ids)}")
        backoff_sleep(0)
    write_jsonl(RAW_OUT / "pubmed.jsonl", rows)

if __name__ == "__main__":
    main()
