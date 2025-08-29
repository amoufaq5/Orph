"""
MedlinePlus Health Topics XML feed -> JSONL
Docs: https://medlineplus.gov/xml.html
"""
from pathlib import Path
import xml.etree.ElementTree as ET
from ._common import RAW_OUT, write_jsonl, http_get

OUT = RAW_OUT / "medlineplus.jsonl"
FEED = "https://medlineplus.gov/xml/mplus_topics_2020-07-01.xml"  # stable sample feed

def parse_topic(node):
    ns = {"m": "http://www.w3.org/1999/xhtml"}
    title = (node.findtext("./title") or "").strip()
    url = (node.findtext("./full-summary/@url") or node.get("url") or "").strip()
    summary = (node.findtext("./full-summary") or "").strip()
    category = (node.findtext("./groupname") or "").strip()
    return {"source":"medlineplus","title":title,"url":url,"summary":summary,"category":category}

def main():
    r = http_get(FEED)
    root = ET.fromstring(r.content)
    rows = []
    for topic in root.findall(".//topic"):
        rows.append(parse_topic(topic))
    write_jsonl(OUT, rows)

if __name__ == "__main__":
    main()
