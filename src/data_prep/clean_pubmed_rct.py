# clean_pubmed_rct.py
from pathlib import Path
from _utils import write_jsonl, norm_text
import csv

RAW_DIR = Path(r"E:\Orph\data\raw\pubmed-rct")  # adjust if needed
OUT = r"E:\Orph\data\interim\pubmed_rct.jsonl"

def main():
    rows = []
    # file names vary; typical: 'train.csv', 'dev.csv', 'test.csv'
    for name in ["train.csv", "dev.csv", "test.csv"]:
        f = RAW_DIR / name
        if not f.exists(): 
            continue
        with f.open("r", encoding="utf-8", errors="ignore") as fh:
            r = csv.DictReader(fh)
            for i, d in enumerate(r):
                title = norm_text(d.get("title") or d.get("ArticleTitle"))
                abstract = norm_text(d.get("abstract") or d.get("AbstractText"))
                if not abstract: 
                    continue
                rows.append({
                    "id": f"pubmedrct_{name}_{i}",
                    "task": "instruction_tuning",
                    "instruction": "Summarize the clinical abstract in 2–3 sentences.",
                    "input": f"Title: {title}\nAbstract: {abstract}",
                    "output": "",
                    "source": "PubMed200kRCT",
                    "split": name.replace(".csv","")
                })
    write_jsonl(rows, OUT)
    print(f"✅ Wrote {len(rows)} rows -> {OUT}")

if __name__ == "__main__":
    main()
