# clean_drugs_reviews.py
from pathlib import Path
from _utils import write_jsonl, norm_text
import csv

RAW_DIR = Path(r"E:\Orph\data\raw\drugscom")
OUT = r"E:\Orph\data\interim\drugs_reviews.jsonl"

def main():
    rows = []
    for f in RAW_DIR.glob("*.tsv"):
        with f.open("r", encoding="utf-8", errors="ignore") as fh:
            r = csv.DictReader(fh, delimiter="\t")
            for i, d in enumerate(r):
                drug = norm_text(d.get("drugName"))
                cond = norm_text(d.get("condition"))
                review = norm_text(d.get("review"))
                if not review: 
                    continue
                inp = f"Drug: {drug}\nCondition: {cond}\nPatient review: {review}"
                rows.append({
                    "id": f"drugs_{f.stem}_{i}",
                    "task": "instruction_tuning",
                    "instruction": "Provide educational advice on use and common side effects. Do NOT give diagnosis; advise seeking care for red-flag symptoms.",
                    "input": inp,
                    "output": "",
                    "source": "Drugs.com",
                    "split": "train"
                })
    write_jsonl(rows, OUT)
    print(f"✅ Wrote {len(rows)} rows -> {OUT}")

if __name__ == "__main__":
    main()
