# clean_pubmedqa.py
from pathlib import Path
from _utils import write_jsonl, norm_text
import json

RAW_DIR = Path(r"E:\Orph\data\raw\pubmedqa")
OUT = r"E:\Orph\data\interim\pubmedqa.jsonl"

def load_json(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)

def main():
    rows = []
    for name in ["pubmedqa_labeled.json", "pubmedqa.json", "data.json"]:
        f = RAW_DIR / name
        if not f.exists(): 
            continue
        data = load_json(f)
        # different mirrors vary; adapt to common fields
        items = data if isinstance(data, list) else data.get("data") or data.get("examples") or []
        for i, d in enumerate(items):
            q = norm_text(d.get("question") or d.get("QUESTION"))
            ctx = norm_text(d.get("context") or d.get("CONTEXT") or d.get("abstract"))
            ans = norm_text(d.get("final_decision") or d.get("answer") or "")
            if not q or not ctx: 
                continue
            rows.append({
                "id": f"pubmedqa_{i}",
                "task": "instruction_tuning",
                "instruction": "Answer the clinical question based on the abstract.",
                "input": f"Question: {q}\nAbstract: {ctx}",
                "output": ans,
                "source": "PubMedQA",
                "split": "train"
            })
    write_jsonl(rows, OUT)
    print(f"✅ Wrote {len(rows)} rows -> {OUT}")

if __name__ == "__main__":
    main()
