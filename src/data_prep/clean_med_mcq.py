# clean_med_mcq.py
from pathlib import Path
from _utils import write_jsonl, norm_text
import json, csv

RAW_DIRS = [
    Path(r"E:\Orph\data\raw\medmcqa"),
    Path(r"E:\Orph\data\raw\medqa")
]
OUT = r"E:\Orph\data\interim\med_mcq.jsonl"

def main():
    rows = []
    # MedMCQA often ships as CSV
    medmcqa_csv = list(RAW_DIRS[0].glob("*.csv"))
    for f in medmcqa_csv:
        with f.open("r", encoding="utf-8", errors="ignore") as fh:
            r = csv.DictReader(fh)
            for i, d in enumerate(r):
                q = norm_text(d.get("question") or d.get("Question"))
                opts = [norm_text(d.get(k)) for k in ["opa","opb","opc","opd","ope"] if d.get(k)]
                ans = norm_text(d.get("cop") or d.get("answer") or "")
                if not q or not opts or not ans: 
                    continue
                opt_str = "\n".join(f"{chr(65+j)}) {o}" for j,o in enumerate(opts))
                rows.append({
                    "id": f"medmcqa_{f.stem}_{i}",
                    "task": "mcq",
                    "instruction": "Reason step-by-step and select the best option.",
                    "input": f"Q: {q}\nOptions:\n{opt_str}",
                    "output": ans.strip()[0].upper(),  # keep A/B/C/...
                    "source": "MedMCQA",
                    "split": "train"
                })
    # MedQA/USMLE (JSON or CSV variants)
    for f in RAW_DIRS[1].glob("*.*"):
        if f.suffix.lower()==".json":
            data = json.loads(f.read_text(encoding="utf-8", errors="ignore"))
            data = data if isinstance(data, list) else data.get("data", [])
            for i, d in enumerate(data):
                q = norm_text(d.get("question"))
                choices = d.get("choices") or []
                ans = norm_text(d.get("answer") or d.get("correct"))
                if not q or not choices or not ans: 
                    continue
                opt_str = "\n".join(f"{chr(65+j)}) {norm_text(c)}" for j,c in enumerate(choices))
                rows.append({
                    "id": f"medqa_{f.stem}_{i}",
                    "task": "mcq",
                    "instruction": "Reason step-by-step and select the best option.",
                    "input": f"Q: {q}\nOptions:\n{opt_str}",
                    "output": ans.strip()[0].upper(),
                    "source": "MedQA",
                    "split": "train"
                })
    write_jsonl(rows, OUT)
    print(f"✅ Wrote {len(rows)} rows -> {OUT}")

if __name__ == "__main__":
    main()
