# clean_mednli.py
from pathlib import Path
from _utils import write_jsonl, norm_text
import json

RAW_DIR = Path(r"E:\Orph\data\raw\mednli")
OUT = r"E:\Orph\data\interim\mednli.jsonl"

def load(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [json.loads(l) for l in f if l.strip()]

def main():
    rows = []
    for name in ["mli_train_v1.jsonl", "mli_dev_v1.jsonl", "mli_test_v1.jsonl"]:
        f = RAW_DIR / name
        if not f.exists(): 
            continue
        data = load(f)
        split = "train" if "train" in name else "dev" if "dev" in name else "test"
        for i, d in enumerate(data):
            premise = norm_text(d.get("sentence1") or d.get("premise"))
            hypo = norm_text(d.get("sentence2") or d.get("hypothesis"))
            label = d.get("gold_label") or d.get("label") or ""
            if not premise or not hypo: 
                continue
            rows.append({
                "id": f"mednli_{split}_{i}",
                "task": "instruction_tuning",
                "instruction": "Does the hypothesis follow from the premise? Reply with entailment, neutral, or contradiction.",
                "input": f"Premise: {premise}\nHypothesis: {hypo}",
                "output": label,
                "source": "MedNLI",
                "split": split
            })
    write_jsonl(rows, OUT)
    print(f"✅ Wrote {len(rows)} rows -> {OUT}")

if __name__ == "__main__":
    main()
