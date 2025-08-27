# make_cot_from_templates.py
import argparse, json, random
from pathlib import Path
from _utils import write_jsonl

TEMPLATES = [
"Step 1: Identify key symptoms and clues.\nStep 2: Match each clue with likely differentials.\nStep 3: Eliminate options contradicting findings.\nTherefore, the best answer is: {ans}.",
"Consider epidemiology and risk factors, then pathophysiology and hallmark signs. Weigh options and select the most consistent answer: {ans}.",
"Given the scenario, evaluate red flags, common vs rare causes, and contraindications. The most likely correct choice is: {ans}."
]

def parse_options(text):
    # expects block like:
    # Options:
    # A) ...
    # B) ...
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    opts = [l for l in lines if l[0:2] in ("A)", "B)", "C)", "D)", "E)")]
    return opts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Comma-separated MCQ jsonl files (from clean_med_mcq.py)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = []
    for path in args.inputs.split(","):
        with open(path.strip(), "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip(): 
                    continue
                row = json.loads(line)
                if row.get("task") != "mcq": 
                    continue
                ans = row.get("output", "").strip()[:1].upper()
                if ans not in {"A","B","C","D","E"}:
                    continue
                rationale = random.choice(TEMPLATES).format(ans=ans)
                out.append({
                    "id": row["id"],
                    "task": "cot_supervision",
                    "instruction": "Reason step-by-step and answer.",
                    "input": row["input"],
                    "rationale": rationale,
                    "output": ans,
                    "source": row.get("source","MCQ"),
                    "split": row.get("split","train")
                })
    write_jsonl(out, args.output)
    print(f"✅ Wrote {len(out)} CoT rows -> {args.output}")

if __name__ == "__main__":
    main()
