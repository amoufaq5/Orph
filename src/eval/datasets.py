import json, csv
from typing import Iterable, Dict, List, Tuple

# ---- PubMedQA (yes/no) ----
# Expect JSONL with fields: question, context (optional), answer ("yes"/"no") or "final_decision"
def load_pubmedqa(jsonl_path: str) -> Iterable[Dict]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            j = json.loads(line)
            q = j.get("question") or j.get("ques") or ""
            ctx = j.get("context") or j.get("abstract") or ""
            ans = (j.get("answer") or j.get("final_decision") or "").strip().lower()
            yield {"type":"yn", "question": q, "context": ctx, "answer": ans}

# ---- MedMCQA / MedQA-USMLE (MCQ) ----
# Expect CSV columns: question, opa, opb, opc, opd, cop  (cop in {a,b,c,d})
def load_mcq_csv(csv_path: str) -> Iterable[Dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            q = r.get("question","")
            options = [r.get("opa",""), r.get("opb",""), r.get("opc",""), r.get("opd","")]
            cop = r.get("cop","").strip().lower()
            label_idx = {"a":0,"b":1,"c":2,"d":3}.get(cop, -1)
            yield {"type":"mcq", "question": q, "options": options, "label": label_idx}

# ---- Generic short answer ----
# JSONL: {question, context, answer}
def load_shortqa(jsonl_path: str) -> Iterable[Dict]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            j = json.loads(line)
            yield {"type":"short", "question": j.get("question",""), "context": j.get("context",""), "answer": j.get("answer","")}
