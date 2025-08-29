# reporter.py
import argparse, json, math, re, statistics, io
from pathlib import Path
from datetime import datetime

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

# Optional HF imports (only if model available)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except Exception:
    AutoTokenizer = AutoModelForCausalLM = torch = None

# ---------- helpers ----------
def _read_jsonl(path: Path):
    if not path or not path.exists(): return []
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except: pass
    return rows

def _avg(nums):
    nums = [x for x in nums if isinstance(x, (int,float))]
    return (sum(nums) / len(nums)) if nums else 0.0

def _token_count(text, tok):
    if not tok or not text: return 0
    return len(tok(text).input_ids)

def _label_counts(rows, field="output"):
    counts = {}
    for r in rows:
        lab = (r.get(field) or "").strip()
        if lab:
            if len(lab) == 1 and lab.upper() in "ABCDE":
                lab = lab.upper()
            counts[lab] = counts.get(lab, 0) + 1
    return counts

def _task_counts(rows):
    counts = {}
    for r in rows:
        t = r.get("task","unknown")
        counts[t] = counts.get(t,0)+1
    return counts

_RED_FLAG_PATTERNS = [
    r"\bchest pain\b", r"\bshortness of breath\b", r"\bdyspnea\b",
    r"\bfaint(ing)?\b", r"\bconfusion\b", r"\bhemoptysis\b",
    r"\brectal bleeding\b", r"\bvaginal bleeding\b", r"\banaphylaxis\b",
    r"\bsevere\b", r"\bunconscious\b"
]
def _red_flag_rate(rows):
    rx = re.compile("|".join(_RED_FLAG_PATTERNS), re.I)
    total, hits = 0, 0
    for r in rows:
        txt = " ".join(str(r.get(k,"")) for k in ("instruction","input","output"))
        if not txt: continue
        total += 1
        if rx.search(txt): hits += 1
    return {"scanned": total, "matches": hits, "rate": (hits/total if total else 0.0)}

def _summarize_jsonl(path, tok=None, kind="SFT"):
    rows = _read_jsonl(path)
    lens_chars = []
    lens_tokens = []
    has_out = 0
    has_rat = 0
    for r in rows:
        inp = (r.get("input") or "")[:20000]
        out = (r.get("output") or "")
        rat = (r.get("rationale") or "")
        lens_chars.append(len(inp))
        if tok: lens_tokens.append(_token_count(inp, tok))
        if out: has_out += 1
        if rat: has_rat += 1
    info = {
        "path": str(path) if path else "",
        "num_rows": len(rows),
        "avg_input_chars": round(_avg(lens_chars), 1),
        "avg_input_tokens": round(_avg(lens_tokens), 1) if lens_tokens else None,
        "with_output_pct": round((has_out/max(1,len(rows)))*100, 1) if rows else 0.0,
        "with_rationale_pct": round((has_rat/max(1,len(rows)))*100, 1) if rows else 0.0,
        "task_counts": _task_counts(rows),
        "label_counts": _label_counts(rows),
        "red_flags": _red_flag_rate(rows),
        "type": kind
    }
    return info

def _quick_eval(model_dir: Path, eval_jsonl: Path, tokenizer_dir: Path, max_len=256, n_samples=50):
    if not (model_dir and model_dir.exists() and eval_jsonl and eval_jsonl.exists()):
        return {"ok": False, "reason": "model or eval set missing"}
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        return {"ok": False, "reason": "HF transformers not available"}
    try:
        tok = AutoTokenizer.from_pretrained(str(tokenizer_dir or model_dir), use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(str(model_dir)).eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        rows = _read_jsonl(eval_jsonl)[:n_samples]
        # very lightweight sanity: ensure generations run
        gen_ok = 0
        for r in rows:
            prompt = r.get("instruction","") + "\n\n" + (r.get("input","") or "")
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=32, do_sample=False, eos_token_id=tok.eos_token_id)
            gen_ok += 1
        return {"ok": True, "tested": gen_ok, "note": "sanity generation passed"}
    except Exception as e:
        return {"ok": False, "reason": str(e)}

def generate_report(job_id: str, user_id: str, out_dir: Path,
                    sft_path: Path = None, cot_path: Path = None,
                    tokenizer_dir: Path = None, model_dir: Path = None,
                    eval_path: Path = None, config: dict = None):
    out_dir.mkdir(parents=True, exist_ok=True)

    # tokenizer (optional for token stats)
    tok = None
    if tokenizer_dir and tokenizer_dir.exists() and AutoTokenizer is not None:
        try:
            tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
            if tok.pad_token is None and "<pad>" in tok.get_vocab():
                tok.pad_token = "<pad>"
        except Exception:
            tok = None

    sft_info = _summarize_jsonl(sft_path, tok, "SFT") if sft_path else None
    cot_info = _summarize_jsonl(cot_path, tok, "CoT") if cot_path else None
    quick = _quick_eval(model_dir, eval_path or sft_path, tokenizer_dir, n_samples=25) if model_dir else {"ok": False, "reason": "no model_dir"}

    report = {
        "meta": {
            "job_id": job_id,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
        },
        "config": config or {},
        "dataset": {
            "sft": sft_info,
            "cot": cot_info
        },
        "quick_eval": quick
    }

    # Write JSON
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write PDF
    pdf_path = out_dir / "report.pdf"
    _write_pdf(report, pdf_path)

    return {"json": str(json_path), "pdf": str(pdf_path)}

def _row_table(title, data: dict):
    if not data: return []
    table_data = [["Metric", "Value"]]
    for k in ["num_rows","avg_input_chars","avg_input_tokens","with_output_pct","with_rationale_pct"]:
        if k in data and data[k] is not None:
            table_data.append([k, str(data[k])])
    # embed small dicts
    if data.get("task_counts"):
        table_data.append(["task_counts", json.dumps(data["task_counts"])])
    if data.get("label_counts"):
        table_data.append(["label_counts", json.dumps(data["label_counts"])])
    if data.get("red_flags"):
        rf = data["red_flags"]
        table_data.append(["red_flags", f"rate={round(rf.get('rate',0)*100,1)}% ({rf.get('matches',0)}/{rf.get('scanned',0)})"])
    return [Paragraph(f"<b>{title}</b>", getSampleStyleSheet()["Heading3"]),
            Spacer(1, 0.2*cm),
            Table(table_data, style=TableStyle([
                ("GRID",(0,0),(-1,-1),0.25,colors.grey),
                ("BACKGROUND",(0,0),(-1,0),colors.HexColor('#f0f0f0')),
            ])),
            Spacer(1, 0.5*cm)]

def _write_pdf(report: dict, out_path: Path):
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    flow = []
    flow += [Paragraph("<b>Orph Studio – Evaluation Report</b>", styles["Title"]), Spacer(1, 0.3*cm)]
    meta = report.get("meta", {})
    flow += [Paragraph(f"Job ID: {meta.get('job_id','')}", styles["Normal"]),
             Paragraph(f"User ID: {meta.get('user_id','')}", styles["Normal"]),
             Paragraph(f"Created: {meta.get('created_at','')}", styles["Normal"]),
             Spacer(1, 0.5*cm)]
    # Config block
    cfg = json.dumps(report.get("config", {}), ensure_ascii=False)
    flow += [Paragraph("<b>Training Config</b>", styles["Heading3"]),
             Spacer(1, 0.2*cm),
             Paragraph(f"<font name='Courier'>{cfg}</font>", styles["Code"]),
             Spacer(1, 0.5*cm)]
    # Datasets
    ds = report.get("dataset", {})
    if ds.get("sft"): flow += _row_table("SFT Dataset", ds["sft"])
    if ds.get("cot"): flow += _row_table("CoT Dataset", ds["cot"])
    # Quick eval
    q = report.get("quick_eval", {})
    flow += [Paragraph("<b>Quick Evaluation</b>", styles["Heading3"]),
             Spacer(1, 0.2*cm),
             Paragraph(json.dumps(q, ensure_ascii=False), styles["Code"]),
             Spacer(1, 0.5*cm)]
    # Footer note
    flow += [Paragraph("<i>Note:</i> This report provides dataset diagnostics and a lightweight generation sanity check. It is not a clinical validation.", styles["Italic"])]
    doc.build(flow)

# ---------- CLI ----------
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_id", required=True)
    ap.add_argument("--user_id", default="default")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sft_path", default="")
    ap.add_argument("--cot_path", default="")
    ap.add_argument("--tokenizer_dir", default="")
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--eval_path", default="")
    ap.add_argument("--config_json", default="")
    args = ap.parse_args()

    out = Path(args.out_dir)
    cfg = json.loads(args.config_json) if args.config_json else {}

    res = generate_report(
        job_id=args.job_id,
        user_id=args.user_id,
        out_dir=out,
        sft_path=Path(args.sft_path) if args.sft_path else None,
        cot_path=Path(args.cot_path) if args.cot_path else None,
        tokenizer_dir=Path(args.tokenizer_dir) if args.tokenizer_dir else None,
        model_dir=Path(args.model_dir) if args.model_dir else None,
        eval_path=Path(args.eval_path) if args.eval_path else None,
        config=cfg
    )
    print("Wrote:", res)

if __name__ == "__main__":
    _cli()