# src/data/pipelines/build_synthetic_supervised.py
import os
import json
import argparse
from pathlib import Path

# Import from within the src package tree
# Requires: E:\Orph\src\data\generate_synthetic_cases.py
try:
    from src.data.generate_synthetic_cases import generate_cases  # preferred
except ImportError as e:
    # Some older versions used a different name for the function
    try:
        from src.data.generate_synthetic_cases import generate_synthetic_cases as generate_cases  # type: ignore
    except Exception:
        raise ImportError(
            "Could not import generate_cases from src.data.generate_synthetic_cases. "
            "Make sure generate_synthetic_cases.py exists under src/data/ and defines "
            "either `generate_cases(n: int) -> list[dict]` or "
            "`generate_synthetic_cases(n: int) -> list[dict]`."
        ) from e

CLEAN_DIR = Path("data/clean")
OUT_PATH = CLEAN_DIR / "text_supervised.jsonl"


def case_to_instruction_records(case: dict) -> list[dict]:
    """
    Convert ONE synthetic case dict to one or more instruction-tuning records.

    Expected (best-effort) keys in `case`:
      - symptoms: list[str] or str
      - duration/time, history, (current_)meds, danger_signs
      - final_diagnosis/diagnosis
      - recommendation (OTC plan or refer)
      - otc_drug/otc, dose, counseling/advice
      - explanation/rationale
      - asmethod_answers

    Output JSONL record schema:
      {
        "task": "triage|otc|diagnosis|counseling",
        "input": "<concise patient summary + explicit task>",
        "target": "<gold response text>",
        "meta": {... free-form audit info ...}
      }
    """
    records = []

    # Safe field extraction
    symptoms = case.get("symptoms", [])
    if isinstance(symptoms, list):
        symptoms_str = ", ".join(symptoms)
    else:
        symptoms_str = str(symptoms) if symptoms else ""

    duration = case.get("duration") or case.get("time") or ""
    history = case.get("history") or ""
    meds = case.get("medication") or case.get("current_meds") or ""
    danger = case.get("danger_signs") or case.get("danger") or []
    if isinstance(danger, list):
        danger_str = ", ".join(danger)
    else:
        danger_str = str(danger) if danger else ""

    final_dx = case.get("final_diagnosis") or case.get("diagnosis") or ""
    reco = case.get("recommendation") or ""
    otc = case.get("otc_drug") or case.get("otc") or ""
    dose = case.get("dose") or ""
    counseling = case.get("counseling") or case.get("advice") or ""
    rationale = case.get("explanation") or case.get("rationale") or ""
    asmethod = case.get("asmethod_answers") or {}

    # Compact summary used in all prompts
    summary_lines = [
        f"Symptoms: {symptoms_str}" if symptoms_str else None,
        f"Duration: {duration}" if duration else None,
        f"History: {history}" if history else None,
        f"Current meds: {meds}" if meds else None,
        f"Danger symptoms: {danger_str}" if danger_str else None,
    ]
    summary = "\n".join([s for s in summary_lines if s])

    # 1) Triage (refer vs OTC)
    if reco:
        records.append({
            "task": "triage",
            "input": f"{summary}\n\nTask: Decide whether to refer to a doctor or provide OTC self-care.",
            "target": reco.strip(),
            "meta": {"source": "synthetic", "rationale": rationale, "asmethod": asmethod, "case": case}
        })

    # 2) Diagnosis (single line)
    if final_dx:
        records.append({
            "task": "diagnosis",
            "input": f"{summary}\n\nTask: Provide the most likely diagnosis (one line).",
            "target": final_dx.strip(),
            "meta": {"source": "synthetic", "rationale": rationale, "asmethod": asmethod, "case": case}
        })

    # 3) OTC recommendation (drug + dose + counseling)
    if otc or counseling:
        rec_lines = []
        if otc:
            rec_lines.append(f"OTC: {otc}")
        if dose:
            rec_lines.append(f"Dose: {dose}")
        if counseling:
            rec_lines.append(f"Advice: {counseling}")
        target = "\n".join(rec_lines) if rec_lines else "Provide safe OTC guidance."

        records.append({
            "task": "otc",
            "input": f"{summary}\n\nTask: Provide OTC recommendation with dosage and cautions.",
            "target": target.strip(),
            "meta": {"source": "synthetic", "rationale": rationale, "asmethod": asmethod, "case": case}
        })

    # 4) Counseling-only (if present without explicit OTC)
    if counseling and not otc:
        records.append({
            "task": "counseling",
            "input": f"{summary}\n\nTask: Provide self-care and red-flag counseling.",
            "target": counseling.strip(),
            "meta": {"source": "synthetic", "rationale": rationale, "asmethod": asmethod, "case": case}
        })

    # Fallback to ensure at least one example per case
    if not records:
        fallback_target = reco or final_dx or "Provide safe self-care guidance or refer if necessary."
        records.append({
            "task": "triage",
            "input": summary if summary else "A patient describes symptoms. Decide next steps.",
            "target": fallback_target,
            "meta": {"source": "synthetic", "case": case}
        })

    return records


def build_synthetic_jsonl(n_cases: int, out_path: Path = OUT_PATH) -> int:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    cases = generate_cases(n_cases)
    if not isinstance(cases, list):
        raise ValueError("generate_cases() must return a list[dict].")

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for case in cases:
            for rec in case_to_instruction_records(case):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Build supervised text JSONL from synthetic cases.")
    parser.add_argument("--n_cases", type=int, default=2000, help="Number of synthetic cases to generate.")
    parser.add_argument("--out", type=str, default=str(OUT_PATH), help="Output JSONL path.")
    args = parser.parse_args()

    out_path = Path(args.out)
    total = build_synthetic_jsonl(args.n_cases, out_path)
    print(f"✅ Wrote {total} records to {out_path}")


if __name__ == "__main__":
    main()
