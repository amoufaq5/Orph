# src/data/pipelines/build_synthetic_supervised.py
import sys, os
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[3]   # goes up from src/data/pipelines/
sys.path.append(str(ROOT))

try:
    from generate_synthetic_cases import generate_cases
except ImportError:
    raise ImportError("Could not import generate_synthetic_cases.py. Make sure it's in project root or src/data/")

    )

CLEAN_DIR = Path("data/clean")
OUT_PATH = CLEAN_DIR / "text_supervised.jsonl"


def case_to_instruction_records(case: dict) -> list[dict]:
    """
    Convert ONE synthetic case into one or more supervised text records.

    Expected keys in `case` (best-effort; adapt to your generator's fields):
      - patient_profile / demographics (optional)
      - symptoms: list[str] or str
      - duration / onset (optional)
      - history / meds (optional)
      - danger_signs: list[str] or str (optional)
      - suspected_diseases: list[str] or str (optional)
      - final_diagnosis: str (optional)
      - recommendation: str  (OTC advice or "refer to doctor")
      - otc_drug / dose / counseling (optional)
      - explanation / rationale (optional)
      - asmethod_answers (optional)

    Output record schema (jsonl):
      {
        "task": "triage|otc|diagnosis|counseling",
        "input": "<concise patient message / structured summary>",
        "target": "<the correct assistant response>",
        "meta": {... free-form, kept for traceability ...}
      }
    """
    records = []

    # Grab fields safely
    symptoms = case.get("symptoms", [])
    if isinstance(symptoms, list):
        symptoms_str = ", ".join(symptoms)
    else:
        symptoms_str = str(symptoms)

    duration = case.get("duration") or case.get("time") or ""
    history = case.get("history") or ""
    meds = case.get("medication") or case.get("current_meds") or ""
    danger = case.get("danger_signs") or case.get("danger") or []
    if isinstance(danger, list):
        danger_str = ", ".join(danger)
    else:
        danger_str = str(danger)

    final_dx = case.get("final_diagnosis") or case.get("diagnosis") or ""
    reco = case.get("recommendation") or ""
    otc = case.get("otc_drug") or case.get("otc") or ""
    dose = case.get("dose") or ""
    counseling = case.get("counseling") or case.get("advice") or ""
    rationale = case.get("explanation") or case.get("rationale") or ""
    asmethod = case.get("asmethod_answers") or {}

    # Create a compact, model-friendly input summary
    summary_lines = [
        f"Symptoms: {symptoms_str}" if symptoms_str else None,
        f"Duration: {duration}" if duration else None,
        f"History: {history}" if history else None,
        f"Current meds: {meds}" if meds else None,
        f"Danger symptoms: {danger_str}" if danger_str else None,
    ]
    summary = "\n".join([s for s in summary_lines if s])

    # 1) Triage record (refer vs self-care)
    if reco:
        records.append({
            "task": "triage",
            "input": f"{summary}\n\nTask: Decide whether to refer to a doctor or provide OTC self-care.",
            "target": reco.strip(),
            "meta": {"source": "synthetic", "rationale": rationale, "asmethod": asmethod, "case": case}
        })

    # 2) Diagnosis record
    if final_dx:
        prompt = f"{summary}\n\nTask: Provide the most likely diagnosis (one line)."
        records.append({
            "task": "diagnosis",
            "input": prompt,
            "target": final_dx.strip(),
            "meta": {"source": "synthetic", "rationale": rationale, "asmethod": asmethod, "case": case}
        })

    # 3) OTC recommendation record
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

    # 4) Counseling-only record (if present)
    if counseling and not otc:
        records.append({
            "task": "counseling",
            "input": f"{summary}\n\nTask: Provide self-care and red-flag counseling.",
            "target": counseling.strip(),
            "meta": {"source": "synthetic", "rationale": rationale, "asmethod": asmethod, "case": case}
        })

    # Fallback: ensure at least one record exists
    if not records:
        # minimal single-turn instruction if generator is sparse
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

    # Generate synthetic cases
    cases = generate_cases(n_cases)
    if not isinstance(cases, list):
        raise ValueError("generate_cases() must return a list of dicts.")

    # Normalize → instruction-tuning records
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for case in cases:
            for rec in case_to_instruction_records(case):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Build supervised text JSONL from synthetic cases.")
    parser.add_argument("--n_cases", type=int, default=2000, help="Number of synthetic base cases to generate.")
    parser.add_argument("--out", type=str, default=str(OUT_PATH), help="Output JSONL path.")
    args = parser.parse_args()

    out_path = Path(args.out)
    total = build_synthetic_jsonl(args.n_cases, out_path)
    print(f"✅ Wrote {total} records to {out_path}")


if __name__ == "__main__":
    main()
