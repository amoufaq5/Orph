# src/data_prep/build_dataset.py
"""
Build the unified Orph training datasets from interim + scraped sources.

Outputs:
  data/clean/text_supervised.jsonl   # instruction-tuning (SFT)
  data/clean/text_cot.jsonl          # chain-of-thought (CoT) when rationale present
  data/clean/stats.json              # counts, dedup, PII heuristic tallies

What it does:
  1) Collects pre-cleaned interim files (from your Kaggle cleaners, etc.)
  2) Folds "scraped" sources into interim/scraped_sft.jsonl (templated instructions)
  3) Validates and normalizes records -> {id, task, instruction, input, output, rationale?, source, split}
  4) De-duplicates by content hash
  5) Simple PII/PHI heuristic flags (counts only; does NOT remove rows)
  6) Random split (train/val/test) if "split" missing
  7) Writes SFT + CoT sets + stats

Usage:
  python src/data_prep/build_dataset.py \
      --seed 42 \
      --train_ratio 0.94 --val_ratio 0.03 --test_ratio 0.03
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# ---------- Paths ----------
PROJ = Path(__file__).resolve().parents[2]
DATA = PROJ / "data"
RAW = DATA / "raw"
SCRAPED = RAW / "scraped"
INTERIM = DATA / "interim"
CLEAN = DATA / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)
INTERIM.mkdir(parents=True, exist_ok=True)

# ---------- Heuristic PII patterns (counts only) ----------
PII_PATTERNS = [
    r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",         # SSN-like
    r"\b\d{10}\b",                                 # 10-digit phone-ish
    r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",  # phone variants
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", # emails
    r"\b(?:MRN|Med Rec|Medical Record Number)[:#]?\s*\w{4,}\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",               # dates mm/dd/yyyy-ish
]
PII_RX = re.compile("|".join(PII_PATTERNS), re.IGNORECASE)

# ---------- Helpers ----------
def _read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n

def _hash_row(instr: str, inp: str, out: str, rat: str = "") -> str:
    m = hashlib.sha256()
    m.update((instr or "").strip().encode("utf-8"))
    m.update(b"\x1f")
    m.update((inp or "").strip().encode("utf-8"))
    m.update(b"\x1f")
    m.update((out or "").strip().encode("utf-8"))
    m.update(b"\x1f")
    m.update((rat or "").strip().encode("utf-8"))
    return m.hexdigest()

def _pii_hits(*texts: str) -> int:
    s = " ".join([t for t in texts if t])
    return len(PII_RX.findall(s))

def _norm_row(r: dict) -> Optional[dict]:
    """Normalize to the canonical schema; return None if invalid/empty."""
    instr = trim(r.get("instruction"))
    inp = trim(r.get("input"))
    out = trim(r.get("output"))
    rat = trim(r.get("rationale"))
    task = r.get("task") or "instruction_tuning"
    source = r.get("source") or "unknown"
    rid = r.get("id") or ""

    # Must have instruction OR input; and output can be empty for SFT (to be generated) but
    # we still keep empty output rows as training "inputs" (common pattern). You can drop if desired.
    if not instr and not inp:
        return None

    return {
        "id": rid,
        "task": task,
        "instruction": instr or "",
        "input": inp or "",
        "output": out or "",
        "rationale": rat or "",
        "source": source,
        "split": r.get("split") or "",  # filled later if missing
    }

def trim(x: Optional[str]) -> str:
    if not x:
        return ""
    x = str(x)
    # truncate pathological long fields to keep JSONL manageable
    return x[:25000].strip()

# ---------- Scraped → Interim folding ----------
def fold_scraped_to_interim() -> Path:
    """
    Takes raw scraped files and converts them to a single interim SFT JSONL.
    Safe to re-run; it overwrites the interim file.
    """
    out_rows: List[dict] = []

    def stream(pname: str) -> Iterable[dict]:
        return _read_jsonl(SCRAPED / pname)

    # MedlinePlus → patient education
    for r in stream("medlineplus.jsonl"):
        text = trim(r.get("summary"))
        if not text:
            continue
        out_rows.append({
            "id": f"medlineplus_{trim(r.get('title'))[:40]}",
            "task": "instruction_tuning",
            "instruction": "Write a clear, layperson-friendly explanation and next steps.",
            "input": text,
            "output": "",
            "source": "MedlinePlus",
            "split": "train",
        })

    # ClinicalTrials → summarize brief summary
    for r in stream("clinicaltrials.jsonl"):
        text = trim(r.get("BriefSummary"))
        if not text:
            continue
        out_rows.append({
            "id": f"ct_{trim(r.get('NCTId'))}",
            "task": "instruction_tuning",
            "instruction": "Summarize the study’s purpose, population, and status in 3 sentences.",
            "input": text,
            "output": "",
            "source": "ClinicalTrials",
            "split": "train",
        })

    # openFDA drug labels → counseling
    for r in stream("openfda_drug_labels.jsonl"):
        sections = []
        for k in [
            "indications_and_usage",
            "dosage_and_administration",
            "warnings",
            "contraindications",
            "adverse_reactions",
        ]:
            v = r.get(k)
            if not v:
                continue
            if isinstance(v, list):
                v = "\n".join([trim(x) for x in v if x])
            sections.append(f"{k.replace('_',' ').title()}:\n{trim(v)}")
        if not sections:
            continue
        out_rows.append({
            "id": f"fda_{trim(r.get('set_id'))}",
            "task": "instruction_tuning",
            "instruction": "Provide patient counseling: indication, typical dosing, and key safety warnings in plain language.",
            "input": "\n\n".join(sections)[:8000],
            "output": "",
            "source": "openFDA",
            "split": "train",
        })

    # PubMed → abstract summarization
    for r in stream("pubmed.jsonl"):
        abst = trim(r.get("abstract"))
        if not abst:
            continue
        out_rows.append({
            "id": f"pm_{trim(r.get('pmid'))}",
            "task": "instruction_tuning",
            "instruction": "Summarize the abstract in 2-3 sentences for a clinical audience.",
            "input": abst,
            "output": "",
            "source": "PubMed",
            "split": "train",
        })

    # PMC Open Access → abstract summarization
    for r in stream("pmc_oa.jsonl"):
        abst = trim(r.get("abstract"))
        if not abst:
            continue
        out_rows.append({
            "id": f"pmc_{trim(r.get('pmcid'))}",
            "task": "instruction_tuning",
            "instruction": "Summarize the findings and clinical implications in 2 sentences.",
            "input": abst,
            "output": "",
            "source": "PMC_OA",
            "split": "train",
        })

    # CDC RSS → public health to layperson
    for r in stream("cdc_rss.jsonl"):
        text = trim(r.get("content") or r.get("summary"))
        if not text:
            continue
        out_rows.append({
            "id": f"cdc_{(trim(r.get('channel'))+'_'+trim(r.get('title')))[:60]}",
            "task": "instruction_tuning",
            "instruction": "Explain the public health notice in simple language and give 3 actionable steps.",
            "input": text,
            "output": "",
            "source": "CDC",
            "split": "train",
        })

    # WHO RSS → clinician summary
    for r in stream("who_rss.jsonl"):
        text = trim(r.get("content") or r.get("summary"))
        if not text:
            continue
        out_rows.append({
            "id": f"who_{(trim(r.get('channel'))+'_'+trim(r.get('title')))[:60]}",
            "task": "instruction_tuning",
            "instruction": "Summarize the key points for clinicians and list recommended precautions.",
            "input": text,
            "output": "",
            "source": "WHO",
            "split": "train",
        })

    # Wikipedia medical portal → explanation + attribution
    for r in stream("wikipedia_med.jsonl"):
        text = trim(r.get("text"))
        if not text:
            continue
        out_rows.append({
            "id": f"wiki_{trim(r.get('title'))[:50]}",
            "task": "instruction_tuning",
            "instruction": "Give a balanced, layperson-friendly explanation and when to seek care. Include a one-line attribution.",
            "input": text,
            "output": f"Source: {trim(r.get('attribution'))} ({trim(r.get('license'))})",
            "source": "Wikipedia",
            "split": "train",
        })

    out_path = INTERIM / "scraped_sft.jsonl"
    _write_jsonl(out_path, out_rows)
    print(f"✅ Folded scraped sources -> {out_path} ({len(out_rows)} rows)")
    return out_path

# ---------- Collect interim files ----------
def discover_interim_inputs() -> List[Path]:
    """
    Finds JSONL files in data/interim/ to unify.
    Includes scraped_sft.jsonl if present (or creates it).
    """
    inputs = []
    # Ensure scraped fold exists (safe to call; will overwrite)
    scraped_path = fold_scraped_to_interim()
    if scraped_path.exists():
        inputs.append(scraped_path)

    # Include any other interim JSONLs produced by your cleaners
    for p in sorted(INTERIM.glob("*.jsonl")):
        if p.name == "scraped_sft.jsonl":
            continue  # already added
        inputs.append(p)
    print(f"🔎 Found {len(inputs)} interim files to unify")
    return inputs

# ---------- Build unified datasets ----------
def unify_and_split(
    paths: List[Path],
    seed: int = 42,
    train_ratio: float = 0.94,
    val_ratio: float = 0.03,
    test_ratio: float = 0.03,
) -> Dict[str, int]:
    rng = random.Random(seed)
    sft_rows: List[dict] = []
    cot_rows: List[dict] = []

    # Collect -> normalize -> dedup
    seen = set()
    stats = {
        "read_rows": 0,
        "kept_rows": 0,
        "deduped": 0,
        "pii_hits": 0,
        "files": {},
    }

    for p in paths:
        cnt = 0
        for r in _read_jsonl(p):
            cnt += 1
            stats["read_rows"] += 1

            nr = _norm_row(r)
            if not nr:
                continue

            h = _hash_row(nr["instruction"], nr["input"], nr["output"], nr.get("rationale", ""))
            if h in seen:
                stats["deduped"] += 1
                continue
            seen.add(h)

            # Assign split if missing
            if not nr["split"]:
                nr["split"] = split_by_hash(h, train_ratio, val_ratio, test_ratio)

            # PII heuristic (count only)
            stats["pii_hits"] += 1 if _pii_hits(nr["instruction"], nr["input"], nr["output"]) else 0

            sft_rows.append(nr)
            if nr.get("rationale"):
                cot_rows.append(nr)

            stats["kept_rows"] += 1

        stats["files"][p.name] = cnt

    # Shuffle each split deterministically for nicer mixing
    sft_rows = stable_shuffle_by_hash(sft_rows)
    cot_rows = stable_shuffle_by_hash(cot_rows)

    # Write outputs
    out_sft = CLEAN / "text_supervised.jsonl"
    out_cot = CLEAN / "text_cot.jsonl"
    n_sft = _write_jsonl(out_sft, sft_rows)
    n_cot = _write_jsonl(out_cot, cot_rows)

    # Persist stats
    stats.update({"sft_rows": n_sft, "cot_rows": n_cot})
    (CLEAN / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Wrote SFT -> {out_sft} ({n_sft} rows)")
    print(f"✅ Wrote CoT -> {out_cot} ({n_cot} rows)")
    print(f"📊 Stats -> {CLEAN / 'stats.json'}")
    return stats

def split_by_hash(h: str, tr: float, vr: float, te: float) -> str:
    """Stable split by hash modulo 1.0."""
    assert abs(tr + vr + te - 1.0) < 1e-6, "ratios must sum to 1"
    frac = int(h[:8], 16) / 0xFFFFFFFF
    if frac < tr:
        return "train"
    if frac < tr + vr:
        return "val"
    return "test"

def stable_shuffle_by_hash(rows: List[dict]) -> List[dict]:
    return sorted(rows, key=lambda r: int(_hash_row(r.get("instruction",""), r.get("input",""), r.get("output",""), r.get("rationale",""))[:12], 16))

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.94)
    ap.add_argument("--val_ratio", type=float, default=0.03)
    ap.add_argument("--test_ratio", type=float, default=0.03)
    return ap.parse_args()

def main():
    args = parse_args()
    paths = discover_interim_inputs()
    unify_and_split(paths, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

if __name__ == "__main__":
    main()
