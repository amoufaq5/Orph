# clean_transcriptions.py
from pathlib import Path
from _utils import write_jsonl, norm_text
import csv, json

RAW_DIR = Path(r"E:\Orph\data\raw\transcriptions")
OUT = r"E:\Orph\data\interim\transcripts.jsonl"

def main():
    rows = []
    # Many mirrors provide a CSV with "description" or "medical_specialty" etc.
    for f in list(RAW_DIR.glob("*.csv")) + list(RAW_DIR.glob("*.tsv")):
        delim = "\t" if f.suffix.lower()==".tsv" else ","
        with f.open("r", encoding="utf-8", errors="ignore") as fh:
            r = csv.DictReader(fh, delimiter=delim)
            for i, d in enumerate(r):
                txt = norm_text(d.get("transcription") or d.get("description") or d.get("text"))
                if not txt: 
                    continue
                rows.append({
                    "id": f"mt_{f.stem}_{i}",
                    "task": "instruction_tuning",
                    "instruction": "Respond as the clinician with a concise, empathetic next step.",
                    "input": f"Transcript snippet:\n{txt}",
                    "output": "",
                    "source": "MedicalTranscriptions",
                    "split": "train"
                })
    write_jsonl(rows, OUT)
    print(f"✅ Wrote {len(rows)} rows -> {OUT}")

if __name__ == "__main__":
    main()
