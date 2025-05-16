# dataset_merger.py

import os
import json
import pandas as pd
from pathlib import Path

INPUT_DIR = Path("data/tagged")
OUTPUT_PATH = Path("data/final/merged_dataset.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def unify_entry(entry):
    return {
        "disease": entry.get("disease") or entry.get("title"),
        "symptoms": entry.get("extracted_terms", []),
        "drugs": [entry.get("drug")] if entry.get("drug") else [],
        "side_effects": entry.get("side_effects", []),
        "overview": entry.get("summary") or entry.get("content"),
        "ICD_code": entry.get("ICD_code") or ""
    }


def merge_tagged_data():
    merged_data = []
    for file in INPUT_DIR.glob("*.json"):
        print(f"🔗 Merging {file.name}...")
        data = load_json(file)
        for entry in data:
            unified = unify_entry(entry)
            if any([unified["disease"], unified["symptoms"], unified["drugs"]]):
                merged_data.append(unified)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2)
    print(f"✅ Final merged dataset saved to {OUTPUT_PATH} with {len(merged_data)} entries")


if __name__ == "__main__":
    merge_tagged_data()
