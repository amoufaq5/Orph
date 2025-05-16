# tag_symptoms.py

import os
import json
import spacy
from pathlib import Path

# Load scispaCy model (run `pip install scispacy && pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz`)
nlp = spacy.load("en_core_sci_sm")

TAGGED_PATH = Path("data/tagged")
TAGGED_PATH.mkdir(parents=True, exist_ok=True)

TARGET_KEYS = ["summary", "content"]

def extract_medical_terms(text):
    doc = nlp(text)
    return list(set(ent.text for ent in doc.ents if ent.label_ != ""))

def tag_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        text_blob = " ".join(entry.get(k, "") for k in TARGET_KEYS if k in entry)
        terms = extract_medical_terms(text_blob)
        entry["extracted_terms"] = terms

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def tag_all_cleaned(input_dir="data/cleaned", output_dir="data/tagged"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for file in input_dir.glob("*.json"):
        print(f"🏷️ Tagging {file.name}...")
        output_file = output_dir / file.name
        tag_file(file, output_file)

    print("✅ Tagging complete. Saved to /data/tagged")


if __name__ == "__main__":
    tag_all_cleaned()
