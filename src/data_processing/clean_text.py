# clean_text.py

import os
import json
import re
from pathlib import Path

CLEANED_PATH = Path("data/cleaned")
CLEANED_PATH.mkdir(parents=True, exist_ok=True)

RE_HTML_TAGS = re.compile(r'<.*?>')
RE_MULTISPACE = re.compile(r'\s+')


def clean_text(text):
    text = re.sub(RE_HTML_TAGS, '', text)
    text = re.sub(RE_MULTISPACE, ' ', text)
    return text.strip()


def clean_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        for key in entry:
            if isinstance(entry[key], str):
                entry[key] = clean_text(entry[key])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def clean_and_save(input_dir="data/raw", output_dir="data/cleaned"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for file in input_dir.glob("*.json"):
        print(f"🧹 Cleaning {file.name}...")
        output_file = output_dir / file.name
        clean_file(file, output_file)
    print("✅ Text cleaning complete. Cleaned files saved to /data/cleaned")


if __name__ == "__main__":
    clean_and_save()
