import pandas as pd
import os
import re

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove newlines and carriage returns
    text = text.replace("\n", " ").replace("\r", " ")
    # Remove special characters except space
    text = re.sub(r"[^a-z0-9\s]", "", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_pubmed_file(input_path, output_path):
    print(f"🔍 Reading: {input_path}")
    df = pd.read_csv(input_path)

    required_columns = {"title", "abstract", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Input file must have columns: {required_columns}")

    print("🧼 Cleaning text...")
    df["title_clean"] = df["title"].astype(str).apply(clean_text)
    df["abstract_clean"] = df["abstract"].astype(str).apply(clean_text)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    input_file = "data/raw/pubmed_sample.csv"  # <- Replace with real filename
    output_file = "data/final/pubmed200k_clean.csv"
    clean_pubmed_file(input_file, output_file)
