import os
import pandas as pd

# Define folder paths
RAW_FOLDER = "data/raw/"
FINAL_FOLDER = "data/final/"

# Supported CSV dataset filenames (adjust as needed)
DATASETS = [
    "drugs_side_effects_drugs_com.csv",
    "drugs.csv",
    "PubMed_abstracts.csv",
    "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
]

# Output format for unified dataset
UNIFIED_COLUMNS = ["title", "text", "label", "source", "type"]


def standardize_dataset(path: str, source: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        df.columns = [col.strip().lower() for col in df.columns]

        # Handle various structures and fallback to defaults
        title = df.get("title") or df.get("drug") or df.get("disease") or ["no title"] * len(df)
        text = df.get("text") or df.get("description") or df.get("content") or df.get("abstract") or df.get("symptoms")
        label = df.get("label") or df.get("category") or df.get("disease") or ["unknown"] * len(df)

        df_clean = pd.DataFrame({
            "title": title,
            "text": text,
            "label": label,
            "source": source,
            "type": "labeled" if "label" in df.columns else "unlabeled"
        })

        df_clean.dropna(subset=["text"], inplace=True)
        return df_clean
    except Exception as e:
        print(f"❌ Error processing {path}: {e}")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)


if __name__ == "__main__":
    os.makedirs(FINAL_FOLDER, exist_ok=True)
    merged = pd.DataFrame(columns=UNIFIED_COLUMNS)

    for file in DATASETS:
        source_name = file.split(".")[0].replace("_", " ")
        full_path = os.path.join(RAW_FOLDER, file)
        if os.path.exists(full_path):
            print(f"✅ Processing: {file}")
            df = standardize_dataset(full_path, source_name)
            merged = pd.concat([merged, df], ignore_index=True)
        else:
            print(f"⚠️ Skipped (not found): {file}")

    # Drop duplicates and save
    merged.drop_duplicates(subset=["text", "label"], inplace=True)
    merged.to_csv(os.path.join(FINAL_FOLDER, "unified_nlp_dataset.csv"), index=False)
    print(f"\n✅ Saved final merged dataset to {FINAL_FOLDER}/unified_nlp_dataset.csv with shape {merged.shape}")
