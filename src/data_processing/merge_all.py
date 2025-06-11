import pandas as pd
import os

CLEAN_DIR = "data/cleaned"
MERGED_PATH = "data/final/merged_dataset.csv"
os.makedirs("data/final", exist_ok=True)

def merge():
    files = [f for f in os.listdir(CLEAN_DIR) if f.endswith(".csv")]
    dfs = [pd.read_csv(os.path.join(CLEAN_DIR, f)) for f in files]

    merged = pd.concat(dfs, ignore_index=True, sort=False)
    merged.dropna(inplace=True)
    merged.to_csv(MERGED_PATH, index=False)
    print(f"✅ Final merged dataset saved to {MERGED_PATH} with {len(merged)} rows.")

if __name__ == "__main__":
    merge()
