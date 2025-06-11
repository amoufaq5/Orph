import pandas as pd
import os

RAW_DIR = "data/raw"
CLEAN_DIR = "data/cleaned"
os.makedirs(CLEAN_DIR, exist_ok=True)

def clean_file(filename):
    path = os.path.join(RAW_DIR, filename)
    df = pd.read_csv(path)

    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    if "age" in df.columns:
        df["age"] = df["age"].apply(lambda x: round(x / 365, 1) if x > 100 else x)

    df = df.dropna()
    clean_path = os.path.join(CLEAN_DIR, f"cleaned_{filename}")
    df.to_csv(clean_path, index=False)
    print(f"✅ Cleaned {filename} -> {clean_path}")

def clean_all():
    for file in ["train.csv", "test.csv"]:
        clean_file(file)

if __name__ == "__main__":
    clean_all()
