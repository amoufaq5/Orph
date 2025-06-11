import pandas as pd
import os

RAW_PATH = "data/raw/cardio_train.csv"
SAVE_PATH = "data/cleaned/cardio_cleaned.csv"

def clean_cardio():
    df = pd.read_csv(RAW_PATH, sep=';')

    # Convert age from days to years
    df["age"] = (df["age"] / 365).round(1)

    # Remove outliers
    df = df[(df["height"] >= 100) & (df["height"] <= 250)]
    df = df[(df["weight"] >= 30) & (df["weight"] <= 200)]
    df = df[(df["ap_hi"] > 0) & (df["ap_lo"] > 0)]

    df.to_csv(SAVE_PATH, index=False)
    print(f"✅ Cleaned dataset saved to {SAVE_PATH}")

if __name__ == "__main__":
    clean_cardio()
