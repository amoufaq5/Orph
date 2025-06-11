import pandas as pd

def clean_processed():
    df = pd.read_csv("data/raw/proccessed_fr_fr.csv")

    df.dropna(inplace=True)
    df.columns = df.columns.str.strip().str.lower()

    df.to_csv("data/cleaned/cleaned_fr.csv", index=False)
    print("✅ Cleaned proccessed_fr_fr.csv saved as cleaned_fr.csv")

if __name__ == "__main__":
    clean_processed()
