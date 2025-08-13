orphtools/preprocessing/text_cleaner.py
import os
import pandas as pd
import re

class TextCleaner:
    def __init__(self, stopwords=None):
        self.stopwords = stopwords or []

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        tokens = [word for word in text.strip().split() if word not in self.stopwords]
        return " ".join(tokens)

    def clean_dataframe(self, df, columns):
        for col in columns:
            if col in df.columns:
                df[col] = df[col].fillna("").apply(self.clean_text)
        return df

    def clean_and_save(self, input_path, output_path, columns_to_clean):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df = pd.read_csv(input_path)
        df = self.clean_dataframe(df, columns_to_clean)
        df.to_csv(output_path, index=False)
        print(f"✅ Cleaned data saved to {output_path}")


# Example usage:
# from orphtools.config_loader import ConfigLoader
# config = ConfigLoader()
# cleaner = TextCleaner(stopwords=["the", "and", "is"])
# cleaner.clean_and_save(
#     input_path=config.get("data", "raw_dir") + "webmd_dataset.csv",
#     output_path=config.get("data", "cleaned_dir") + "webmd_cleaned.csv",
#     columns_to_clean=["overview", "symptoms"]
# )
