# orphtools/preprocessing/dataset_merger.py
import os
import pandas as pd

class DatasetMerger:
    def __init__(self):
        self.columns = ["disease", "drug_name", "symptoms", "interactions", "side_effects", "overview", "source"]

    def standardize_columns(self, df, source_name):
        mapped = {col: col for col in df.columns if col in self.columns}
        df = df.rename(columns=mapped)
        for col in self.columns:
            if col not in df.columns:
                df[col] = ""
        df["source"] = source_name
        return df[self.columns]

    def merge_files(self, input_files, output_file):
        all_dfs = []
        for file_path, source_name in input_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df = self.standardize_columns(df, source_name)
                all_dfs.append(df)
            else:
                print(f"❌ Skipping missing file: {file_path}")

        if not all_dfs:
            print("❌ No data to merge.")
            return

        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df.drop_duplicates(inplace=True)
        merged_df.to_csv(output_file, index=False)
        print(f"✅ Merged dataset saved to {output_file}")


# Example usage:
# merger = DatasetMerger()
# merger.merge_files(
#     input_files=[
#         ("data/cleaned/webmd_cleaned.csv", "webmd"),
#         ("data/cleaned/drugs_com_cleaned.csv", "drugs_com")
#     ],
#     output_file="data/final/merged_dataset.csv"
