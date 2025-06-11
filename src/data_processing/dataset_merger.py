import pandas as pd
import os

input_files = [
    "data/raw/cardio_train.csv",
    "data/raw/train.csv",
    "data/raw/test.csv",
    "data/raw/processed_fr_fr.csv",
    "data/raw/sample_submission.csv",
    "data/raw/submissions.csv"
]

dfs = []
for file in input_files:
    if not os.path.exists(file):
        print(f"❌ File not found: {file}")
        continue

    try:
        df = pd.read_csv(file)
        print(f"✅ Loaded {file} with shape {df.shape}")
        # Add filename source
        df['source_file'] = os.path.basename(file)
        dfs.append(df)
    except Exception as e:
        print(f"⚠️ Failed to read {file}: {e}")

if not dfs:
    raise Exception("❌ No datasets loaded. Cannot proceed.")

# Combine only columns that exist in at least 2 datasets
from collections import Counter
all_cols = sum([list(df.columns) for df in dfs], [])
col_counts = Counter(all_cols)
common_cols = [col for col, count in col_counts.items() if count >= 2]

print(f"📊 Common columns: {common_cols}")

# Filter to only common columns and merge
merged_df = pd.concat([df[common_cols + ['source_file']] for df in dfs if set(common_cols).issubset(df.columns)], ignore_index=True)

# Drop empty rows
merged_df = merged_df.dropna(how='all')
print(f"✅ Merged dataset shape: {merged_df.shape}")

# Save final
output_path = "data/final/merged_dataset.csv"
merged_df.to_csv(output_path, index=False)
print(f"✅ Final merged dataset saved to {output_path}")
