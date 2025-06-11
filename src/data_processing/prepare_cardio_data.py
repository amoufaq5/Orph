import pandas as pd
import os

# Input/output paths
input_path = "data/raw/cardio_train.csv"
output_path = "data/final/labeled_cardio.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load data
df = pd.read_csv(input_path, sep=';')

# Drop duplicates if any
df = df.drop_duplicates()

# Drop unrealistic values (optional)
df = df[(df['height'] > 100) & (df['height'] < 250)]
df = df[(df['weight'] > 30) & (df['weight'] < 250)]

# Optional: Normalize or scale features later
df.to_csv(output_path, index=False)
print(f"✅ Cleaned cardio dataset saved to {output_path} with shape {df.shape}")
