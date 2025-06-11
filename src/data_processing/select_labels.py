import pandas as pd

df = pd.read_csv("data/final/merged_dataset.csv")

# Try to find a column that is suitable as label
possible_labels = ["cardio", "target", "diagnosis", "label"]
label_col = None
for col in df.columns:
    if col.lower() in possible_labels:
        label_col = col
        break

if label_col is None:
    raise ValueError("❌ No target/label column found!")

print(f"✅ Using label column: {label_col}")

# Drop rows with missing labels
df = df.dropna(subset=[label_col])

# Save new dataset
df.to_csv("data/final/train_ready.csv", index=False)
print(f"✅ Saved train_ready.csv with {df.shape[0]} rows and {df.shape[1]} columns")
