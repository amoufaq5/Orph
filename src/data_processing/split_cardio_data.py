import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the cleaned cardio dataset
df = pd.read_csv("data/final/labeled_cardio.csv")

# Separate features and label
X = df.drop(columns=["cardio"])
y = df["cardio"]

# Perform train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save splits
os.makedirs("data/processed", exist_ok=True)
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print(f"✅ Split complete. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
