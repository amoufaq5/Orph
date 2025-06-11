import os
import pandas as pd

raw_folder = "data/raw_excel"
output_folder = "data/raw_csv"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(raw_folder):
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        excel_path = os.path.join(raw_folder, filename)
        try:
            # Load first sheet
            df = pd.read_excel(excel_path, engine="openpyxl")
            csv_name = filename.replace(".xlsx", ".csv").replace(".xls", ".csv")
            output_path = os.path.join(output_folder, csv_name)
            df.to_csv(output_path, index=False)
            print(f"✅ Converted {filename} to CSV.")
        except Exception as e:
            print(f"❌ Failed to convert {filename}: {e}")
