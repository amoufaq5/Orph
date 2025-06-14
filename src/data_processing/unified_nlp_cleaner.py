import os
import pandas as pd

# Updated file paths
file_paths = {
    "drugs_com": "data/raw/drugs_side_effects_drugs_com.csv",
    "drug_reviews": "data/raw/drugs.csv",
    "pubmed": "data/raw/PubMed abstracts.csv",
    "symptoms": "data/raw/Final_Augmented_dataset_Diseases_and_Symptoms.csv"
}

unified = []

def clean(text):
    return ' '.join(''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in str(text)).split())

def is_valid(text, label):
    return isinstance(text, str) and isinstance(label, str) and text.strip() != "" and label.strip() != ""

# Loader functions per file
def load_drugs_com():
    df = pd.read_csv(file_paths["drugs_com"])
    for _, row in df.iterrows():
        text = clean(row.get("Description", ""))
        label = str(row.get("Purpose", "")).strip()
        if is_valid(text, label):
            unified.append({
                "title": clean(row.get("Drug", "")),
                "text": text,
                "label": label,
                "source": "drugs.com",
                "type": "labeled"
            })

def load_drug_reviews():
    df = pd.read_csv(file_paths["drug_reviews"])
    for _, row in df.iterrows():
        text = clean(row.get("review", ""))
        label = str(row.get("condition", "")).strip()
        if is_valid(text, label):
            unified.append({
                "title": clean(row.get("drugName", "")),
                "text": text,
                "label": label,
                "source": "drug_reviews",
                "type": "labeled"
            })

def load_pubmed():
    df = pd.read_csv(file_paths["pubmed"])
    for _, row in df.iterrows():
        text = clean(row.get("abstract", ""))
        if isinstance(text, str) and text.strip():
            unified.append({
                "title": clean(row.get("title", "")),
                "text": text,
                "label": None,
                "source": "pubmed",
                "type": "unlabeled"
            })

def load_symptoms():
    df = pd.read_csv(file_paths["symptoms"])
    for _, row in df.iterrows():
        text = clean(row.get("symptoms", ""))
        label = str(row.get("drug name", "")).strip()
        if is_valid(text, label):
            unified.append({
                "title": clean(row.get("disease", "")),
                "text": text,
                "label": label,
                "source": "disease_symptoms",
                "type": "labeled"
            })

# Run all
try: load_drugs_com()
except Exception as e: print("❌ Drugs.com:", e)
try: load_drug_reviews()
except Exception as e: print("❌ Drug Reviews:", e)
try: load_pubmed()
except Exception as e: print("❌ PubMed Abstracts:", e)
try: load_symptoms()
except Exception as e: print("❌ Disease Symptoms:", e)

# Save final result
os.makedirs("data/final", exist_ok=True)
df_final = pd.DataFrame(unified)
df_final.to_csv("data/final/unified_nlp_dataset.csv", index=False)
print(f"✅ Cleaned NLP dataset saved to data/final/unified_nlp_dataset.csv with {len(df_final)} valid rows.")
