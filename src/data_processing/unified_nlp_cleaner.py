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

# Loader functions per file
def load_drugs_com():
    df = pd.read_csv(file_paths["drugs_com"])
    for _, row in df.iterrows():
        unified.append({
            "title": clean(row.get("Drug", "")),
            "text": clean(row.get("Description", "")),
            "label": row.get("Purpose", ""),
            "source": "drugs.com",
            "type": "labeled"
        })

def load_drug_reviews():
    df = pd.read_csv(file_paths["drug_reviews"])
    for _, row in df.iterrows():
        unified.append({
            "title": clean(row.get("drugName", "")),
            "text": clean(row.get("review", "")),
            "label": row.get("condition", ""),
            "source": "drug_reviews",
            "type": "labeled"
        })

def load_pubmed():
    df = pd.read_csv(file_paths["pubmed"])
    for _, row in df.iterrows():
        unified.append({
            "title": clean(row.get("title", "")),
            "text": clean(row.get("abstract", "")),
            "label": None,
            "source": "pubmed",
            "type": "unlabeled"
        })

def load_symptoms():
    df = pd.read_csv(file_paths["symptoms"])
    for _, row in df.iterrows():
        unified.append({
            "title": clean(row.get("disease", "")),
            "text": clean(row.get("symptoms", "")),
            "label": row.get("drug name", ""),
            "source": "disease_symptoms",
            "type": "labeled"
        })

# Execute loaders
try: load_drugs_com()
except Exception as e: print("❌ Drugs.com:", e)

try: load_drug_reviews()
except Exception as e: print("❌ Drug Reviews:", e)

try: load_pubmed()
except Exception as e: print("❌ PubMed Abstracts:", e)

try: load_symptoms()
except Exception as e: print("❌ Disease Symptoms:", e)

# Save final cleaned dataset
os.makedirs("data/final", exist_ok=True)
df_final = pd.DataFrame(unified)
df_final.to_csv("data/final/unified_nlp_dataset.csv", index=False)
print(f"✅ Final NLP dataset saved to data/final/unified_nlp_dataset.csv with {len(df_final)} rows.")
