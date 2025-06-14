import os
import pandas as pd

# Set up raw file paths (rename your files if needed)
file_paths = {
    "mednli": "data/raw/mednli.csv",
    "drugs": "data/raw/drugs.csv",
    "pubmed_abstracts": "data/raw/pubmed_abstracts.csv",
    "pubmed_200k": "data/raw/pubmed_200k.csv",
    "drug_reviews": "data/raw/drug_reviews.csv"
}

unified = []

def clean(text):
    return ' '.join(''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in str(text)).split())

# Load and format each dataset
def load_mednli():
    df = pd.read_csv(file_paths["mednli"])
    for _, row in df.iterrows():
        unified.append({
            "title": clean(row.get("sentence1")),
            "text": clean(row.get("sentence2")),
            "label": row.get("gold_label"),
            "source": "mednli",
            "type": "labeled"
        })

def load_drugs():
    df = pd.read_csv(file_paths["drugs"])
    for _, row in df.iterrows():
        unified.append({
            "title": clean(row.get("Drug")),
            "text": clean(row.get("Description")),
            "label": row.get("Purpose"),
            "source": "drugs.com",
            "type": "labeled"
        })

def load_pubmed_abstracts():
    df = pd.read_csv(file_paths["pubmed_abstracts"])
    for _, row in df.iterrows():
        unified.append({
            "title": clean(row.get("title")),
            "text": clean(row.get("abstract")),
            "label": None,
            "source": "pubmed",
            "type": "unlabeled"
        })

def load_pubmed_200k():
    df = pd.read_csv(file_paths["pubmed_200k"])
    for _, row in df.iterrows():
        unified.append({
            "title": clean(row.get("title")),
            "text": clean(row.get("abstract")),
            "label": row.get("label"),
            "source": "pubmed_200k",
            "type": "labeled"
        })

def load_drug_reviews():
    df = pd.read_csv(file_paths["drug_reviews"])
    for _, row in df.iterrows():
        unified.append({
            "title": clean(row.get("drugName")),
            "text": clean(row.get("review")),
            "label": row.get("condition"),
            "source": "drug_reviews",
            "type": "labeled"
        })

# Run all loaders
try: load_mednli()
except Exception as e: print("❌ MedNLI:", e)
try: load_drugs()
except Exception as e: print("❌ Drugs.com:", e)
try: load_pubmed_abstracts()
except Exception as e: print("❌ PubMed Abstracts:", e)
try: load_pubmed_200k()
except Exception as e: print("❌ PubMed 200k:", e)
try: load_drug_reviews()
except Exception as e: print("❌ Drug Reviews:", e)

# Save cleaned output
os.makedirs("data/final", exist_ok=True)
df_final = pd.DataFrame(unified)
df_final.to_csv("data/final/unified_nlp_dataset.csv", index=False)
print(f"✅ Final NLP dataset saved to data/final/unified_nlp_dataset.csv with {len(df_final)} rows.")
