import os
import pandas as pd

# File paths (make sure these match actual file names)
paths = {
    "side_effects": "data/raw/drugs_side_effects_drugs_com.csv",
    "openfda_drugs": "data/raw/drugs.csv",
    "pubmed_rct": "data/raw/pubmed_200k.csv",
    "symptoms_bool": "data/raw/final_augmented_dataset_diseases_and_symptoms.csv",
    "symptoms_list": "data/raw/disease_symptom_pairs.csv",
    "pubmed_abstracts": "data/raw/pubmed_abstracts.csv"
}

unified = []

def clean(text):
    return ' '.join(''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in str(text)).split())

def is_valid(text, label):
    return isinstance(text, str) and isinstance(label, str) and text.strip() != "" and label.strip() != ""

# 1. Side Effects & Conditions (drugs.com-like)
def load_drugs_side_effects():
    try:
        df = pd.read_csv(paths["side_effects"])
        for _, row in df.iterrows():
            text = clean(row.get("side_effects", ""))
            label = row.get("medical_condition", "")
            if is_valid(text, label):
                unified.append({
                    "title": clean(row.get("drug_name", "")),
                    "text": text,
                    "label": label,
                    "source": "drugs.com",
                    "type": "labeled"
                })
    except Exception as e:
        print("❌ drugs_side_effects:", e)

# 2. OpenFDA structured drugs list
def load_openfda_drugs():
    try:
        df = pd.read_csv(paths["openfda_drugs"])
        for _, row in df.iterrows():
            name = row.get("brand_name", "")
            ingredients = row.get("active_ingredients", "")
            if is_valid(ingredients, name):
                unified.append({
                    "title": clean(name),
                    "text": clean(ingredients),
                    "label": row.get("marketing_status", ""),
                    "source": "openfda",
                    "type": "labeled"
                })
    except Exception as e:
        print("❌ openfda_drugs:", e)

# 3. PubMed 200k RCT
def load_pubmed_rct():
    try:
        df = pd.read_csv(paths["pubmed_rct"])
        for _, row in df.iterrows():
            text = clean(row.get("abstract_text", ""))
            label = row.get("target", "")
            if is_valid(text, label):
                unified.append({
                    "title": f"PubMed Abstract {row.get('abstract_id', '')}",
                    "text": text,
                    "label": label,
                    "source": "pubmed_rct",
                    "type": "labeled"
                })
    except Exception as e:
        print("❌ pubmed_rct:", e)

# 4. Boolean Disease-Symptom Dataset
def load_symptoms_boolean():
    try:
        df = pd.read_csv(paths["symptoms_bool"])
        for _, row in df.iterrows():
            disease = row.get("diseases", "")
            symptom_list = ", ".join([col for col in row.index if row[col] == 1])
            if is_valid(symptom_list, disease):
                unified.append({
                    "title": disease,
                    "text": clean(symptom_list),
                    "label": disease,
                    "source": "symptoms_boolean",
                    "type": "labeled"
                })
    except Exception as e:
        print("❌ symptoms_bool:", e)

# 5. Symptom List Format
def load_symptoms_list():
    try:
        df = pd.read_csv(paths["symptoms_list"])
        for _, row in df.iterrows():
            disease = row.get("Disease", "")
            symptoms = ', '.join([str(val) for val in row.values[1:] if pd.notna(val)])
            if is_valid(symptoms, disease):
                unified.append({
                    "title": disease,
                    "text": clean(symptoms),
                    "label": disease,
                    "source": "symptoms_list",
                    "type": "labeled"
                })
    except Exception as e:
        print("❌ symptoms_list:", e)

# 6. PubMed Abstracts (unlabeled)
def load_pubmed_abstracts():
    try:
        df = pd.read_csv(paths["pubmed_abstracts"])
        for _, row in df.iterrows():
            text = clean(row.get("abstract_text", ""))
            if isinstance(text, str) and text.strip():
                unified.append({
                    "title": clean(row.get("title", "")),
                    "text": text,
                    "label": None,
                    "source": "pubmed",
                    "type": "unlabeled"
                })
    except Exception as e:
        print("❌ pubmed_abstracts:", e)

# Run all loaders
load_drugs_side_effects()
load_openfda_drugs()
load_pubmed_rct()
load_symptoms_boolean()
load_symptoms_list()
load_pubmed_abstracts()

# Save output
os.makedirs("data/final", exist_ok=True)
df_all = pd.DataFrame(unified)
df_all.to_csv("data/final/unified_nlp_dataset.csv", index=False)
print(f"✅ Saved {len(df_all)} rows to data/final/unified_nlp_dataset.csv")
