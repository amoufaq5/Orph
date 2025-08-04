# generate_synthetic_cases.py

import random
import csv
import numpy as np

symptom_pool = [
    "fever", "dry cough", "chest pain", "headache", "rash", "fatigue", "nausea", "shortness of breath",
    "abdominal pain", "diarrhea", "muscle aches", "dizziness", "weight loss"
]

disease_labels = [
    "common cold", "pneumonia", "covid-19", "gastritis", "flu", "migraine", "asthma", "bronchitis", "allergy", "GERD"
]

def generate_case():
    symptoms = random.sample(symptom_pool, k=random.randint(2, 5))
    symptom_text = "Patient reports " + ", ".join(symptoms) + "."

    # Simulate ASMETHOD (8 fields)
    asmethod_vector = [
        random.randint(10, 80),                 # Age
        random.randint(1, 14),                  # Duration (days)
        random.randint(0, 1),                   # Taking medication?
        random.randint(0, 1),                   # Extra medicine?
        random.randint(0, 1),                   # History of illness?
        random.randint(0, 1),                   # Danger signs?
        random.randint(0, 1),                   # Other symptoms?
        random.randint(0, 1),                   # Referred before?
    ]

    # Simulate CNN image embedding (2048 dims)
    image_vector = np.random.rand(2048).round(4).tolist()

    # Diagnosis label (multi-hot)
    true_labels = random.sample(range(len(disease_labels)), k=random.randint(1, 2))
    diagnosis_label = [1 if i in true_labels else 0 for i in range(len(disease_labels))]

    # Referral (1 = refer, 0 = OTC)
    refer_label = 1 if asmethod_vector[5] == 1 or random.random() < 0.25 else 0

    return {
        "symptoms": symptom_text,
        "image_vector": image_vector,
        "asmethod_vector": asmethod_vector,
        "diagnosis_label": diagnosis_label,
        "refer_label": refer_label
    }

def save_dataset(n=10000, path="data/final/train_multimodal.csv"):
    rows = []
    for _ in range(n):
        row = generate_case()
        rows.append(row)

    keys = ["symptoms", "image_vector", "asmethod_vector", "diagnosis_label", "refer_label"]

    with open(path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "symptoms": row["symptoms"],
                "image_vector": str(row["image_vector"]),
                "asmethod_vector": str(row["asmethod_vector"]),
                "diagnosis_label": str(row["diagnosis_label"]),
                "refer_label": row["refer_label"]
            })

    print(f"✅ Saved {n} synthetic cases to {path}")

if __name__ == "__main__":
    save_dataset()
