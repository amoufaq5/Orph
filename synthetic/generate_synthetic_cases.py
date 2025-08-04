# orphtools/synthetic/generate_synthetic_cases.py

import random
import json
import os
from pathlib import Path
from faker import Faker

fake = Faker()

COMMON_SYMPTOMS = ["fever", "cough", "fatigue", "headache", "shortness of breath"]
COMMON_DISEASES = ["common cold", "influenza", "pneumonia", "COVID-19", "bronchitis"]

class SyntheticCaseGenerator:
    def __init__(self, output_dir="data/synthetic/", num_cases=100):
        self.output_dir = Path(output_dir)
        self.num_cases = num_cases
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_case(self):
        age = random.randint(1, 90)
        duration = random.randint(1, 14)
        symptoms = random.sample(COMMON_SYMPTOMS, k=random.randint(1, 4))
        disease = random.choice(COMMON_DISEASES)
        danger = random.choice(["", "chest pain", "loss of consciousness", "labored breathing"])
        meta = {
            "age": age,
            "duration_days": duration,
            "danger_symptoms": danger,
            "email": fake.email()
        }
        return {
            "input": f"Patient reports: {', '.join(symptoms)}",
            "meta": meta,
            "output": {
                "disease": disease,
                "score": round(random.uniform(0.6, 0.95), 2),
                "otc": disease in ["common cold", "influenza"]
            }
        }

    def generate_and_save(self):
        cases = [self.generate_case() for _ in range(self.num_cases)]
        file_path = self.output_dir / "synthetic_cases.json"
        with open(file_path, "w") as f:
            json.dump(cases, f, indent=2)
        print(f"✅ Generated {self.num_cases} synthetic cases at {file_path}")
        return file_path
