"""
Synthetic case generator for Orph project.
Produces dictionaries representing synthetic patient cases
for training the supervised dataset.
"""

import random

# --- Example vocab (expand as needed) ---
SYMPTOMS = [
    "fever", "cough", "sore throat", "headache",
    "abdominal pain", "diarrhea", "fatigue", "rash"
]

DIAGNOSES = [
    "Common cold", "Influenza", "Migraine",
    "Gastroenteritis", "Allergic rhinitis", "COVID-19"
]

OTC_DRUGS = [
    ("paracetamol", "500 mg every 6–8h (max 4g/day)"),
    ("ibuprofen", "400 mg every 6–8h (max 1.2g/day OTC)"),
    ("loratadine", "10 mg once daily"),
    ("oral rehydration salts", "as needed after each loose stool"),
]

COUNSELING = [
    "Stay hydrated and rest.",
    "Monitor fever and seek care if it persists beyond 3 days.",
    "Avoid close contact to prevent transmission.",
    "Seek immediate care if shortness of breath or chest pain develops."
]


def generate_case() -> dict:
    """Generate ONE synthetic patient case dictionary."""
    # Pick random symptoms
    symptoms = random.sample(SYMPTOMS, k=random.randint(1, 3))
    duration = f"{random.randint(1, 7)} days"
    history = random.choice(["no significant history", "asthma", "hypertension", "diabetes"])
    danger_signs = random.choice([[], ["severe chest pain"], ["persistent vomiting"], ["confusion"]])

    diagnosis = random.choice(DIAGNOSES)
    drug, dose = random.choice(OTC_DRUGS)
    counseling_text = random.choice(COUNSELING)

    recommendation = random.choice([
        "Provide OTC self-care.",
        "Refer to doctor if symptoms persist or worsen."
    ])

    case = {
        "symptoms": symptoms,
        "duration": duration,
        "history": history,
        "danger_signs": danger_signs,
        "final_diagnosis": diagnosis,
        "recommendation": recommendation,
        "otc_drug": drug,
        "dose": dose,
        "counseling": counseling_text,
        "explanation": f"Case generated for training with {diagnosis}.",
        "asmethod_answers": {
            "Age/appearance": "Adult, 30-40",
            "Self/Someone else": "Self",
            "Medication": "None",
            "Extra medicines": "None",
            "Time": duration,
            "History": history,
            "Other symptoms": ", ".join(symptoms),
            "Danger symptoms": ", ".join(danger_signs) if danger_signs else "None"
        }
    }
    return case


def generate_cases(n: int) -> list[dict]:
    """Generate a list of n synthetic patient cases."""
    return [generate_case() for _ in range(n)]


# Manual test
if __name__ == "__main__":
    cases = generate_cases(3)
    for c in cases:
        print(c)
