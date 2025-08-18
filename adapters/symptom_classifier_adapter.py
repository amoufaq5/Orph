from orph_tools.models.symptom_classifier import SymptomClassifier

_classifier = SymptomClassifier()

def classify_symptoms(text: str) -> dict:
    """Adapter for symptom classification"""
    return _classifier.classify(text)
