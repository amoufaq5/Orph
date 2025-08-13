# orphtools/models/diagnosis_engine.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class DiagnosisEngine:
    def __init__(self, model_name, threshold=0.75, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy().flatten()
        top_indices = np.argsort(probs)[::-1]
        results = [(int(i), float(probs[i])) for i in top_indices if probs[i] >= self.threshold]
        return results


# Example usage:
# from orphtools.models.diagnosis_engine import DiagnosisEngine
# engine = DiagnosisEngine("dmis-lab/biobert-base-cased-v1.1", threshold=0.75)
# engine.predict("patient has fever and fatigue")
