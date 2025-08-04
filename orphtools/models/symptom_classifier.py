# orphtools/models/symptom_classifier.py
import torch
import torch.nn as nn
import numpy as np

class SymptomClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SymptomClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.classifier(x)

    def predict(self, x_tensor, threshold=0.75):
        with torch.no_grad():
            logits = self.forward(x_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return [(i, float(prob)) for i, prob in enumerate(probs) if prob >= threshold]


# Example usage:
# model = SymptomClassifier(input_dim=300, output_dim=10)
# x = torch.rand((1, 300))
# preds = model.predict(x, threshold=0.75)
