# evaluate_orphgpt.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from train_orphgpt import OrphDataset
from orphgpt import OrphGPT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model_path, data_path, num_classes=10):
    dataset = OrphDataset(data_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = OrphGPT(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()

    y_true_diag, y_pred_diag = [], []
    y_true_ref, y_pred_ref = [], []

    with torch.no_grad():
        for batch in loader:
            outputs = model(batch['symptoms'], batch['image_vector'].to(DEVICE), batch['asmethod_vector'].to(DEVICE))
            logits_diag = outputs["diagnosis_logits"]
            logits_ref = outputs["refer_logits"]

            probs_diag = torch.sigmoid(logits_diag).cpu().numpy()
            preds_diag = (probs_diag > 0.5).astype(int)

            probs_ref = torch.sigmoid(logits_ref).cpu().numpy().flatten()
            preds_ref = (probs_ref > 0.5).astype(int)

            y_true_diag.extend(batch["diagnosis_label"].numpy())
            y_pred_diag.extend(preds_diag)
            y_true_ref.extend(batch["refer_label"].numpy())
            y_pred_ref.extend(preds_ref)

    # Compute metrics
    f1 = f1_score(y_true_diag, y_pred_diag, average="micro")
    acc = accuracy_score(y_true_ref, y_pred_ref)
    auc = roc_auc_score(y_true_ref, probs_ref)

    print(f"\n🧪 Diagnosis F1 Score: {f1:.3f}")
    print(f"📈 Referral Accuracy: {acc:.3f}")
    print(f"🔍 Referral AUC: {auc:.3f}")

    return y_true_diag, y_pred_diag, y_true_ref, y_pred_ref

if __name__ == "__main__":
    evaluate("orphgpt_multimodal.pt", "data/final/test_multimodal.csv")
