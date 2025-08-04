# plot_predictions.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import numpy as np

def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true.argmax(axis=1), np.array(y_pred).argmax(axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("🧠 Diagnosis Confusion Matrix")
    plt.show()

def plot_roc(y_true_ref, y_scores_ref):
    fpr, tpr, _ = roc_curve(y_true_ref, y_scores_ref)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("📊 Referral ROC Curve")
    plt.legend()
    plt.show()
