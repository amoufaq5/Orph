import pandas as pd
import shap
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import os

# Load data (X_test only)
X_test = pd.read_csv("data/processed/X_test.csv")

# Load XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgboost_cardio.json")

# SHAP wrapper for XGBoost
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Summary Plot (Feature Importance Globally)
os.makedirs("outputs", exist_ok=True)
plt.title("SHAP Feature Importance (Global)")
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("outputs/shap_summary_xgboost.png")
print("✅ Global SHAP summary saved to outputs/shap_summary_xgboost.png")

# Explain single sample (first sample in X_test)
sample_idx = 0
sample = X_test.iloc[[sample_idx]]
sample_shap_values = explainer.shap_values(sample)

# Force Plot (Single Prediction Explanation)
shap.initjs()
force_plot = shap.force_plot(explainer.expected_value, sample_shap_values, sample)

# Save force plot as HTML (can view in browser)
shap.save_html("outputs/shap_force_sample.html", force_plot)
print("✅ SHAP force plot for single sample saved to outputs/shap_force_sample.html")
