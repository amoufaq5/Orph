from flask import Flask, request, jsonify
import pandas as pd
import xgboost as xgb
import shap

app = Flask(__name__)

# Load XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgboost_cardio.json")

# Load feature names (from X_train)
feature_names = pd.read_csv("data/processed/X_train.csv").columns.tolist()

# SHAP Explainer
explainer = shap.TreeExplainer(xgb_model)

@app.route("/predict_cardio", methods=["POST"])
def predict_cardio():
    # Parse input JSON
    input_data = request.json
    try:
        # Convert to DataFrame
        df_input = pd.DataFrame([input_data], columns=feature_names)

        # Predict
        dmatrix = xgb.DMatrix(df_input)
        y_pred_prob = xgb_model.predict(dmatrix)
        y_pred = int(y_pred_prob[0] > 0.5)

        # SHAP values
        shap_values = explainer.shap_values(df_input)[0]
        explanation = dict(zip(feature_names, shap_values.tolist()))

        result = {
            "prediction": "Cardio Disease Risk" if y_pred == 1 else "Healthy",
            "probability": float(y_pred_prob[0]),
            "shap_explanation": explanation
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
