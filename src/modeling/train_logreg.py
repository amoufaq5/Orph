import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("data/final/train_ready.csv")

# Remove non-numeric or object columns except the label
label_col = "cardio" if "cardio" in df.columns else df.columns[-1]
X = df.drop(columns=[label_col])
X = X.select_dtypes(include=["int64", "float64"])
y = df[label_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("✅ Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
import joblib
joblib.dump(model, "models/logreg_model.joblib")
print("✅ Model saved to models/logreg_model.joblib")
