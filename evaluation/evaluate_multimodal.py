# evaluate_multimodal.py
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load trained model
model = tf.keras.models.load_model("checkpoints/orph_multimodal")

# Load validation set (same process as training)
from train_multimodal import prepare_dataset, df_val, BATCH_SIZE

val_data, val_labels = prepare_dataset(df_val)
val_ds = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(BATCH_SIZE)

# Predict
pred_probs = model.predict(val_ds)
preds = (pred_probs > 0.5).astype(int)

# Evaluate
print("\nClassification Report:\n")
print(classification_report(val_labels, preds))

print("\nConfusion Matrix:\n")
print(confusion_matrix(val_labels, preds))
