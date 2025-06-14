import tensorflow as tf
from transformers import TFBertModel, AutoTokenizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# Load dataset
df = pd.read_csv("data/final/unified_nlp_dataset.csv")
df = df.dropna(subset=['text', 'label'])  # Removed 'type == labeled' filter

# Check shape
if df.shape[0] == 0:
    raise ValueError("Dataset is empty after dropping null values. Please check your 'text' and 'label' columns.")

# Label encoding
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])
num_labels = len(label_encoder.classes_)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
encodings = tokenizer(
    df['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors='tf'
)

# TF Dataset
dataset = tf.data.Dataset.from_tensor_slices((
    dict(encodings),
    df['label_encoded'].values
)).shuffle(1000).batch(32)

# Load BioBERT model
bert = TFBertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Build classification model
input_ids = tf.keras.Input(shape=(256,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.Input(shape=(256,), dtype=tf.int32, name="attention_mask")

bert_outputs = bert(input_ids, attention_mask=attention_mask)[1]  # Pooled output
output = tf.keras.layers.Dense(num_labels, activation="softmax")(bert_outputs)

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(dataset, epochs=3)

# Save the trained model
os.makedirs("artifacts", exist_ok=True)
model.save("artifacts/research_gpt_classifier_tf")

# Save label encoding map
pd.DataFrame(
    list(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))),
    columns=["Label", "Encoded"]
).to_csv("artifacts/label_map.csv", index=False)

print("✅ Model training complete and saved to 'artifacts/research_gpt_classifier_tf'")
