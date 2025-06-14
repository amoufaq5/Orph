import tensorflow as tf
from transformers import TFBertModel, AutoTokenizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load dataset
df = pd.read_csv("data/final/unified_nlp_dataset.csv")
df = df[df['type'] == 'labeled'].dropna(subset=['text', 'label'])

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

# Load BioBERT
bert = TFBertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Build Model
input_ids = tf.keras.Input(shape=(256,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.Input(shape=(256,), dtype=tf.int32, name="attention_mask")

bert_outputs = bert(input_ids, attention_mask=attention_mask)[1]  # pooled output
output = tf.keras.layers.Dense(num_labels, activation="softmax")(bert_outputs)

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train
model.fit(dataset, epochs=3)

# Save model
model.save("artifacts/research_gpt_classifier_tf")
print("✅ Model trained and saved to artifacts/research_gpt_classifier_tf")
