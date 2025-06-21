# train_multimodal.py
import os
import tensorflow as tf
from transformers import AutoTokenizer
from models.multimodal_model import OrphMultimodal
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- Config ---
MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1'
MAX_LEN = 256
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 3

# --- Load Text & Image Data ---
df = pd.read_csv('data/final/unified_nlp_dataset.csv')
df = df.dropna(subset=['text', 'label'])

# If image_path column is not present, you can add dummy images later
def load_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    return img_to_array(img) / 255.0

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def encode(texts):
    return tokenizer(
        list(texts),
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='tf'
    )

# Add dummy image if image not present
def prepare_dataset(df):
    enc = encode(df['text'])
    input_ids = enc['input_ids']
    segment_ids = tf.zeros_like(input_ids)
    modality_ids = tf.zeros_like(input_ids)

    if 'image_path' in df.columns:
        images = np.stack([load_image(p) for p in df['image_path']])
    else:
        images = np.zeros((len(df), IMG_SIZE[0], IMG_SIZE[1], 3))  # dummy blank

    labels = df['label'].astype('float32').values
    return (input_ids, segment_ids, modality_ids, images), labels

# --- Split ---
df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
train_data, train_labels = prepare_dataset(df_train)
val_data, val_labels = prepare_dataset(df_val)

train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- Model ---
model = OrphMultimodal()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
model.save("checkpoints/orph_multimodal")
