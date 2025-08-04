# orphtools/train/train_orphgpt.py

import json
import os
import tensorflow as tf
from pathlib import Path
from orphtools.utils.config_loader import load_config
from orphtools.synthetic.generate_synthetic_cases import SyntheticCaseGenerator
from orphtools.preprocessing.cleaner import clean_data
from orphtools.preprocessing.dataset_merger import merge_datasets
from orphtools.models.orphgpt import OrphGPT
from sklearn.model_selection import train_test_split

def load_and_prepare_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    texts = [entry["input"] for entry in data]
    labels = [entry["output"]["disease"] for entry in data]
    return texts, labels

def main():
    config = load_config("config.yaml")

    # Step 1: Clean real dataset
    cleaned_data_path = Path(config["paths"]["clean_data"])
    if not cleaned_data_path.exists():
        print("🧹 Cleaning real data...")
        clean_data(config["paths"]["raw_data"], str(cleaned_data_path))

    # Step 2: Generate synthetic cases (optional)
    if config.get("synthetic", {}).get("use", True):
        print("🧪 Generating synthetic data...")
        generator = SyntheticCaseGenerator(
            output_dir=config["paths"]["synthetic_data"],
            num_cases=config["synthetic"]["num_cases"]
        )
        synthetic_file = generator.generate_and_save()
    else:
        synthetic_file = None

    # Step 3: Merge datasets
    print("📦 Merging datasets...")
    sources = [str(cleaned_data_path)]
    if synthetic_file:
        sources.append(str(synthetic_file))

    merged_path = config["paths"]["merged_data"]
    merged_file = merge_datasets(sources, merged_path)

    # Step 4: Load merged dataset
    print("📚 Loading training data...")
    texts, labels = load_and_prepare_data(merged_file)

    # Step 5: Encode labels
    label_tokenizer = tf.keras.preprocessing.text.Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    y = label_tokenizer.texts_to_sequences(labels)
    y = tf.keras.utils.to_categorical([item[0] - 1 for item in y], num_classes=len(label_tokenizer.word_index))

    # Step 6: Tokenize input texts
    text_tokenizer = tf.keras.preprocessing.text.Tokenizer()
    text_tokenizer.fit_on_texts(texts)
    X = text_tokenizer.texts_to_sequences(texts)
    X = tf.keras.preprocessing.sequence.pad_sequences(X, padding="post")

    # Step 7: Split and train
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    model = OrphGPT(
        vocab_size=len(text_tokenizer.word_index) + 1,
        num_classes=len(label_tokenizer.word_index)
    )

    model.train(X_train, y_train, X_val, y_val)

    # Step 8: Save tokenizers and model
    model.save(config["paths"]["model_dir"])
    with open(os.path.join(config["paths"]["model_dir"], "label_tokenizer.json"), "w") as f:
        json.dump(label_tokenizer.word_index, f)
    with open(os.path.join(config["paths"]["model_dir"], "text_tokenizer.json"), "w") as f:
        json.dump(text_tokenizer.word_index, f)

    print("✅ Training complete and saved.")

if __name__ == "__main__":
    main()
