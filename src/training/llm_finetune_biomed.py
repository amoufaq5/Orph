# llm_finetune_biomed.py

"""
This script fine-tunes BioMedLM or BioGPT using the merged dataset.
Recommended base model: https://huggingface.co/stanford-crfm/BioMedLM
"""

import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

MODEL_NAME = "stanford-crfm/BioMedLM"  # or use "microsoft/BioGPT"
DATA_PATH = "data/final/merged_dataset.json"
OUTPUT_DIR = "models/biomed_finetuned"

def prepare_dataset():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_data = []
    for item in data:
        prompt = f"Disease: {item['disease']}\n"
        prompt += f"Symptoms: {', '.join(item['symptoms'])}\n"
        prompt += f"Drugs: {', '.join(item['drugs'])}\n"
        prompt += f"Overview: {item['overview']}\n"
        prompt += f"ICD Code: {item['ICD_code']}\n"
        prompt += f"Diagnosis and Advice:"  # acts as prompt for next token generation

        formatted_data.append({"text": prompt})

    return Dataset.from_list(formatted_data)

def main():
    dataset = prepare_dataset()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    tokenized_dataset = dataset.map(lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=512), batched=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir="logs",
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    print("🚀 Starting fine-tuning...")
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
