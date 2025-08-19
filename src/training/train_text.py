# src/training/train_text.py
from __future__ import annotations
import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from src.utils.config import Config
from src.data.builders import TextPretrainDataset, TextSupervisedDataset
from src.explain import GradCAM

def main():
    # Load config
    cfg = Config("conf/config.yaml")
    model_name = cfg.get("model.name", "gpt2")
    mode = cfg.get("training.mode", "pretrain")  # "pretrain" or "supervised"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ... after you do: tokenizer = AutoTokenizer.from_pretrained(...); model = AutoModelForCausalLM.from_pretrained(...)

# GPT-2 has no pad token by default → align pad/eos
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = tokenizer.pad_token_id

# Use a causal LM collator that handles dynamic padding
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # causal LM
)

    # Dataset
    if mode == "pretrain":
        dataset = TextPretrainDataset(
            cfg.require("text_datasets.pretrain_jsonl"),
            tokenizer,
            max_length=cfg.get("training.max_length", 512),
        )
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    elif mode == "supervised":
        dataset = TextSupervisedDataset(
            cfg.require("text_datasets.supervised_jsonl"),
            tokenizer,
            max_length=cfg.get("training.max_length", 512),
        )
        collator = None
    else:
        raise ValueError(f"Unknown training mode: {mode}")

    # Model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Training args
    training_args = TrainingArguments(
        output_dir=cfg.get("training.output_dir", "./checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=cfg.get("training.epochs", 3),
        per_device_train_batch_size=cfg.get("training.batch_size", 2),
        save_steps=cfg.get("training.save_steps", 500),
        save_total_limit=2,
        logging_dir=cfg.get("training.log_dir", "./logs"),
        logging_steps=cfg.get("training.logging_steps", 50),
        fp16=torch.cuda.is_available(),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save
    out_dir = cfg.get("training.output_dir", "./checkpoints")
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"✅ Training finished. Model saved to {out_dir}")

    # ---- Optional GradCAM demo ----
    if cfg.get("training.demo_gradcam", False):
        print("Running GradCAM demo...")
        cam = GradCAM(model, target_layer_name="transformer.h.0")
        # Example forward pass with one token
        inputs = tokenizer("Test input", return_tensors="pt")
        logits = model(**inputs).logits
        heatmap = cam(logits, class_idx=0)
        print("GradCAM heatmap:", heatmap.shape)


if __name__ == "__main__":
    main()
