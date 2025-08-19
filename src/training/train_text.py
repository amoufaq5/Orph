# src/training/train_text.py
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

# ------------------------------------------------------------------------------------
# Config
DATA_JSONL = Path("data/clean/text_supervised.jsonl")
PIPELINE_MODULE = "src.data.pipelines.build_synthetic_supervised"  # builds the jsonl
DEFAULT_MODEL = "gpt2"
SEED = 42

# ------------------------------------------------------------------------------------
# Dataset (expects you already have this builder as we discussed)
# If your class lives elsewhere, adjust the import path accordingly.
from src.data.builders.text_supervised_builder import TextSupervisedDataset  # noqa: E402


def ensure_supervised_dataset(n_cases: int = 3000):
    """Ensure data/clean/text_supervised.jsonl exists; if not, build from synthetic cases."""
    if DATA_JSONL.exists():
        return
    print("⚠️  text_supervised.jsonl not found. Building from synthetic cases...")
    cmd = [sys.executable, "-m", PIPELINE_MODULE, "--n_cases", str(n_cases)]
    subprocess.run(cmd, check=True)
    if not DATA_JSONL.exists():
        raise FileNotFoundError(str(DATA_JSONL))
    print(f"✅ Built {DATA_JSONL}")


def load_tokenizer_and_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # GPT‑2 has no pad token by default → align pad/eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Reduce memory footprint
    model.gradient_checkpointing_enable()

    return tokenizer, model


def build_datasets(tokenizer, max_length: int, eval_split: float):
    """Simple in‑file split without extra libs."""
    # Read all rows once to split deterministically
    rows = []
    with open(DATA_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    n_total = len(rows)
    n_eval = int(n_total * eval_split) if eval_split > 0 else 0
    n_train = n_total - n_eval

    # Write temporary split files to reuse your dataset class unmodified
    tmp_dir = Path("data/clean/_splits")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    train_path = tmp_dir / "train.jsonl"
    eval_path = tmp_dir / "eval.jsonl"

    with open(train_path, "w", encoding="utf-8") as ft:
        for i in range(n_train):
            ft.write(json.dumps(rows[i], ensure_ascii=False) + "\n")

    if n_eval > 0:
        with open(eval_path, "w", encoding="utf-8") as fe:
            for i in range(n_train, n_total):
                fe.write(json.dumps(rows[i], ensure_ascii=False) + "\n")

    train_ds = TextSupervisedDataset(str(train_path), tokenizer=tokenizer, max_length=max_length)
    eval_ds = TextSupervisedDataset(str(eval_path), tokenizer=tokenizer, max_length=max_length) if n_eval > 0 else None
    return train_ds, eval_ds


def main():
    parser = argparse.ArgumentParser(description="Train causal LM on Orph supervised text data.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help="HF model name or path (e.g., gpt2)")
    parser.add_argument("--output_dir", type=str, default="out/text_gpt2")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--train_batch_size", type=int, default=1)  # per device
    parser.add_argument("--grad_accum", type=int, default=8)        # effective batch size ~8
    parser.add_argument("--eval_split", type=float, default=0.02)   # 2% for quick sanity eval
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    # 1) Ensure dataset exists (generate from synthetic cases if missing)
    ensure_supervised_dataset(n_cases=3000)

    # 2) Load tokenizer + model (pad handling + gradient checkpointing)
    tokenizer, model = load_tokenizer_and_model(args.model_name)

    # 3) Datasets
    train_ds, eval_ds = build_datasets(tokenizer, args.max_length, args.eval_split)

    # 4) Collator: dynamic padding, labels auto‑aligned by dataset
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5) Training args (tuned for ~8GB GPU)
    fp16 = torch.cuda.is_available()  # use fp16 if CUDA present
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=max(1, args.train_batch_size),
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,

        fp16=fp16,
        bf16=False,  # 30‑series cards prefer fp16
        torch_compile=False,

        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        save_steps=args.save_steps,
        save_total_limit=2,

        report_to=["none"],  # set to ["tensorboard"] if you want TB
        dataloader_pin_memory=True,
    )

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=None,  # optional: add perplexity later
    )

    # 7) Train
    trainer.train()

    # 8) Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Finished. Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
