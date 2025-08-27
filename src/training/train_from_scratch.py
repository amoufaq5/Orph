# src/training/train_from_scratch.py
from __future__ import annotations
import os
from typing import Optional
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch

from src.utils.config import Config
from src.data.builders import TextPretrainDataset, TextSupervisedDataset

def load_tokenizer(tok_dir: str, max_len: int) -> GPT2TokenizerFast:
    tok = GPT2TokenizerFast.from_pretrained(tok_dir, padding_side="right", truncation_side="right")
    # Ensure required special tokens exist; GPT2 uses EOS but we add PAD for batching
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<|pad|>"})
    tok.model_max_length = max_len
    return tok

def build_gpt2_config(vocab_size: int, arch_cfg: dict, max_len: int) -> GPT2Config:
    return GPT2Config(
        vocab_size=int(vocab_size),
        n_embd=int(arch_cfg.get("n_embd", 768)),
        n_layer=int(arch_cfg.get("n_layer", 12)),
        n_head=int(arch_cfg.get("n_head", 12)),
        n_positions=int(arch_cfg.get("n_positions", max_len)),
        n_ctx=int(arch_cfg.get("n_positions", max_len)),
        resid_pdrop=float(arch_cfg.get("dropout", 0.0)),
        embd_pdrop=float(arch_cfg.get("dropout", 0.0)),
        attn_pdrop=float(arch_cfg.get("dropout", 0.0)),
    )

def main(cfg_path: Optional[str] = None):
    cfg = Config(cfg_path or "conf/config.yaml")

    tok_dir = cfg.get("model.tokenizer_dir", "outputs/tokenizer/orph_bpe_32k")
    max_len = int(cfg.get("training.max_length", 1024))
    arch = cfg.get("model.arch", {"n_embd": 768, "n_layer": 12, "n_head": 12, "n_positions": max_len})
    mode = cfg.get("training.mode", "pretrain")  # pretrain | supervised
    out_dir = cfg.get("training.output_dir", "outputs/checkpoints/orph-125m")

    # 1) Tokenizer
    tokenizer = load_tokenizer(tok_dir, max_len)

    # 2) Model (random init)
    gcfg = build_gpt2_config(len(tokenizer), arch, max_len)
    model = GPT2LMHeadModel(gcfg)
    # IMPORTANT: resize token embeddings to match tokenizer (for added PAD etc.)
    model.resize_token_embeddings(len(tokenizer))

    # 3) Dataset(s)
    if mode == "pretrain":
        ds = TextPretrainDataset(cfg.require("text_datasets.pretrain_jsonl"), tokenizer, max_length=max_len)
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    elif mode == "supervised":
        ds = TextSupervisedDataset(cfg.require("text_datasets.supervised_jsonl"), tokenizer, max_length=max_len)
        data_collator = None
    else:
        raise ValueError("training.mode must be 'pretrain' or 'supervised' for train_from_scratch.py")

    # 4) Training args
    args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=int(cfg.get("training.epochs", 3)),
        per_device_train_batch_size=int(cfg.get("training.batch_size", 2)),
        gradient_accumulation_steps=int(cfg.get("training.grad_accum", 8)),
        learning_rate=float(cfg.get("training.learning_rate", 5e-5)),
        weight_decay=float(cfg.get("training.weight_decay", 0.0)),
        warmup_ratio=float(cfg.get("training.warmup_ratio", 0.03)),
        logging_steps=int(cfg.get("training.logging_steps", 50)),
        save_steps=int(cfg.get("training.save_steps", 1000)),
        save_total_limit=int(cfg.get("training.save_total_limit", 3)),
        fp16=torch.cuda.is_available(),
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # 5) Save model + tokenizer together for easy reload
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"✅ From-scratch training ({mode}) finished. Saved to: {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()
    main(args.config)
