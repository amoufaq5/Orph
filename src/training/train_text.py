from __future__ import annotations
import os
import random
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from src.utils.config import Config
from src.utils.logging import init_logger
from src.data.builders.text_pretrain_builder import TextPretrainDataset
from src.data.builders.text_supervised_builder import TextSupervisedDataset
from src.modeling.tokenizer import load_or_train_tokenizer
from transformers import DataCollatorForLanguageModeling
from src.data.builders import TextPretrainDataset, TextSupervisedDataset

# pretrain
ds = TextPretrainDataset(cfg.require("text_datasets.pretrain_jsonl"), tok, max_len)
collator = DataCollatorForLanguageModeling(tok, mlm=False)

# supervised
ds = TextSupervisedDataset(cfg.require("text_datasets.supervised_jsonl"), tok, max_len)
collator = None
logger = init_logger("train_text")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(cfg: Config):
    set_seed(cfg.get("run.seed", 42))

    # Tokenizer (load or train from corpus)
    tok = load_or_train_tokenizer(
        pretrained_name=cfg.get("text_tokenizer.pretrained_name"),
        train_from_corpus=cfg.get("text_tokenizer.train_from_corpus", False),
        vocab_size=cfg.get("text_tokenizer.vocab_size", 32000),
        sp_model_type=cfg.get("text_tokenizer.sp_model_type", "bpe"),
        corpus_jsonl=cfg.get("text_tokenizer.corpus_jsonl"),
        save_dir=cfg.require("paths.tokenizer_dir"),
    )

    stage = cfg.get("train_text.stage", "pretrain")
    max_len = cfg.get("model.max_seq_len", 1024)

    if stage == "pretrain":
        ds = TextPretrainDataset(cfg.require("text_datasets.pretrain_jsonl"), tok, max_len)
        data_collator = DataCollatorForLanguageModeling(tok, mlm=False)
    else:
        ds = TextSupervisedDataset(cfg.require("text_datasets.supervised_jsonl"), tok, max_len)
        data_collator = None

    # Model warm-start
    pretrained_name = cfg.get("model.pretrained_name") or "gpt2"
    model = AutoModelForCausalLM.from_pretrained(pretrained_name)
    model.resize_token_embeddings(len(tok))

    out_dir = cfg.require("train_text.output_dir")
    os.makedirs(out_dir, exist_ok=True)

    use_bf16 = cfg.get("run.mixed_precision", "no") == "bf16"
    use_fp16 = cfg.get("run.mixed_precision", "no") == "fp16"

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg.get("train_text.batch_size", 8),
        gradient_accumulation_steps=cfg.get("train_text.grad_accum_steps", 8),
        learning_rate=cfg.get("train_text.lr", 2e-4),
        weight_decay=cfg.get("train_text.weight_decay", 0.01),
        warmup_ratio=cfg.get("train_text.warmup_ratio", 0.03),
        max_steps=cfg.get("train_text.max_steps", 20000),
        logging_steps=cfg.get("train_text.log_every", 50),
        save_steps=cfg.get("train_text.save_every", 1000),
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=data_collator,
        tokenizer=tok,
    )

    logger.info(f"Starting text training stage={stage} on {torch.cuda.device_count()} GPU(s)")
    trainer.train()

    logger.info("Saving final checkpoint")
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="conf/config.yaml")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    main(cfg)
