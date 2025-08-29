import argparse, json, math, random, time
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)

# -----------------------------
# Utilities
# -----------------------------
def read_jsonl(path: Path):
    if not path or not path.exists(): return []
    out = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line=line.strip()
            if line:
                try: out.append(json.loads(line))
                except Exception: continue
    return out

def build_prompt_sft(ex: Dict) -> str:
    instr = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    out = (ex.get("output") or "").strip()
    return f"<s>\nInstruction:\n{instr}\n\nInput:\n{inp}\n\nResponse:\n{out}\n</s>"

def build_prompt_cot(ex: Dict) -> str:
    instr = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    rationale = (ex.get("rationale") or "").strip()
    out = (ex.get("output") or "").strip()
    return f"<s>\nInstruction:\n{instr}\n\nInput:\n{inp}\n\nReasoning:\n{rationale}\n\nFinal Answer:\n{out}\n</s>"

class JsonlCausalDataset(Dataset):
    def __init__(self, records: List[Dict], tokenizer, max_length: int):
        self.records = records
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        ex = self.records[idx]
        text = ex["__prompt__"]
        enc = self.tok(text, max_length=self.max_length, truncation=True, padding=False, return_tensors=None)
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(enc["input_ids"], dtype=torch.long),
        }

# -----------------------------
# Mixing / curriculum
# -----------------------------
def parse_curriculum(s: Optional[str]):
    default = {"sft": 1.0, "cot": 0.0}
    if not s: return default
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = {}
    for p in parts:
        if ":" in p:
            k, v = p.split(":", 1)
            try: out[k.strip().lower()] = float(v)
            except: pass
    if not out: return default
    tot = sum(out.values()) or 1.0
    return {k: v / tot for k, v in out.items()}

def mix_records(sft: List[Dict], cot: List[Dict], ratio: Dict[str, float], target_size: Optional[int] = None):
    sft = sft or []; cot = cot or []
    if not sft and not cot: return []
    if target_size is None: target_size = len(sft) + len(cot)
    nsft = int(target_size * ratio.get("sft", 0.0))
    ncot = int(target_size * ratio.get("cot", 0.0))
    def expand(pool, n):
        if not pool: return []
        k = math.ceil(n / max(len(pool),1))
        arr = (pool * max(k,1))[:n]
        random.shuffle(arr); return arr
    mixed = expand(sft, nsft) + expand(cot, ncot)
    if not mixed: mixed = sft or cot
    random.shuffle(mixed)
    return mixed

# -----------------------------
# ETA callback
# -----------------------------
class ETACallback(TrainerCallback):
    def __init__(self): self.start = None
    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.time()
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.start or not state.max_steps or not state.global_step: return
        elapsed = time.time() - self.start
        rate = state.global_step / max(elapsed,1e-9)
        remaining_steps = max(state.max_steps - state.global_step, 0)
        eta_sec = remaining_steps / max(rate,1e-9)
        def fmt(sec):
            m, s = divmod(int(sec), 60); h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        print(f"[ETA] step {state.global_step}/{state.max_steps} | elapsed {fmt(elapsed)} | remaining ~{fmt(eta_sec)}")

# -----------------------------
# Auto batch-size tuner
# -----------------------------
def try_batch_size(model, tok, max_length, candidate, device):
    try:
        model.zero_grad(set_to_none=True)
        model.to(device)
        ids = tok("<s>\n", max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        ids = {k: v.to(device) for k, v in ids.items()}
        # repeat to candidate batch
        ids = {k: v.expand(candidate, -1) for k, v in ids.items()}
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out = model(**ids, labels=ids["input_ids"])
            loss = out.loss
            loss.backward()  # try backward to truly allocate grads
        return True
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            return False
        raise
    finally:
        torch.cuda.empty_cache()

def autotune_batch_size(model, tok, max_length, desired, device):
    if device.type != "cuda": return desired  # autotune most useful on GPU
    for bs in [desired, max(1, desired-1), max(1, desired-2), 1]:
        ok = try_batch_size(model, tok, max_length, bs, device)
        if ok:
            if bs < desired:
                print(f"⚠️  Auto-tune: reducing per_device_train_batch_size {desired} → {bs} to fit VRAM")
            return bs
    return 1

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--train_files", required=True)
    ap.add_argument("--cot_files", default="")
    ap.add_argument("--curriculum", default="sft:1.0,cot:0.0")
    ap.add_argument("--model_name", default="gpt2")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_length", type=int, default=1024)

    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--train_batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--eval_split", default="")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--gradient_checkpointing", type=str, default="true",
                    help="true/false to enable grad checkpointing")
    ap.add_argument("--auto_batch", type=str, default="true",
                    help="true/false to auto-tune batch size to VRAM")

    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    if tokenizer.pad_token is None:
        if "<pad>" in tokenizer.get_vocab(): tokenizer.pad_token = "<pad>"
        else: tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # ---- data ----
    sft = read_jsonl(Path(args.train_files))
    cot = read_jsonl(Path(args.cot_files)) if args.cot_files else []
    for ex in sft: ex["__prompt__"] = build_prompt_sft(ex)
    for ex in cot: ex["__prompt__"] = build_prompt_cot(ex)
    ratio = parse_curriculum(args.curriculum)
    mixed = mix_records(sft, cot, ratio)
    print(f"Loaded: SFT={len(sft)} | COT={len(cot)} | Mixed train={len(mixed)}")
    if not mixed: raise SystemExit("No training data found.")

    # optional eval
    eval_data = []
    if args.eval_split:
        raw = read_jsonl(Path(args.eval_split))
        for ex in raw:
            if ex.get("task") == "instruction_tuning": ex["__prompt__"] = build_prompt_sft(ex)
            elif ex.get("task") == "cot_supervision": ex["__prompt__"] = build_prompt_cot(ex)
            else: continue
            eval_data.append(ex)

    train_ds = JsonlCausalDataset(mixed, tokenizer, args.max_length)
    eval_ds = JsonlCausalDataset(eval_data, tokenizer, args.max_length) if eval_data else None

    # ---- model ----
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # gradient checkpointing (saves RAM)
    if args.gradient_checkpointing.lower() == "true":
        model.gradient_checkpointing_enable()
        # disable cache to avoid incompatibility
        model.config.use_cache = False

    # resize if tokenizer grew
    model.resize_token_embeddings(len(tokenizer))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- auto batch-size tuning ----
    per_device_bs = args.train_batch_size
    if args.auto_batch.lower() == "true":
        print(f"🔍 Auto-tuning batch size for max_length={args.max_length} ...")
        per_device_bs = autotune_batch_size(model, tokenizer, args.max_length, desired=args.train_batch_size, device=device)

    # ---- trainer ----
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=max(1, per_device_bs // 2),
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        evaluation_strategy="steps" if eval_ds else "no",
        eval_steps=max(args.logging_steps, 100) if eval_ds else None,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),
        report_to=["none"],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[ETACallback()],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Finished. Model saved to:", args.output_dir)

if __name__ == "__main__":
    main()
