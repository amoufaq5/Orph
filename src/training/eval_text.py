from __future__ import annotations
import os, json, math
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# Helpers
# ----------------------------
def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def _prepare_example_pretrain(row: Dict[str, Any]) -> str:
    return (row.get("text") or "").strip()

def _prepare_example_supervised(row: Dict[str, Any]) -> Tuple[str, str]:
    prompt = row.get("input")
    if not prompt:
        prompt = ((row.get("title") or "") + "\n" + (row.get("text") or "")).strip()
    target = row.get("target") or row.get("label") or ""
    return prompt, target

def _chunk_ids(ids: torch.Tensor, max_len: int) -> Iterable[torch.Tensor]:
    # yield non-overlapping chunks up to max_len
    for i in range(0, ids.size(0), max_len):
        yield ids[i : i + max_len]

def _safe_to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

# ----------------------------
# Core eval
# ----------------------------
@torch.inference_mode()
def eval_pretrain_jsonl(
    model, tok, jsonl_path: str, max_length: int = 1024, stride: Optional[int] = None
) -> Dict[str, float]:
    """
    Computes average negative log-likelihood (cross-entropy) and perplexity over `text` fields.
    Uses windowed evaluation for long docs: sliding stride or simple chunks.
    """
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0

    for row in _read_jsonl(jsonl_path):
        text = _prepare_example_pretrain(row)
        if not text:
            continue

        enc = tok(text, return_tensors="pt")
        input_ids = enc["input_ids"][0].to(device)

        # Windowed evaluation to avoid OOM for long sequences
        if stride is None:
            # simple chunking
            for chunk in _chunk_ids(input_ids, max_length):
                out = model(chunk.unsqueeze(0), labels=chunk.unsqueeze(0))
                # loss is mean over tokens in the batch; scale by #tokens
                n_tokens = chunk.numel()
                total_nll += _safe_to_float(out.loss) * n_tokens
                total_tokens += n_tokens
        else:
            # sliding window (overlapping) – more precise but a bit slower
            # Follows the approach in HF examples
            seq_len = input_ids.size(0)
            start = 0
            while start < seq_len:
                end = min(start + max_length, seq_len)
                trg_len = end - start  # number of tokens we predict here
                input_chunk = input_ids[start:end]
                out = model(input_chunk.unsqueeze(0), labels=input_chunk.unsqueeze(0))
                total_nll += _safe_to_float(out.loss) * trg_len
                total_tokens += trg_len
                if end == seq_len:
                    break
                start += stride

    ppl = math.exp(total_nll / max(1, total_tokens)) if total_tokens else float("inf")
    return {"tokens": float(total_tokens), "nll": float(total_nll), "ppl": float(ppl)}

@torch.inference_mode()
def eval_supervised_jsonl(
    model, tok, jsonl_path: str, max_length: int = 1024
) -> Dict[str, float]:
    """
    Computes loss/perplexity ONLY on the target portion.
    We mask the prompt tokens with -100 so they don't contribute to the loss.
    """
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0

    for row in _read_jsonl(jsonl_path):
        prompt, target = _prepare_example_supervised(row)
        if not target:
            continue

        # tokenize separately so we can build labels that mask the prompt
        enc_prompt = tok(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        enc_target = tok(target, return_tensors="pt", truncation=True, max_length=max_length)

        input_ids = torch.cat([enc_prompt["input_ids"], enc_target["input_ids"]], dim=1)[0]
        # labels: mask prompt with -100, keep target ids as labels
        labels = torch.full_like(input_ids, fill_value=-100)
        tgt_len = enc_target["input_ids"].size(1)
        labels[-tgt_len:] = input_ids[-tgt_len:]

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        out = model(input_ids.unsqueeze(0), labels=labels.unsqueeze(0))
        # Count only target tokens
        n_tokens = (labels != -100).sum().item()
        total_nll += _safe_to_float(out.loss) * n_tokens
        total_tokens += n_tokens

    ppl = math.exp(total_nll / max(1, total_tokens)) if total_tokens else float("inf")
    return {"tokens": float(total_tokens), "nll": float(total_nll), "ppl": float(ppl)}

def save_report(report: Dict[str, Any], out_path: str):
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

# ----------------------------
# Entrypoint
# ----------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate causal LM on JSONL datasets.")
    ap.add_argument("--mode", choices=["pretrain", "supervised"], required=True,
                    help="Evaluation mode to match your dataset type.")
    ap.add_argument("--jsonl", type=str, required=True,
                    help="Path to validation JSONL.")
    ap.add_argument("--ckpt", type=str, default="outputs/checkpoints/text",
                    help="Model checkpoint dir or HF model name.")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=0,
                    help="For pretrain: sliding stride (0 = simple chunking).")
    ap.add_argument("--report", type=str, default="outputs/reports/eval_text.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.ckpt)
    model.to(device)
    model.eval()

    if args.mode == "pretrain":
        rep = eval_pretrain_jsonl(
            model, tok, args.jsonl, max_length=args.max_length,
            stride=None if args.stride <= 0 else args.stride
        )
    else:
        rep = eval_supervised_jsonl(
            model, tok, args.jsonl, max_length=args.max_length
        )

    rep["mode"] = args.mode
    rep["jsonl"] = args.jsonl
    rep["ckpt"] = args.ckpt
    save_report(rep, args.report)

    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
