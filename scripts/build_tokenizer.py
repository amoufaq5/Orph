# scripts/build_tokenizer.py
from __future__ import annotations
import argparse, json, os, tempfile
from pathlib import Path

# Uses HF "tokenizers" (fast) + GPT-2 compatible byte-level BPE
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import ByteLevel

SPECIAL_TOKENS = {
    "unk_token": "<|unk|>",
    "bos_token": "<|bos|>",
    "eos_token": "<|eos|>",
    "pad_token": "<|pad|>",
}

def jsonl_to_txt(jsonl_path: Path) -> Path:
    """
    Extracts 'text' field lines from a JSONL into a flat .txt file for tokenizer training.
    Falls back to 'input' and 'target' fields when 'text' is missing.
    """
    tmp = Path(tempfile.mkstemp(suffix=".txt")[1])
    n = 0
    with open(jsonl_path, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            s = row.get("text")
            if not s:
                inp = row.get("input", "")
                tgt = row.get("target", "")
                s = (inp + "\n" + tgt).strip()
            if s:
                fout.write(s.replace("\r", " ") + "\n")
                n += 1
    if n == 0:
        raise ValueError(f"No usable lines found in {jsonl_path}")
    return tmp

def save_tokenizer_files(outdir: Path, tokenizer: ByteLevelBPETokenizer):
    """
    Save vocab/merges and minimal HF-compatible config so AutoTokenizer/GPT2TokenizerFast can load it.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(outdir))  # writes vocab.json + merges.txt

    # tokenizer_config.json (lets AutoTokenizer infer GPT2TokenizerFast)
    tk_cfg = {
        "model_max_length": 1024,
        "padding_side": "right",
        "truncation_side": "right",
        "special_tokens_map": SPECIAL_TOKENS,
        "tokenizer_class": "GPT2TokenizerFast"
    }
    (outdir / "tokenizer_config.json").write_text(json.dumps(tk_cfg, indent=2), encoding="utf-8")

    # special_tokens_map.json
    (outdir / "special_tokens_map.json").write_text(json.dumps(SPECIAL_TOKENS, indent=2), encoding="utf-8")

    # Add a minimal config.json for GPT2TokenizerFast compatibility
    if not (outdir / "config.json").exists():
        (outdir / "config.json").write_text(json.dumps({"model_type": "gpt2"}, indent=2), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to pretrain JSONL (expects 'text' or 'input/target').")
    ap.add_argument("--outdir", required=True, help="Directory to save tokenizer files.")
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--min_frequency", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=1024)
    args = ap.parse_args()

    src = Path(args.input)
    outdir = Path(args.outdir)

    txt_path = jsonl_to_txt(src)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(txt_path)],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=list(SPECIAL_TOKENS.values()),
    )
    tokenizer._tokenizer.post_processor = ByteLevel(trim_offsets=False)
    tokenizer.enable_truncation(args.max_length)
    tokenizer.enable_padding(length=None)

    save_tokenizer_files(outdir, tokenizer)

    print(f"✅ Tokenizer trained and saved to: {outdir}")
    print(f"   Files: vocab.json, merges.txt, tokenizer_config.json, special_tokens_map.json, config.json")

if __name__ == "__main__":
    main()
