# E:\Orph\src\training\train_tokenizer.py
import argparse
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

def train_tokenizer(input_file, outdir, vocab_size, min_frequency, max_length):
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[input_file],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )

    # Save merges & vocab (for HF-compatible BPE)
    tokenizer.save_model(str(out_path))

    # Also save a tokenizer.json (useful for fast tokenizers)
    tokenizer.save(str(out_path / "tokenizer.json"))

    # Save a tiny config for your reference
    (out_path / "config.json").write_text(
        (
            '{\n'
            f'  "max_length": {max_length},\n'
            f'  "vocab_size": {vocab_size},\n'
            f'  "min_frequency": {min_frequency}\n'
            '}\n'
        ),
        encoding="utf-8"
    )

    print(f"✅ Tokenizer trained and saved to: {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--vocab_size", type=int, default=32000)
    p.add_argument("--min_frequency", type=int, default=2)
    p.add_argument("--max_length", type=int, default=1024)
    args = p.parse_args()
    train_tokenizer(args.input, args.outdir, args.vocab_size, args.min_frequency, args.max_length)
