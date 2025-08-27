import argparse
from tokenizers import ByteLevelBPETokenizer

def train_tokenizer(input_file, outdir, vocab_size, min_frequency, max_length):
    # Initialize a Byte-Pair Encoding tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Train tokenizer on dataset
    tokenizer.train(
        files=[input_file],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )

    # Save tokenizer
    tokenizer.save_model(outdir)

    # Also save config.json with max length
    with open(f"{outdir}/config.json", "w", encoding="utf-8") as f:
        f.write(str({
            "max_length": max_length,
            "vocab_size": vocab_size,
            "min_frequency": min_frequency
        }))

    print(f"✅ Tokenizer trained and saved to: {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to training file (.jsonl or .txt)")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for tokenizer files")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=1024)

    args = parser.parse_args()

    train_tokenizer(args.input, args.outdir, args.vocab_size, args.min_frequency, args.max_length)
