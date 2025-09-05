import argparse, sentencepiece as spm, os, tempfile
from src.utils.io import read_jsonl
from src.utils.logger import get_logger
log = get_logger("tokenizer")

def write_corpus_txt(jsonl_path: str, out_txt: str, max_lines: int|None):
    n=0
    with open(out_txt, "w", encoding="utf-8") as f:
        for r in read_jsonl(jsonl_path):
            t = r.get("text") or ""
            if not t: continue
            f.write(t.replace("\n"," ") + "\n")
            n += 1
            if max_lines and n >= max_lines: break
    log.info(f"Tokenizer corpus lines: {n}")

def train_spm(corpus_txt: str, out_dir: str, vocab_size: int, model_type: str):
    os.makedirs(out_dir, exist_ok=True)
    model_prefix = os.path.join(out_dir, "orph_spm")
    spm.SentencePieceTrainer.Train(
        input=corpus_txt,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,     # bpe | unigram
        character_coverage=0.9995,
        input_sentence_size=10000000,
        shuffle_input_sentence=True,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3
    )
    log.info(f"Saved tokenizer to {model_prefix}.model / .vocab")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="data/cleaned/text_corpus.jsonl")
    ap.add_argument("--out_dir", default="data/artifacts/tokenizer")
    ap.add_argument("--vocab_size", type=int, default=48000)
    ap.add_argument("--model_type", default="bpe")
    ap.add_argument("--max_lines", type=int, default=2_000_000)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        corpus_txt = os.path.join(td, "tok_corpus.txt")
        write_corpus_txt(args.jsonl, corpus_txt, args.max_lines)
        train_spm(corpus_txt, args.out_dir, args.vocab_size, args.model_type)
