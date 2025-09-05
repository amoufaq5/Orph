import argparse, os
from datasketch import MinHash, MinHashLSH
from src.utils.io import read_jsonl, write_jsonl, ensure_dir
from src.utils.logger import get_logger
from rapidfuzz.utils import default_process
log = get_logger("dedup")

def shingles(text: str, n=5):
    s = default_process(text or "")
    return {s[i:i+n] for i in range(max(0, len(s)-n+1))}

def minhash(text: str, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for g in shingles(text, n=5):
        m.update(g.encode("utf-8"))
    return m

def dedup(in_path: str, out_path: str, threshold=0.85, num_perm=128):
    ensure_dir(os.path.dirname(out_path))
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept, idx = [], 0
    mh_store = []

    for r in read_jsonl(in_path):
        txt = r.get("text","")
        if not txt:
            kept.append(r); continue
        m = minhash(txt, num_perm=num_perm)
        # Query near-dups
        hits = lsh.query(m)
        if hits:
            continue
        name = f"m{idx}"; idx += 1
        lsh.insert(name, m); mh_store.append((name, m))
        kept.append(r)

        if idx % 5000 == 0:
            log.info(f"Processed {idx} items; kept={len(kept)}")

    write_jsonl(out_path, kept)
    log.info(f"Dedup done. Input→Output: ? → {len(kept)} rows.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--num_perm", type=int, default=128)
    args = ap.parse_args()
    dedup(**vars(args))
