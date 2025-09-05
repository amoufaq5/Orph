import random, json
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logger import get_logger
log = get_logger("split")

def split(in_path, out_train, out_val, out_test, ratios=(0.8,0.1,0.1), seed=1337):
    rng = random.Random(seed)
    train, val, test = [], [], []
    for r in read_jsonl(in_path):
        x = rng.random()
        if x < ratios[0]: train.append(r)
        elif x < ratios[0]+ratios[1]: val.append(r)
        else: test.append(r)
    write_jsonl(out_train, train)
    write_jsonl(out_val, val)
    write_jsonl(out_test, test)
    log.info(f"Split: train={len(train)} val={len(val)} test={len(test)}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_val", required=True)
    ap.add_argument("--out_test", required=True)
    split(**vars(ap.parse_args()))
