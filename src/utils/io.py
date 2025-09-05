import json, os, gzip
from typing import Iterable, Dict, Any, Union

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: str):
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def save_json(path: str, obj: Union[dict, list]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
