# _utils.py
import json, os
from pathlib import Path

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def write_jsonl(rows, out_path: str):
    p = Path(out_path)
    ensure_parent(p)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_lines(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()

def norm_text(s):
    return " ".join(str(s).split()) if s is not None else ""
