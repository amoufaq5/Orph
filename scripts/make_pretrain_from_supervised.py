from __future__ import annotations
import json, sys
from pathlib import Path

inp = Path(sys.argv[1])  # data/clean/text_supervised.jsonl
out = Path(sys.argv[2])  # data/clean/text_pretrain.jsonl
out.parent.mkdir(parents=True, exist_ok=True)

n = 0
with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = (row.get("input") or "").strip()
        target = (row.get("target") or "").strip()
        text = (prompt + ("\n" if prompt and target else "") + target).strip()
        if text:
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1
print(f"wrote {n} lines to {out}")
