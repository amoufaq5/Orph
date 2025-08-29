import os, time, json, math, requests
from pathlib import Path
from typing import Iterable, Dict, Any

ROOT = Path(__file__).resolve().parents[3]  # E:\Orph
RAW_OUT = ROOT / "data" / "raw" / "scraped"
RAW_OUT.mkdir(parents=True, exist_ok=True)

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"✅ Wrote {n} rows -> {path}")
    return n

def backoff_sleep(i: int):
    time.sleep(min(60, 2 ** i))

def http_get(url: str, params=None, headers=None, max_retries=5, timeout=30):
    params = params or {}
    headers = headers or {}
    for i in range(max_retries):
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            backoff_sleep(i)
            continue
        r.raise_for_status()
    r.raise_for_status()
