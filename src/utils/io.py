# E:\Orph\src\utils\io.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Dict, Any, List, Union

PathLike = Union[str, Path]

def ensure_dir(p: PathLike) -> None:
    """Create parent directory for a file or the directory itself."""
    p = Path(p)
    target = p if p.suffix == "" else p.parent
    target.mkdir(parents=True, exist_ok=True)

def read_jsonl(path: PathLike) -> Iterable[Dict[str, Any]]:
    """Yield dicts from a JSONL file; skips blank lines."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: PathLike, rows: Iterable[Dict[str, Any]], append: bool = False) -> None:
    """Write an iterable of dicts to JSONL."""
    p = Path(path)
    ensure_dir(p)
    mode = "a" if append else "w"
    with p.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def read_text(path: PathLike) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8")

def write_text(path: PathLike, text: str) -> None:
    p = Path(path)
    ensure_dir(p)
    p.write_text(text, encoding="utf-8")
