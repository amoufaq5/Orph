from abc import ABC, abstractmethod
from typing import Dict, Iterator
from src.utils.logger import get_logger
from src.utils.io import write_jsonl, ensure_dir
import os, uuid, time

log = get_logger("scraper")

class Scraper(ABC):
    name = "base"

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        ensure_dir(out_dir)

    @abstractmethod
    def stream(self) -> Iterator[Dict]:
        ...

    def run(self, shard_size: int = 5000):
        buf, n, shard = [], 0, 0
        for row in self.stream():
            buf.append(row)
            n += 1
            if len(buf) >= shard_size:
                self._flush(buf, shard); shard += 1; buf = []
        if buf:
            self._flush(buf, shard)
        log.info(f"[{self.name}] done: {n} rows")

    def _flush(self, rows, shard):
        out = os.path.join(self.out_dir, f"{self.name}.{shard:05d}.jsonl")
        write_jsonl(out, rows)
        log.info(f"[{self.name}] wrote {len(rows)} → {out}")

def mk_id(prefix="row"):
    return f"{prefix}-{uuid.uuid4()}"
