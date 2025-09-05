# src/data_prep/scrapers/base.py
from __future__ import annotations
import os, json, time, math, tempfile, random, threading
from abc import ABC, abstractmethod
from typing import Dict, Iterator, Optional, Iterable, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.io import ensure_dir
from src.utils.logger import get_logger

log = get_logger("scraper")

def mk_id(prefix: str) -> str:
    import uuid
    return f"{prefix}-{uuid.uuid4()}"

class RateLimiter:
    """
    Token-bucket limiter. calls_per_sec with optional burst.
    Thread-safe; sleep() blocks until a token is available.
    """
    def __init__(self, calls_per_sec: float, burst: int = 1):
        self.capacity = max(1, burst)
        self.tokens = self.capacity
        self.rate = float(calls_per_sec)
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def sleep(self):
        with self.lock:
            now = time.monotonic()
            # refill tokens
            elapsed = now - self.last
            self.last = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens >= 1:
                self.tokens -= 1
                return
            # need to wait
            need = 1 - self.tokens
            wait = need / self.rate
        time.sleep(wait)
        # after sleeping, consume token
        with self.lock:
            self.tokens = max(0, self.tokens - 1)

class HttpClient:
    """
    requests.Session with robust retries and backoff.
    - Retries 429/5xx with exponential backoff
    - Honors Retry-After
    - Adds jitter to sleeps
    """
    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 45,
        total_retries: int = 5,
        backoff_factor: float = 0.5,
        status_forcelist: Iterable[int] = (429, 500, 502, 503, 504),
        allowed_methods: Iterable[str] = ("GET", "POST"),
        headers: Optional[Dict[str, str]] = None,
    ):
        self.base = base_url.rstrip("/") if base_url else None
        self.timeout = timeout
        self.sess = requests.Session()
        retry = Retry(
            total=total_retries,
            read=total_retries,
            connect=total_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=frozenset(m.upper() for m in allowed_methods),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=32)
        self.sess.mount("https://", adapter)
        self.sess.mount("http://", adapter)
        if headers:
            self.sess.headers.update(headers)

    def _url(self, path_or_url: str) -> str:
        if self.base and not path_or_url.startswith(("http://","https://")):
            return f"{self.base}/{path_or_url.lstrip('/')}"
        return path_or_url

    def get(self, path_or_url: str, params: Dict[str, Any] | None = None, **kw) -> requests.Response:
        r = self.sess.get(self._url(path_or_url), params=params, timeout=self.timeout, **kw)
        return r

    def json(self, path_or_url: str, params: Dict[str, Any] | None = None, **kw) -> Dict[str, Any]:
        r = self.get(path_or_url, params=params, **kw)
        r.raise_for_status()
        return r.json()

    def text(self, path_or_url: str, params: Dict[str, Any] | None = None, **kw) -> str:
        r = self.get(path_or_url, params=params, **kw)
        r.raise_for_status()
        return r.text

class Scraper(ABC):
    """
    Base scraper:
      - call self.stream() to yield rows (dicts)
      - automatic sharded JSONL writing
      - supports --max_docs and --shard_size
    """
    name = "base"

    def __init__(
        self,
        out_dir: str,
        client: Optional[HttpClient] = None,
        shard_size: int = 5000,
        max_docs: Optional[int] = None,
    ):
        self.out_dir = out_dir
        self.client = client
        self.shard_size = max(1, shard_size)
        self.max_docs = max_docs
        ensure_dir(out_dir)

    @abstractmethod
    def stream(self) -> Iterator[Dict]:
        ...

    def run(self):
        shard_rows = []
        total = 0
        shard_idx = 0
        try:
            for row in self.stream():
                shard_rows.append(row)
                total += 1
                if self.max_docs and total >= self.max_docs:
                    self._flush(shard_rows, shard_idx); shard_rows = []
                    break
                if len(shard_rows) >= self.shard_size:
                    self._flush(shard_rows, shard_idx)
                    shard_rows = []
                    shard_idx += 1
            if shard_rows:
                self._flush(shard_rows, shard_idx)
        except KeyboardInterrupt:
            log.warning(f"[{self.name}] interrupted; flushing partial shard.")
            if shard_rows:
                self._flush(shard_rows, shard_idx)
        log.info(f"[{self.name}] DONE rows={total}")

    def _flush(self, rows: list[Dict], shard_idx: int):
        if not rows: return
        fname = f"{self.name}.{shard_idx:05d}.jsonl"
        out_path = os.path.join(self.out_dir, fname)
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=f".{fname}.", dir=self.out_dir)
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp_path, out_path)
        log.info(f"[{self.name}] wrote {len(rows)} → {out_path}")
