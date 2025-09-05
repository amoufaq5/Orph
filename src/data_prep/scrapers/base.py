# src/data_prep/scrapers/base.py
from __future__ import annotations
import os, time, random, requests
from typing import Iterator, Dict, Any, Optional, Tuple

DEFAULT_USER_AGENT = f"Orph/1.0 ({os.getenv('SCRAPER_EMAIL','noreply@example.com')})"

def _backoff_sleep(delay: float) -> float:
    time.sleep(delay + random.uniform(0, 0.3))
    return min(delay * 2, 8.0)

def req_json(
    url: str,
    params: Dict[str, Any],
    *,
    min_sleep: float = 0.34,
    tries: int = 6,
    timeout: int = 30,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Robust JSON GET:
      - Retries on 429/5xx and non-JSON responses
      - Parses only when Content-Type includes application/json
      - Polite pacing via min_sleep on success
    """
    hdrs = {
        "Accept": "application/json",
        "User-Agent": headers.get("User-Agent", DEFAULT_USER_AGENT) if headers else DEFAULT_USER_AGENT,
    }
    if headers:
        hdrs.update(headers)

    delay = min_sleep
    last_exc: Optional[Exception] = None
    for _ in range(tries):
        r = requests.get(url, params=params, headers=hdrs, timeout=timeout)
        ctype = (r.headers.get("Content-Type") or "").lower()

        if r.status_code in (429, 500, 502, 503, 504):
            last_exc = RuntimeError(f"{r.status_code} from server: {r.text[:200]}")
            delay = _backoff_sleep(delay)
            continue

        if "application/json" in ctype:
            try:
                js = r.json()
                time.sleep(min_sleep)
                return js
            except ValueError as e:
                last_exc = e
                delay = _backoff_sleep(delay)
                continue

        if "text/html" in ctype or "xml" in ctype:
            last_exc = RuntimeError(f"Non-JSON response ({ctype}): {r.text[:200]}")
            delay = _backoff_sleep(delay)
            continue

        r.raise_for_status()

    raise last_exc or RuntimeError(f"Failed after {tries} attempts: {url}")

class Scraper:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def stream(self) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError

    def run(self) -> None:
        for _ in self.stream():
            pass
