# src/data_prep/scrapers/base.py
import os, time, random, requests
from typing import Iterator, Dict, Any

# Shared, hardened HTTP helpers for all scrapers
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
    headers: Dict[str, str] | None = None,
    extra_ok_content_types: tuple[str, ...] = (),
) -> Dict[str, Any]:
    """
    Robust JSON request:
    - Retries on 429/5xx and non-JSON (HTML/XML) transient responses.
    - Only parses when Content-Type includes application/json.
    - Polite pacing via min_sleep even on success.
    """
    hdrs = {
        "Accept": "application/json",
        "User-Agent": headers.get("User-Agent", DEFAULT_USER_AGENT) if headers else DEFAULT_USER_AGENT,
    }
    if headers:
        hdrs.update(headers)

    delay = min_sleep
    last_exc: Exception | None = None
    for _ in range(tries):
        r = requests.get(url, params=params, headers=hdrs, timeout=timeout)
        ctype = (r.headers.get("Content-Type") or "").lower()

        if r.status_code in (429, 500, 502, 503, 504):
            last_exc = RuntimeError(f"{r.status_code} from server: {r.text[:200]}")
            delay = _backoff_sleep(delay)
            continue

        if "application/json" in ctype or any(t in ctype for t in extra_ok_content_types):
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
    """Base class for streaming rows into the pipeline."""
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def stream(self) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError

    def run(self) -> None:
        for _ in self.stream():
            pass  # your concrete scrapers should write rows to disk within stream()
