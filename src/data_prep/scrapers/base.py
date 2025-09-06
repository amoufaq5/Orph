# src/data_prep/scrapers/base.py
from __future__ import annotations
import os, time, random, requests, json, re
from typing import Dict, Any, Optional, Iterator

__all__ = ["Scraper", "req_json"]

DEFAULT_USER_AGENT = f"Orph/1.0 ({os.getenv('SCRAPER_EMAIL','noreply@example.com')})"
_CTL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')  # keep \t \n \r

def _backoff_sleep(delay: float) -> float:
    time.sleep(delay + random.uniform(0, 0.3))
    return min(delay * 2, 8.0)

def _try_parse_json_text(text: str) -> Any:
    try:
        return json.loads(text, strict=False)
    except Exception:
        clean = _CTL_RE.sub('', text)
        return json.loads(clean, strict=False)

def req_json(
    url: str,
    params: Dict[str, Any],
    *,
    min_sleep: float = 0.34,
    tries: int = 8,
    timeout: int = 45,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Robust JSON GET with retry/backoff and tolerant parsing."""
    hdrs = {"Accept": "application/json", "User-Agent": DEFAULT_USER_AGENT}
    if headers: hdrs.update(headers)

    delay = min_sleep
    last_exc: Optional[Exception] = None
    for _ in range(tries):
        r = requests.get(url, params=params, headers=hdrs, timeout=timeout)
        ctype = (r.headers.get("Content-Type") or "").lower()
        body = r.text

        if r.status_code in (429, 500, 502, 503, 504):
            last_exc = RuntimeError(f"{r.status_code} from server: {body[:200]}")
            delay = _backoff_sleep(delay); continue

        if "text/html" in ctype or "xml" in ctype:
            last_exc = RuntimeError(f"Non-JSON response ({ctype}): {body[:200]}")
            delay = _backoff_sleep(delay); continue

        if "application/json" in ctype:
            try:
                js = r.json()
            except Exception:
                if body.lstrip().startswith("<"):
                    last_exc = RuntimeError(f"Claimed JSON but got HTML: {body[:200]}")
                    delay = _backoff_sleep(delay); continue
                try:
                    js = _try_parse_json_text(body)
                except Exception as e:
                    last_exc = e
                    delay = _backoff_sleep(delay); continue
            time.sleep(min_sleep)
            return js  # type: ignore[return-value]

        try:
            r.raise_for_status()
        except Exception as e:
            last_exc = e
            delay = _backoff_sleep(delay); continue

        try:
            js = _try_parse_json_text(body)
            time.sleep(min_sleep)
            return js
        except Exception as e:
            last_exc = e
            delay = _backoff_sleep(delay)

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
