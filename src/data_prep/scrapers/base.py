# src/data_prep/scrapers/base.py
from __future__ import annotations
import os, time, random, requests, json, re
from typing import Iterator, Dict, Any, Optional

DEFAULT_USER_AGENT = f"Orph/1.0 ({os.getenv('SCRAPER_EMAIL','noreply@example.com')})"

_CTL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')  # keep \t \n \r

def _backoff_sleep(delay: float) -> float:
    time.sleep(delay + random.uniform(0, 0.3))
    return min(delay * 2, 8.0)

def _try_parse_json_text(text: str) -> Any:
    # First try permissive decode (allows control chars inside strings)
    try:
        return json.loads(text, strict=False)
    except Exception:
        # Sanitize illegal ASCII controls outside of whitespace, then try again
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
    """
    Robust JSON GET:
      - Retries on 429/5xx and non-JSON (HTML/XML) transient responses.
      - Parses JSON with tolerant fallback and sanitation.
      - Polite pacing via min_sleep on success.
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
        body = r.text

        # throttle / server errors → retry
        if r.status_code in (429, 500, 502, 503, 504):
            last_exc = RuntimeError(f"{r.status_code} from server: {body[:200]}")
            delay = _backoff_sleep(delay)
            continue

        # If we expected JSON but get HTML/XML (error page), retry
        if "text/html" in ctype or "xml" in ctype:
            last_exc = RuntimeError(f"Non-JSON response ({ctype}): {body[:200]}")
            delay = _backoff_sleep(delay)
            continue

        # If JSON: try normal parse, then tolerant fallbacks
        if "application/json" in ctype:
            try:
                js = r.json()
            except ValueError:
                # If it looks like HTML anyway, treat as transient
                if body.lstrip().startswith("<"):
                    last_exc = RuntimeError(f"Claimed JSON but got HTML: {body[:200]}")
                    delay = _backoff_sleep(delay)
                    continue
                try:
                    js = _try_parse_json_text(body)
                except Exception as e:
                    last_exc = e
                    delay = _backoff_sleep(delay)
                    continue
            time.sleep(min_sleep)
            # type: ignore[return-value]
            return js  # Dict[str, Any]

        # Unexpected content-type but 2xx
        try:
            r.raise_for_status()
        except Exception as e:
            last_exc = e
            delay = _backoff_sleep(delay)
            continue

        # Last-resort tolerant parse if server mislabeled JSON
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
