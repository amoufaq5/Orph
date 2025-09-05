import re

def normalize_text(s: str) -> str:
    s = s.replace("\u00A0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def is_language_ok(s: str, allow=("en",)):
    # TODO: plug fastText langid; now naive
    return True
