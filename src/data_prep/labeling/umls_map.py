import json, os
from functools import lru_cache
from rapidfuzz import process, fuzz

_ART = os.path.join
ROOT = "data/artifacts"

def _load_map(path):
    if not os.path.exists(path): return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@lru_cache
def _icd10():  return _load_map(_ART(ROOT, "icd10_map.json"))
@lru_cache
def _snomed(): return _load_map(_ART(ROOT, "snomed_map.json"))
@lru_cache
def _rxnorm(): return _load_map(_ART(ROOT, "rxnorm_map.json"))
@lru_cache
def _meddra(): return _load_map(_ART(ROOT, "meddra_map.json"))

def _best_keys(q: str, keys: list[str], limit=3, score_cutoff=86):
    res = process.extract(q, keys, scorer=fuzz.WRatio, limit=limit, score_cutoff=score_cutoff)
    return [k for k,score,idx in res]

def map_icd10(text: str) -> list:
    m = _icd10()
    if not text: return []
    keys = _best_keys(text, list(m.keys()))
    out = []
    for k in keys: out.extend(m.get(k, []))
    return sorted(set(out))

def map_snomed(text: str) -> list:
    m = _snomed()
    if not text: return []
    keys = _best_keys(text, list(m.keys()))
    out = []
    for k in keys: out.extend(m.get(k, []))
    return sorted(set(out))

def map_rxnorm(drug_name: str) -> list:
    m = _rxnorm()
    if not drug_name: return []
    keys = _best_keys(drug_name, list(m.keys()))
    out = []
    for k in keys: out.extend(m.get(k, []))
    return sorted(set(out))

def map_meddra(text: str) -> list:
    m = _meddra()
    if not text: return []
    keys = _best_keys(text, list(m.keys()))
    out = []
    for k in keys: out.extend(m.get(k, []))
    return sorted(set(out))
