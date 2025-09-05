import json, os, itertools
from typing import List

RULES_PATH = "data/artifacts/ddi_rules.json"

def _load_rules():
    if not os.path.exists(RULES_PATH):
        return {"synonyms":{}, "pairs":[], "classes":[], "class_members":{}}
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

RULES = _load_rules()

def _normalize(name: str) -> str:
    n = name.lower().strip()
    # map synonyms to canonical
    for canon, syns in RULES.get("synonyms", {}).items():
        if n == canon or n in syns: return canon
    return n

def _to_classes(drug: str) -> List[str]:
    classes = []
    cm = RULES.get("class_members", {})
    for cls, members in cm.items():
        if drug in [m.lower() for m in members]:
            classes.append(cls)
    return classes

def check_interactions(drugs: List[str]) -> List[str]:
    if not drugs: return []
    d = [_normalize(x) for x in drugs if x]
    out = []

    # Pairwise exact rules
    pairset = RULES.get("pairs", [])
    for a,b in itertools.combinations(d, 2):
        for rule in pairset:
            ra, rb = rule["a"], rule["b"]
            if set([a,b]) == set([ra,rb]):
                out.append(f"{ra} + {rb}: {rule['note']}")

    # Class-based
    classes = {x: _to_classes(x) for x in d}
    for a,b in itertools.combinations(d, 2):
        ca, cb = classes.get(a, []), classes.get(b, [])
        for rule in RULES.get("classes", []):
            if any(x in rule["classA"] for x in ca) and any(y in rule["classB"] for y in cb):
                out.append(f"{a} (class {','.join(ca)}) + {b} (class {','.join(cb)}): {rule['note']}")

    return sorted(set(out))
