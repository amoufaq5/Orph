import re
from typing import List, Tuple

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\.]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0

def f1(pred: str, gold: str) -> float:
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p or not g:
        return 1.0 if not p and not g else 0.0
    common = {}
    for tok in p:
        if tok in g:
            common[tok] = min(p.count(tok), g.count(tok))
    num_same = sum(common.values()) if common else 0
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)

def mcq_acc(pred_idx: int, gold_idx: int) -> float:
    return 1.0 if pred_idx == gold_idx else 0.0
