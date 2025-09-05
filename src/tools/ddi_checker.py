# Minimal offline DDI checker (placeholder). Replace with RxNorm/ONCHigh-level later.
# Input: list of drug names; Output: list of potential interactions (string notes).
KNOWN = {
  ("ibuprofen","warfarin"): "↑ bleeding risk (avoid or monitor INR)",
  ("clarithromycin","simvastatin"): "↑ statin levels (rhabdomyolysis risk) — avoid"
}

def check_interactions(drugs: list[str]) -> list[str]:
    d = [x.lower().strip() for x in drugs]
    out = []
    for a in range(len(d)):
        for b in range(a+1, len(d)):
            key = tuple(sorted((d[a], d[b])))
            # normalize to our keys
            for k,v in KNOWN.items():
                if set(k) == set(key): out.append(f"{k[0]} + {k[1]}: {v}")
    return out
