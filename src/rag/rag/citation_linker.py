def format_citations(hits):
    cites = []
    for h in hits:
        src = h["meta"].get("source","unknown")
        cites.append(f"[{src}] score={h['score']:.3f}")
    return " | ".join(cites)
