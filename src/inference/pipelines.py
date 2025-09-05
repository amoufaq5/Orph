from src.rag.retriever import Retriever
from src.rag.citation_linker import format_citations
from src.tools.ddi_checker import check_interactions

class Pipeline:
    def __init__(self, index_dir: str, top_k: int = 5):
        self.retriever = Retriever(index_dir, top_k=top_k)

    def answer(self, role: str, query: str, drugs: list[str] | None = None):
        # 1) Retrieve evidence
        hits = self.retriever.search(query)
        citations = format_citations(hits)

        # 2) Simple heuristic “reasoning”: pick the most relevant snippet
        summary = hits[0]["text"] if hits else "No evidence found."

        # 3) Optional DDI tool
        ddi = check_interactions(drugs) if drugs else []

        # 4) Compose response (Replace this with your LLM call)
        reply = f"{summary}\n\nEvidence: {citations}"
        if ddi:
            reply += "\n\nPotential drug interactions:\n- " + "\n- ".join(ddi)

        # 5) Safety: patient role gets simpler language
        if role == "patient":
            reply = "Here’s what I found (simple):\n" + reply

        return {"role": role, "answer": reply, "citations": hits, "ddi": ddi}
