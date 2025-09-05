from typing import List, Optional
from src.rag.retriever import Retriever
from src.rag.citation_linker import format_citations
from src.tools.ddi_checker import check_interactions
from src.inference.llm import OrphLLM
from src.utils.config import load_config

PROMPT_TMPL = """You are Orph Research, a medical assistant. Answer the user's query using only the EVIDENCE provided. 
Cite sources inline as [#] indices corresponding to the evidence items. Be concise, evidence-first, and include a short, verifiable rationale.
If safety flags or missing information appear, say so and recommend next steps.

User Query:
{query}

Evidence:
{evidence}

Output:
"""

def build_evidence_block(hits) -> str:
    lines = []
    for i,h in enumerate(hits, start=1):
        src = h["meta"].get("source","unknown")
        lines.append(f"[{i}] ({src}) {h['text'][:800]}")
    return "\n".join(lines)

class Pipeline:
    def __init__(self, index_dir: str, top_k: int = 5):
        cfg = load_config()
        inf = cfg.main.get("inference", {})
        self.top_k = top_k
        self.retriever = Retriever(index_dir, top_k=top_k)
        self.llm = OrphLLM(inf.get("model_dir","./out/text_orphgpt"), device=inf.get("device","auto"))
        self.gen_args = {
            "max_new_tokens": inf.get("max_new_tokens", 256),
            "temperature": inf.get("temperature", 0.4),
            "top_p": inf.get("top_p", 0.95),
        }

    def answer(self, role: str, query: str, drugs: Optional[List[str]] = None):
        # 1) Retrieve evidence
        hits = self.retriever.search(query)
        citations = format_citations(hits)
        ev_block = build_evidence_block(hits) if hits else "No relevant passages were retrieved."

        # 2) Tool call: DDI if drugs provided
        ddi = check_interactions(drugs) if drugs else []

        # 3) Compose LLM prompt
        prompt = PROMPT_TMPL.format(query=query, evidence=ev_block)
        llm_text = self.llm.generate(prompt, **self.gen_args)

        # 4) Append DDI findings
        if ddi:
            llm_text += "\n\nDrug–Drug Interactions detected:\n- " + "\n- ".join(ddi)

        # 5) Patient simplification handled at UI level; we keep medical fidelity here
        return {"role": role, "answer": llm_text, "citations": hits, "ddi": ddi}
