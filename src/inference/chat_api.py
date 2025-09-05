from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.pipelines import Pipeline
from src.inference.safety import disclaimers

app = FastAPI(title="Orph Research API")

pipeline = Pipeline(index_dir="data/artifacts/rag", top_k=5)

class ChatIn(BaseModel):
    role: str  # patient|clinician|pharma|student
    query: str
    drugs: list[str] | None = None

@app.post("/chat")
def chat(inp: ChatIn):
    out = pipeline.answer(inp.role, inp.query, inp.drugs)
    out["disclaimer"] = disclaimers(inp.role)
    return out

@app.get("/health")
def health():
    return {"ok": True}
