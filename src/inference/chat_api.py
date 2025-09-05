from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import io, base64, os
from zipfile import ZipFile
import tempfile
import nibabel as nib

from src.inference.pipelines import Pipeline
from src.inference.safety import disclaimers
from src.inference.vision_stub import classify_image  # real ViT Grad-CAM if ORPH_USE_VIT_CAM=1
from src.utils.config import load_config

app = FastAPI(title="Orph Research API", version="1.0")

# CORS for local frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ORPH_CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config + pipeline (RAG + OrphGPT)
_cfg = load_config()
pipeline = Pipeline(index_dir=_cfg.main.get("paths", {}).get("rag_index", "data/artifacts/rag"), top_k=_cfg.main.get("rag", {}).get("top_k", 5))

class ChatIn(BaseModel):
    role: str        # patient|clinician|pharma|student
    query: str
    drugs: list[str] | None = None

class ChatOut(BaseModel):
    role: str
    answer: str
    disclaimer: str

class VQAOut(BaseModel):
    role: str
    finding: str
    probability: float
    disclaimer: str
    heatmap_png_b64: str | None = None

class VQAMRIOut(BaseModel):
    role: str
    summary: str
    disclaimer: str
    heatmap_png_b64: str | None = None

def _encode_heatmap_rgba(img: Image.Image, heatmap_0_1: np.ndarray) -> str:
    im = img.convert("L").resize((heatmap_0_1.shape[1], heatmap_0_1.shape[0]))
    arr = np.array(im).astype(np.float32) / 255.0
    alpha = 0.5
    overlay = (1 - alpha) * arr + alpha * heatmap_0_1
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    out = Image.fromarray(overlay)
    bio = io.BytesIO(); out.save(bio, format="PNG")
    return base64.b64encode(bio.getvalue()).decode("ascii")

def _center_slice_png_from_zip(content: bytes) -> str | None:
    with tempfile.TemporaryDirectory() as td:
        zf = ZipFile(io.BytesIO(content))
        zf.extractall(td)
        slices = []
        for root,_,files in os.walk(td):
            for fn in files:
                if fn.lower().endswith((".nii",".nii.gz")):
                    arr = nib.load(os.path.join(root, fn)).get_fdata()
                    if arr.ndim != 3:  # skip non-3D volumes
                        continue
                    z = arr.shape[2] // 2
                    sl = arr[:, :, z]
                    p1, p99 = np.percentile(sl, [1, 99])
                    sl = np.clip((sl - p1) / (p99 - p1 + 1e-6), 0, 1)
                    slices.append((sl * 255).astype(np.uint8))
        if not slices:
            return None
        h = min(s.shape[0] for s in slices)
        slices = [s[:h, :] for s in slices]
        comp = np.concatenate(slices, axis=1)
        im = Image.fromarray(comp).convert("L")
        bio = io.BytesIO(); im.save(bio, format="PNG")
        return base64.b64encode(bio.getvalue()).decode("ascii")

@app.get("/")
def root():
    return {"name": "Orph Research API", "version": app.version, "endpoints": ["/chat", "/vqa", "/vqa_mri", "/health"]}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat", response_model=ChatOut)
def chat(inp: ChatIn):
    out = pipeline.answer(inp.role, inp.query, inp.drugs)
    return ChatOut(role=inp.role, answer=out["answer"], disclaimer=disclaimers(inp.role))

@app.post("/vqa", response_model=VQAOut)
async def vqa(role: str, file: UploadFile = File(...)):
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    meta, hm = classify_image(img)  # returns ({"finding","prob"}, heatmap[0..1])
    hm_b64 = _encode_heatmap_rgba(img, hm)
    return VQAOut(role=role, finding=meta["finding"], probability=float(meta["prob"]), disclaimer=disclaimers(role), heatmap_png_b64=hm_b64)

@app.post("/vqa_mri", response_model=VQAMRIOut)
async def vqa_mri(role: str, file: UploadFile = File(...)):
    content = await file.read()
    b64 = _center_slice_png_from_zip(content)
    if b64 is None:
        return VQAMRIOut(role=role, summary="No NIfTI volumes detected in the archive.", disclaimer=disclaimers(role))
    return VQAMRIOut(role=role, summary="Composite center-slice preview generated (research mode).", disclaimer=disclaimers(role), heatmap_png_b64=b64)
