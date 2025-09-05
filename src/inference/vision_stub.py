from PIL import Image
import numpy as np
import os

_USE_REAL = os.getenv("ORPH_USE_VIT_CAM", "0") == "1"

if _USE_REAL:
    from .gradcam_vit import ViTExplainer
    _explainer = ViTExplainer()
else:
    _explainer = None

def classify_image(img: Image.Image):
    if _explainer is None:
        # Fallback: same behavior as before
        w,h = img.size
        return {"finding": "Placeholder finding (enable ORPH_USE_VIT_CAM=1 for ViT Grad-CAM)", "prob": 0.5, "size": [w,h]}, _fake_heatmap(img)
    label, conf, cam = _explainer.predict_and_cam(img)
    return {"finding": f"Top-1: {label}", "prob": conf}, cam

def _fake_heatmap(img: Image.Image) -> np.ndarray:
    w,h = img.size
    hm = np.zeros((h,w), dtype=np.float32)
    cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    hm = np.exp(-(dist/(0.25*min(h,w)))**2)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-9)
    return hm
