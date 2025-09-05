from PIL import Image
import numpy as np

def classify_image_placeholder(img: Image.Image) -> dict:
    # TODO: replace with real model inferencing
    w,h = img.size
    finding = "No acute cardiopulmonary abnormality (placeholder)"
    prob = 0.51
    return {"finding": finding, "prob": prob, "size": [w,h]}

def gradcam_placeholder(img: Image.Image) -> np.ndarray:
    # Returns a normalized heatmap 0..1 same size as image
    w,h = img.size
    hm = np.zeros((h,w), dtype=np.float32)
    # simple radial blob for demo
    cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    hm = np.exp(-(dist/(0.25*min(h,w)))**2)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-9)
    return hm
