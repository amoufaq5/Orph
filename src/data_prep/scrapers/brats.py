import os, glob, argparse
import numpy as np
import nibabel as nib
from PIL import Image
from typing import Iterator, Dict
from .base import Scraper, mk_id
from src.utils.io import ensure_dir
from src.utils.logger import get_logger
log = get_logger("brats")

"""
Looks for subject folders containing *_flair.nii.gz / *_t1ce.nii.gz / *_t1.nii.gz / *_t2.nii.gz
Emits a center axial slice PNG for each available modality and a multimodal composite.
Labels are not attached here (varies by release); set answer=None, keep meta.
"""

def _load_center_slice(nii_path: str) -> np.ndarray:
    img = nib.load(nii_path)
    data = img.get_fdata()
    z = data.shape[2] // 2
    sl = data[:, :, z]
    # normalize robustly
    p1, p99 = np.percentile(sl, [1, 99])
    sl = np.clip((sl - p1) / (p99 - p1 + 1e-6), 0, 1)
    return (sl * 255).astype(np.uint8)

def _save_png(arr: np.ndarray, out_path: str):
    Image.fromarray(arr).save(out_path)

class BraTSScraper(Scraper):
    name = "brats"

    def __init__(self, out_dir: str, cases_root: str, out_img_dir: str):
        super().__init__(out_dir)
        self.cases_root = cases_root
        self.out_img_dir = out_img_dir
        ensure_dir(self.out_img_dir)

    def stream(self) -> Iterator[Dict]:
        subs = [d for d in glob.glob(os.path.join(self.cases_root, "*")) if os.path.isdir(d)]
        count = 0
        for sub in subs:
            modal = {}
            for key in ["flair","t1ce","t1","t2"]:
                cand = glob.glob(os.path.join(sub, f"*_{key}.nii*"))
                if cand:
                    modal[key] = cand[0]
            if not modal:
                continue

            # Generate per-modality slice PNGs
            pngs = {}
            for k, p in modal.items():
                arr = _load_center_slice(p)
                op = os.path.join(self.out_img_dir, f"{os.path.basename(sub)}_{k}.png")
                _save_png(arr, op)
                pngs[k] = op

            # Optional composite (stacked horizontally if >=2)
            if len(pngs) >= 2:
                arrs = [np.array(Image.open(v)) for v in pngs.values()]
                h = min(a.shape[0] for a in arrs)
                arrs = [a[:h, :] for a in arrs]
                comp = np.concatenate(arrs, axis=1)
                comp_path = os.path.join(self.out_img_dir, f"{os.path.basename(sub)}_composite.png")
                _save_png(comp, comp_path)
                target_img = comp_path
            else:
                target_img = list(pngs.values())[0]

            yield {
                "id": mk_id("brats"),
                "modality": ["image"],
                "task": "classification",
                "text": None,
                "image_path": target_img.replace("\\","/"),
                "answer": None,
                "rationale": None,
                "labels": {},
                "meta": {"source":"brats","license":"research-only","modalities": list(pngs.keys())},
                "split":"train"
            }
            count += 1
        log.info(f"[BraTS] emitted {count} rows")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases_root", required=True)
    ap.add_argument("--out_img_dir", default="data/images/brats_png")
    args = ap.parse_args()
    BraTSScraper(args.out, args.cases_root, args.out_img_dir).run()
