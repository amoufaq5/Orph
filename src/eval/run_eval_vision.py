import os, csv, argparse
from typing import Tuple
from PIL import Image
from src.inference.vision_stub import classify_image  # ViT Grad-CAM if enabled

def run_chexpert_eval(images_root: str, csv_path: str, limit: int | None):
    # Expect CheXpert train.csv-like with "Path" and "Cardiomegaly" columns
    n, correct = 0, 0
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            img_rel = row["Path"]
            y = 1 if row.get("Cardiomegaly","") == "1.0" else 0
            img_path = os.path.join(images_root, img_rel)
            if not os.path.exists(img_path): continue
            img = Image.open(img_path).convert("RGB")
            meta, _ = classify_image(img)
            pred_has = 1 if "cardio" in meta["finding"].lower() else 0  # naive parse of placeholder/label
            correct += 1 if pred_has == y else 0
            n += 1
            if limit and n >= limit: break
    acc = correct / max(1,n)
    print(f"CheXpert Cardiomegaly top-1 acc: {acc:.4f} (n={n})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run_chexpert_eval(args.images_root, args.train_csv, args.limit)
