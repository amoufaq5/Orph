from __future__ import annotations
import torch
from typing import Dict, Any
from PIL import Image
import torchvision.transforms as T

class ImageInference:
    def __init__(self, ckpt_path: str, backbone_factory, device: str = "cuda"):
        self.model = backbone_factory()
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    @torch.inference_mode()
    def predict(self, img: Image.Image) -> Dict[str, Any]:
        x = self.tf(img).unsqueeze(0).to(self.device)
        logits = self.model(x)
        prob = torch.softmax(logits, dim=1)[0]
        conf, idx = prob.max(dim=0)
        return {"label_index": int(idx.item()), "confidence": float(conf.item()), "probs": prob.tolist()}
