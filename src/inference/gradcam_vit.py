import torch
import numpy as np
from PIL import Image
from typing import Tuple
import timm
from torchvision import transforms
from grad_cam import GradCAM
from grad_cam.utils.model_targets import ClassifierOutputTarget
from grad_cam.utils.reshape_transforms import vit_reshape_transform

# (The pypi package is "grad-cam"; it exposes import path "grad_cam")

class ViTExplainer:
    def __init__(self, model_name: str = "vit_base_patch16_224", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval().to(self.device)
        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])
        # Target layer: final block norm works well with ViT + Grad-CAM package
        self.target_layers = [self.model.blocks[-1].norm1]

    @torch.inference_mode(False)
    def predict_and_cam(self, img: Image.Image) -> Tuple[str, float, np.ndarray]:
        x = self.tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
        prob = torch.softmax(logits, dim=-1)[0]
        cls_id = int(prob.argmax().item())
        confidence = float(prob[cls_id].item())

        # Grad-CAM needs gradients; enable grad context
        cam = GradCAM(model=self.model, target_layers=self.target_layers, reshape_transform=vit_reshape_transform)
        grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(cls_id)])[0]  # (H, W) in 0..1

        # class label name (ImageNet)
        try:
            labels = timm.data.resolve_data_config({}, model=self.model)
            # not strictly needed; we can return id only if label map missing
            label_name = f"class_{cls_id}"
        except Exception:
            label_name = f"class_{cls_id}"

        return label_name, confidence, grayscale_cam
