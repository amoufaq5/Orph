from __future__ import annotations
import torch
import torch.nn.functional as F

class GradCAM:
    """
    Generic Grad-CAM for CNNs/ViTs. You must point to a feature map layer.
    Example:
        cam = GradCAM(model, target_layer_name="layer4")   # ResNet50
        logits = model(x)
        heat = cam(logits, class_idx=1)  # (B,H,W) in [0,1]
    """
    def __init__(self, model, target_layer_name: str):
        self.model = model
        modules = dict(model.named_modules())
        if target_layer_name not in modules:
            raise ValueError(f"Layer {target_layer_name} not found.")
        self.layer = modules[target_layer_name]
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()
        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.layer.register_forward_hook(fwd_hook)
        self.layer.register_full_backward_hook(bwd_hook)

    def __call__(self, logits: torch.Tensor, class_idx: int) -> torch.Tensor:
        # logits shape: (B, C)
        score = logits[:, class_idx].sum()
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        grads = self.gradients      # (B,C,H,W)
        acts  = self.activations    # (B,C,H,W)
        if grads is None or acts is None:
            raise RuntimeError("Hooks failed; check target layer name.")

        weights = grads.mean(dim=(2, 3), keepdim=True)     # (B,C,1,1)
        cam = (weights * acts).sum(dim=1)                  # (B,H,W)
        cam = F.relu(cam)

        # normalize to [0,1] per-sample
        B = cam.shape[0]
        cam = cam.view(B, -1)
        minv = cam.min(dim=1, keepdim=True)[0]
        maxv = cam.max(dim=1, keepdim=True)[0]
        cam = (cam - minv) / (maxv - minv + 1e-6)
        return cam.view(-1, *self.activations.shape[2:])   # (B,H,W)
