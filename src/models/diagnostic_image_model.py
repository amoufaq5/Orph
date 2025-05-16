# diagnostic_image_model.py with Grad-CAM image saving and multi-class support

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Example disease classes (can be extended)
CLASS_NAMES = ["Normal", "Pneumonia", "COVID-19", "Tuberculosis"]

class DiagnosticModel(nn.Module):
    def __init__(self, num_classes=len(CLASS_NAMES)):
        super(DiagnosticModel, self).__init__()
        self.backbone = models.densenet121(pretrained=True)
        self.backbone.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.backbone(x)


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(model, image_tensor, class_names=CLASS_NAMES):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.to(device))
        probs = torch.softmax(output, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        return class_names[pred_idx], probs.squeeze().tolist()


from torchvision.models.feature_extraction import create_feature_extractor

def grad_cam(model, image_tensor, target_layer='features.denseblock4', output_path=None):
    model.eval()
    extractor = create_feature_extractor(model.backbone, return_nodes={target_layer: 'feat'})
    image_tensor.requires_grad_()
    feat_out = extractor(image_tensor.to(device))['feat']

    output = model(image_tensor.to(device))
    pred_class = output.argmax(dim=1)
    one_hot = torch.zeros_like(output).to(device)
    one_hot[0, pred_class] = 1
    output.backward(gradient=one_hot)

    gradients = image_tensor.grad
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    cam = torch.zeros(feat_out.shape[2:]).to(device)
    for i in range(pooled_gradients.shape[0]):
        cam += pooled_gradients[i] * feat_out[0, i, :, :]
    cam = torch.clamp(cam, min=0).cpu().detach().numpy()
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cam = np.expand_dims(cam, axis=2)
    cam = np.repeat(cam, 3, axis=2)

    original = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    original = (original * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    original = np.clip(original, 0, 1)

    fig, ax = plt.subplots()
    ax.imshow(original)
    ax.imshow(cam, cmap='jet', alpha=0.4)
    ax.axis('off')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiagnosticModel()
    model.load_state_dict(torch.load("models/xray_model.pt", map_location=device))
    model.to(device)

    test_img = "uploads/test_xray.jpg"
    img_tensor = load_image(test_img)
    prediction, prob = predict(model, img_tensor)
    print(f"Prediction: {prediction}, Probabilities: {prob}")

    grad_cam(model, img_tensor, output_path="outputs/gradcam_overlay.jpg")
    print("Grad-CAM image saved to outputs/gradcam_overlay.jpg")
