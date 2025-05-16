# diagnostic_image_model.py

"""
This script loads a CNN-based model (e.g., DenseNet121) for X-ray/CT image classification
and provides Grad-CAM visualization for explainability.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Define preprocessing pipeline for medical images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load a pretrained DenseNet121 model
class DiagnosticModel(nn.Module):
    def __init__(self, num_classes=2):  # Binary (e.g., Normal vs Pneumonia)
        super(DiagnosticModel, self).__init__()
        self.backbone = models.densenet121(pretrained=True)
        self.backbone.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.backbone(x)


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension


def predict(model, image_tensor, class_names=["Normal", "Pneumonia"]):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.to(device))
        probs = torch.softmax(output, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        return class_names[pred_idx], probs.squeeze().tolist()


# Grad-CAM visualization
from torchvision.models.feature_extraction import create_feature_extractor

def grad_cam(model, image_tensor, target_layer='features.denseblock4'):
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
    return cam


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiagnosticModel()
    model.load_state_dict(torch.load("models/xray_model.pt", map_location=device))
    model.to(device)

    test_img = "uploads/test_xray.jpg"  # replace with uploaded file
    img_tensor = load_image(test_img)
    prediction, prob = predict(model, img_tensor)

    print(f"Prediction: {prediction}, Probabilities: {prob}")

    heatmap = grad_cam(model, img_tensor)
    plt.imshow(Image.open(test_img))
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title(f"{prediction} Diagnosis")
    plt.axis('off')
    plt.show()
