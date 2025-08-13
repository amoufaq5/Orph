# orphtools/models/visual_diagnosis.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class VisualDiagnosis:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Identity()  # Use as feature extractor
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image_path):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor).squeeze(0)
        return features.cpu()


# Example usage:
# visual = VisualDiagnosis()
# image_vec = visual.extract_features("data/raw/sample_ct_image.png")
