# orphtools/models/fusion_engine.py
import torch
import torch.nn as nn

class FusionEngine(nn.Module):
    def __init__(self, text_dim, image_dim, structured_dim, output_dim):
        super(FusionEngine, self).__init__()
        self.text_proj = nn.Linear(text_dim, 256)
        self.image_proj = nn.Linear(image_dim, 256)
        self.struct_proj = nn.Linear(structured_dim, 64)

        self.fusion_layer = nn.Sequential(
            nn.Linear(256 + 256 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, text_vec, image_vec, struct_vec):
        t = self.text_proj(text_vec)
        i = self.image_proj(image_vec)
        s = self.struct_proj(struct_vec)
        combined = torch.cat([t, i, s], dim=-1)
        return self.fusion_layer(combined)


# Example usage:
# fusion = FusionEngine(text_dim=768, image_dim=512, structured_dim=10, output_dim=5)
# output = fusion(text_vec, image_vec, struct_vec)
