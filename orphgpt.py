# orphgpt.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration

class OrphGPT(nn.Module):
    def __init__(self, text_model_name="google/byt5-small", image_dim=2048, struct_dim=8, fusion_dim=768, num_classes=20):
        super().__init__()

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        text_hidden_dim = self.text_encoder.config.hidden_size

        # Image adapter
        self.image_proj = nn.Linear(image_dim, fusion_dim)

        # Structured input adapter (ASMETHOD, etc.)
        self.struct_proj = nn.Linear(struct_dim, fusion_dim)

        # Fusion layer (text + image + structured)
        self.fusion_layer = nn.Linear(text_hidden_dim + fusion_dim * 2, fusion_dim)

        # Heads
        self.diagnosis_head = nn.Linear(fusion_dim, num_classes)  # multi-label
        self.refer_head = nn.Linear(fusion_dim, 1)                # binary
        self.summary_head = T5ForConditionalGeneration.from_pretrained("google/byt5-small")

    def forward(self, input_text, image_vector=None, struct_vector=None):
        # Encode text
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        text_out = self.text_encoder(**inputs).last_hidden_state[:, 0, :]  # [CLS] token

        # Encode image
        image_feat = self.image_proj(image_vector) if image_vector is not None else torch.zeros_like(text_out)

        # Encode structured
        struct_feat = self.struct_proj(struct_vector) if struct_vector is not None else torch.zeros_like(text_out)

        # Fuse
        combined = torch.cat([text_out, image_feat, struct_feat], dim=1)
        fused = self.fusion_layer(combined)

        # Outputs
        diagnosis_logits = self.diagnosis_head(fused)
        refer_logits = self.refer_head(fused)

        return {
            "fused": fused,
            "diagnosis_logits": diagnosis_logits,
            "refer_logits": refer_logits
        }

    def generate_summary(self, input_text, max_len=128):
        tokens = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        out_ids = self.summary_head.generate(**tokens, max_length=max_len)
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
