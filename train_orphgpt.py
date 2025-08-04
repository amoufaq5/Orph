# train_orphgpt.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from orphgpt import OrphGPT

# ---------- Hyperparameters ----------
EPOCHS = 5
BATCH_SIZE = 16
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Custom Dataset ----------
class OrphDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.fillna('')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        return {
            'symptoms': row['symptoms'],
            'image_vector': torch.tensor(eval(row['image_vector']), dtype=torch.float),
            'asmethod_vector': torch.tensor(eval(row['asmethod_vector']), dtype=torch.float),
            'diagnosis_label': torch.tensor(eval(row['diagnosis_label']), dtype=torch.float),
            'refer_label': torch.tensor([row['refer_label']], dtype=torch.float)
        }

# ---------- Training Function ----------
def train():
    dataset = OrphDataset("data/final/train_multimodal.csv")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = OrphGPT(num_classes=10)  # change class count as needed
    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_diag = nn.BCEWithLogitsLoss()
    loss_ref = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        total_loss = 0

        for batch in pbar:
            symptoms = batch['symptoms']
            image_vector = batch['image_vector'].to(DEVICE)
            struct_vector = batch['asmethod_vector'].to(DEVICE)
            y_diag = batch['diagnosis_label'].to(DEVICE)
            y_ref = batch['refer_label'].to(DEVICE)

            outputs = model(symptoms, image_vector, struct_vector)
            pred_diag = outputs["diagnosis_logits"]
            pred_ref = outputs["refer_logits"]

            loss1 = loss_diag(pred_diag, y_diag)
            loss2 = loss_ref(pred_ref, y_ref)
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})

    torch.save(model.state_dict(), "orphgpt_multimodal.pt")
    print("✅ Model saved to orphgpt_multimodal.pt")

# ---------- Entry Point ----------
if __name__ == "__main__":
    train()
