# Minimal stub; replace with your real implementation later.
from torch.utils.data import Dataset

class ImageClassificationDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.items = []
    def __len__(self):
        return 0
    def __getitem__(self, idx):
        raise IndexError("No items in ImageClassificationDataset (stub).")
