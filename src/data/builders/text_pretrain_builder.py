from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import Dataset

class TextPretrainDataset(Dataset):
    """
    JSONL -> causal LM dataset.
    Each line must have: {"text": "..."}.
    Use with DataCollatorForLanguageModeling(mlm=False).
    """
    def __init__(self, path_jsonl: str, tokenizer, max_length: int = 1024):
        self.path = Path(path_jsonl)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rows = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.rows[idx].get("text", "")
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # For causal LM, labels == input_ids
        enc["labels"] = enc["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in enc.items()}
