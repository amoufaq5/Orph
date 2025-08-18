from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import Dataset

class TextSupervisedDataset(Dataset):
    """
    Supervised instruction/QA dataset.
    Preferred per-line format:
      {"input": "...", "target": "<rationale>\\n<final answer>"}
    Fallback if missing 'input':
      prompt := (title + "\\n" + text), and use 'label' as target if present.
    """
    def __init__(self, path_jsonl: str, tokenizer, max_length: int = 768):
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
        row = self.rows[idx]
        prompt = row.get("input")
        if not prompt:
            prompt = ((row.get("title") or "") + "\n" + (row.get("text") or "")).strip()
        target = row.get("target") or row.get("label") or ""

        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )["input_ids"]
        enc["labels"] = labels.squeeze(0)
        return {k: v.squeeze(0) for k, v in enc.items()}
