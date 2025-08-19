# src/data/builders/text_supervised_builder.py
from torch.utils.data import Dataset
from typing import Dict
import json

class TextSupervisedDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"No rows in {path}")

        # Optional: validate keys
        # expected: task,input,target
        # you can relax this if your jsonl differs

    def __len__(self):
        return len(self.rows)

    def _build_texts(self, row: Dict) -> Dict[str, str]:
        # Prompt format — keep it stable
        task = row.get("task", "instruction")
        user = row.get("input", "")
        target = row.get("target", "")

        prompt = (
            f"[TASK]: {task}\n"
            f"[INPUT]: {user}\n"
            f"[RESPONSE]: "
        )
        answer = target

        return {"prompt": prompt, "answer": answer}

    def __getitem__(self, idx: int) -> Dict[str, list]:
        row = self.rows[idx]
        parts = self._build_texts(row)
        prompt, answer = parts["prompt"], parts["answer"]

        # Tokenize separately to know prompt length
        prompt_enc = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length
        )
        answer_enc = self.tokenizer(
            answer if len(answer) else " ",  # avoid empty
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length
        )

        # Concatenate within max_length
        input_ids = prompt_enc["input_ids"] + answer_enc["input_ids"]
        attention_mask = [1] * len(input_ids)

        # Truncate to max_length (important!)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        # Labels: ignore prompt tokens (no loss), learn on answer tokens only
        labels = [-100] * len(prompt_enc["input_ids"]) + answer_enc["input_ids"]
        if len(labels) > self.max_length:
            labels = labels[: self.max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
