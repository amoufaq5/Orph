from __future__ import annotations
import torch
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_GEN = dict(max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True)

class TextInference:
    def __init__(self, ckpt_dir: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        self.model = AutoModelForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16 if device=="cuda" else None)
        self.model.to(device)
        self.device = device

    @torch.inference_mode()
    def generate(self, prompt: str, n: int = 3, gen_kwargs: Dict[str, Any] | None = None) -> List[str]:
        gen_kwargs = {**DEFAULT_GEN, **(gen_kwargs or {})}
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outs = []
        for _ in range(n):
            out = self.model.generate(**inputs, **gen_kwargs)
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            outs.append(text[len(prompt):].strip())
        return outs

    def self_consistent_answer(self, prompt: str, n: int = 5) -> Dict[str, Any]:
        samples = self.generate(prompt, n=n)
        finals = [s.split('
')[-1].strip() for s in samples]
        from collections import Counter
        vote = Counter(finals)
        best, freq = vote.most_common(1)[0]
        conf = freq / max(1, len(samples))
        return {"answer": best, "confidence": conf, "samples": samples}
