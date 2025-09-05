import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class OrphLLM:
    def __init__(self, model_dir: str, device: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="auto" if device=="auto" else None)
        if device != "auto":
            self.model.to(device)

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens=256, temperature=0.4, top_p=0.95):
        toks = self.tokenizer(prompt, return_tensors="pt")
        toks = {k: v.to(self.model.device) for k,v in toks.items()}
        out_ids = self.model.generate(
            **toks,
            max_new_tokens=max_new_tokens,
            do_sample=temperature>0,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id
        )
        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return text[len(prompt):].strip()
