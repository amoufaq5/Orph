from transformers import AutoTokenizer

def load_biobert_tokenizer():
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ Loaded BioBERT tokenizer: {model_name}")
        return tokenizer
    except Exception as e:
        print(f"❌ Failed to load BioBERT tokenizer: {e}")
        raise
