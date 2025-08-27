# Makefile for Orph LLM project

# Default config
CONFIG = conf/config.yaml
TOKENIZER_DIR = outputs/tokenizer/orph_bpe_32k

.PHONY: tokenizer pretrain sft serve clean

tokenizer:
	python scripts/build_tokenizer.py \
	  --input data/clean/text_pretrain.jsonl \
	  --outdir $(TOKENIZER_DIR) \
	  --vocab_size 32000 --min_frequency 2 --max_length 1024

pretrain:
	python -m src.training.train_from_scratch --config $(CONFIG)

sft:
	python -m src.training.train_from_scratch --config $(CONFIG)

serve:
	@echo "Serving model from outputs/checkpoints/orph-125m-sft ..."
	set MODEL_DIR=outputs/checkpoints/orph-125m-sft && python -m src.server.app

clean:
	rm -rf outputs/checkpoints/* outputs/reports/*
