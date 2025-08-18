# Orph – Backend (Production Skeleton)

A clean, modular backend for Orph: training, inference, adapters, and a Flask API ready for your existing orphtools logic.

---

## ✨ Features

- **Unified config** (`conf/config.yaml`) + **.env overrides** (no secrets in code)
- **Datasets**: pretrain & supervised builders (`TextPretrainDataset`, `TextSupervisedDataset`)
- **Training** with Hugging Face `Trainer`
- **Inference**: text (LLM) + image hooks, self-consistency sampling
- **Clinical logic**: adapters for referral/clarification/ASMETHOD
- **API**: Flask + middleware (API key), typed schemas, CORS
- **Explainability**: Grad-CAM for vision
- **Adapters**: bridge your existing `orph_tools` into `src/*`

---

## 📁 Directory Layout

