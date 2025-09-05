# Orph Research (bootstrap)
1) `python -m venv .venv && . .venv/Scripts/activate` (Windows PowerShell) or `. .venv/bin/activate` (Linux/Mac)
2) `pip install -r requirements.txt`
3) Run scrapers + index: `./scripts/run_orph.ps1` (Windows) or `bash scripts/run_orph.sh`
4) Launch API: `uvicorn src.inference.chat_api:app --reload --host 0.0.0.0 --port 8000`
5) Frontend dev: proxy `/api` to `http://localhost:8000` and run your React app(s).
