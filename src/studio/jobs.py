import threading, time, uuid, json, subprocess, sys
from pathlib import Path
from .storage import model_path, log_path

# in-memory job registry for demo
_JOBS = {}  # job_id -> dict(status, user_id, config, logs_path, model_dir)

def _run_local_training(job_id, user_id, config: dict):
    """Simulate or call a real training process."""
    logs = log_path(user_id, job_id)
    model_dir = model_path(user_id, job_id)
    _JOBS[job_id]["status"] = "running"
    _JOBS[job_id]["logs_path"] = logs
    _JOBS[job_id]["model_dir"] = model_dir

    with open(logs, "w", encoding="utf-8") as lf:
        try:
            lf.write(f"[job {job_id}] starting...\n")
            lf.flush()

            # Example: call your train_text.py with CPU/tiny settings
            cmd = [
                sys.executable, "src/training/train_text.py",
                "--tokenizer_dir", "outputs/tokenizer/orph_bpe_32k",
                "--train_files", config.get("sft_path", "data/clean/text_supervised.jsonl"),
                "--cot_files", config.get("cot_path", ""),
                "--curriculum", config.get("curriculum", "sft:1.0,cot:0.0"),
                "--model_name", config.get("base_model", "gpt2"),
                "--output_dir", model_dir,
                "--max_length", str(config.get("max_length", 512)),
                "--epochs", str(config.get("epochs", 1)),
                "--train_batch_size", "1",
                "--grad_accum", "1",
                "--lr", "5e-5",
                "--gradient_checkpointing", "false",
                "--auto_batch", "false",
            ]
            lf.write(f"cmd: {' '.join(cmd)}\n"); lf.flush()

            # For MacBook CPU dev, you can simulate training time:
            if config.get("simulate", True):
                for i in range(10):
                    time.sleep(1)
                    lf.write(f"[{i+1}/10] simulated step...\n"); lf.flush()
                # fake “trained” marker
                Path(model_dir).mkdir(parents=True, exist_ok=True)
                (Path(model_dir) / "MODEL_READY").write_text("ok", encoding="utf-8")
            else:
                subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[2]), check=True)

            _JOBS[job_id]["status"] = "completed"
            lf.write("done.\n")
        except Exception as e:
            _JOBS[job_id]["status"] = "failed"
            lf.write(f"ERROR: {e}\n")

def submit_job(user_id: str, config: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {"status": "queued", "user_id": user_id, "config": config}
    t = threading.Thread(target=_run_local_training, args=(job_id, user_id, config), daemon=True)
    t.start()
    return job_id

def job_status(job_id: str) -> dict:
    return _JOBS.get(job_id, {"status": "unknown"})

def job_logs(job_id: str) -> str:
    job = _JOBS.get(job_id)
    if not job: return ""
    p = job.get("logs_path")
    return Path(p).read_text(encoding="utf-8", errors="ignore") if p and Path(p).exists() else ""

def job_model_dir(job_id: str) -> str:
    job = _JOBS.get(job_id)
    return job.get("model_dir", "") if job else ""
