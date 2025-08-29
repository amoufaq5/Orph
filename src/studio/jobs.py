import threading, time, uuid, subprocess, sys
from pathlib import Path

from .config import TOKENIZER_DIR, ROOT
from .storage import model_dir, logs_path, upsert_registry_entry, set_status, set_model_dir
from . import reporter

# In-memory runtime job state (lightweight)
_JOBS = {}  # job_id -> dict(status, user_id, config, logs_path, model_dir)

def _write_log(logfile: Path, text: str):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a", encoding="utf-8") as lf:
        lf.write(text.rstrip() + "\n")
        lf.flush()

def _run_local_training(job_id: str, user_id: str, config: dict):
    """
    Simulated (or real) training runner.
    If config['simulate'] is True (default), writes 10 simulated steps.
    Otherwise, calls src/training/train_text.py with CPU-safe flags.
    """
    status = "running"
    _JOBS[job_id]["status"] = status
    set_status(job_id, status)

    logf = logs_path(user_id, job_id)
    out_dir = model_dir(user_id, job_id)
    _JOBS[job_id]["logs_path"] = str(logf)
    _JOBS[job_id]["model_dir"] = str(out_dir)
    set_model_dir(job_id, str(out_dir))

    _write_log(logf, f"[{job_id}] starting; user={user_id}")
    _write_log(logf, f"config={config}")

    try:
        simulate = config.get("simulate", True)

        if simulate:
            steps = int(config.get("sim_steps", 10))
            for i in range(steps):
                time.sleep(1)
                _write_log(logf, f"[{i+1}/{steps}] simulated step...")
            # mark as “trained”
            (out_dir / "MODEL_READY").write_text("ok", encoding="utf-8")
        else:
            # Real call (CPU-safe defaults)
            cmd = [
                sys.executable, "src/training/train_text.py",
                "--tokenizer_dir", str(TOKENIZER_DIR),
                "--train_files", config.get("sft_path", "data/clean/text_supervised.jsonl"),
                "--cot_files", config.get("cot_path", ""),
                "--curriculum", config.get("curriculum", "sft:1.0,cot:0.0"),
                "--model_name", config.get("base_model", "gpt2"),
                "--output_dir", str(out_dir),
                "--max_length", str(config.get("max_length", 512)),
                "--epochs", str(config.get("epochs", 1)),
                "--train_batch_size", str(config.get("train_batch_size", 1)),
                "--grad_accum", str(config.get("grad_accum", 1)),
                "--lr", str(config.get("lr", 5e-5)),
                "--gradient_checkpointing", str(config.get("gradient_checkpointing", "false")),
                "--auto_batch", "false",
            ]
            _write_log(logf, "cmd: " + " ".join(cmd))
            subprocess.run(cmd, cwd=str(ROOT), check=True)

        # Generate report (works even without a real model; will skip quick gen)
        reporter.generate_report(
            job_id=job_id,
            user_id=user_id,
            out_dir=out_dir,
            sft_path=Path(config.get("sft_path","")) if config.get("sft_path") else None,
            cot_path=Path(config.get("cot_path","")) if config.get("cot_path") else None,
            tokenizer_dir=TOKENIZER_DIR,
            model_dir=out_dir if (out_dir / "MODEL_READY").exists() else None,
            eval_path=Path(config.get("eval_path","")) if config.get("eval_path") else None,
            config=config
        )

        status = "completed"
        _JOBS[job_id]["status"] = status
        set_status(job_id, status)
        _write_log(logf, "done.")
    except Exception as e:
        status = "failed"
        _JOBS[job_id]["status"] = status
        set_status(job_id, status)
        _write_log(logf, f"ERROR: {e}")

def submit_job(user_id: str, config: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {"status": "queued", "user_id": user_id, "config": config}
    upsert_registry_entry(job_id, user_id, "queued", "", config)
    t = threading.Thread(target=_run_local_training, args=(job_id, user_id, config), daemon=True)
    t.start()
    return job_id

def job_status(job_id: str) -> dict:
    job = _JOBS.get(job_id)
    return {"job_id": job_id, **job} if job else {"job_id": job_id, "status": "unknown"}

def job_logs(job_id: str) -> str:
    job = _JOBS.get(job_id, {})
    p = Path(job.get("logs_path",""))
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""

def job_model_dir(job_id: str) -> str:
    job = _JOBS.get(job_id, {})
    return job.get("model_dir","")