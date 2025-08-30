import threading, time, uuid, subprocess, sys, os
from pathlib import Path

from .config import TOKENIZER_DIR, ROOT, CLOUD_BACKEND
from .storage import model_dir, logs_path, upsert_registry_entry, set_status, set_model_dir
from . import reporter
import shutil

# Optional: RunPod adapter
from .adapters import runpod_adapter as rp

_JOBS = {}  # job_id -> dict(status, user_id, config, logs_path, model_dir)

def _write_log(logfile: Path, text: str):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a", encoding="utf-8") as lf:
        lf.write(text.rstrip() + "\n")
        lf.flush()

def _finish_and_report(job_id, user_id, out_dir: Path, logf: Path, config: dict, ok: bool):
    # Report (works even without a real model; quick eval will be skipped)
    try:
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
    except Exception as e:
        _write_log(logf, f"[report] ERROR: {e}")

    status = "completed" if ok else "failed"
    _JOBS[job_id]["status"] = status
    set_status(job_id, status)
    _write_log(logf, status)


# -------------------- DATASET BUILD BACKEND (local) --------------------
def _run_build_dataset(job_id: str, user_id: str, config: dict):
    """
    Runs src/data_prep/build_dataset.py and captures artifacts into a per-job dir.
    Config accepts:
      {
        "seed": 42,
        "train_ratio": 0.94, "val_ratio": 0.03, "test_ratio": 0.03
      }
    """
    _JOBS[job_id]["status"] = "running"; set_status(job_id, "running")
    logf = logs_path(user_id, job_id)
    out_dir = build_job_dir(user_id, job_id)
    _JOBS[job_id]["logs_path"] = str(logf)
    _JOBS[job_id]["model_dir"] = str(out_dir)   # reuse for downloads/report endpoints
    set_model_dir(job_id, str(out_dir))
    add_tag(job_id, "build")

    def _log(s): _write_log(logf, s)

    try:
        _log(f"[{job_id}] starting dataset build; user={user_id}")
        _log(f"config={config}")

        # Assemble CLI
        args = [
            sys.executable, "src/data_prep/build_dataset.py",
            "--seed", str(config.get("seed", 42)),
            "--train_ratio", str(config.get("train_ratio", 0.94)),
            "--val_ratio", str(config.get("val_ratio", 0.03)),
            "--test_ratio", str(config.get("test_ratio", 0.03)),
        ]
        _log("cmd: " + " ".join(args))
        r = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True)
        if r.stdout: _log(r.stdout.strip())
        if r.stderr: _log("[stderr] " + r.stderr.strip())
        if r.returncode != 0:
            raise RuntimeError(f"build_dataset.py exited {r.returncode}")

        # Copy artifacts into this job's dir for easy retrieval
        clean_dir = ROOT / "data" / "clean"
        for name in ["text_supervised.jsonl", "text_cot.jsonl", "stats.json"]:
            src = clean_dir / name
            if src.exists():
                shutil.copy2(src, out_dir / name)
        # Write a tiny manifest
        (out_dir / "BUILD_READY").write_text("ok", encoding="utf-8")
        (out_dir / "build_report.json").write_text(
            json.dumps({
                "job_id": job_id,
                "user_id": user_id,
                "artifacts": [p.name for p in out_dir.glob("*") if p.is_file()],
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        _JOBS[job_id]["status"] = "completed"; set_status(job_id, "completed")
        _log("done.")
    except Exception as e:
        _JOBS[job_id]["status"] = "failed"; set_status(job_id, "failed")
        _log(f"ERROR: {e}")

# -------------------- LOCAL BACKEND --------------------
def _run_local_training(job_id: str, user_id: str, config: dict):
    _JOBS[job_id]["status"] = "running"; set_status(job_id, "running")
    logf = logs_path(user_id, job_id)
    out_dir = model_dir(user_id, job_id)
    _JOBS[job_id]["logs_path"] = str(logf)
    _JOBS[job_id]["model_dir"] = str(out_dir)
    set_model_dir(job_id, str(out_dir))

    _write_log(logf, f"[{job_id}] starting local; user={user_id}")
    _write_log(logf, f"config={config}")

    try:
        simulate = config.get("simulate", True)
        if simulate:
            steps = int(config.get("sim_steps", 10))
            for i in range(steps):
                time.sleep(1)
                _write_log(logf, f"[{i+1}/{steps}] simulated step...")
            (out_dir / "MODEL_READY").write_text("ok", encoding="utf-8")
        else:
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

        _finish_and_report(job_id, user_id, out_dir, logf, config, ok=True)
    except Exception as e:
        _write_log(logf, f"ERROR: {e}")
        _finish_and_report(job_id, user_id, out_dir, logf, config, ok=False)

# -------------------- RUNPOD BACKEND --------------------
def _run_runpod_training(job_id: str, user_id: str, config: dict):
    _JOBS[job_id]["status"] = "queued"; set_status(job_id, "queued")
    logf = logs_path(user_id, job_id)
    out_dir = model_dir(user_id, job_id)
    _JOBS[job_id]["logs_path"] = str(logf)
    _JOBS[job_id]["model_dir"] = str(out_dir)
    set_model_dir(job_id, str(out_dir))

    _write_log(logf, f"[{job_id}] submitting to RunPod; user={user_id}")
    _write_log(logf, f"config={config}")

    # Build command for worker image (Dockerfile.train)
    cmd = [
        "python", "src/training/train_text.py",
        "--tokenizer_dir", "/app/outputs/tokenizer/orph_bpe_32k",
        "--train_files", config.get("sft_path", "/app/data/clean/text_supervised.jsonl"),
        "--cot_files", config.get("cot_path", ""),
        "--curriculum", config.get("curriculum", "sft:1.0,cot:0.0"),
        "--model_name", config.get("base_model", "gpt2"),
        "--output_dir", f"/app/outputs/user_models/{user_id}/{job_id}",
        "--max_length", str(config.get("max_length", 1024)),
        "--epochs", str(config.get("epochs", 1)),
        "--train_batch_size", str(config.get("train_batch_size", 2)),
        "--grad_accum", str(config.get("grad_accum", 8)),
        "--lr", str(config.get("lr", 5e-5)),
        "--gradient_checkpointing", str(config.get("gradient_checkpointing", "true")),
        "--auto_batch", "true",
    ]

    # Submit job
    api_key = os.getenv("RUNPOD_API_KEY", "")
    image = os.getenv("RUNPOD_IMAGE", "ghcr.io/you/orph-train:latest")  # build & push your train image
    env = {
        "RUN_ID": job_id,
        "USER_ID": user_id,
        # add storage creds (S3/GCS) here if needed
    }

    handle = rp.submit_training(api_key=api_key, image=image, command=cmd, env=env)
    _write_log(logf, f"[runpod] submitted backend_id={handle.backend_id}")
    _JOBS[job_id]["status"] = "running"; set_status(job_id, "running")

    try:
        # Simplified poll loop (every ~15s). Replace with webhook or better polling.
        for _ in range(6):
            time.sleep(15)
            status = rp.poll_status(handle.job_id)
            _write_log(logf, rp.fetch_logs(handle.job_id))
            if status in ("completed","failed"):
                break

        ok = (rp.poll_status(handle.job_id) == "completed")
        if ok:
            # In a real setup, artifacts are written to shared storage (S3/GCS).
            # For demo, mark MODEL_READY so reporter runs.
            (out_dir / "MODEL_READY").write_text("ok", encoding="utf-8")

        _finish_and_report(job_id, user_id, out_dir, logf, config, ok=ok)

    except Exception as e:
        _write_log(logf, f"[runpod] ERROR: {e}")
        _finish_and_report(job_id, user_id, out_dir, logf, config, ok=False)

# -------------------- PUBLIC API --------------------
def submit_job(user_id: str, config: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {"status": "queued", "user_id": user_id, "config": config}
    upsert_registry_entry(job_id, user_id, "queued", "", config)

    backend = os.getenv("ORPH_CLOUD_BACKEND", CLOUD_BACKEND).lower()
    runner = _run_runpod_training if backend == "runpod" else _run_local_training
    t = threading.Thread(target=runner, args=(job_id, user_id, config), daemon=True)
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
