import os, sys, time, uuid, subprocess, threading, shutil, json
from pathlib import Path

from .config import TOKENIZER_DIR, ROOT, CLOUD_BACKEND
from .storage import (
    model_dir, logs_path, upsert_registry_entry, set_status, set_model_dir, add_tag,
    scrape_job_dir, build_job_dir
)
from . import reporter
from .scrape_reporter import build_scrape_summary, write_report
from .adapters import runpod_adapter as rp  # stub adapter (replace TODOs later)

# In-memory runtime job state
_JOBS = {}  # job_id -> dict(status, user_id, config, logs_path, model_dir)

# -------------------- utils --------------------
def _write_log(logfile: Path, text: str):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a", encoding="utf-8") as lf:
        lf.write(text.rstrip() + "\n")
        lf.flush()

def _finish_and_report(job_id, user_id, out_dir: Path, logf: Path, config: dict, ok: bool):
    """Training report (JSON+PDF); graceful if no model produced."""
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

# -------------------- TRAINING: local --------------------
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

# -------------------- TRAINING: runpod (stub adapter) --------------------
def _run_runpod_training(job_id: str, user_id: str, config: dict):
    _JOBS[job_id]["status"] = "queued"; set_status(job_id, "queued")
    logf = logs_path(user_id, job_id)
    out_dir = model_dir(user_id, job_id)
    _JOBS[job_id]["logs_path"] = str(logf)
    _JOBS[job_id]["model_dir"] = str(out_dir)
    set_model_dir(job_id, str(out_dir))

    _write_log(logf, f"[{job_id}] submitting to RunPod; user={user_id}")
    _write_log(logf, f"config={config}")

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

    api_key = os.getenv("RUNPOD_API_KEY", "")
    image = os.getenv("RUNPOD_IMAGE", "ghcr.io/you/orph-train:latest")
    env = {"RUN_ID": job_id, "USER_ID": user_id}

    handle = rp.submit_training(api_key=api_key, image=image, command=cmd, env=env)
    _write_log(logf, f"[runpod] submitted backend_id={handle.backend_id}")
    _JOBS[job_id]["status"] = "running"; set_status(job_id, "running")

    try:
        # Simple poll loop (replace with real polling/webhooks)
        for _ in range(6):
            time.sleep(15)
            status = rp.poll_status(handle.job_id)
            _write_log(logf, rp.fetch_logs(handle.job_id))
            if status in ("completed", "failed"):
                break

        ok = (rp.poll_status(handle.job_id) == "completed")
        if ok:
            # In real adapter, artifacts live on shared storage (S3/GCS). For now, mark ready.
            (out_dir / "MODEL_READY").write_text("ok", encoding="utf-8")

        _finish_and_report(job_id, user_id, out_dir, logf, config, ok=ok)

    except Exception as e:
        _write_log(logf, f"[runpod] ERROR: {e}")
        _finish_and_report(job_id, user_id, out_dir, logf, config, ok=False)

# -------------------- SCRAPE (local) --------------------
def _run_scrape(job_id: str, user_id: str, config: dict):
    """
    Config:
      { "sources": ["all"] | ["pubmed_fetch.py", ...],
        "env": {"NCBI_EMAIL":"...", "NCBI_API_KEY":"...", "WIKI_LANG":"en"} }
    """
    _JOBS[job_id]["status"] = "running"; set_status(job_id, "running")
    logf = logs_path(user_id, job_id)
    out_dir = scrape_job_dir(user_id, job_id)    # where scrape_report.json is written
    _JOBS[job_id]["logs_path"] = str(logf)
    _JOBS[job_id]["model_dir"] = str(out_dir)    # reuse report/download endpoints
    set_model_dir(job_id, str(out_dir))
    add_tag(job_id, "scrape")

    def _log(s): _write_log(logf, s)

    try:
        _log(f"[{job_id}] starting scrape; user={user_id}")
        _log(f"config={config}")

        env = os.environ.copy()
        for k, v in (config.get("env") or {}).items():
            if isinstance(v, str):
                env[k] = v

        all_scripts = [
            "medlineplus_fetch.py",
            "clinicaltrials_fetch.py",
            "openfda_labels_fetch.py",
            "pubmed_fetch.py",
            "pmc_oa_fetch.py",
            "cdc_rss_fetch.py",
            "who_rss_fetch.py",
            "wikipedia_med_portal_fetch.py",
        ]
        sources = config.get("sources") or ["all"]
        if len(sources) == 1 and str(sources[0]).lower() == "all":
            todo = all_scripts
        else:
            sset = {str(s).lower() for s in sources}
            todo = [s for s in all_scripts if s.lower() in sset] or all_scripts

        scr_dir = ROOT / "src" / "data_prep" / "scrapers"
        for name in todo:
            cmd = [sys.executable, str(scr_dir / name)]
            _log("▶ " + " ".join(cmd))
            r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
            if r.stdout: _log(r.stdout.strip())
            if r.stderr: _log("[stderr] " + r.stderr.strip())

        scraped_dir = (ROOT / "data" / "raw" / "scraped")
        summary = build_scrape_summary(scraped_dir)
        path = write_report(out_dir, summary)
        _log(f"scrape_report.json -> {path}")

        _JOBS[job_id]["status"] = "completed"; set_status(job_id, "completed")
        _log("done.")
    except Exception as e:
        _JOBS[job_id]["status"] = "failed"; set_status(job_id, "failed")
        _log(f"ERROR: {e}")

# -------------------- DATASET BUILD (local) --------------------
def _run_build_dataset(job_id: str, user_id: str, config: dict):
    """
    Run src/data_prep/build_dataset.py and copy artifacts to per-job dir.
    Config: { "seed": 42, "train_ratio": 0.94, "val_ratio": 0.03, "test_ratio": 0.03 }
    """
    _JOBS[job_id]["status"] = "running"; set_status(job_id, "running")
    logf = logs_path(user_id, job_id)
    out_dir = build_job_dir(user_id, job_id)
    _JOBS[job_id]["logs_path"] = str(logf)
    _JOBS[job_id]["model_dir"] = str(out_dir)   # reuse report/download endpoints
    set_model_dir(job_id, str(out_dir))
    add_tag(job_id, "build")

    def _log(s): _write_log(logf, s)

    try:
        _log(f"[{job_id}] starting dataset build; user={user_id}")
        _log(f"config={config}")

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

        clean_dir = ROOT / "data" / "clean"
        for name in ["text_supervised.jsonl", "text_cot.jsonl", "stats.json"]:
            src = clean_dir / name
            if src.exists():
                shutil.copy2(src, out_dir / name)

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

# -------------------- PUBLIC submitters & accessors --------------------
def submit_job(user_id: str, config: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {"status": "queued", "user_id": user_id, "config": config}
    upsert_registry_entry(job_id, user_id, "queued", "", config)
    backend = os.getenv("ORPH_CLOUD_BACKEND", CLOUD_BACKEND).lower()
    runner = _run_runpod_training if backend == "runpod" else _run_local_training
    t = threading.Thread(target=runner, args=(job_id, user_id, config), daemon=True)
    t.start()
    return job_id

def submit_scrape_job(user_id: str, config: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {"status": "queued", "user_id": user_id, "config": config}
    upsert_registry_entry(job_id, user_id, "queued", "", {"type": "scrape", **(config or {})})
    t = threading.Thread(target=_run_scrape, args=(job_id, user_id, config), daemon=True)
    t.start()
    return job_id

def submit_build_job(user_id: str, config: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {"status": "queued", "user_id": user_id, "config": config}
    upsert_registry_entry(job_id, user_id, "queued", "", {"type": "build", **(config or {})})
    t = threading.Thread(target=_run_build_dataset, args=(job_id, user_id, config), daemon=True)
    t.start()
    return job_id

def job_status(job_id: str) -> dict:
    job = _JOBS.get(job_id)
    return {"job_id": job_id, **job} if job else {"job_id": job_id, "status": "unknown"}

def job_logs(job_id: str) -> str:
    job = _JOBS.get(job_id, {})
    p = Path(job.get("logs_path", ""))
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""

def job_model_dir(job_id: str) -> str:
    job = _JOBS.get(job_id, {})
    return job.get("model_dir", "")
