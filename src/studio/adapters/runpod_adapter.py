# src/studio/adapters/runpod_adapter.py
"""
RunPod adapter stub for Orph Studio.
Replace TODO blocks with real RunPod API calls:
- create a job/pod with Docker image built from Dockerfile.train
- mount dataset/model storage (S3/GCS) or pass presigned URLs
- tail logs and map statuses to 'queued'|'running'|'completed'|'failed'
"""
from dataclasses import dataclass
from typing import Optional, Dict
import time, uuid

@dataclass
class RunpodJobHandle:
    job_id: str
    backend_id: str   # provider-side id (pod/run id)
    status: str       # queued/running/completed/failed

# In-memory sim store
_STATE: Dict[str, RunpodJobHandle] = {}

def submit_training(api_key: str, image: str, command: list, env: dict) -> RunpodJobHandle:
    """Submit a training job to RunPod (stub)."""
    # TODO: Use requests to call RunPod API, send image/command/env.
    job_id = uuid.uuid4().hex[:12]
    backend_id = "sim-" + job_id
    h = RunpodJobHandle(job_id=job_id, backend_id=backend_id, status="queued")
    _STATE[job_id] = h
    return h

def poll_status(job_id: str) -> str:
    """Poll job status from RunPod (stub)."""
    h = _STATE.get(job_id)
    if not h:
        return "unknown"
    # naive time-based progression
    if h.status == "queued":
        h.status = "running"
    elif h.status == "running":
        h.status = "completed"
    _STATE[job_id] = h
    return h.status

def fetch_logs(job_id: str, last_bytes: int = 4096) -> str:
    """Fetch recent logs (stub)."""
    h = _STATE.get(job_id)
    if not h: return ""
    # TODO: Tail logs from RunPod API / storage
    if h.status == "queued":
        return "[runpod] queued..."
    if h.status == "running":
        return "[runpod] running..."
    if h.status == "completed":
        return "[runpod] completed."
    if h.status == "failed":
        return "[runpod] failed."
    return ""

def cancel(job_id: str) -> bool:
    """Cancel job (stub)."""
    h = _STATE.get(job_id)
    if not h: return False
    h.status = "failed"
    _STATE[job_id] = h
    # TODO: call RunPod cancel API
    return True