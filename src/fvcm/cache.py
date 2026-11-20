"""Crash-resume cache and run manifests.

Deliberately minimal. Function vectors are always refit from the current data,
so there is no incremental-scoring or snapshot-delta machinery here: if the
dataset changes, the run is redone. The cache exists only so that a SLURM job
killed at hour three does not start from zero.

A stage writes one JSONL file per unit of work, keyed by `config_hash`. On
restart it reads back the keys it already has and skips them.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import REPO_ROOT

CACHE_ROOT = REPO_ROOT / "results" / "cache"


def _path(model: str, stage: str, cfg_hash: str) -> Path:
    return CACHE_ROOT / model / stage / f"{cfg_hash}.jsonl"


def done_keys(model: str, stage: str, cfg_hash: str) -> set[str]:
    """Keys already recorded for this stage and configuration."""
    p = _path(model, stage, cfg_hash)
    if not p.exists():
        return set()
    keys = set()
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue          # a torn final line from a killed job
    return keys


def append(model: str, stage: str, cfg_hash: str, key: str, payload: dict) -> None:
    p = _path(model, stage, cfg_hash)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"key": key, "ts": datetime.now(timezone.utc).isoformat(), "payload": payload}
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())      # survive a hard kill mid-write


def read_all(model: str, stage: str, cfg_hash: str) -> dict[str, dict]:
    """key -> payload, last write winning."""
    p = _path(model, stage, cfg_hash)
    out: dict[str, dict] = {}
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[rec["key"]] = rec["payload"]
    return out


# --------------------------------------------------------------------------- #
# Manifests
# --------------------------------------------------------------------------- #

def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_manifest(path: Path, *, stage: str, model: str, cfg_hash: str,
                   started: float, extra: dict | None = None) -> Path:
    """Stamp every output with enough provenance to reproduce or discard it."""
    import torch

    peak = None
    if torch.cuda.is_available():
        peak = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    manifest = {
        "stage": stage,
        "model": model,
        "config_hash": cfg_hash,
        "git_sha": git_sha(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - started, 1),
        "peak_vram_gib": peak,
        "torch": torch.__version__,
        **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
