"""Shared setup for the stage scripts: argument parsing, paths, cell enumeration."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fvcm import ALL_TASKS, PART1_TASKS, PART2_TASKS, conditions_for  # noqa: E402
from fvcm.cache import write_manifest  # noqa: E402
from fvcm.config import ExpConfig, ModelConfig, config_hash  # noqa: E402

RESULTS = REPO_ROOT / "results"


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", default="configs/model/gemma3_4b.yaml",
                   help="path to a model config under configs/model/")
    p.add_argument("--exp", default="configs/exp/default.yaml",
                   help="path to an experiment config under configs/exp/")
    p.add_argument("--tasks", default="all",
                   help="comma-separated task names, or all / part1 / part2")
    p.add_argument("--conditions", default="all",
                   help="comma-separated conditions, or all")
    p.add_argument("--out", default=None, help="override the output directory")
    p.add_argument("--no-progress", action="store_true")
    return p


def resolve_tasks(spec: str) -> list[str]:
    if spec == "all":
        return list(ALL_TASKS)
    if spec == "part1":
        return list(PART1_TASKS)
    if spec == "part2":
        return list(PART2_TASKS)
    names = [t.strip() for t in spec.split(",") if t.strip()]
    bad = [t for t in names if t not in ALL_TASKS]
    if bad:
        raise SystemExit(f"unknown task(s): {bad}\nknown: {list(ALL_TASKS)}")
    return names


def cells(tasks: list[str], condition_spec: str) -> list[tuple[str, str]]:
    """Every (task, condition) pair in scope, skipping conditions a task lacks."""
    wanted = None
    if condition_spec != "all":
        wanted = {c.strip() for c in condition_spec.split(",") if c.strip()}
    out = []
    for task in tasks:
        for cond in conditions_for(task):
            if wanted is None or cond in wanted:
                out.append((task, cond))
    return out


def setup(args, stage: str):
    """Load configs and derive the run's identity. Returns everything a stage needs."""
    mcfg = ModelConfig.load(args.model)
    exp = ExpConfig.load(args.exp)
    chash = config_hash(mcfg, exp, stage=stage)
    out = Path(args.out) if args.out else RESULTS / stage / mcfg.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{stage}] model={mcfg.name} exp={exp.name} config_hash={chash}")
    print(f"[{stage}] writing to {out.relative_to(REPO_ROOT)}")
    return mcfg, exp, chash, out, time.time()


def finish(out: Path, *, stage: str, mcfg, chash: str, started: float, **extra):
    path = write_manifest(out / "manifest.json", stage=stage, model=mcfg.name,
                          cfg_hash=chash, started=started, extra=extra)
    print(f"[{stage}] done in {time.time() - started:.1f}s -> {path.name}")
