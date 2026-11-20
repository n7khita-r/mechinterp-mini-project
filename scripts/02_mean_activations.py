#!/usr/bin/env python3
"""Stage 02 -- mean attention-head activations per (task, condition).

The `ā_{l,h}` of the method: what each head outputs at the last token when the
task is clear. Used twice downstream -- as the patch in causal mediation
(stage 03) and as the raw material of the function vector (stage 04).

Queries are drawn from the competence-filtered items written by stage 01,
matching the reference, which passes its validation filter set here.
"""

from __future__ import annotations

import sys

import torch
from fvcm.progress import track

from _common import REPO_ROOT, RESULTS, cells, finish, parser, resolve_tasks, setup  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))

from fvcm import activations as A, data as D, model as M  # noqa: E402
from fvcm.cache import read_json, write_json  # noqa: E402


def main() -> int:
    args = parser("mean head activations per (task, condition)").parse_args()
    mcfg, exp, chash, out, started = setup(args, "02_mean_activations")

    filt_path = RESULTS / "01_baselines" / mcfg.name / "competence_filter.json"
    filters = read_json(filt_path) if filt_path.exists() else {}
    if not filters:
        print("  note: no competence filter found; run 01_baselines first for "
              "results comparable with the paper. Proceeding unfiltered.")

    model, tok, info = M.load_model(mcfg)
    todo = cells(resolve_tasks(args.tasks), args.conditions)

    index = {}
    for task, cond in track(todo, "cells", stage="02_mean_activations", enabled=not args.no_progress, label=lambda tc: f"{tc[0]}/{tc[1]}"):
        items = D.headline(D.load_items(task, cond, exp), exp)
        split = D.split_items(items, exp)
        if not split.valid or len(split.train) < exp.n_shots:
            print(f"  skipping {task}/{cond}: train={len(split.train)} valid={len(split.valid)}")
            continue

        keep = set(filters.get(f"{task}|{cond}", []))
        acts = A.mean_head_activations(model, tok, info, split, exp,
                                       filter_ids=keep, task=task, condition=cond)
        path = out / cond / f"{task}.pt"
        A.save(acts, path)
        index[f"{task}|{cond}"] = str(path.relative_to(REPO_ROOT))

    write_json(out / "index.json", index)
    print(f"\nwrote {len(index)} mean-activation tensors, shape "
          f"({info.n_layers}, {info.n_heads}, {info.head_dim})")
    finish(out, stage="02_mean_activations", mcfg=mcfg, chash=chash, started=started,
           n_cells=len(index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
