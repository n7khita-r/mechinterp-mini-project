#!/usr/bin/env python3
"""Stage 06 -- cross-condition transfer.

Apply the function vector built in one condition to zero-shot prompts in
another. This is the experiment that separates the two explanations for a drop
on Telugu or Tenglish:

* the English vector steers Telugu prompts about as well as the Telugu vector
  does -> the task representation is language-independent, and what fails is
  reading the input;
* it does not transfer -> the task representation itself differs by condition.
"""

from __future__ import annotations

import sys

import numpy as np
from fvcm.progress import track

from _common import REPO_ROOT, RESULTS, finish, parser, resolve_tasks, setup  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))

from fvcm import conditions_for, data as D, model as M  # noqa: E402
from fvcm.cache import read_json, write_json  # noqa: E402
from fvcm.fv import FunctionVector  # noqa: E402
from fvcm.intervene import transfer  # noqa: E402


def main() -> int:
    args = parser("cross-condition function-vector transfer").parse_args()
    mcfg, exp, chash, out, started = setup(args, "06_transfer")

    interv = RESULTS / "05_intervene" / mcfg.name / "intervention.json"
    if not interv.exists():
        raise SystemExit("run 05_intervene first (it chooses the edit layer)")
    interv_rows = read_json(interv)
    layers = {f"{r['task']}|{r['condition']}": r["edit_layer"] for r in interv_rows}
    alphas = {f"{r['task']}|{r['condition']}": float(r.get("fv_rel_norm", 0.0))
              for r in interv_rows}
    default_alpha = float(exp.intervene.get("fv_rel_norm", 0.1))
    filt_path = RESULTS / "01_baselines" / mcfg.name / "competence_filter.json"
    filters = read_json(filt_path) if filt_path.exists() else {}

    model, tok, info = M.load_model(mcfg)
    rows = []

    for task in track(resolve_tasks(args.tasks), "tasks", stage="06_transfer",
                     enabled=not args.no_progress, label=str):
        conds = conditions_for(task)
        for target in conds:
            items = D.headline(D.load_items(task, target, exp), exp)
            split = D.split_items(items, exp)
            keep = set(filters.get(f"{task}|{target}", []))
            pool = [i for i in split.valid if not keep or i.id in keep]
            if not pool:
                continue
            # Hold the edit layer fixed at the target condition's own choice, so
            # the comparison is about the vector, not about where it is injected.
            layer = layers.get(f"{task}|{target}", info.mid_layer)
            # Same reasoning for the scale: the target cell's alpha, so a source
            # vector is never handicapped by being injected at the wrong norm.
            alpha = alphas.get(f"{task}|{target}") or default_alpha
            for source in conds:
                try:
                    fv = FunctionVector.load(RESULTS / "fv", mcfg.name, source, task)
                except FileNotFoundError:
                    continue
                r = transfer(model, tok, info, pool, exp, fv, layer, alpha)
                rows.append({k: v for k, v in r.items() if k != "scores"})

    write_json(out / "transfer.json", rows)

    conds = sorted({r["source_condition"] for r in rows})
    if conds:
        mat = np.full((len(conds), len(conds)), np.nan)
        for i, s in enumerate(conds):
            for j, t in enumerate(conds):
                vals = [r["acc"] for r in rows
                        if r["source_condition"] == s and r["target_condition"] == t]
                if vals:
                    mat[i, j] = float(np.mean(vals))
        write_json(out / "transfer_matrix.json",
                   {"conditions": conds, "acc": mat.tolist()})
        print("\ntransfer (rows: FV source, cols: prompt condition), mean over tasks")
        print("          " + " ".join(f"{c:>6s}" for c in conds))
        for i, s in enumerate(conds):
            print(f"  {s:7s} " + " ".join(f"{mat[i, j]:6.2f}" for j in range(len(conds))))

    finish(out, stage="06_transfer", mcfg=mcfg, chash=chash, started=started,
           n_pairs=len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
