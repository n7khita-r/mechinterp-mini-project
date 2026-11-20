#!/usr/bin/env python3
"""Stage 05 -- add each function vector to zero-shot prompts.

The test of whether a vector carries the task: with no demonstrations at all,
does adding v_t make the model do the task?

The edit layer is chosen on a dev slice of the evaluation pool and the headline
number is then computed at that fixed layer on the held-out remainder. The
reference reports the maximum over all layers, which is fine with thousands of
items but optimistically biased with the few dozen hand-authored data affords.
The full sweep curve is still saved -- its shape is a result in itself.
"""

from __future__ import annotations

import sys

from fvcm.progress import track

from _common import REPO_ROOT, RESULTS, cells, finish, parser, resolve_tasks, setup  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))

from fvcm import data as D, model as M  # noqa: E402
from fvcm.cache import append, done_keys, read_all, read_json, write_json  # noqa: E402
from fvcm.fv import FunctionVector  # noqa: E402
from fvcm.intervene import run_cell  # noqa: E402

STAGE = "05_intervene"


def main() -> int:
    args = parser("function-vector intervention on zero-shot prompts").parse_args()
    mcfg, exp, chash, out, started = setup(args, STAGE)

    filt_path = RESULTS / "01_baselines" / mcfg.name / "competence_filter.json"
    filters = read_json(filt_path) if filt_path.exists() else {}

    model, tok, info = M.load_model(mcfg)
    todo = cells(resolve_tasks(args.tasks), args.conditions)
    already = done_keys(mcfg.name, STAGE, chash)
    if already:
        print(f"  resuming: {len(already)} cell(s) already done")

    for task, cond in track(todo, "cells", stage="05_intervene", enabled=not args.no_progress, label=lambda tc: f"{tc[0]}/{tc[1]}"):
        key = f"{task}|{cond}"
        if key in already:
            continue
        try:
            fv = FunctionVector.load(RESULTS / "fv", mcfg.name, cond, task)
        except FileNotFoundError:
            continue

        items = D.headline(D.load_items(task, cond, exp), exp)
        split = D.split_items(items, exp)
        keep = set(filters.get(key, []))
        pool = [i for i in split.valid if not keep or i.id in keep]
        if not pool:
            continue

        append(mcfg.name, STAGE, chash, key,
               run_cell(model, tok, info, pool, exp, fv,
                        progress=not args.no_progress))

    results = read_all(mcfg.name, STAGE, chash)
    rows = [{k: v for k, v in r.items() if not k.startswith("scores_")}
            for r in results.values()]
    write_json(out / "intervention.json", rows)
    write_json(out / "scores.json",
               {k: {"with_fv": v["scores_with_fv"], "zero_shot": v["scores_zero_shot"]}
                for k, v in results.items()})

    print(f"\n{'task':20s} {'cond':5s} {'layer':>5s} {'0-shot':>7s} {'+FV':>7s} {'gain':>7s}")
    for r in sorted(rows, key=lambda r: (r["task"], r["condition"])):
        print(f"{r['task']:20s} {r['condition']:5s} {r['edit_layer']:5d} "
              f"{r['acc_zero_shot']:7.2f} {r['acc_with_fv']:7.2f} "
              f"{r['acc_with_fv'] - r['acc_zero_shot']:+7.2f}")

    finish(out, stage=STAGE, mcfg=mcfg, chash=chash, started=started, n_cells=len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
