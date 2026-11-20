#!/usr/bin/env python3
"""Stage 08 -- controls: random vector of matched norm, unrelated-task vector,
and the shuffled-label n-shot floor. A function vector must beat all three.
"""

from __future__ import annotations

import sys

from fvcm.progress import track

from _common import REPO_ROOT, RESULTS, cells, finish, parser, resolve_tasks, setup  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))

from fvcm import activations as A, data as D, model as M  # noqa: E402
from fvcm.cache import read_json, write_json  # noqa: E402
from fvcm.controls import run_controls  # noqa: E402
from fvcm.data import UNRELATED_TASK  # noqa: E402
from fvcm.fv import FunctionVector, build_function_vector  # noqa: E402


def main() -> int:
    args = parser("controls: random vector, unrelated-task FV, shuffled floor").parse_args()
    mcfg, exp, chash, out, started = setup(args, "08_controls")

    interv = RESULTS / "05_intervene" / mcfg.name / "intervention.json"
    if not interv.exists():
        raise SystemExit("run 05_intervene first (it chooses the edit layer)")
    # Reuse stage 05's layer and scale, so controls sit in the same regime.
    med_rows = read_json(interv)
    layers = {f"{r['task']}|{r['condition']}": r["edit_layer"] for r in med_rows}
    alphas = {f"{r['task']}|{r['condition']}": float(r.get("fv_rel_norm", 0.0))
              for r in med_rows}
    default_alpha = float(exp.intervene.get("fv_rel_norm", 0.1))
    filters_path = RESULTS / "01_baselines" / mcfg.name / "competence_filter.json"
    filters = read_json(filters_path) if filters_path.exists() else {}

    med = RESULTS / "03_mediation" / mcfg.name / "head_sets.json"
    head_sets = {k: [tuple(h) for h in v] for k, v in read_json(med).items()}

    model, tok, info = M.load_model(mcfg)

    unrelated: dict[str, FunctionVector] = {}
    for cond in ("en", "te", "rom"):
        if cond not in head_sets:
            continue
        items = D.headline(D.load_items(UNRELATED_TASK, cond, exp), exp)
        split = D.split_items(items, exp)
        if len(split.train) < exp.n_shots or not split.valid:
            continue
        acts = A.mean_head_activations(model, tok, info, split, exp,
                                       task=UNRELATED_TASK, condition=cond)
        unrelated[cond] = build_function_vector(model, info, acts, head_sets[cond],
                                                UNRELATED_TASK, cond, cond)

    rows = []
    for task, cond in track(cells(resolve_tasks(args.tasks), args.conditions),
                           "cells", stage="08_controls", enabled=not args.no_progress,
                           label=lambda tc: f"{tc[0]}/{tc[1]}"):
        if task == UNRELATED_TASK:
            continue
        key = f"{task}|{cond}"
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
        rows.append(run_controls(model, tok, info, split, pool, exp, fv,
                                 unrelated.get(cond), layers.get(key, info.mid_layer),
                                 alphas.get(key) or default_alpha))

    write_json(out / "controls.json", rows)
    hdr = (f"\n{'task':20s} {'cond':5s} {'FV':>6s} {'rand':>6s} {'unrel':>6s} "
           f"{'shuf':>6s} {'FV-rand':>8s}")
    print(hdr)
    for r in sorted(rows, key=lambda r: (r["task"], r["condition"])):
        gap = r["acc_fv"] - r["acc_random"]
        # Identical FV and random columns mean the comparison is uninformative.
        flag = "  <- identical" if abs(gap) < 1e-9 else ""
        print(f"{r['task']:20s} {r['condition']:5s} {r['acc_fv']:6.2f} "
              f"{r['acc_random']:6.2f} {r['acc_unrelated']:6.2f} "
              f"{r['acc_shuffled']:6.2f} {gap:+8.2f}{flag}")

    tied = sum(1 for r in rows if abs(r["acc_fv"] - r["acc_random"]) < 1e-9)
    if tied > len(rows) // 2:
        print(f"\nWARNING: {tied}/{len(rows)} cells have FV == random exactly.")
        print("That is the signature of an intervention too small to change the")
        print("argmax, not of a non-specific vector. Check fv_norm_scaled against")
        print("resid_norm_at_layer before reporting these.")

    finish(out, stage="08_controls", mcfg=mcfg, chash=chash, started=started,
           n_cells=len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
