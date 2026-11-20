#!/usr/bin/env python3
"""Stage 03 -- causal mediation: the indirect effect of every attention head.

One sweep per condition (plus a pooled sweep over all of them), with each trial
drawing a random task from that condition. The paper averages the indirect
effect over tasks anyway to get a single universal head set; doing it per
condition instead makes "do Telugu and English recruit the same heads?" a
measurable quantity rather than an assumption.

This is the expensive stage. It is batched over heads -- every patch at a given
layer applies to the same prompt, so the prompt is replicated across the batch
and each element carries a different head. That turns n_layers x n_heads
forwards per trial into n_layers: 1,984 -> 62 on the 27B model.
"""

from __future__ import annotations

import sys

import torch

from _common import REPO_ROOT, RESULTS, finish, parser, resolve_tasks, setup  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))

from fvcm import activations as A, data as D, model as M  # noqa: E402
from fvcm.cache import read_json, write_json  # noqa: E402
from fvcm.mediation import head_overlap, indirect_effects, top_heads  # noqa: E402


def main() -> int:
    p = parser("causal mediation / indirect effect sweep")
    p.add_argument("--pooled", action="store_true",
                   help="also run a sweep pooling every condition together")
    args = p.parse_args()
    mcfg, exp, chash, out, started = setup(args, "03_mediation")

    acts_dir = RESULTS / "02_mean_activations" / mcfg.name
    if not (acts_dir / "index.json").exists():
        raise SystemExit("run 02_mean_activations first")
    filt_path = RESULTS / "01_baselines" / mcfg.name / "competence_filter.json"
    filters = read_json(filt_path) if filt_path.exists() else {}

    model, tok, info = M.load_model(mcfg)
    tasks = resolve_tasks(args.tasks)

    wanted = None if args.conditions == "all" else {
        c.strip() for c in args.conditions.split(",")
    }

    def cells_for(conds: set[str]) -> dict:
        out_cells = {}
        for task in tasks:
            from fvcm import conditions_for
            for cond in conditions_for(task):
                if cond not in conds:
                    continue
                path = acts_dir / cond / f"{task}.pt"
                if not path.exists():
                    continue
                items = D.headline(D.load_items(task, cond, exp), exp)
                split = D.split_items(items, exp)
                if len(split.train) < exp.n_shots or not split.valid:
                    continue
                key = task if len(conds) == 1 else f"{task}|{cond}"
                out_cells[key] = (split, A.load(path),
                                  set(filters.get(f"{task}|{cond}", [])))
        return out_cells

    from fvcm import CONDITIONS
    conds = [c for c in CONDITIONS if wanted is None or c in wanted]

    sets, summary = {}, {}
    n_top = int(exp.fv["n_top_heads"])

    for cond in conds:
        cell_map = cells_for({cond})
        if not cell_map:
            print(f"  skipping {cond}: no cells with mean activations")
            continue
        aie = indirect_effects(model, tok, info, cell_map, exp, cond,
                               progress=not args.no_progress)
        torch.save(aie, out / f"aie_{cond}.pt")
        heads = top_heads(aie, n_top)
        sets[cond] = heads
        summary[cond] = {"n_tasks": len(cell_map), "tasks": sorted(cell_map),
                         "top_heads": heads}
        print(f"  {cond}: top heads " + ", ".join(f"{l}.{h}" for l, h, _ in heads))

    if args.pooled:
        cell_map = cells_for(set(conds))
        if cell_map:
            aie = indirect_effects(model, tok, info, cell_map, exp, "pooled",
                                   progress=not args.no_progress)
            torch.save(aie, out / "aie_pooled.pt")
            sets["pooled"] = top_heads(aie, n_top)
            summary["pooled"] = {"n_tasks": len(cell_map), "tasks": sorted(cell_map),
                                 "top_heads": sets["pooled"]}

    overlaps = {}
    present = [c for c in conds if c in sets]
    for a in present:
        for b in present:
            overlaps[f"{a}|{b}"] = head_overlap(sets[a], sets[b])

    write_json(out / "head_sets.json", {k: [list(h) for h in v] for k, v in sets.items()})
    write_json(out / "summary.json", summary)
    write_json(out / "head_overlap.json", overlaps)

    if len(present) > 1:
        print("\nhead-set overlap (Jaccard):")
        print("        " + " ".join(f"{c:>6s}" for c in present))
        for a in present:
            print(f"  {a:5s} " + " ".join(
                f"{overlaps[f'{a}|{b}']['jaccard']:6.2f}" for b in present))

    finish(out, stage="03_mediation", mcfg=mcfg, chash=chash, started=started,
           conditions=present, n_top_heads=n_top,
           forwards_per_trial=info.n_layers,
           forwards_per_trial_if_unbatched=info.n_layers * info.n_heads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
