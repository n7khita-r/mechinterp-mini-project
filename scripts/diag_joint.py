#!/usr/bin/env python3
"""Joint (layer, scale) grid search for one cell, on the dev slice.

Reports whether the joint best beats what stage 05's sequential choice found.
"""

from __future__ import annotations

import sys

from _common import REPO_ROOT, RESULTS, parser, setup  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))

from fvcm import data as D  # noqa: E402
from fvcm import model as M  # noqa: E402
from fvcm.cache import read_json, write_json  # noqa: E402
from fvcm.evaluate import accuracy  # noqa: E402
from fvcm.fv import FunctionVector  # noqa: E402
from fvcm.intervene import dev_test_split, residual_norms, score_zero_shot  # noqa: E402
from fvcm.progress import track  # noqa: E402


def main() -> int:
    p = parser("joint (layer, scale) grid for one cell")
    p.add_argument("--task", default="present_past")
    p.add_argument("--condition", default="te")
    p.add_argument("--layer-stride", type=int, default=4)
    p.add_argument("--alphas", default="0.05,0.1,0.2,0.4,0.8")
    p.add_argument("--max-items", type=int, default=40)
    args = p.parse_args()
    mcfg, exp, chash, out, started = setup(args, "diag_joint")

    alphas = [float(a) for a in args.alphas.split(",") if a]
    model, tok, info = M.load_model(mcfg)
    fv = FunctionVector.load(RESULTS / "fv", mcfg.name, args.condition, args.task)

    # Dev slice only, to compare against stage 05's choice on the same slice.
    items = D.headline(D.load_items(args.task, args.condition, exp), exp)
    split = D.split_items(items, exp)
    filt = RESULTS / "01_baselines" / mcfg.name / "competence_filter.json"
    keep = set(read_json(filt).get(f"{args.task}|{args.condition}", [])) if filt.exists() else set()
    pool = [i for i in split.valid if not keep or i.id in keep]
    dev, _ = dev_test_split(pool, exp)
    dev = dev[: args.max_items]
    if not dev:
        print("no dev items for this cell")
        return 1

    layers = list(range(0, info.n_layers, max(1, args.layer_stride)))
    norms = residual_norms(model, tok, info, dev, exp)
    base = float(fv.vector.float().norm()) or 1.0
    print(f"\ncell: {args.task}/{args.condition}   {len(dev)} dev items   "
          f"{len(layers)} layers x {len(alphas)} scales\n")

    grid: dict[tuple[int, float], float] = {}
    for layer in track(layers, "layers", stage="diag_joint",
                       enabled=not args.no_progress, label=lambda l: f"layer {l}"):
        for a in alphas:
            vec = fv.vector * (a * norms[layer] / base)
            grid[(layer, a)] = accuracy(
                score_zero_shot(model, tok, info, dev, exp, vec, layer), k=1)

    print(f"\n{'layer':>6s}" + "".join(f"{f'a={a}':>9s}" for a in alphas))
    for layer in layers:
        print(f"{layer:6d}" + "".join(f"{grid[(layer, a)]:9.2f}" for a in alphas))

    (best_l, best_a), best = max(grid.items(), key=lambda kv: kv[1])

    interv = RESULTS / "05_intervene" / mcfg.name / "intervention.json"
    seq = None
    if interv.exists():
        for r in read_json(interv):
            if r["task"] == args.task and r["condition"] == args.condition:
                seq = r
                break

    print(f"\njoint best:      layer {best_l}, alpha {best_a}  ->  {best:.2f} (dev)")
    if seq:
        print(f"stage 05 chose:  layer {seq['edit_layer']}, alpha {seq['fv_rel_norm']}"
              f"  ->  {max(v for _, v in seq['alpha_sweep']):.2f} (dev)")
        gap = best - max(v for _, v in seq["alpha_sweep"])
        print()
        if gap > 0.05:
            print(f"The joint search beats the sequential one by {gap:+.2f} on the dev")
            print("slice. The sequential procedure MISSED a working combination, and")
            print("that is a limitation of our method, not a property of the language.")
        else:
            print(f"The joint search gains only {gap:+.2f}. The sequential procedure")
            print("did NOT miss anything: this cell simply does not steer well, and")
            print("the failure is not an artefact of how layer and scale are chosen.")

    write_json(out / f"joint_{args.task}_{args.condition}.json",
               [{"layer": l, "alpha": a, "acc": v} for (l, a), v in grid.items()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
