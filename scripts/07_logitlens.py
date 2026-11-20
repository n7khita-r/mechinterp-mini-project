#!/usr/bin/env python3
"""Stage 07 -- logit-lens cross-check: per-layer rank of the gold first token.

No intervention. Early layers are shaded and not interpreted.
"""

from __future__ import annotations

import sys

from fvcm.progress import track

from _common import REPO_ROOT, RESULTS, cells, finish, parser, resolve_tasks, setup  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))

from fvcm import data as D, model as M  # noqa: E402
from fvcm.cache import read_json, write_json  # noqa: E402
from fvcm.fv import FunctionVector  # noqa: E402
from fvcm.logitlens import decode_vector, lens_curve  # noqa: E402


def main() -> int:
    p = parser("logit-lens layer curves and function-vector vocabulary decoding")
    p.add_argument("--max-items", type=int, default=40,
                   help="items per cell; the lens needs one forward each")
    args = p.parse_args()
    mcfg, exp, chash, out, started = setup(args, "07_logitlens")

    filt_path = RESULTS / "01_baselines" / mcfg.name / "competence_filter.json"
    filters = read_json(filt_path) if filt_path.exists() else {}

    model, tok, info = M.load_model(mcfg)
    todo = cells(resolve_tasks(args.tasks), args.conditions)

    curves, decoded = [], {}
    for task, cond in track(todo, "cells", stage="07_logitlens", enabled=not args.no_progress, label=lambda tc: f"{tc[0]}/{tc[1]}"):
        items = D.headline(D.load_items(task, cond, exp), exp)
        split = D.split_items(items, exp)
        keep = set(filters.get(f"{task}|{cond}", []))
        pool = [i for i in split.valid if not keep or i.id in keep][: args.max_items]
        if not pool or len(split.train) < exp.n_shots:
            continue
        curves.append(lens_curve(model, tok, info, split, pool, exp, task, cond))

        try:
            fv = FunctionVector.load(RESULTS / "fv", mcfg.name, cond, task)
            decoded[f"{task}|{cond}"] = decode_vector(model, info, fv.vector, tok, k=10)
        except FileNotFoundError:
            pass

    write_json(out / "lens_curves.json", curves)
    write_json(out / "fv_vocabulary.json", decoded)

    print(f"\n{'task':20s} {'cond':5s} {'n':>4s} {'emerges at layer':>17s}")
    for c in sorted(curves, key=lambda c: (c["task"], c["condition"])):
        em = c.get("emergence_layer")
        print(f"{c['task']:20s} {c['condition']:5s} {c['n']:4d} "
              f"{(str(em) if em is not None else 'never'):>17s}")

    finish(out, stage="07_logitlens", mcfg=mcfg, chash=chash, started=started,
           n_curves=len(curves))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
