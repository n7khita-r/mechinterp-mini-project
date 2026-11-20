#!/usr/bin/env python3
"""Stage 04 -- build function vectors.

    v_t = sum over the condition's top heads of  W_O^(L) [ e_H * ā_{L,H,last} ]

The head *set* comes from that condition's mediation sweep and is shared across
tasks; the *values* are the task's own mean activations. So the vector is
task-specific while the circuit carrying it is not.
"""

from __future__ import annotations

import sys

from fvcm.progress import track

from _common import REPO_ROOT, RESULTS, cells, finish, parser, resolve_tasks, setup  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))

from fvcm import activations as A, model as M  # noqa: E402
from fvcm.cache import read_json, write_json  # noqa: E402
from fvcm.fv import FunctionVector, build_function_vector, cosine_grid  # noqa: E402


def main() -> int:
    p = parser("build function vectors from head sets and mean activations")
    p.add_argument("--head-set", default="per_condition",
                   choices=["per_condition", "pooled"],
                   help="which mediation sweep supplies the head set")
    args = p.parse_args()
    mcfg, exp, chash, out, started = setup(args, "04_build_fv")

    med = RESULTS / "03_mediation" / mcfg.name
    acts_dir = RESULTS / "02_mean_activations" / mcfg.name
    if not (med / "head_sets.json").exists():
        raise SystemExit("run 03_mediation first")
    head_sets = {k: [tuple(h) for h in v]
                 for k, v in read_json(med / "head_sets.json").items()}

    model, tok, info = M.load_model(mcfg)
    todo = cells(resolve_tasks(args.tasks), args.conditions)

    built, skipped = {}, []
    for task, cond in track(todo, "vectors", stage="04_build_fv", enabled=not args.no_progress, label=lambda tc: f"{tc[0]}/{tc[1]}"):
        path = acts_dir / cond / f"{task}.pt"
        scope = "pooled" if args.head_set == "pooled" else cond
        if not path.exists() or scope not in head_sets:
            skipped.append(f"{task}/{cond}")
            continue
        fv = build_function_vector(model, info, A.load(path), head_sets[scope],
                                   task, cond, head_set_scope=scope)
        fv.save(RESULTS / "fv")
        built[(task, cond)] = fv

    if not built:
        raise SystemExit(f"built nothing; missing inputs for {skipped}")

    keys, sim = cosine_grid(built)
    write_json(out / "similarity.json",
               {"keys": [list(k) for k in keys], "cosine": sim.tolist()})
    write_json(out / "index.json",
               {f"{t}|{c}": {"norm": fv.norm, "head_set_scope": fv.head_set_scope,
                             "heads": [list(h) for h in fv.heads]}
                for (t, c), fv in built.items()})

    print(f"\nbuilt {len(built)} function vectors "
          f"(head set: {args.head_set}); skipped {len(skipped)}")
    if skipped:
        print("  skipped:", ", ".join(skipped[:10]) + (" ..." if len(skipped) > 10 else ""))
    finish(out, stage="04_build_fv", mcfg=mcfg, chash=chash, started=started,
           n_vectors=len(built), head_set=args.head_set, skipped=skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
