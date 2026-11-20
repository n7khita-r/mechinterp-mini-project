#!/usr/bin/env python3
"""Stage 01 -- n-shot and zero-shot baselines, and the competence filter.

For every (task, condition) cell: score the validation items with a clean
n-shot prompt and with no prompt at all, then record which items the model gets
right from demonstrations. Everything downstream runs on that filtered set --
a function vector cannot be blamed for failing a task the model cannot do even
when shown examples.

Read the `kept` column first. It is simultaneously the headline question (can
Gemma do these tasks in Telugu at all?) and the thing that decides whether any
downstream number has enough items behind it to mean anything.
"""

from __future__ import annotations

import sys

import torch
from fvcm.progress import track

from _common import REPO_ROOT, cells, finish, parser, resolve_tasks, setup  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))

from fvcm import data as D, model as M  # noqa: E402
from fvcm.cache import write_json  # noqa: E402
from fvcm.evaluate import (  # noqa: E402
    batched, competence_filter, rng_for, score_batch, summarise,
)
from fvcm.prompts import build, sample_shots, zero_shot  # noqa: E402

MIN_KEPT_WARN = 15


def main() -> int:
    args = parser("n-shot / zero-shot baselines and the competence filter").parse_args()
    mcfg, exp, chash, out, started = setup(args, "01_baselines")

    model, tok, info = M.load_model(mcfg)
    todo = cells(resolve_tasks(args.tasks), args.conditions)

    # Per-item scores, so accuracy can be broken down by tier afterwards without
    # re-running the model. The aggregate rows below cannot support that.
    rows, filters, thin, per_item = [], {}, [], []
    for task, cond in track(todo, "cells", stage="01_baselines", enabled=not args.no_progress, label=lambda tc: f"{tc[0]}/{tc[1]}"):
        items = D.headline(D.load_items(task, cond, exp), exp)
        split = D.split_items(items, exp)
        if not split.valid:
            continue

        rng = rng_for("baseline", task, cond, exp.seed)
        nshot_prompts, zs_prompts = [], []
        for item in split.valid:
            shots = sample_shots(split.train, exp.n_shots, rng, exclude=item)
            nshot_prompts.append(build(item, shots, exp))
            zs_prompts.append(zero_shot(item, exp))

        bs = int(info.batch.get("score", 8))
        nshot, zs = [], []
        for chunk in batched(nshot_prompts, bs):
            nshot += score_batch(model, tok, chunk, model.device)
        for chunk in batched(zs_prompts, bs):
            zs += score_batch(model, tok, chunk, model.device)

        kept = competence_filter(nshot)
        filters[f"{task}|{cond}"] = sorted(kept)

        row = {
            "task": task, "condition": cond,
            "n_items": len(split.valid), "n_train": len(split.train),
            "n_kept": len(kept),
            "kept_frac": len(kept) / len(split.valid) if split.valid else float("nan"),
            **{f"nshot_{k}": v for k, v in summarise(nshot, exp).items()},
            **{f"zs_{k}": v for k, v in summarise(zs, exp).items()},
        }
        row["top1"] = row["nshot_top1"]
        row["mean_prompt_tokens"] = row["nshot_mean_prompt_tokens"]
        row["mean_answer_tokens"] = row["nshot_mean_answer_tokens"]
        rows.append(row)
        per_item += [{"task": task, "condition": cond, "item_id": s.item_id,
                      "tier": s.tier, "rank": s.rank} for s in nshot]
        if len(kept) < MIN_KEPT_WARN:
            thin.append((task, cond, len(kept)))

    write_json(out / "baselines.json", rows)
    write_json(out / "scores.json", per_item)
    write_json(out / "competence_filter.json", filters)

    print(f"\n{'task':20s} {'cond':5s} {'n':>4s} {'10-shot':>8s} {'0-shot':>7s} {'kept':>6s}")
    for r in rows:
        print(f"{r['task']:20s} {r['condition']:5s} {r['n_items']:4d} "
              f"{r['nshot_top1']:8.2f} {r['zs_top1']:7.2f} "
              f"{r['n_kept']:4d} ({r['kept_frac']:.0%})")

    if thin:
        print(f"\nWARNING: {len(thin)} cell(s) keep fewer than {MIN_KEPT_WARN} items. "
              "Downstream intervals there will be too wide to interpret.")
        for task, cond, n in thin:
            print(f"  {task}/{cond}: {n}")
        print("Remedies: add rows to the task's TSV, lower data.test_split, or "
              "report these cells pooled rather than per task.")

    finish(out, stage="01_baselines", mcfg=mcfg, chash=chash, started=started,
           n_cells=len(rows), thin_cells=thin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
