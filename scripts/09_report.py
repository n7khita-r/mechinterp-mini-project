#!/usr/bin/env python3
"""Stage 09 -- turn every stage's JSON into the paper's figures and tables.

CPU only. Missing stages are skipped with a note rather than crashing.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

from _common import REPO_ROOT, RESULTS, finish, parser, setup  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))

from fvcm import ALL_TASKS, TIERS, conditions_for  # noqa: E402
from fvcm.analysis import figures as F, tables as T  # noqa: E402
from fvcm.analysis.style import CONDITION_ORDER  # noqa: E402
from fvcm.cache import read_json, write_json  # noqa: E402
from fvcm.config import ExpConfig  # noqa: E402
from fvcm.data import headline, load_items  # noqa: E402
from fvcm.evaluate import Score, bootstrap_ci, paired_delta  # noqa: E402

PAPER_FIG = REPO_ROOT / "paper" / "figures"
PAPER_TAB = REPO_ROOT / "paper" / "tables"
SLIDE_IMG = REPO_ROOT / "presentation" / "images"


# Withheld from the reported tables: `tense` is flat at zero in every condition
# including English, so it measures nothing about code-mixing. Raw numbers stay
# in results/ and the omission is stated in the write-up.
EXCLUDE_TASKS = {"tense"}


def _drop_excluded(obj):
    """Strip withheld tasks from a loaded result file, whatever its shape."""
    if isinstance(obj, list):
        return [r for r in obj
                if not (isinstance(r, dict) and r.get("task") in EXCLUDE_TASKS)]
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items()
                if str(k).split("|")[0] not in EXCLUDE_TASKS}
    return obj


def _load(path: Path):
    if not path.exists():
        print(f"  skip: {path.relative_to(REPO_ROOT)} not found")
        return None
    return _drop_excluded(read_json(path))


def _scores(dicts: list[dict]) -> list[Score]:
    return [Score(**d) for d in dicts]


def _publish(pdf: Path) -> None:
    """One figure, two destinations: PDF for the paper, PNG for the slides."""
    SLIDE_IMG.mkdir(parents=True, exist_ok=True)
    png = pdf.with_suffix(".png")
    if png.exists():
        shutil.copy2(png, SLIDE_IMG / png.name)


def main() -> int:
    p = parser("generate all paper figures and tables")
    p.add_argument("--headline-task", default="antonym",
                   help="task used for the single-task illustrative figures")
    args = p.parse_args()
    mcfg, exp, chash, out, started = setup(args, "09_report")

    for d in (PAPER_FIG, PAPER_TAB, SLIDE_IMG):
        d.mkdir(parents=True, exist_ok=True)
    R = RESULTS
    m = mcfg.name
    made = []

    # ---- dataset summary (no model needed) --------------------------------- #
    ds_rows = []
    for task in ALL_TASKS:
        if task in EXCLUDE_TASKS:
            continue
        cond = conditions_for(task)[0]
        items = load_items(task, cond, ExpConfig.load(args.exp))
        counts = {t: sum(1 for i in items if i.tier == t) for t in TIERS}
        ds_rows.append({"task": task, **counts})
    made.append(T.dataset_summary(ds_rows, PAPER_TAB / "dataset.tex"))

    # ---- 01 baselines ------------------------------------------------------ #
    base = _load(R / "01_baselines" / m / "baselines.json")
    if base:
        made.append(T.baselines(base, PAPER_TAB / "baselines.tex"))
        made.append(T.token_lengths(base, PAPER_TAB / "token_lengths.tex"))
        _publish(F.baseline_bars(base, PAPER_FIG / "baselines",
                                 "10-shot accuracy and items kept by the competence filter"))
        made.append(PAPER_FIG / "baselines.pdf")

    # ---- 03 mediation ------------------------------------------------------ #
    head_sets = _load(R / "03_mediation" / m / "head_sets.json")
    if head_sets:
        sets = {k: [tuple(h) for h in v] for k, v in head_sets.items()}
        made.append(T.head_sets(sets, PAPER_TAB / "head_sets.tex"))

        overlaps = _load(R / "03_mediation" / m / "head_overlap.json") or {}
        conds = [c for c in CONDITION_ORDER if c in sets]
        if len(conds) > 1:
            J = np.array([[overlaps.get(f"{a}|{b}", {}).get("jaccard", np.nan)
                           for b in conds] for a in conds])
            _publish(F.head_overlap_grid(conds, J, PAPER_FIG / "head_overlap",
                                         "Top-head overlap between conditions"))
            made.append(PAPER_FIG / "head_overlap.pdf")

        import torch
        for cond in conds:
            pt = R / "03_mediation" / m / f"aie_{cond}.pt"
            if pt.exists():
                aie = torch.load(pt, map_location="cpu")
                mean = aie.mean(dim=0) if aie.dim() == 3 else aie
                _publish(F.aie_heatmap(mean.numpy(), PAPER_FIG / f"aie_{cond}",
                                       cond, top_heads=sets[cond]))
                made.append(PAPER_FIG / f"aie_{cond}.pdf")

    # ---- 04 similarity ----------------------------------------------------- #
    sim = _load(R / "04_build_fv" / m / "similarity.json")
    if sim:
        keys = [tuple(k) for k in sim["keys"]]
        # Task-major: task clustering then reads as diagonal blocks.
        order = sorted(range(len(keys)),
                       key=lambda i: (keys[i][0], CONDITION_ORDER.index(keys[i][1])))
        M_ = np.array(sim["cosine"])[np.ix_(order, order)]
        _publish(F.similarity_grid([keys[i] for i in order], M_,
                                   PAPER_FIG / "similarity",
                                   "Cosine similarity between function vectors"))
        made.append(PAPER_FIG / "similarity.pdf")

    # ---- 05 intervention --------------------------------------------------- #
    interv = _load(R / "05_intervene" / m / "intervention.json")
    scores = _load(R / "05_intervene" / m / "scores.json")
    if interv:
        n_boot = int(exp.eval.get("n_bootstrap", 2000))
        rows = []
        for r in interv:
            key = f"{r['task']}|{r['condition']}"
            s = _scores(scores[key]["with_fv"]) if scores and key in scores else []
            lo, hi = bootstrap_ci(s, k=1, n_boot=n_boot, seed=exp.seed)
            rows.append({**r, "lo": lo, "hi": hi})
        made.append(T.fv_strength(rows, PAPER_TAB / "fv_strength.tex"))
        _publish(F.fv_strength(rows, PAPER_FIG / "fv_strength"))
        made.append(PAPER_FIG / "fv_strength.pdf")

        curves = [{"condition": r["condition"], "layers": r["sweep_layers"],
                   "acc": r["sweep_acc"]}
                  for r in interv if r["task"] == args.headline_task]
        if curves:
            _publish(F.layer_sweep(curves, PAPER_FIG / "layer_sweep",
                                   args.headline_task.replace("_", " ")))
            made.append(PAPER_FIG / "layer_sweep.pdf")

        if scores:
            deltas = []
            for task in sorted({r["task"] for r in interv}):
                conds = [c for c in CONDITION_ORDER
                         if f"{task}|{c}" in scores]
                for b in conds[1:]:
                    d = paired_delta(_scores(scores[f"{task}|{conds[0]}"]["with_fv"]),
                                     _scores(scores[f"{task}|{b}"]["with_fv"]),
                                     k=1, n_boot=n_boot, seed=exp.seed)
                    deltas.append({"task": task, "a": conds[0], "b": b, **d})
            made.append(T.paired_gaps(deltas, PAPER_TAB / "paired_gaps.tex"))
            write_json(out / "paired_gaps.json", deltas)

            tier_rows = []
            for cond in CONDITION_ORDER:
                allsc = [s for k, v in scores.items() if k.endswith(f"|{cond}")
                         for s in _scores(v["with_fv"])]
                for tier in TIERS:
                    sub = [s for s in allsc if s.tier == tier]
                    if len(sub) < 3:
                        continue
                    lo, hi = bootstrap_ci(sub, k=1, n_boot=n_boot, seed=exp.seed)
                    tier_rows.append({"tier": tier, "condition": cond,
                                      "acc": float(np.mean([s.rank == 0 for s in sub])),
                                      "lo": lo, "hi": hi, "n": len(sub)})
            if tier_rows:
                made.append(T.tier_degradation(tier_rows, PAPER_TAB / "tier.tex"))
                _publish(F.tier_degradation(tier_rows, PAPER_FIG / "tier",
                                            "Accuracy by complexity tier"))
                made.append(PAPER_FIG / "tier.pdf")

    # ---- 06 transfer ------------------------------------------------------- #
    tm = _load(R / "06_transfer" / m / "transfer_matrix.json")
    if tm:
        conds = tm["conditions"]
        made.append(T.transfer(conds, conds, tm["acc"], PAPER_TAB / "transfer.tex"))
        _publish(F.transfer_grid(conds, conds, tm["acc"], PAPER_FIG / "transfer",
                                 "Cross-condition transfer"))
        made.append(PAPER_FIG / "transfer.pdf")

    # ---- 07 logit lens ----------------------------------------------------- #
    lens = _load(R / "07_logitlens" / m / "lens_curves.json")
    if lens:
        sel = [c for c in lens if c["task"] == args.headline_task]
        if sel:
            _publish(F.logit_lens(sel, PAPER_FIG / "logit_lens",
                                  args.headline_task.replace("_", " "),
                                  interpret_from=sel[0].get("interpret_from_layer")))
            made.append(PAPER_FIG / "logit_lens.pdf")

    # ---- 08 controls ------------------------------------------------------- #
    ctrl = _load(R / "08_controls" / m / "controls.json")
    if ctrl:
        made.append(T.controls(ctrl, PAPER_TAB / "controls.tex"))

    print(f"\nwrote {len(made)} artefacts")
    for p_ in made:
        print(f"  {Path(p_).relative_to(REPO_ROOT)}")
    finish(out, stage="09_report", mcfg=mcfg, chash=chash, started=started,
           n_artefacts=len(made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
