"""Figure generation. Each function takes plain dicts and writes one PDF + PNG.

Form is chosen by the job the data does, not by habit:

* magnitude across a few labelled groups -> grouped bars with CIs
* magnitude across depth -> lines (depth is ordered and continuous)
* magnitude across a matrix of pairs -> single-hue heat grid
* a signed effect across a matrix -> diverging heat grid, neutral grey midpoint
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .style import (
    CONDITION_COLOR,
    CONDITION_DASH,
    CONDITION_LABEL,
    CONDITION_MARKER,
    CONDITION_ORDER,
    COL_W,
    DIV,
    FULL_W,
    GRID,
    INK,
    INK_2,
    SEQ,
    TIER_LABEL,
    TIER_ORDER,
    condition_legend,
    finish,
    use_paper_style,
)

import matplotlib.pyplot as plt


def _grid(ax, axis="y"):
    ax.grid(axis=axis, zorder=0)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #

def fv_strength(rows: list[dict], out: Path, title: str = "") -> Path:
    """Zero-shot accuracy with the function vector added, by task and condition.

    `rows`: {task, condition, acc_with_fv, acc_zero_shot, lo, hi, n}
    Grouped bars because the comparison is between a handful of labelled
    categories; the zero-shot floor is drawn as a tick on each bar so the
    function vector's contribution is read as a distance, not an absolute.
    """
    use_paper_style()
    tasks = sorted({r["task"] for r in rows})
    conds = [c for c in CONDITION_ORDER if any(r["condition"] == c for r in rows)]
    by = {(r["task"], r["condition"]): r for r in rows}

    x = np.arange(len(tasks))
    w = 0.8 / max(1, len(conds))
    fig, ax = plt.subplots(figsize=(FULL_W, 2.5))
    _grid(ax)

    for j, cond in enumerate(conds):
        vals, los, his, floors = [], [], [], []
        for t in tasks:
            r = by.get((t, cond), {})
            vals.append(r.get("acc_with_fv", np.nan))
            los.append(r.get("acc_with_fv", np.nan) - r.get("lo", np.nan))
            his.append(r.get("hi", np.nan) - r.get("acc_with_fv", np.nan))
            floors.append(r.get("acc_zero_shot", np.nan))
        pos = x - 0.4 + w * (j + 0.5)
        ax.bar(pos, vals, width=w * 0.88, color=CONDITION_COLOR[cond],
               label=CONDITION_LABEL[cond], zorder=3, linewidth=0)
        ax.errorbar(pos, vals, yerr=[los, his], fmt="none", ecolor=INK_2,
                    elinewidth=0.7, capsize=1.6, zorder=4)
        # Zero-shot floor: a short horizontal tick, not a second axis.
        for p, f in zip(pos, floors):
            if np.isfinite(f):
                ax.plot([p - w * 0.44, p + w * 0.44], [f, f], color=INK,
                        linewidth=0.9, zorder=5)

    ax.set_xticks(x, [t.replace("_", " ") for t in tasks], rotation=30, ha="right")
    ax.set_ylabel("zero-shot top-1 with FV")
    ax.set_ylim(0, 1.02)
    if title:
        ax.set_title(title, loc="left", pad=16)
    condition_legend(ax, conds, loc="lower left", ncols=len(conds),
                     bbox_to_anchor=(0, 1.01, 1, 0.1), mode="expand", borderaxespad=0)
    ax.text(0.0, -0.52, "black tick: zero-shot accuracy without the function vector",
            transform=ax.transAxes, fontsize=6.5, color=INK_2)
    return finish(fig, out)


def layer_sweep(curves: list[dict], out: Path, title: str = "") -> Path:
    """Accuracy against the layer the function vector is added at.

    `curves`: {condition, layers, acc}. Lines, because depth is ordered; the
    peak's position is the thing being read.
    """
    use_paper_style()
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    _grid(ax, axis="both")
    conds = [c["condition"] for c in curves if c["condition"] in CONDITION_ORDER]

    for c in curves:
        cond = c["condition"]
        dash = CONDITION_DASH[cond]
        line, = ax.plot(c["layers"], c["acc"], color=CONDITION_COLOR[cond],
                        marker=CONDITION_MARKER[cond], markevery=max(1, len(c["layers"]) // 7),
                        label=CONDITION_LABEL[cond], zorder=3)
        if dash[0] is not None:
            line.set_dashes(list(dash))

    ax.set_xlabel("edit layer")
    ax.set_ylabel("top-1 accuracy")
    ax.set_ylim(bottom=0)
    if title:
        ax.set_title(title, loc="left")
    condition_legend(ax, conds, loc="upper right")
    return finish(fig, out)


def logit_lens(curves: list[dict], out: Path, title: str = "",
               interpret_from: int | None = None) -> Path:
    """Mean rank of the gold token across depth, log scale, lower is better."""
    use_paper_style()
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    _grid(ax, axis="both")
    conds = [c["condition"] for c in curves]

    for c in curves:
        cond = c["condition"]
        dash = CONDITION_DASH[cond]
        line, = ax.plot(c["layers"], np.maximum(c["mean_rank"], 0.5),
                        color=CONDITION_COLOR[cond], marker=CONDITION_MARKER[cond],
                        markevery=max(1, len(c["layers"]) // 7),
                        label=CONDITION_LABEL[cond], zorder=3)
        if dash[0] is not None:
            line.set_dashes(list(dash))

    ax.set_yscale("log")
    ax.set_xlabel("layer")
    ax.set_ylabel("mean rank of gold token")
    if interpret_from:
        # The lens is unreliable in early layers; shade what we do not interpret.
        ax.axvspan(0, interpret_from, color=GRID, alpha=0.7, zorder=1, linewidth=0)
        ax.text(interpret_from * 0.5, ax.get_ylim()[1] * 0.4, "not\ninterpreted",
                fontsize=6, color=INK_2, ha="center", va="top")
    if title:
        ax.set_title(title, loc="left")
    condition_legend(ax, conds, loc="lower left")
    return finish(fig, out)


def _heat(matrix, row_labels, col_labels, out: Path, *, cmap, vmin, vmax,
          cbar_label: str, title: str = "", annotate: bool = True,
          width: float = COL_W, fmt: str = "{:.2f}") -> Path:
    use_paper_style()
    m = np.asarray(matrix, dtype=float)
    h = 0.26 * len(row_labels) + 1.1
    fig, ax = plt.subplots(figsize=(width, h))
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(col_labels)), col_labels, rotation=40, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    # 2px surface gap between cells, so adjacent fills never touch.
    ax.grid(which="minor", color="#fcfcfb", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    if annotate:
        mid = (vmin + vmax) / 2
        span = max(vmax - mid, mid - vmin) or 1.0
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if not np.isfinite(m[i, j]):
                    continue
                # Direct labels satisfy the relief rule and make the grid legible
                # in greyscale print.
                strong = abs(m[i, j] - mid) / span > 0.55
                ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center",
                        fontsize=5.6, color="#ffffff" if strong else INK)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(cbar_label, fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    if title:
        ax.set_title(title, loc="left")
    return finish(fig, out)


def similarity_grid(keys: list[tuple[str, str]], sim, out: Path,
                    title: str = "") -> Path:
    """Cosine similarity between function vectors: do they group by task or language?

    Magnitude, so a single hue light-to-dark. Rows are ordered task-major, which
    makes task clustering appear as blocks on the diagonal and language
    clustering as a stripe pattern -- the two hypotheses are visually distinct.
    """
    labels = [f"{t.replace('_', ' ')} / {c}" for t, c in keys]
    return _heat(np.asarray(sim), labels, labels, out, cmap=SEQ, vmin=0, vmax=1,
                 cbar_label="cosine similarity", title=title,
                 annotate=len(labels) <= 16, width=FULL_W)


def head_overlap_grid(conds: list[str], jaccard, out: Path, title: str = "") -> Path:
    """Jaccard overlap of each condition's top-head set."""
    labels = [CONDITION_LABEL[c] for c in conds]
    return _heat(np.asarray(jaccard), labels, labels, out, cmap=SEQ, vmin=0, vmax=1,
                 cbar_label="Jaccard overlap of top heads", title=title)


def transfer_grid(sources: list[str], targets: list[str], acc, out: Path,
                  title: str = "") -> Path:
    """Accuracy when a vector built in `source` steers prompts in `target`."""
    return _heat(np.asarray(acc), [CONDITION_LABEL[s] for s in sources],
                 [CONDITION_LABEL[t] for t in targets], out, cmap=SEQ,
                 vmin=0, vmax=1, cbar_label="top-1 accuracy", title=title)


def aie_heatmap(aie, out: Path, condition: str = "", top_heads=None) -> Path:
    """Indirect effect per (layer, head).

    Signed, so a diverging map with a neutral grey midpoint at zero and a
    symmetric range -- otherwise the sign of a head's effect is unreadable.
    """
    use_paper_style()
    m = np.asarray(aie, dtype=float)
    lim = float(np.nanmax(np.abs(m))) or 1.0
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    im = ax.imshow(m.T, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto", origin="lower")
    ax.set_xlabel("layer")
    ax.set_ylabel("head")
    for sp in ax.spines.values():
        sp.set_visible(False)

    if top_heads:
        ls, hs = zip(*[(l, h) for l, h, _ in top_heads])
        # 2px surface ring on overlapping marks, so the highlight reads over any fill.
        ax.scatter(ls, hs, s=26, facecolors="none", edgecolors="#ffffff",
                   linewidths=1.6, zorder=3)
        ax.scatter(ls, hs, s=26, facecolors="none", edgecolors=INK,
                   linewidths=0.7, zorder=4)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("indirect effect", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    ax.set_title(f"{CONDITION_LABEL.get(condition, condition)}"
                 + ("  (circled: top heads)" if top_heads else ""), loc="left")
    return finish(fig, out)


def tier_degradation(rows: list[dict], out: Path, title: str = "") -> Path:
    """Accuracy by complexity/register tier, one line per condition.

    Tiers are ordered by intended difficulty, so this is a profile across an
    ordered factor -- lines, not bars.
    """
    use_paper_style()
    tiers = [t for t in TIER_ORDER if any(r["tier"] == t for r in rows)]
    conds = [c for c in CONDITION_ORDER if any(r["condition"] == c for r in rows)]
    by = {(r["tier"], r["condition"]): r for r in rows}

    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    _grid(ax, axis="both")
    x = np.arange(len(tiers))
    for cond in conds:
        ys = [by.get((t, cond), {}).get("acc", np.nan) for t in tiers]
        los = [by.get((t, cond), {}).get("lo", np.nan) for t in tiers]
        his = [by.get((t, cond), {}).get("hi", np.nan) for t in tiers]
        dash = CONDITION_DASH[cond]
        line, = ax.plot(x, ys, color=CONDITION_COLOR[cond],
                        marker=CONDITION_MARKER[cond], label=CONDITION_LABEL[cond],
                        zorder=3)
        if dash[0] is not None:
            line.set_dashes(list(dash))
        ax.fill_between(x, los, his, color=CONDITION_COLOR[cond], alpha=0.12,
                        linewidth=0, zorder=2)

    ax.set_xticks(x, [TIER_LABEL[t] for t in tiers], rotation=25, ha="right")
    ax.set_ylabel("top-1 accuracy")
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title, loc="left")
    condition_legend(ax, conds, loc="best")
    return finish(fig, out)


def baseline_bars(rows: list[dict], out: Path, title: str = "") -> Path:
    """10-shot accuracy and how many items survive the competence filter.

    Two panels sharing an x-axis rather than one chart with two y-scales.
    """
    use_paper_style()
    tasks = sorted({r["task"] for r in rows})
    conds = [c for c in CONDITION_ORDER if any(r["condition"] == c for r in rows)]
    by = {(r["task"], r["condition"]): r for r in rows}

    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.4), sharex=True,
                             gridspec_kw={"height_ratios": [1, 0.72], "hspace": 0.12})
    x = np.arange(len(tasks))
    w = 0.8 / max(1, len(conds))

    for j, cond in enumerate(conds):
        pos = x - 0.4 + w * (j + 0.5)
        acc = [by.get((t, cond), {}).get("top1", np.nan) for t in tasks]
        kept = [by.get((t, cond), {}).get("n_kept", np.nan) for t in tasks]
        axes[0].bar(pos, acc, width=w * 0.88, color=CONDITION_COLOR[cond],
                    label=CONDITION_LABEL[cond], zorder=3, linewidth=0)
        axes[1].bar(pos, kept, width=w * 0.88, color=CONDITION_COLOR[cond],
                    zorder=3, linewidth=0)

    for a in axes:
        _grid(a)
    axes[0].set_ylabel("10-shot top-1")
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylabel("items kept")
    axes[1].set_xticks(x, [t.replace("_", " ") for t in tasks], rotation=30, ha="right")
    if title:
        axes[0].set_title(title, loc="left")
    condition_legend(axes[0], conds, loc="lower left", ncols=len(conds),
                     bbox_to_anchor=(0, 1.01, 1, 0.1), mode="expand", borderaxespad=0)
    return finish(fig, out)
