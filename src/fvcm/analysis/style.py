"""Shared figure style for the paper and the slides.

Palette and rules follow the project's data-visualisation method:

* The four conditions are a **categorical** encoding, so they get a fixed hue
  order that is never cycled. The chosen four -- blue, orange, aqua, violet --
  were validated with the all-pairs colour-vision-deficiency check (worst CVD
  Delta E 9.2, worst normal-vision Delta E 16.3 against a light surface), because
  in line charts and heat grids every series is on screen at once, so the
  adjacent-pair relaxation does not apply.
* Aqua sits below 3:1 contrast on a light surface, which obliges **relief**:
  every figure here is emitted alongside the same numbers as a LaTeX table, and
  line charts carry distinct markers and dash patterns. That also makes the
  figures readable in greyscale print, which matters for a paper.
* Magnitude (cosine similarity, head overlap, transfer) uses a **single hue,
  light to dark**. Signed quantities (indirect effects, which can be negative)
  use a **diverging** pair with a neutral grey midpoint -- never a rainbow, and
  never a hue at the midpoint.
* No figure in this project has two y-axes.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8982"
GRID = "#e3e2dd"

# Fixed categorical order. Index by condition name, never by position in a loop.
CONDITION_COLOR = {
    "en": "#2a78d6",     # blue
    "te": "#eb6834",     # orange
    "rom": "#1baf7a",    # aqua
    "tg": "#4a3aa7",     # violet
}
CONDITION_MARKER = {"en": "o", "te": "s", "rom": "^", "tg": "D"}
CONDITION_DASH = {"en": (None, None), "te": (4, 1.5), "rom": (1.5, 1.5), "tg": (6, 1.5, 1.5, 1.5)}
CONDITION_LABEL = {
    "en": "English",
    "te": "Telugu (script)",
    "rom": "Telugu (romanized)",
    "tg": "Tenglish",
}
CONDITION_ORDER = ("en", "te", "rom", "tg")

TIER_ORDER = ("core", "complex", "sanskritized", "rare", "literary_en")
TIER_LABEL = {
    "core": "core",
    "complex": "complex",
    "sanskritized": "sanskritized",
    "rare": "rare",
    "literary_en": "rare English",
}

# Sequential: one hue, light to dark. For magnitudes in [0, 1].
SEQ = LinearSegmentedColormap.from_list("fvcm_seq", ["#f2f7fd", "#2a78d6", "#123a69"])
# Diverging: two hues, neutral grey midpoint. For signed effects.
DIV = LinearSegmentedColormap.from_list(
    "fvcm_div", ["#123a69", "#2a78d6", "#dedcd5", "#eb6834", "#7a2f10"]
)


def use_paper_style() -> None:
    """Serif, small, recessive chrome -- sized to drop into an ACL column."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# ACL single column is 3.17in wide; the full text block is 6.3in.
COL_W = 3.17
FULL_W = 6.3


def finish(fig, path, also_png: bool = True):
    """Save as PDF for LaTeX and PNG for the slides, from one figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    if also_png:
        fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return path.with_suffix(".pdf")


def condition_legend(ax, conditions, **kw):
    """Legend is always present for two or more series, so identity is never
    carried by colour alone."""
    if len(conditions) >= 2:
        ax.legend(loc=kw.pop("loc", "best"), ncols=kw.pop("ncols", 1), **kw)
