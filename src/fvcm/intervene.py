"""Adding a function vector to a zero-shot prompt, and the layer sweep.

    h_l <- h_l + v,   with  v rescaled to  alpha * ||h_l||

Layer and alpha are chosen on a dev slice; accuracy is reported on the held-out
remainder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from .progress import nested

from .config import ExpConfig
from .data import Item
from .evaluate import Score, accuracy, batched, score_batch
from .fv import FunctionVector
from .model import ModelInfo, add_to_residual, capture_residual
from .prompts import zero_shot


@dataclass
class LayerSweep:
    """Zero-shot accuracy at every candidate edit layer."""

    layers: list[int]
    acc: list[float]
    n_items: int

    @property
    def best_layer(self) -> int:
        return self.select_layer()[0]

    def select_layer(self, min_items: int = 8) -> tuple[int, str]:
        """Pick the edit layer from a 3-point smoothed curve. Returns (layer, reason).

        Ties break toward one third of depth, so a flat or tiny dev slice falls
        back to that prior instead of collapsing to layer 0.
        """
        prior = max(self.layers) // 3

        if self.n_items < min_items:
            return prior, f"prior (dev slice {self.n_items} < {min_items} items)"

        acc = self.acc
        smoothed = [
            sum(acc[max(0, i - 1):min(len(acc), i + 2)])
            / len(acc[max(0, i - 1):min(len(acc), i + 2)])
            for i in range(len(acc))
        ]
        peak = max(smoothed)
        if peak <= min(smoothed) + 1e-9:
            return prior, "prior (sweep curve is flat)"

        tol = max(1e-9, 0.02 * peak)
        near = [i for i, v in enumerate(smoothed) if v >= peak - tol]
        best = min(near, key=lambda i: abs(self.layers[i] - prior))
        return self.layers[best], f"smoothed peak ({len(near)} layers within tolerance)"


def _ctx_factory(model, info: ModelInfo, layer: int, vec: torch.Tensor):
    return lambda: add_to_residual(model, info, layer, vec)


@torch.no_grad()
def residual_norms(model, tokenizer, info: ModelInfo, items: list[Item],
                   exp: ExpConfig, max_items: int = 8) -> list[float]:
    """Mean ||residual|| at the last token, per layer.

    Gemma's residual grows ~780 to ~1e5 with depth, so every injected vector is
    rescaled relative to it.
    """
    prompts = [zero_shot(i, exp) for i in items[:max_items]]
    enc = tokenizer([p.text for p in prompts], return_tensors="pt",
                    padding=True).to(model.device)
    store: dict[int, torch.Tensor] = {}
    with capture_residual(model, info, store):
        model(**enc)
    out = []
    for l in range(info.n_layers):
        r = store[l].float()
        r = r[:, -1, :] if r.dim() == 3 else r
        out.append(float(r.norm(dim=-1).mean()))
    return out


@torch.no_grad()
def score_zero_shot(model, tokenizer, info: ModelInfo, items: list[Item],
                    exp: ExpConfig, vec: torch.Tensor | None = None,
                    layer: int | None = None) -> list[Score]:
    """Score zero-shot prompts; `vec=None` gives the zero-shot floor."""
    prompts = [zero_shot(i, exp) for i in items]
    ctx = None if vec is None else _ctx_factory(model, info, layer, vec)
    bs = int(info.batch.get("score", 8))
    out: list[Score] = []
    for chunk in batched(prompts, bs):
        out += score_batch(model, tokenizer, chunk, model.device, ctx=ctx)
    return out


@torch.no_grad()
def layer_sweep(model, tokenizer, info: ModelInfo, items: list[Item], exp: ExpConfig,
                vec: torch.Tensor, progress: bool = True,
                rel_norm: float | None = None) -> LayerSweep:
    """Zero-shot accuracy with the vector added at each layer in turn."""
    layers = list(range(info.n_layers))
    accs = []
    norms = residual_norms(model, tokenizer, info, items, exp) if rel_norm else None
    base = float(vec.float().norm()) or 1.0
    for layer in nested(layers, "layer sweep", stage="05_intervene",
                        enabled=progress, label=lambda l: f"layer {l}"):
        v = vec if not rel_norm else vec * (rel_norm * norms[layer] / base)
        scores = score_zero_shot(model, tokenizer, info, items, exp, v, layer)
        accs.append(accuracy(scores, k=1))
    return LayerSweep(layers=layers, acc=accs, n_items=len(items))


@torch.no_grad()
def joint_select(model, tokenizer, info: ModelInfo, items: list[Item], exp: ExpConfig,
                 fv: FunctionVector, norms: list[float], alphas: list[float],
                 stride: int, progress: bool = True) -> tuple[int, float, dict, list]:
    """Search layer and scale together on the dev slice, over a strided layer grid.

    The best layer depends on the scale, so a sequential search can miss it.
    """
    layers = list(range(0, info.n_layers, max(1, stride)))
    base = float(fv.vector.float().norm()) or 1.0
    grid: dict[tuple[int, float], float] = {}

    for layer in nested(layers, "joint grid", stage="05_intervene",
                        enabled=progress, label=lambda l: f"layer {l}"):
        for a in alphas:
            vec = fv.vector * (a * norms[layer] / base)
            grid[(layer, a)] = accuracy(
                score_zero_shot(model, tokenizer, info, items, exp, vec, layer), k=1)

    # Ties go to the smallest scale, then to mid-depth.
    best = max(grid.values())
    prior = info.n_layers // 3
    near = [(l, a) for (l, a), v in grid.items() if v >= best - 1e-9]
    layer, alpha = min(near, key=lambda la: (la[1], abs(la[0] - prior)))

    # Refine locally: the stride may have stepped over the peak.
    lo, hi = max(0, layer - stride + 1), min(info.n_layers - 1, layer + stride - 1)
    fine = [l for l in range(lo, hi + 1) if l not in layers]
    for l in fine:
        vec = fv.vector * (alpha * norms[l] / base)
        grid[(l, alpha)] = accuracy(
            score_zero_shot(model, tokenizer, info, items, exp, vec, l), k=1)
    best_fine = max((v, -abs(l - prior), l) for (l, a), v in grid.items() if a == alpha)
    layer = best_fine[2]

    curve_layers = sorted({l for (l, a) in grid if a == alpha})
    curve = [grid[(l, alpha)] for l in curve_layers]
    return layer, alpha, grid, [curve_layers, curve]


def dev_test_split(items: list[Item], exp: ExpConfig) -> tuple[list[Item], list[Item]]:
    """Split the evaluation pool into a dev slice and a held-out rest, by item uid."""
    frac = float(exp.intervene.get("dev_slice_frac", 0.3))
    ordered = sorted(items, key=lambda i: i.uid)
    n_dev = max(1, int(round(len(ordered) * frac))) if ordered else 0
    return ordered[:n_dev], ordered[n_dev:]


@torch.no_grad()
def run_cell(model, tokenizer, info: ModelInfo, items: list[Item], exp: ExpConfig,
             fv: FunctionVector, progress: bool = True) -> dict:
    """Full intervention protocol for one (task, condition) cell.

    Returns the sweep curve, the selected layer, and held-out scores with and
    without the function vector.
    """
    dev, test = dev_test_split(items, exp)

    cap = exp.intervene.get("layer_sweep_max_items")
    sweep_items = dev[: int(cap)] if cap else dev

    alpha0 = float(exp.intervene.get("fv_rel_norm", 0.1))
    grid = [float(a) for a in exp.intervene.get("fv_rel_norm_grid",
                                                [0.05, 0.1, 0.2, 0.4, 0.8])]
    norms = residual_norms(model, tokenizer, info, sweep_items, exp)
    base = float(fv.vector.float().norm()) or 1.0
    requested = int(exp.intervene.get("edit_layer", -1))
    joint = bool(exp.intervene.get("joint_selection", True))

    if joint and requested < 0:
        stride = int(exp.intervene.get("joint_layer_stride", 4))
        layer, alpha, jgrid, curve = joint_select(
            model, tokenizer, info, sweep_items, exp, fv, norms, grid, stride,
            progress=progress)
        reason = f"joint (layer, scale) grid, stride {stride}"
        sweep = LayerSweep(layers=curve[0], acc=curve[1], n_items=len(sweep_items))
        alpha_acc = sorted({(a, v) for (l, a), v in jgrid.items() if l == layer})
    else:
        sweep = layer_sweep(model, tokenizer, info, sweep_items, exp, fv.vector,
                            progress=progress, rel_norm=alpha0)
        if requested < 0:
            layer, reason = sweep.select_layer()
        else:
            layer, reason = requested, "fixed by config"

        def scaled_at(a: float) -> torch.Tensor:
            return fv.vector * (a * norms[layer] / base)

        # Ties go to the smallest alpha.
        alpha_acc = [(a, accuracy(score_zero_shot(model, tokenizer, info, sweep_items,
                                                  exp, scaled_at(a), layer), k=1))
                     for a in grid]
        if alpha_acc:
            best_acc = max(v for _, v in alpha_acc)
            alpha = min(a for a, v in alpha_acc if v >= best_acc - 1e-9)
        else:
            alpha = alpha0

    def scaled(a: float) -> torch.Tensor:
        return fv.vector * (a * norms[layer] / base)

    held = test or dev
    with_fv = score_zero_shot(model, tokenizer, info, held, exp, scaled(alpha), layer)
    without = score_zero_shot(model, tokenizer, info, held, exp, None, None)

    return {
        "task": fv.task,
        "condition": fv.condition,
        "edit_layer": layer,
        "layer_reason": reason,
        "selection": "joint" if joint and requested < 0 else "sequential",
        "layer_requested": requested,
        "fv_rel_norm": alpha,
        "fv_scale_multiplier": round(alpha * norms[layer] / base, 2),
        "fv_norm_raw": round(base, 3),
        "resid_norm_at_layer": round(norms[layer], 1),
        "alpha_sweep": [[a, v] for a, v in alpha_acc],
        "sweep_layers": sweep.layers,
        "sweep_acc": sweep.acc,
        "sweep_n_items": sweep.n_items,
        "n_held_out": len(held),
        "acc_with_fv": accuracy(with_fv, k=1),
        "acc_zero_shot": accuracy(without, k=1),
        "scores_with_fv": [s.as_dict() for s in with_fv],
        "scores_zero_shot": [s.as_dict() for s in without],
    }


@torch.no_grad()
def transfer(model, tokenizer, info: ModelInfo, items: list[Item], exp: ExpConfig,
             source_fv: FunctionVector, layer: int, rel_norm: float) -> dict:
    """Apply a function vector built in one condition to prompts in another.

    Rescaled to the target cell's own `rel_norm * ||h_layer||`, so source and
    target vectors differ only in direction.
    """
    norms = residual_norms(model, tokenizer, info, items, exp)
    base = float(source_fv.vector.float().norm()) or 1.0
    vec = source_fv.vector * (rel_norm * norms[layer] / base)
    scores = score_zero_shot(model, tokenizer, info, items, exp, vec, layer)
    target = items[0].condition if items else "?"
    return {
        "source_condition": source_fv.condition,
        "target_condition": target,
        "task": source_fv.task,
        "edit_layer": layer,
        "n": len(scores),
        "acc": accuracy(scores, k=1),
        "fv_rel_norm": rel_norm,
        "resid_norm_at_layer": round(norms[layer], 1),
        "scores": [s.as_dict() for s in scores],
    }
