"""Adding a function vector to a zero-shot prompt, and the layer sweep.

The intervention is the paper's test of whether a function vector really carries
the task: take a prompt with **no demonstrations**, add `v_t` to the residual
stream at one layer's output at the last token, and see whether the model now
performs the task.

Layer selection departs from the reference. The reference sweeps every layer and
reports the best. With hundreds to thousands of items that is harmless; with the
~20-80 evaluation items hand-authored data affords, taking the maximum over
34-62 layers is badly optimistically biased -- the winning layer is partly
fitted to noise. So the sweep runs on a **dev slice** of the evaluation pool,
the layer is fixed from that, and the headline number is computed at that fixed
layer on the **held-out remainder**. The full sweep curve is still reported,
because its shape (a mid-depth peak) is itself a result.
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
        """Pick the edit layer, robustly. Returns (layer, reason).

        A plain argmax is wrong here in two ways, and both bit us:

        1. **Ties resolve to index 0.** When a small dev slice scores 0.0 at
           every layer, `max` returns the first index -- layer 0, which is
           precisely where adding a function vector does nothing. Cells then
           report a ~0.00 delta that looks like a null result but is an artefact.
        2. **Single-layer argmax overfits.** With tens of items the curve is
           noisy, and the winning layer is partly fitted to that noise.

        So: smooth the curve with a 3-point moving average before choosing, and
        break ties toward one third of depth, which is where Todd et al. find the
        effect peaks and where the proposal said we would intervene. If the dev
        slice is too small to support any choice, or the curve is flat, say so
        and fall back to that prior rather than pretending to have measured it.
        """
        prior = max(self.layers) // 3          # ~1/3 depth, as in the paper

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

        # Among layers within noise of the peak, take the one nearest the prior
        # rather than the first, so ties never silently collapse to layer 0.
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

    Needed because Gemma's residual stream grows by two orders of magnitude with
    depth (~780 at layer 0 to ~100k at layer 33, a consequence of scaling
    embeddings by sqrt(d_model) and never renormalising the stream itself). A
    raw function vector has norm ~16, so adding it unscaled is a ~0.1% nudge at
    layer 11 and ~0.02% by layer 32 -- arithmetically real, numerically
    invisible. Todd et al. did not hit this because GPT-J's residual is far
    smaller relative to head outputs.
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
    """Score zero-shot prompts, optionally with a vector added at `layer`.

    With `vec=None` this is the zero-shot floor: what the model does with no
    demonstrations and no intervention.
    """
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
    # Scale the vector to a fixed fraction of the residual norm at each layer,
    # so the sweep compares layers on equal footing instead of comparing a
    # constant vector against a stream that grows 100x with depth.
    norms = residual_norms(model, tokenizer, info, items, exp) if rel_norm else None
    base = float(vec.float().norm()) or 1.0
    for layer in nested(layers, "layer sweep", stage="05_intervene",
                        enabled=progress, label=lambda l: f"layer {l}"):
        v = vec if not rel_norm else vec * (rel_norm * norms[layer] / base)
        scores = score_zero_shot(model, tokenizer, info, items, exp, v, layer)
        accs.append(accuracy(scores, k=1))
    return LayerSweep(layers=layers, acc=accs, n_items=len(items))


def dev_test_split(items: list[Item], exp: ExpConfig) -> tuple[list[Item], list[Item]]:
    """Split the evaluation pool into a layer-selection slice and a held-out rest.

    Deterministic and content-addressed, so it does not move when rows are added.
    """
    frac = float(exp.intervene.get("dev_slice_frac", 0.3))
    ordered = sorted(items, key=lambda i: i.uid)
    n_dev = max(1, int(round(len(ordered) * frac))) if ordered else 0
    return ordered[:n_dev], ordered[n_dev:]


@torch.no_grad()
def run_cell(model, tokenizer, info: ModelInfo, items: list[Item], exp: ExpConfig,
             fv: FunctionVector, progress: bool = True) -> dict:
    """Full intervention protocol for one (task, condition) cell.

    Returns the sweep curve, the selected layer, and held-out scores both with
    and without the function vector, so the paper's number is a difference
    measured on items that played no part in choosing the layer.
    """
    dev, test = dev_test_split(items, exp)

    cap = exp.intervene.get("layer_sweep_max_items")
    sweep_items = dev[: int(cap)] if cap else dev

    # The vector is scaled to a fraction `alpha` of the residual norm at the edit
    # layer. A raw (alpha-free) vector is ~1000x smaller than Gemma's residual
    # stream and does nothing; a fixed multiplier would not transfer between
    # models whose residual norms differ. Both the layer and alpha are chosen on
    # the dev slice and reported on the held-out remainder.
    alpha0 = float(exp.intervene.get("fv_rel_norm", 0.1))
    grid = [float(a) for a in exp.intervene.get("fv_rel_norm_grid",
                                                [0.05, 0.1, 0.2, 0.4, 0.8])]

    sweep = layer_sweep(model, tokenizer, info, sweep_items, exp, fv.vector,
                        progress=progress, rel_norm=alpha0)

    requested = int(exp.intervene.get("edit_layer", -1))
    if requested < 0:
        layer, reason = sweep.select_layer()
    else:
        layer, reason = requested, "fixed by config"

    norms = residual_norms(model, tokenizer, info, sweep_items, exp)
    base = float(fv.vector.float().norm()) or 1.0

    def scaled(a: float) -> torch.Tensor:
        return fv.vector * (a * norms[layer] / base)

    # Pick alpha on the dev slice at the chosen layer. Ties go to the smallest
    # alpha: the least intervention that achieves the effect.
    alpha_acc = [(a, accuracy(score_zero_shot(model, tokenizer, info, sweep_items,
                                              exp, scaled(a), layer), k=1))
                 for a in grid]
    if alpha_acc:
        best_acc = max(v for _, v in alpha_acc)
        alpha = min(a for a, v in alpha_acc if v >= best_acc - 1e-9)
    else:
        alpha = alpha0

    held = test or dev
    with_fv = score_zero_shot(model, tokenizer, info, held, exp, scaled(alpha), layer)
    without = score_zero_shot(model, tokenizer, info, held, exp, None, None)

    return {
        "task": fv.task,
        "condition": fv.condition,
        "edit_layer": layer,
        "layer_reason": reason,
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

    If the English vector steers Telugu prompts as well as the Telugu vector
    does, the representation is language-independent and the gap lies in reading
    the input. If it does not transfer, the task representation itself differs.
    That contrast is the point of the experiment.

    The vector is rescaled to `rel_norm * ||residual|| at `layer` -- the scale the
    *target* cell selected -- so source and target vectors are injected at equal
    magnitude and the only thing that varies is direction. Applied raw, every
    source vector is far too small to move the argmax and the matrix degenerates
    into one row repeated per target: a null that reads like a finding.
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
