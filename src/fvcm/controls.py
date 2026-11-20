"""Controls: random vector, unrelated-task vector, and shuffled-label floor.

Every vector is scaled to the same fraction of the residual norm at the edit
layer, so the interventions differ only in direction.
"""

from __future__ import annotations

import torch

from .config import ExpConfig
from .data import Item, Split
from .evaluate import Score, accuracy, batched, rng_for, score_batch
from .fv import FunctionVector, random_like
from .intervene import residual_norms, score_zero_shot
from .model import ModelInfo
from .prompts import build, sample_shots


@torch.no_grad()
def shuffled_label_floor(model, tokenizer, info: ModelInfo, split: Split,
                         items: list[Item], exp: ExpConfig) -> list[Score]:
    """n-shot accuracy with the demonstration labels permuted."""
    rng = rng_for("shuffled_floor", items[0].task if items else "", exp.seed)
    prompts = []
    for item in items:
        shots = sample_shots(split.train, exp.n_shots, rng, exclude=item)
        prompts.append(build(item, shots, exp, shuffle_labels=True, rng=rng))
    out: list[Score] = []
    for chunk in batched(prompts, int(info.batch.get("score", 8))):
        out += score_batch(model, tokenizer, chunk, model.device)
    return out


@torch.no_grad()
def run_controls(model, tokenizer, info: ModelInfo, split: Split, items: list[Item],
                 exp: ExpConfig, fv: FunctionVector,
                 unrelated: FunctionVector | None, layer: int,
                 rel_norm: float) -> dict:
    """All controls on one cell, each vector rescaled to rel_norm * ||h_layer||."""
    norms = residual_norms(model, tokenizer, info, items, exp)
    target = rel_norm * norms[layer]

    def at_scale(v: torch.Tensor) -> torch.Tensor:
        n = float(v.float().norm())
        return v * (target / n) if n else v

    fv_scores = score_zero_shot(model, tokenizer, info, items, exp,
                                at_scale(fv.vector), layer)
    rnd = at_scale(random_like(fv, seed=exp.seed))
    rnd_scores = score_zero_shot(model, tokenizer, info, items, exp, rnd, layer)
    unrel_scores = (
        score_zero_shot(model, tokenizer, info, items, exp,
                        at_scale(unrelated.vector), layer)
        if unrelated is not None else []
    )
    floor = shuffled_label_floor(model, tokenizer, info, split, items, exp)

    return {
        "task": fv.task,
        "condition": fv.condition,
        "edit_layer": layer,
        "n": len(items),
        "acc_fv": accuracy(fv_scores),
        "acc_random": accuracy(rnd_scores),
        "acc_unrelated": accuracy(unrel_scores) if unrel_scores else float("nan"),
        "acc_shuffled": accuracy(floor),
        "random_vector_norm": round(float(rnd.norm()), 3),
        "fv_norm": fv.norm,
        "fv_rel_norm": rel_norm,
        "fv_norm_scaled": round(target, 3),
        "resid_norm_at_layer": round(norms[layer], 1),
    }
