"""Controls: what else could explain a function vector that works?

Four of them, each closing a specific alternative explanation.

* **random vector, matched norm** -- rules out "adding any large vector to the
  residual stream at that depth jolts the model into the right answer".
* **unrelated-task function vector** -- a vector built the same way from a task
  nobody asked about (vegetable-vs-object). Rules out "any well-formed function
  vector helps", leaving only task-specific content as the explanation.
* **shuffled-label floor** -- n-shot accuracy with permuted demonstration
  labels. The level the model reaches when the prompt carries the task's *form*
  but not its content; a function vector should beat it.
* **zero-shot floor** -- no vector at all, measured in `intervene.run_cell`.

All three vectors are scaled to the *same* fraction of the residual norm at the
edit layer -- the fraction stage 05 chose for that cell. This matters more than
it sounds: applied raw, a function vector is ~0.1% of Gemma's residual norm and
changes nothing, so FV, random vector and unrelated FV all score identically and
the controls appear to show the effect is not task-specific. They are really all
showing the same null. Norm-matching the controls to the scaled vector is what
makes the comparison a comparison.
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
    """All controls on one cell, at the layer and scale chosen for that cell.

    `rel_norm` is the alpha stage 05 selected on its dev slice: every vector here
    is rescaled to `alpha * ||residual|| at `layer`, so the three interventions
    differ only in direction.
    """
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
