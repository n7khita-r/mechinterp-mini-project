"""Causal mediation: average indirect effect (AIE) of every attention head.

    AIE(L, H) = E_prompts [ p_patched(gold) - p_corrupted(gold) ]
"""

from __future__ import annotations

import torch
from .progress import track

from .config import ExpConfig
from .data import Item, Split
from .evaluate import rng_for
from .model import ModelInfo, patch_heads
from . import model as M
from .prompts import answer_token_ids, build, sample_shots


@torch.no_grad()
def indirect_effects(model, tokenizer, info: ModelInfo,
                     cells: dict[str, tuple[Split, torch.Tensor, set[str]]],
                     exp: ExpConfig, condition: str,
                     progress: bool = True) -> torch.Tensor:
    """AIE for every head, over trials spanning all tasks in one condition.

    `cells` maps task -> (split, mean_activations, kept_ids). Returns
    (n_trials, n_layers, n_heads) float32; the caller averages over trials.
    """
    n_trials = int(exp.mediation["n_trials"])
    tasks = sorted(cells)
    if not tasks:
        raise ValueError(f"no tasks supplied for condition {condition}")

    rng = rng_for("aie", condition, exp.seed)
    out = torch.zeros(n_trials, info.n_layers, info.n_heads, dtype=torch.float32)

    for t in track(range(n_trials), f"AIE[{condition}]", stage="03_mediation",
                   enabled=progress, total=n_trials):
        task = tasks[rng.randrange(len(tasks))]
        split, mean_acts, keep = cells[task]

        pool = [i for i in split.valid if not keep or i.id in keep] or split.valid
        query: Item = pool[rng.randrange(len(pool))]
        shots = sample_shots(split.train, exp.n_shots, rng, exclude=query)

        # Corrupted run: labels shuffled, so the demonstrations carry no mapping.
        prompt = build(query, shots, exp, shuffle_labels=True, rng=rng)
        gold = answer_token_ids(prompt.text, query.out, tokenizer)[0]

        enc = tokenizer(prompt.text, return_tensors="pt").to(model.device)
        base = torch.softmax(M.last_logits(model, enc)[0], dim=-1)[gold].item()

        mean_acts = mean_acts.to(model.device)
        for layer in range(info.n_layers):
            # One batch element per head of this layer.
            batch = {k: v.repeat(info.n_heads, 1) for k, v in enc.items()}
            patches = [(layer, h) for h in range(info.n_heads)]
            with patch_heads(model, info, patches, mean_acts):
                logits = M.last_logits(model, batch)
            probs = torch.softmax(logits, dim=-1)[:, gold]
            out[t, layer] = (probs - base).cpu()

    return out


def top_heads(aie: torch.Tensor, n_top: int) -> list[tuple[int, int, float]]:
    """The n_top heads by mean indirect effect, as (layer, head, effect)."""
    mean = aie.mean(dim=0) if aie.dim() == 3 else aie
    flat = mean.reshape(-1)
    vals, idx = torch.topk(flat, k=min(n_top, flat.numel()), largest=True)
    n_heads = mean.shape[1]
    return [(int(i) // n_heads, int(i) % n_heads, round(float(v), 6))
            for i, v in zip(idx, vals)]


def head_overlap(a: list[tuple[int, int, float]], b: list[tuple[int, int, float]]) -> dict:
    """Jaccard overlap between two head sets, ignoring effect sizes."""
    sa = {(l, h) for l, h, _ in a}
    sb = {(l, h) for l, h, _ in b}
    inter = len(sa & sb)
    union = len(sa | sb)
    return {
        "intersection": inter,
        "union": union,
        "jaccard": inter / union if union else float("nan"),
        "shared": sorted(sa & sb),
    }
