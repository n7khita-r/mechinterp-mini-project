"""Causal mediation: the average indirect effect (AIE) of every attention head.

For a head (L, H): run a **shuffled-label** n-shot prompt, where the task cannot
be inferred from the demonstrations, replace that one head's output at the last
token with its clean mean, and measure how much the gold answer's probability
rises:

    AIE(L, H) = E_prompts [ p_patched(gold) - p_corrupted(gold) ]

Two departures from the reference, both deliberate:

1. **One sweep per condition, not per (task, condition).** The paper averages AIE
   over tasks anyway to obtain a single universal head set, so each trial here
   samples a random task from the condition. That makes the per-condition head
   sets a first-class result -- "do Telugu and English recruit the same heads?"
   is exactly the overlap between these sets -- and cuts the sweep cost by the
   number of tasks (~8x).

2. **Batched over heads.** Every patch for a given layer applies to the same
   prompt, so the prompt is replicated across the batch and each element carries
   a different head. Forwards per trial drop from n_layers x n_heads to
   n_layers: 1,984 -> 62 on the 27B model. The reference has a `batched_input`
   path that does this, but its own sweep does not use it; that single choice is
   the difference between a ~4-hour and a ~50-hour run.
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
    """AIE for every head, averaged over trials that span all tasks.

    `cells` maps task -> (split, mean_activations, competence_filtered_ids) for
    this condition. Returns a (n_trials, n_layers, n_heads) float32 tensor; the
    caller averages over trials, keeping the per-trial values so the variability
    of a head's effect can be inspected.
    """
    n_trials = int(exp.mediation["n_trials"])
    tasks = sorted(cells)
    if not tasks:
        raise ValueError(f"no tasks supplied for condition {condition}")

    rng = rng_for("aie", condition, exp.seed)
    out = torch.zeros(n_trials, info.n_layers, info.n_heads, dtype=torch.float32)

    # The dominant cost of the whole project: n_trials * n_layers batched
    # forwards per condition. Report on a wall-clock interval so a stalled sweep
    # is distinguishable from a merely slow one.
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
    """Jaccard overlap between two head sets, ignoring effect sizes.

    This is the direct readout for "are the same heads carrying the task in
    both conditions?", and the reason AIE is computed per condition.
    """
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
