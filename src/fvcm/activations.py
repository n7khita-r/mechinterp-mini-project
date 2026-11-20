"""Mean attention-head activations.

For each (task, condition) we average every head's output at the last token over
`n_trials` clean n-shot prompts. This is the `ā_{l,h}` of the method: the
"this is what this head does when the task is clear" reference value, used both
as the patch in causal mediation and as the raw material of the function vector.

Simplification against the reference: it stores means for *every* token position
of a canonical template layout, which requires mapping multi-token words onto
shared slots (`compute_duplicated_labels` / `idx_map`). Every use in this
project -- the mediation patch and the function vector -- reads only the last
token, so we store only that. This drops a few hundred lines of index mapping
that could go wrong without being noticed, and changes no result.
"""

from __future__ import annotations

import torch

from .config import ExpConfig
from .data import Split
from .evaluate import rng_for
from .model import ModelInfo, capture_head_outputs
from .prompts import build, sample_shots


@torch.no_grad()
def mean_head_activations(model, tokenizer, info: ModelInfo, split: Split,
                          exp: ExpConfig, filter_ids: set[str] | None = None,
                          task: str = "", condition: str = "",
                          shuffle_labels: bool = False) -> torch.Tensor:
    """Average head outputs at the last token over clean n-shot prompts.

    Returns a float32 CPU tensor of shape (n_layers, n_heads, head_dim).

    `filter_ids` restricts the query drawn for each prompt to competence-filtered
    items, matching the reference, which passes `filter_set_validation` here.
    """
    n_trials = int(exp.mean_activations["n_trials"])
    rng = rng_for("mean_acts", task, condition, exp.seed)

    pool = split.valid
    if filter_ids:
        kept = [i for i in pool if i.id in filter_ids]
        if kept:
            pool = kept
    if not pool:
        raise ValueError(f"no query items available for {task}/{condition}")

    total = torch.zeros(info.n_layers, info.n_heads, info.head_dim, dtype=torch.float32)
    for _ in range(n_trials):
        query = pool[rng.randrange(len(pool))]
        shots = sample_shots(split.train, exp.n_shots, rng, exclude=query)
        prompt = build(query, shots, exp, shuffle_labels=shuffle_labels, rng=rng)

        enc = tokenizer(prompt.text, return_tensors="pt").to(model.device)
        store: dict[int, torch.Tensor] = {}
        with capture_head_outputs(model, info, store):
            model(**enc)
        total += torch.stack([store[l][0] for l in range(info.n_layers)])

    return total / n_trials


def save(acts: torch.Tensor, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(acts, path)


def load(path) -> torch.Tensor:
    return torch.load(path, map_location="cpu")
