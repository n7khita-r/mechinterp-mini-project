"""Mean attention-head activations.

    ā_{l,h} = (1/N) sum over N clean n-shot prompts of head (l,h)'s last-token output
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

    Returns float32 CPU tensor of shape (n_layers, n_heads, head_dim).
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
