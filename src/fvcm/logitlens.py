"""Logit lens: where across depth does the answer become available?

    rank_l(gold) from  unembed(final_norm(h_l))

Gemma ties the unembedding to the input embeddings and needs the final norm
applied first, or every layer reads as noise. Early layers are not interpreted.
"""

from __future__ import annotations

import torch

from .config import ExpConfig
from .data import Item
from .evaluate import rng_for
from .model import ModelInfo, capture_residual
from .prompts import answer_token_ids, build, sample_shots


def _unembed(model, info: ModelInfo):
    norm = dict(model.named_modules())[info.norm_name]
    head = model.get_output_embeddings()
    if head is None:
        raise RuntimeError("model exposes no output embedding to unembed with")
    return norm, head


@torch.no_grad()
def lens_curve(model, tokenizer, info: ModelInfo, split, items: list[Item],
               exp: ExpConfig, task: str, condition: str) -> dict:
    """Per-layer rank and probability of the gold first token, averaged over items."""
    norm, head = _unembed(model, info)
    rng = rng_for("lens", task, condition, exp.seed)

    ranks = torch.zeros(info.n_layers, dtype=torch.float64)
    probs = torch.zeros(info.n_layers, dtype=torch.float64)
    n = 0

    for item in items:
        shots = sample_shots(split.train, exp.n_shots, rng, exclude=item)
        prompt = build(item, shots, exp)
        gold = answer_token_ids(prompt.text, item.out, tokenizer)[0]

        enc = tokenizer(prompt.text, return_tensors="pt").to(model.device)
        store: dict[int, torch.Tensor] = {}
        with capture_residual(model, info, store):
            model(**enc)

        for layer in range(info.n_layers):
            h = store[layer]
            logits = head(norm(h)).float()[0]
            lp = torch.log_softmax(logits, dim=-1)
            ranks[layer] += int((lp > lp[gold]).sum().item())
            probs[layer] += float(lp[gold].exp().item())
        n += 1

    if n == 0:
        return {"task": task, "condition": condition, "n": 0,
                "layers": list(range(info.n_layers)), "mean_rank": [], "mean_prob": []}

    return {
        "task": task,
        "condition": condition,
        "n": n,
        "layers": list(range(info.n_layers)),
        "mean_rank": (ranks / n).tolist(),
        "mean_prob": (probs / n).tolist(),
        # Layer where the gold token first enters the top 10; None if never.
        "emergence_layer": _emergence(ranks / n),
        "interpret_from_layer": info.n_layers // 4,
    }


def _emergence(mean_ranks: torch.Tensor, threshold: float = 10.0) -> int | None:
    for layer, r in enumerate(mean_ranks.tolist()):
        if r < threshold:
            return layer
    return None


@torch.no_grad()
def decode_vector(model, info: ModelInfo, vec: torch.Tensor, tokenizer,
                  k: int = 10) -> list[tuple[str, float]]:
    """Top-k tokens a function vector promotes, read through the same lens."""
    norm, head = _unembed(model, info)
    h = vec.to(next(model.parameters()).dtype).to(model.device).reshape(1, 1, -1)
    logits = head(norm(h)).float()[0, 0]
    probs = torch.softmax(logits, dim=-1)
    vals, idx = torch.topk(probs, k=k)
    return [(tokenizer.decode([int(i)]), round(float(v), 5)) for i, v in zip(idx, vals)]
