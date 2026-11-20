"""Scoring, the competence filter, and bootstrap confidence intervals.

Headline metric is the rank of the gold answer's first token. A full-answer mean
log-probability is logged alongside it, since first-token rank is weak where the
gold first token is a prefix shared by many words.
"""

from __future__ import annotations

import contextlib
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch

from . import model as M
from .config import ExpConfig
from .data import Item
from .prompts import Prompt, answer_token_ids


@dataclass
class Score:
    """Per-item result."""

    item_id: str
    task: str
    condition: str
    tier: str
    uid: str
    rank: int                  # rank of the gold first token, 0 = top-1 correct
    prob: float                # probability of the gold first token
    full_logprob: float        # mean log-prob of the full gold answer
    n_answer_tokens: int
    n_prompt_tokens: int

    @property
    def correct(self) -> bool:
        return self.rank == 0

    def as_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def score_batch(model, tokenizer, prompts: list[Prompt], device,
                ctx=None) -> list[Score]:
    """Score a batch of prompts in one forward pass, left-padded to position -1.

    `ctx` is a zero-argument callable returning a context manager, so an
    intervention stays open for both the first-token and full-answer passes.
    """
    if not prompts:
        return []
    ctx = ctx or (lambda: contextlib.nullcontext())

    texts = [p.text for p in prompts]
    answers = [answer_token_ids(p.text, p.query.out, tokenizer) for p in prompts]

    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    padding_side="left").to(device)
    with ctx():
        last = M.last_logits(model, enc)                    # (B, V)
    last = torch.log_softmax(last, dim=-1)

    out = []
    for b, (p, ans) in enumerate(zip(prompts, answers)):
        first = ans[0]
        lp = last[b]
        rank = int((lp > lp[first]).sum().item())
        prob = float(lp[first].exp().item())

        full = _full_answer_logprob(model, tokenizer, p.text, ans, device, ctx)

        out.append(Score(
            item_id=p.query.id, task=p.query.task, condition=p.query.condition,
            tier=p.query.tier, uid=p.query.uid, rank=rank, prob=prob,
            full_logprob=full, n_answer_tokens=len(ans),
            n_prompt_tokens=int(enc["attention_mask"][b].sum().item()),
        ))
    return out


@torch.no_grad()
def _full_answer_logprob(model, tokenizer, prompt_text: str, answer_ids: list[int],
                         device, ctx=None) -> float:
    """Mean log-probability of the gold answer tokens, teacher-forced."""
    ctx = ctx or (lambda: contextlib.nullcontext())
    if len(answer_ids) == 1:
        ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        with ctx():
            lp = torch.log_softmax(model(ids).logits[0, -1].float(), dim=-1)
        return float(lp[answer_ids[0]].item())

    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    ans = torch.tensor([answer_ids], device=device)
    ids = torch.cat([prompt_ids, ans], dim=1)
    with ctx():
        logits = model(ids).logits[0].float()
    start = prompt_ids.shape[1] - 1
    lps = torch.log_softmax(logits[start:start + len(answer_ids)], dim=-1)
    tot = sum(float(lps[i, t].item()) for i, t in enumerate(answer_ids))
    return tot / len(answer_ids)


def batched(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# --------------------------------------------------------------------------- #
# Competence filter
# --------------------------------------------------------------------------- #

def competence_filter(scores: list[Score]) -> set[str]:
    """Item ids the model already answers correctly from n-shot demonstrations.

    Everything downstream is restricted to these, so a function vector is never
    blamed for a task the model cannot do even when shown examples.
    """
    return {s.item_id for s in scores if s.rank == 0}


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def accuracy(scores: list[Score], k: int = 1) -> float:
    if not scores:
        return float("nan")
    return float(np.mean([s.rank < k for s in scores]))


def bootstrap_ci(scores: list[Score], k: int = 1, n_boot: int = 2000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI over items."""
    if not scores:
        return (float("nan"), float("nan"))
    hits = np.array([s.rank < k for s in scores], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(hits), size=(n_boot, len(hits)))
    means = hits[draws].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_delta(a: list[Score], b: list[Score], k: int = 1, n_boot: int = 2000,
                 seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired comparison of two conditions on the same items; bootstrap resamples items."""
    ba = {s.item_id: s for s in a}
    bb = {s.item_id: s for s in b}
    shared = sorted(set(ba) & set(bb))
    if not shared:
        return {"n": 0, "delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "acc_a": float("nan"), "acc_b": float("nan")}

    da = np.array([ba[i].rank < k for i in shared], dtype=float)
    db = np.array([bb[i].rank < k for i in shared], dtype=float)
    diff = da - db
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[draws].mean(axis=1)
    return {
        "n": len(shared),
        "acc_a": float(da.mean()),
        "acc_b": float(db.mean()),
        "delta": float(diff.mean()),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
    }


def summarise(scores: list[Score], exp: ExpConfig) -> dict:
    """Per-cell summary: accuracies with CIs, plus the token-length diagnostic."""
    n_boot = int(exp.eval.get("n_bootstrap", 2000))
    out = {"n": len(scores)}
    for k in exp.eval.get("topk", [1, 10]):
        lo, hi = bootstrap_ci(scores, k=k, n_boot=n_boot, seed=exp.seed)
        out[f"top{k}"] = accuracy(scores, k=k)
        out[f"top{k}_lo"], out[f"top{k}_hi"] = lo, hi
    if scores:
        out["mean_full_logprob"] = float(np.mean([s.full_logprob for s in scores]))
        out["mean_prompt_tokens"] = float(np.mean([s.n_prompt_tokens for s in scores]))
        out["mean_answer_tokens"] = float(np.mean([s.n_answer_tokens for s in scores]))
    return out


def rng_for(*parts) -> random.Random:
    """Deterministic RNG keyed by whatever identifies this unit of work."""
    return random.Random("|".join(str(p) for p in parts))
