"""Function vector construction.

    v_t = sum over the top heads of  W_O^(L) [ e_H * ā_{L,H,last} ]

i.e. for each selected head, build a zero vector of width `o_proj.in_features`,
write that head's mean last-token activation into its own slot, push it through
that layer's output projection, and sum the results in residual-stream space.

The head *set* comes from the indirect-effect sweep and is shared across tasks
within a condition; the *values* are the per-task mean activations. So the
function vector is task-specific while the circuit carrying it is not -- which
is the paper's central claim and what makes cross-condition head overlap
meaningful.

The `attn_in_dim` width is the Gemma 3 fix: the reference sizes this buffer as
`resid_dim` and offsets by `resid_dim // n_heads`, which on Gemma 3 is both the
wrong width (2048 vs 2560) and the wrong stride (320 vs 256).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .model import ModelInfo, head_to_resid


@dataclass
class FunctionVector:
    """A function vector plus everything needed to reproduce it."""

    vector: torch.Tensor                      # (resid_dim,)
    task: str
    condition: str
    head_set_scope: str                       # which AIE sweep chose the heads
    heads: list[tuple[int, int, float]]
    model: str

    @property
    def norm(self) -> float:
        return float(self.vector.norm().item())

    def save(self, root: Path) -> Path:
        d = root / self.model / self.condition
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.vector.cpu(), d / f"{self.task}.pt")
        meta = {
            "task": self.task,
            "condition": self.condition,
            "model": self.model,
            "head_set_scope": self.head_set_scope,
            "heads": [[l, h, e] for l, h, e in self.heads],
            "norm": self.norm,
            "resid_dim": int(self.vector.shape[0]),
        }
        (d / f"{self.task}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return d / f"{self.task}.pt"

    @classmethod
    def load(cls, root: Path, model: str, condition: str, task: str) -> "FunctionVector":
        d = root / model / condition
        meta = json.loads((d / f"{task}.json").read_text(encoding="utf-8"))
        return cls(
            vector=torch.load(d / f"{task}.pt", map_location="cpu"),
            task=meta["task"], condition=meta["condition"],
            head_set_scope=meta["head_set_scope"],
            heads=[tuple(x) for x in meta["heads"]], model=meta["model"],
        )


def build_function_vector(model, info: ModelInfo, mean_acts: torch.Tensor,
                          heads: list[tuple[int, int, float]], task: str,
                          condition: str, head_set_scope: str) -> FunctionVector:
    """Sum the selected heads' mean outputs, each projected to residual space."""
    vec = torch.zeros(info.resid_dim, dtype=torch.float32)
    for layer, head, _ in heads:
        contrib = head_to_resid(model, info, layer, head, mean_acts[layer, head])
        vec += contrib.float().cpu()
    return FunctionVector(vector=vec, task=task, condition=condition,
                          head_set_scope=head_set_scope, heads=heads, model=info.name)


def cosine_grid(fvs: dict[tuple[str, str], FunctionVector]) -> tuple[list, "torch.Tensor"]:
    """Cosine similarity between every pair of function vectors.

    Keys are (task, condition). The question this answers is whether vectors
    cluster by *task* -- the same task in different languages landing together --
    or by *language*, which would mean the representation is about the input's
    surface form rather than the function being computed.
    """
    keys = sorted(fvs)
    mat = torch.stack([fvs[k].vector for k in keys])
    mat = mat / mat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return keys, mat @ mat.T


def random_like(fv: FunctionVector, seed: int = 0) -> torch.Tensor:
    """A random vector with the same norm -- the control for "any big vector helps".

    Without this, a function vector that improves zero-shot accuracy proves only
    that perturbing the residual stream at that layer does something.
    """
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(fv.vector.shape, generator=g)
    return v / v.norm() * fv.vector.norm()
