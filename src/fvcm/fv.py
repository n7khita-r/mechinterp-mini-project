"""Function vector construction.

    v_t = sum over the top heads of  W_O^(L) [ e_H * ā_{L,H,last} ]
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
    head_set_scope: str
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
    """Cosine similarity between every pair of function vectors, keyed by (task, condition)."""
    keys = sorted(fvs)
    mat = torch.stack([fvs[k].vector for k in keys])
    mat = mat / mat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return keys, mat @ mat.T


def random_like(fv: FunctionVector, seed: int = 0) -> torch.Tensor:
    """A random vector matched to `fv`'s norm. Control for "any large nudge works"."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(fv.vector.shape, generator=g)
    return v / v.norm() * fv.vector.norm()
