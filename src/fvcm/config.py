"""Config loading. Model and experiment configs are plain YAML; see configs/."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Participates in the config hash: bump when a change alters computed numbers,
# so cached results from the old code are not replayed.
CODE_VERSION = 3

# Per-stage overrides, so a fix confined to one stage invalidates only that
# stage's cache instead of forcing a full recompute.
STAGE_CODE_VERSION = {"08_controls": 2, "06_transfer": 2, "05_intervene": 2}


def _load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class ModelConfig:
    """Model facts needed before loading; `expect` is asserted in model.describe."""

    name: str
    hf_id: str | None
    dtype: str
    device_map: str
    prepend_bos: bool
    expect: dict
    batch: dict
    random_init: bool = False
    tokenizer_id: str | None = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        d = _load_yaml(path)
        return cls(
            name=d["name"],
            hf_id=d.get("hf_id"),
            dtype=d.get("dtype", "bfloat16"),
            device_map=d.get("device_map", "cuda"),
            prepend_bos=bool(d.get("prepend_bos", True)),
            expect=d.get("expect", {}),
            batch=d.get("batch", {}),
            random_init=bool(d.get("random_init", False)),
            tokenizer_id=d.get("tokenizer_id"),
            raw=d,
        )


@dataclass
class ExpConfig:
    """Experiment hyperparameters."""

    name: str
    seed: int
    prompt: dict
    data: dict
    mean_activations: dict
    mediation: dict
    fv: dict
    intervene: dict
    eval: dict
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "ExpConfig":
        d = _load_yaml(path)
        return cls(
            name=d["name"],
            seed=int(d.get("seed", 42)),
            prompt=d["prompt"],
            data=d["data"],
            mean_activations=d["mean_activations"],
            mediation=d["mediation"],
            fv=d["fv"],
            intervene=d["intervene"],
            eval=d["eval"],
            raw=d,
        )

    @property
    def n_shots(self) -> int:
        return int(self.prompt["n_shots"])

    @property
    def prefixes(self) -> dict:
        return dict(self.prompt["prefixes"])

    @property
    def separators(self) -> dict:
        return dict(self.prompt["separators"])


def config_hash(model: ModelConfig, exp: ExpConfig, stage: str | None = None,
                **extra: Any) -> str:
    """Stable short hash over everything that could change a number.

    Keys the resume cache and is stamped into every run manifest.
    """
    payload = {
        "model": {k: v for k, v in model.raw.items() if k != "device_map"},
        "exp": exp.raw,
        "code": CODE_VERSION,
        "extra": extra,
    }
    if stage and stage in STAGE_CODE_VERSION:
        payload["stage_code"] = {stage: STAGE_CODE_VERSION[stage]}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
