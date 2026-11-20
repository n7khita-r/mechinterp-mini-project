"""Config loading. Model and experiment configs are plain YAML; see configs/."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Bump when a change alters computed numbers, so cached results from the old
# code are not replayed. History:
#   1  initial
#   2  robust layer selection (smoothed curve, mid-depth tie-break, minimum dev
#      slice) -- a plain argmax resolved all-zero ties to layer 0
#   3  residual-relative function vector scaling. A raw vector (norm ~16) against
#      Gemma's residual stream (780 at layer 0, 100k at layer 33) is a ~0.1%
#      nudge and changed nothing; the vector is now scaled to a fraction of the
#      residual norm at the edit layer, with that fraction chosen on the dev slice
CODE_VERSION = 3

# Per-stage overrides, for a fix that changes one stage's numbers and no others.
# A bump to CODE_VERSION invalidates every cached stage, which means rerunning
# hours of GPU work to reproduce numbers that did not change. A stage listed here
# gets its own extra key in the hash, so only that stage recomputes. Stages that
# are absent hash exactly as they did before this mechanism existed.
#   08_controls 2  the controls applied the *raw* function vector while stage 05
#                  applied the residual-relative scaled one, so FV, random vector
#                  and unrelated FV were all in the invisible regime and came out
#                  identical to two decimals. All three are now scaled to the same
#                  fraction of the residual norm that stage 05 chose for the cell.
#   06_transfer 2  the transfer matrix applied the source vector raw, for the same
#                  reason, so every row collapsed onto the target condition's own
#                  zero-shot behaviour and three rows came out bit-identical
#   05_intervene 2  layer and scale are now selected jointly, not sequentially
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
    """What we need to know about a model before loading it.

    `expect` is asserted against the loaded model in model.load_model, so a
    transformers upgrade that reshapes Gemma 3 fails the job rather than silently
    indexing the wrong heads.
    """

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
    """Experiment hyperparameters. Defaults mirror Todd et al. unless noted."""

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

    Used to key the resume cache and stamped into every run manifest, so a
    result can always be traced back to the configuration that produced it.
    """
    payload = {
        "model": {k: v for k, v in model.raw.items() if k != "device_map"},
        "exp": exp.raw,
        # Code version participates in the key. Without it, changing an algorithm
        # without touching a config leaves the hash unchanged, so cached results
        # from the old code are silently replayed -- which happened once with the
        # layer-selection fix and cost a confusing debugging round. Bump
        # CODE_VERSION whenever a change alters numbers.
        "code": CODE_VERSION,
        "extra": extra,
    }
    # Added only for stages that carry an override, so every other stage's hash
    # is bit-identical to what it was before STAGE_CODE_VERSION was introduced.
    if stage and stage in STAGE_CODE_VERSION:
        payload["stage_code"] = {stage: STAGE_CODE_VERSION[stage]}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
