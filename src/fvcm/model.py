"""Model loading, module discovery, and the three hooks the method needs:
read a head's output, overwrite one head's output, add a vector to the residual.

Module paths are discovered, not hardcoded: Gemma 3 4B/27B move the text stack
under `model.language_model`.
"""

from __future__ import annotations

import re
import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from .config import ModelConfig

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class ModelInfo:
    """Resolved architecture facts and module paths for one loaded model."""

    name: str
    name_or_path: str
    n_layers: int
    n_heads: int
    head_dim: int
    resid_dim: int
    attn_in_dim: int             # n_heads * head_dim; NOT resid_dim on Gemma 3
    layer_prefix: str
    attn_hook_names: list[str]
    layer_hook_names: list[str]
    norm_name: str
    prepend_bos: bool
    batch: dict = field(default_factory=dict)

    @property
    def mid_layer(self) -> int:
        """Roughly one third of the depth, where the paper adds function vectors."""
        return self.n_layers // 3


def _find_layer_prefix(model: torch.nn.Module) -> str:
    """Longest run of `<prefix>.<int>` modules that look like decoder layers."""
    runs: dict[str, set[int]] = {}
    for name, mod in model.named_modules():
        m = re.match(r"^(.*)\.(\d+)$", name)
        if m and mod.__class__.__name__.endswith(("DecoderLayer", "Block", "Layer")):
            runs.setdefault(m.group(1), set()).add(int(m.group(2)))
    if not runs:
        raise RuntimeError("could not locate the decoder layer stack")
    prefix = max(runs, key=lambda k: len(runs[k]))
    idx = runs[prefix]
    if idx != set(range(len(idx))):
        raise RuntimeError(f"decoder layer indices under {prefix!r} are not contiguous")
    return prefix


def _find_norm(model: torch.nn.Module, layer_prefix: str) -> str:
    """The final norm applied before the unembedding; skipping it makes the logit lens noise."""
    stack_root = layer_prefix.rsplit(".", 1)[0]
    names = dict(model.named_modules())
    for cand in (f"{stack_root}.norm", f"{stack_root}.final_layernorm", f"{stack_root}.ln_f"):
        if cand in names:
            return cand
    raise RuntimeError(f"could not locate the final norm under {stack_root!r}")


def describe(model: torch.nn.Module, mcfg: ModelConfig) -> ModelInfo:
    """Resolve paths and shapes, then assert them against the config's `expect`."""
    layer_prefix = _find_layer_prefix(model)
    mods = dict(model.named_modules())
    n_layers = sum(1 for n in mods if re.fullmatch(rf"{re.escape(layer_prefix)}\.\d+", n))

    attn_hooks = [f"{layer_prefix}.{i}.self_attn.o_proj" for i in range(n_layers)]
    missing = [n for n in attn_hooks if n not in mods]
    if missing:
        raise RuntimeError(f"missing attention output projections, e.g. {missing[0]!r}")

    tcfg = getattr(model.config, "text_config", model.config)
    n_heads = int(tcfg.num_attention_heads)
    resid_dim = int(tcfg.hidden_size)
    attn_in_dim = int(mods[attn_hooks[0]].in_features)
    # Never resid_dim // n_heads: on Gemma 3 those differ (4B: 8*256=2048 vs 2560).
    head_dim = int(getattr(tcfg, "head_dim", attn_in_dim // n_heads))
    if n_heads * head_dim != attn_in_dim:
        raise RuntimeError(
            f"n_heads*head_dim ({n_heads}*{head_dim}={n_heads * head_dim}) != "
            f"o_proj.in_features ({attn_in_dim})"
        )

    info = ModelInfo(
        name=mcfg.name,
        name_or_path=str(mcfg.hf_id or mcfg.name),
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        resid_dim=resid_dim,
        attn_in_dim=attn_in_dim,
        layer_prefix=layer_prefix,
        attn_hook_names=attn_hooks,
        layer_hook_names=[f"{layer_prefix}.{i}" for i in range(n_layers)],
        norm_name=_find_norm(model, layer_prefix),
        prepend_bos=mcfg.prepend_bos,
        batch=dict(mcfg.batch),
    )

    for key, want in (mcfg.expect or {}).items():
        got = {"n_layers": n_layers, "hidden_size": resid_dim, "n_heads": n_heads,
               "head_dim": head_dim,
               "n_kv_heads": int(getattr(tcfg, "num_key_value_heads", -1))}.get(key)
        if got is not None and int(want) != got:
            raise RuntimeError(
                f"{mcfg.name}: expected {key}={want} but the loaded model has {got}. "
                "Update configs/model/*.yaml deliberately -- do not just widen the check."
            )
    return info


def load_model(mcfg: ModelConfig):
    """Load (or randomly build) a model plus tokenizer, and describe it."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    dtype = DTYPES[mcfg.dtype]

    if mcfg.random_init:
        # Tiny CPU stand-in; shapes mirror Gemma 3, head_dim != hidden_size // n_heads.
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig

        e = mcfg.expect
        tok = AutoTokenizer.from_pretrained(mcfg.tokenizer_id or "gpt2")
        cfg = Gemma3TextConfig(
            vocab_size=len(tok),
            hidden_size=e["hidden_size"],
            intermediate_size=2 * e["hidden_size"],
            num_hidden_layers=e["n_layers"],
            num_attention_heads=e["n_heads"],
            num_key_value_heads=e["n_kv_heads"],
            head_dim=e["head_dim"],
            max_position_embeddings=1024,
            sliding_window=64,
        )
        torch.manual_seed(0)
        model = Gemma3ForCausalLM(cfg).to(dtype).eval()
    else:
        tok = AutoTokenizer.from_pretrained(mcfg.hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            mcfg.hf_id, dtype=dtype, device_map=mcfg.device_map,
        ).eval()
        if mcfg.raw.get("drop_vision_tower", True):
            # Best-effort: on some transformers versions these are read-only.
            dropped = []
            for holder in (getattr(model, "model", None), model):
                if holder is None:
                    continue
                for attr in ("vision_tower", "multi_modal_projector"):
                    if getattr(holder, attr, None) is None:
                        continue
                    try:
                        setattr(holder, attr, None)
                        dropped.append(attr)
                    except (AttributeError, TypeError):
                        pass
            if dropped:
                print(f"[model] dropped unused {', '.join(sorted(set(dropped)))}")

    # Left padding keeps position -1 the last real token of every sequence.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    return model, tok, describe(model, mcfg)


# --------------------------------------------------------------------------- #
# Forward helpers
# --------------------------------------------------------------------------- #

_LOGIT_KWARG: str | None | bool = False   # False = not yet probed


def _logit_kwarg(model) -> str | None:
    """Name of the 'only compute the last N positions' kwarg, probed once and cached.

    Renamed `num_logits_to_keep` -> `logits_to_keep` in transformers 4.50.
    """
    global _LOGIT_KWARG
    if _LOGIT_KWARG is not False:
        return _LOGIT_KWARG
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        params = {}
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            _LOGIT_KWARG = name
            break
    else:
        _LOGIT_KWARG = None
    return _LOGIT_KWARG


def last_logits(model, enc: dict) -> torch.Tensor:
    """Logits at the final position only, as float32, shape (B, vocab)."""
    kw = _logit_kwarg(model)
    out = model(**enc, **({kw: 1} if kw else {}))
    return out.logits[:, -1, :].float()


# --------------------------------------------------------------------------- #
# Hooks
# --------------------------------------------------------------------------- #

def _module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    return dict(model.named_modules())[name]


@contextmanager
def capture_head_outputs(model, info: ModelInfo, store: dict):
    """Capture every head's output (its slice of o_proj's input) at the last token.

    After the forward pass, `store[layer]` has shape (batch, n_heads, head_dim).
    """
    handles = []

    def make(layer: int):
        def pre_hook(_mod, args):
            x = args[0]                                   # (batch, tokens, attn_in_dim)
            store[layer] = (
                x[:, -1, :]
                .reshape(x.shape[0], info.n_heads, info.head_dim)
                .detach()
                .float()
                .cpu()
            )
            return None
        return pre_hook

    try:
        for layer, name in enumerate(info.attn_hook_names):
            handles.append(_module(model, name).register_forward_pre_hook(make(layer)))
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def patch_heads(model, info: ModelInfo, patches: list[tuple[int, int] | None],
                mean_acts: torch.Tensor):
    """Replace one head's last-token output per batch element.

    `patches[b]` is `(layer, head)` for element `b`, or None to leave it clean.
    Batching one head per element turns the AIE sweep from n_layers * n_heads
    forwards per trial into n_layers.
    """
    handles = []
    by_layer: dict[int, list[tuple[int, int]]] = {}
    for b, p in enumerate(patches):
        if p is not None:
            by_layer.setdefault(p[0], []).append((b, p[1]))

    def make(layer: int, targets: list[tuple[int, int]]):
        def pre_hook(_mod, args):
            x = args[0].clone()
            heads = x.reshape(*x.shape[:-1], info.n_heads, info.head_dim)
            for b, h in targets:
                heads[b, -1, h] = mean_acts[layer, h].to(heads.dtype).to(heads.device)
            return (heads.reshape(x.shape),) + tuple(args[1:])
        return pre_hook

    try:
        for layer, targets in by_layer.items():
            handles.append(
                _module(model, info.attn_hook_names[layer])
                .register_forward_pre_hook(make(layer, targets))
            )
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def add_to_residual(model, info: ModelInfo, layer: int, vec: torch.Tensor, idx: int = -1):
    """Add `vec` to the residual stream at layer `layer`'s output, at token `idx`."""
    def hook(_mod, _args, output):
        if isinstance(output, tuple):
            hidden = output[0]
            hidden[:, idx] = hidden[:, idx] + vec.to(hidden.dtype).to(hidden.device)
            return (hidden,) + tuple(output[1:])
        output[:, idx] = output[:, idx] + vec.to(output.dtype).to(output.device)
        return output

    handle = _module(model, info.layer_hook_names[layer]).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def capture_residual(model, info: ModelInfo, store: dict):
    """Capture every decoder layer's output at the last token, for the logit lens."""
    handles = []

    def make(layer: int):
        def hook(_mod, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            store[layer] = hidden[:, -1, :].detach()
            return output
        return hook

    try:
        for layer, name in enumerate(info.layer_hook_names):
            handles.append(_module(model, name).register_forward_hook(make(layer)))
        yield store
    finally:
        for h in handles:
            h.remove()


def head_to_resid(model, info: ModelInfo, layer: int, head: int,
                  value: torch.Tensor) -> torch.Tensor:
    """Project one head's output into the residual stream: W_O^(h) @ value.

    The buffer is sized from `attn_in_dim`, not `resid_dim` -- they differ on Gemma 3.
    """
    proj = _module(model, info.attn_hook_names[layer])
    w = proj.weight
    x = torch.zeros(1, info.attn_in_dim, dtype=w.dtype, device=w.device)
    lo = head * info.head_dim
    x[0, lo:lo + info.head_dim] = value.to(w.dtype).to(w.device)
    return proj(x).squeeze(0)
