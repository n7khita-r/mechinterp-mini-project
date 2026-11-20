"""Tests for the hooking layer, against a tiny randomly-initialised stand-in.

These guard the four Gemma-3 traps described in CONTEXT.md. The stand-in is
deliberately shaped so that `n_heads * head_dim != hidden_size` (4 * 24 = 96 vs
64), which is exactly the case the reference implementation gets wrong by
computing `head_dim = hidden_size // n_heads`.
"""

from __future__ import annotations

import pytest
import torch

from fvcm import model as M
from fvcm.config import ModelConfig


@pytest.fixture(scope="module")
def loaded():
    mcfg = ModelConfig.load("configs/model/debug_tiny.yaml")
    model, tok, info = M.load_model(mcfg)
    enc = tok("Q:big\nA:small\n\nQ:hot\nA:", return_tensors="pt")
    return model, tok, info, enc


def test_head_dim_is_not_derived_from_hidden_size(loaded):
    """The Gemma-3 trap: head_dim must come from the config, not division."""
    _, _, info, _ = loaded
    assert info.head_dim == 24
    assert info.resid_dim == 64
    assert info.head_dim != info.resid_dim // info.n_heads, (
        "the fixture no longer exercises the head_dim mismatch")
    assert info.attn_in_dim == info.n_heads * info.head_dim == 96


def test_attn_in_dim_matches_the_o_proj_weight(loaded):
    """attn_in_dim must equal o_proj.in_features, not hidden_size."""
    model, _, info, _ = loaded
    for layer in (0, info.n_layers - 1):
        o_proj = dict(model.named_modules())[info.attn_hook_names[layer]]
        assert o_proj.in_features == info.attn_in_dim
        assert o_proj.out_features == info.resid_dim


def test_capture_shapes(loaded):
    model, _, info, enc = loaded
    store: dict[int, torch.Tensor] = {}
    with M.capture_head_outputs(model, info, store):
        model(**enc)
    assert set(store) == set(range(info.n_layers))
    for v in store.values():
        assert v.shape == (1, info.n_heads, info.head_dim)


def test_patching_one_head_changes_logits_and_leaves_others_alone(loaded):
    """Each batch element is patched at its own head, independently."""
    model, _, info, enc = loaded
    clean = M.last_logits(model, enc)

    store: dict[int, torch.Tensor] = {}
    with M.capture_head_outputs(model, info, store):
        model(**enc)
    mean = torch.stack([store[l][0] for l in range(info.n_layers)]) + 5.0

    layer = info.mid_layer
    # One batch element per head, plus a final element patched at no head --
    # a control that must come back bit-identical to the clean run.
    patches = [(layer, h) for h in range(info.n_heads)] + [None]
    batch = {k: v.repeat(len(patches), 1) for k, v in enc.items()}
    with M.patch_heads(model, info, patches, mean):
        out = M.last_logits(model, batch)

    for h in range(info.n_heads):
        assert (out[h] - clean[0]).abs().max() > 1e-4, f"head {h} had no effect"
    assert (out[-1] - clean[0]).abs().max() < 1e-4, (
        "the unpatched control element changed, so the patch is leaking across "
        "the batch")


def test_residual_add_is_last_token_only(loaded):
    model, _, info, enc = loaded
    clean = M.last_logits(model, enc)
    vec = torch.randn(info.resid_dim) * 3
    with M.add_to_residual(model, info, info.mid_layer, vec):
        added = M.last_logits(model, enc)
    assert (added - clean).abs().max() > 1e-4

    # Adding at a layer after the last one would be a no-op; adding zero must be.
    with M.add_to_residual(model, info, info.mid_layer, torch.zeros(info.resid_dim)):
        zero = M.last_logits(model, enc)
    assert (zero - clean).abs().max() < 1e-5


def test_logit_lens_reproduces_the_models_own_logits(loaded):
    """Catches a missing final norm or an untied unembedding.

    Gemma scales embeddings by sqrt(d_model) and uses (1 + w) RMSNorm, so
    unembedding a raw residual stream without the final norm gives noise at every
    layer. The last layer is the one case where the lens must match exactly.
    """
    model, _, info, enc = loaded
    clean = M.last_logits(model, enc)
    resid: dict[int, torch.Tensor] = {}
    with M.capture_residual(model, info, resid):
        model(**enc)
    norm = dict(model.named_modules())[info.norm_name]
    head = model.get_output_embeddings()
    lens = head(norm(resid[info.n_layers - 1])).float()
    assert (lens - clean).abs().max() < 1e-4


def test_head_to_resid_projects_through_the_right_slice(loaded):
    """The function vector sums o_proj(e_h * a_h); the slice must match head h."""
    model, _, info, _ = loaded
    layer = 1
    o_proj = dict(model.named_modules())[info.attn_hook_names[layer]]
    value = torch.ones(info.head_dim)

    for head in range(info.n_heads):
        got = M.head_to_resid(model, info, layer, head, value)
        lo = head * info.head_dim
        expect = o_proj.weight[:, lo:lo + info.head_dim].float() @ value.float()
        assert torch.allclose(got.float().flatten(), expect, atol=1e-4), (
            f"head {head} projected through the wrong columns of W_O")


def test_hooks_are_removed_on_exit(loaded):
    """A leaked hook would silently contaminate every later forward pass."""
    model, _, info, enc = loaded
    before = M.last_logits(model, enc).clone()

    store: dict[int, torch.Tensor] = {}
    with M.capture_head_outputs(model, info, store):
        model(**enc)
    with M.add_to_residual(model, info, info.mid_layer, torch.randn(info.resid_dim) * 5):
        model(**enc)

    after = M.last_logits(model, enc)
    assert torch.allclose(before, after, atol=1e-6), "a hook outlived its context"
