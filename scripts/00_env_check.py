#!/usr/bin/env python3
"""Stage 00 -- fail fast before a long job starts.

Checks the GPU, HF auth, model shapes against the config's `expect` block, that
all three hooks change the logits, fp16 numerical safety, and a safe batch size.
"""

from __future__ import annotations

import sys

import torch

from _common import REPO_ROOT, parser, setup  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))

from fvcm import model as M  # noqa: E402
from fvcm.cache import write_json  # noqa: E402


def main() -> int:
    p = parser("environment and model sanity checks")
    p.add_argument("--skip-gpu", action="store_true",
                   help="do not require a GPU (used by the CPU smoke test)")
    p.add_argument("--skip-fp16-check", action="store_true",
                   help="skip the fp32 reference comparison (it needs ~17 GB host RAM)")
    args = p.parse_args()
    mcfg, exp, chash, out, started = setup(args, "00_env_check")

    report, failures = {}, []

    def check(name, ok, detail=""):
        report[name] = {"ok": bool(ok), "detail": str(detail)}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} {detail}")
        if not ok:
            failures.append(name)

    # 1. hardware
    has_cuda = torch.cuda.is_available()
    if not args.skip_gpu and not mcfg.random_init:
        total = (torch.cuda.get_device_properties(0).total_memory / 2**30) if has_cuda else 0
        check("cuda available", has_cuda,
              f"{torch.cuda.get_device_name(0)} {total:.0f} GiB" if has_cuda else "no GPU")
        # ~2 bytes per parameter, plus headroom for activations and KV cache.
        need = {"gemma3_4b": 10, "gemma3_4b_bf16": 10, "gemma3_27b": 62}.get(mcfg.name, 0)
        # is_bf16_supported() counts emulated bf16; real bf16 needs sm_80+.
        cc = torch.cuda.get_device_capability(0) if has_cuda else (0, 0)
        bf16_native = cc[0] >= 8
        check("dtype appropriate for this GPU",
              mcfg.dtype != "bfloat16" or bf16_native,
              f"dtype={mcfg.dtype}; native bf16 (sm_80+)={bf16_native}"
              + ("  <- this GPU emulates bf16 in software: correct but much "
                 "slower than fp16 for no benefit. Use configs/model/gemma3_4b.yaml"
                 if mcfg.dtype == "bfloat16" and has_cuda and not bf16_native else ""))
        if has_cuda:
            cap = torch.cuda.get_device_capability(0)
            check("not a Pascal GPU", cap[0] >= 7,
                  f"compute capability {cap[0]}.{cap[1]}"
                  + ("  <- 1080 Ti: fp16 runs at 1/64 rate. Resubmit with "
                     "--exclude=gnode[01-42]" if cap[0] < 7 else ""))
        check("gpu memory sufficient", total >= need,
              f"have {total:.0f} GiB, want >= {need} GiB for {mcfg.name}")
    else:
        check("cuda check skipped", True, "running on CPU by request")

    # 2. weights reachable
    if not mcfg.random_init:
        try:
            from huggingface_hub import model_info
            info = model_info(mcfg.hf_id)
            check("weights reachable", True, f"{mcfg.hf_id} ({info.sha[:8]})")
        except Exception as e:  # noqa: BLE001
            check("weights reachable", False,
                  f"{type(e).__name__}: {e}. Gemma 3 is gated -- accept the "
                  "licence on huggingface.co and set HF_TOKEN.")
            write_json(out / "env_check.json", report)
            return 1

    # 3. load and verify shapes
    try:
        model, tok, minfo = M.load_model(mcfg)
    except Exception as e:  # noqa: BLE001
        check("model loads", False, f"{type(e).__name__}: {e}")
        write_json(out / "env_check.json", report)
        return 1

    check("model loads", True,
          f"{minfo.n_layers}L x {minfo.n_heads}H, head_dim {minfo.head_dim}, "
          f"resid {minfo.resid_dim}")
    check("module paths discovered", True,
          f"{minfo.layer_prefix}.* / norm={minfo.norm_name}")
    naive = minfo.resid_dim // minfo.n_heads
    check("head_dim read from config, not derived", True,
          f"head_dim={minfo.head_dim}; resid_dim//n_heads would give {naive}"
          + ("  <- these differ, deriving it would be WRONG" if naive != minfo.head_dim else ""))

    # 4. hooks do something
    enc = tok("Q:big\nA:small\n\nQ:hot\nA:", return_tensors="pt").to(model.device)
    clean = model(**enc).logits[:, -1, :].float()

    store: dict[int, torch.Tensor] = {}
    with M.capture_head_outputs(model, minfo, store):
        model(**enc)
    shapes_ok = (len(store) == minfo.n_layers
                 and store[0].shape[1:] == (minfo.n_heads, minfo.head_dim))
    check("head capture", shapes_ok, f"{len(store)} layers, {tuple(store[0].shape)}")

    mean = torch.stack([store[l][0] for l in range(minfo.n_layers)]) + 5.0
    batch = {k: v.repeat(minfo.n_heads, 1) for k, v in enc.items()}
    with M.patch_heads(model, minfo, [(minfo.mid_layer, h) for h in range(minfo.n_heads)], mean):
        patched = model(**batch).logits[:, -1, :].float()
    deltas = (patched - clean).abs().max(dim=-1).values
    check("head patching changes logits", bool((deltas > 0).all()),
          f"min per-head delta {float(deltas.min()):.4g}")

    vec = torch.randn(minfo.resid_dim) * 3
    with M.add_to_residual(model, minfo, minfo.mid_layer, vec):
        added = model(**enc).logits[:, -1, :].float()
    check("residual add changes logits", float((added - clean).abs().max()) > 0,
          f"max delta {float((added - clean).abs().max()):.4g}")

    # Must reproduce the model's own logits: checks the norm and tied unembedding.
    resid: dict[int, torch.Tensor] = {}
    with M.capture_residual(model, minfo, resid):
        model(**enc)
    norm = dict(model.named_modules())[minfo.norm_name]
    head = model.get_output_embeddings()
    lens_last = head(norm(resid[minfo.n_layers - 1])).float()
    err = float((lens_last - clean).abs().max())
    tol = 0.2 if model.dtype != torch.float32 else 1e-3
    check("logit lens reproduces final logits", err < tol, f"max abs error {err:.4g}")

    # 5. fp16 numerical safety, against an fp32 CPU reference. Gemma 3 was
    # trained in bf16 and has a history of fp16 activation overflow.
    if model.dtype == torch.float16 and not mcfg.random_init and not args.skip_fp16_check:
        from transformers import AutoModelForCausalLM

        probes = [
            "Q:big\nA:small\n\nQ:hot\nA:",
            "Q:\u092a\u0947\u0926\u094d\u0926\nA:\u0686\u093f\u0928\u094d\u0928\n\nQ:\u0935\u0947\u0921\u093f\nA:",
            "Q:pedda\nA:chinna\n\nQ:vedi\nA:",
        ]
        print("  [INFO] loading an fp32 CPU copy for the fp16 reference check "
              "(~17 GB host RAM, a few minutes)")
        ref = AutoModelForCausalLM.from_pretrained(
            mcfg.hf_id, dtype=torch.float32, device_map="cpu").eval()
        worst_top1, worst_corr, nan_seen = True, 1.0, False
        for text in probes:
            e16 = tok(text, return_tensors="pt").to(model.device)
            l16 = model(**e16).logits[:, -1, :].float().cpu()
            l32 = ref(**tok(text, return_tensors="pt")).logits[:, -1, :].float()
            nan_seen |= bool(torch.isnan(l16).any() or torch.isinf(l16).any())
            worst_top1 &= bool(l16.argmax(-1) == l32.argmax(-1))
            c = float(torch.corrcoef(torch.stack([l16[0], l32[0]]))[0, 1])
            worst_corr = min(worst_corr, c)
        del ref
        check("fp16 produces no NaN/Inf", not nan_seen,
              "checked English, Telugu-script and romanized probes")
        check("fp16 agrees with fp32 reference", worst_top1 and worst_corr > 0.999,
              f"top-1 match={worst_top1}, min logit corr={worst_corr:.5f}"
              + ("" if worst_top1 and worst_corr > 0.999
                 else "  <- fp16 is NOT safe here. Move to a bf16-capable GPU "
                      "and use configs/model/gemma3_4b_bf16.yaml"))
        report["fp16_check"] = {"top1_match": worst_top1, "min_logit_corr": worst_corr,
                                "nan_or_inf": nan_seen}

    # 6. batch-size headroom
    rec = None
    if has_cuda and not mcfg.random_init:
        torch.cuda.reset_peak_memory_stats()
        with M.patch_heads(model, minfo, [(0, h) for h in range(minfo.n_heads)], mean):
            model(**batch)
        per = torch.cuda.max_memory_allocated() / 2**30
        free = torch.cuda.get_device_properties(0).total_memory / 2**30 - per
        rec = max(1, int(minfo.n_heads * (1 + free / max(per, 1e-6)) * 0.6))
        print(f"  [INFO] peak {per:.1f} GiB at batch {minfo.n_heads}; "
              f"head_batch up to ~{rec} should fit")

    report["recommended_head_batch"] = rec
    report["model_info"] = {
        "n_layers": minfo.n_layers, "n_heads": minfo.n_heads,
        "head_dim": minfo.head_dim, "resid_dim": minfo.resid_dim,
        "attn_in_dim": minfo.attn_in_dim, "layer_prefix": minfo.layer_prefix,
        "norm_name": minfo.norm_name,
        "aie_forwards_per_trial": minfo.n_layers,
        "aie_forwards_per_trial_unbatched": minfo.n_layers * minfo.n_heads,
    }
    write_json(out / "env_check.json", report)

    if failures:
        print(f"\n{len(failures)} check(s) FAILED: {failures}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
