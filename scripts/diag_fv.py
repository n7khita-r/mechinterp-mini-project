#!/usr/bin/env python3
"""Diagnose a function vector with no effect: is it too small to move the residual?

Prints the norm ratio, then sweeps a scale factor over one cell.
"""

from __future__ import annotations

import sys

import torch

from _common import REPO_ROOT, parser, setup  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))

from fvcm import data as D  # noqa: E402
from fvcm import model as M  # noqa: E402
from fvcm.evaluate import accuracy  # noqa: E402
from fvcm.fv import FunctionVector  # noqa: E402
from fvcm.intervene import score_zero_shot  # noqa: E402
from fvcm.prompts import zero_shot  # noqa: E402
from _common import RESULTS  # noqa: E402

SCALES = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]


def main() -> int:
    p = parser("diagnose function vector magnitude")
    p.add_argument("--task", default="present_past")
    p.add_argument("--condition", default="te")
    p.add_argument("--max-items", type=int, default=40)
    p.add_argument("--layers", default="", help="comma-separated; default: 1/3 depth and mid")
    args = p.parse_args()
    mcfg, exp, chash, out, started = setup(args, "diag_fv")

    model, tok, info = M.load_model(mcfg)
    fv = FunctionVector.load(RESULTS / "fv", mcfg.name,
                             args.condition, args.task)

    items = D.headline(D.load_items(args.task, args.condition, exp), exp)
    split = D.split_items(items, exp)
    eval_items = split.valid[: args.max_items]
    print(f"\ncell: {args.task}/{args.condition}   {len(eval_items)} items\n")

    # 1. How big is the vector, relative to what it is added to?
    prompts = [zero_shot(i, exp) for i in eval_items[:8]]
    enc = tok([p.text for p in prompts], return_tensors="pt",
              padding=True).to(model.device)
    resid: dict[int, torch.Tensor] = {}
    with M.capture_residual(model, info, resid):
        model(**enc)

    fv_norm = float(fv.vector.float().norm())
    print(f"||function vector|| = {fv_norm:.3f}\n")
    print(f"{'layer':>6s}  {'||resid||':>10s}  {'ratio fv/resid':>15s}")
    for l in range(0, info.n_layers, max(1, info.n_layers // 8)):
        r = resid[l].float()
        r = r[:, -1, :] if r.dim() == 3 else r   # already last-token
        rn = float(r.norm(dim=-1).mean())
        print(f"{l:6d}  {rn:10.2f}  {fv_norm / max(rn, 1e-9):15.4f}")
    print("\nA ratio below ~0.01 means the vector is numerically invisible: the")
    print("addition happens, but it cannot change the argmax over 262k tokens.\n")

    # 2. Does scaling it up recover an effect?
    layers = ([int(x) for x in args.layers.split(",") if x]
              or sorted({info.n_layers // 3, info.n_layers // 2}))
    base = accuracy(score_zero_shot(model, tok, info, eval_items, exp), k=1)
    print(f"zero-shot baseline (no vector): {base:.3f}\n")
    print(f"{'layer':>6s}  " + "  ".join(f"x{s:<5g}" for s in SCALES))
    for layer in layers:
        row = []
        for s in SCALES:
            vec = fv.vector * s
            acc = accuracy(score_zero_shot(model, tok, info, eval_items, exp, vec, layer), k=1)
            row.append(f"{acc:6.3f}")
        print(f"{layer:6d}  " + "  ".join(row))

    print("\nReading this:")
    print("  * a hump as the scale rises  -> the method works, it needed scaling")
    print("  * flat across every scale    -> the vector does not carry the task here")
    print("  * accuracy collapses at high scale only -> normal; it swamps the prompt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
