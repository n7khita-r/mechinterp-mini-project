#!/bin/bash
# Environment setup on a rented bf16-capable GPU box (H100 / H200 / A100).
#
#   bash scripts/setup_env_gpu.sh
#
# For a bf16-capable GPU. An older cluster needing torch 2.3.1+cu118 for
# Ada's constraints; none of those apply here, and two of them would actively
# hurt:
#   * H100/H200 are sm_90. Use a current torch with cu12x -- a cu118 build is
#     the wrong target.
#   * No glibc ceiling, so no version pinning.
#
# Why we are on a rented GPU at all: Ada's RTX 2080 Ti has no native bf16, and
# scripts/00_env_check.py showed Gemma 3 produces NaN in fp16 on it (hooks and
# logit lens all returned nan). Gemma 3 was trained in bf16 and needs bf16.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "ERROR: a virtualenv is active ($VIRTUAL_ENV). Run 'deactivate' first."
  echo "Building a venv from inside a venv produces a self-referential"
  echo "pyvenv.cfg whose site-packages never reaches sys.path."
  exit 1
fi

# Rented boxes often have a small root disk and a large attached volume. Weights
# are 8.6 GB for 4B and ~54 GB for 27B, so put the cache somewhere with room and
# check before downloading rather than after.
: "${HF_HOME:=$HOME/.cache/huggingface}"
export HF_HOME
mkdir -p "$HF_HOME"
echo "HF_HOME: $HF_HOME"
df -h "$HF_HOME" | tail -1

PYBIN=""
for c in python3.12 python3.11 python3.10 python3; do
  cand=$(command -v "$c" 2>/dev/null) || continue
  case "$cand" in "$REPO"/*) continue ;; esac
  if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,9) else 1)' 2>/dev/null; then
    PYBIN="$cand"; break
  fi
done
[ -n "$PYBIN" ] || { echo "ERROR: no python >= 3.9 found"; exit 1; }
echo "base interpreter: $PYBIN ($("$PYBIN" -c 'import platform; print(platform.python_version())'))"

rm -rf .venv
"$PYBIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel

# Current torch for the host's CUDA. No constraints file: unlike Ada, nothing
# here forces an old version.
pip install torch
pip install -r requirements.txt

python - <<'PY'
import sys, torch, transformers
print(f"python {sys.version.split()[0]}")
print(f"torch {torch.__version__} (cuda {torch.version.cuda})")
print(f"transformers {transformers.__version__}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    native_bf16 = p.major >= 8
    print(f"gpu {p.name}  {p.total_memory / 2**30:.0f} GiB  sm_{p.major}{p.minor}")
    print(f"native bf16 (sm_80+): {native_bf16}")
    if not native_bf16:
        print("WARNING: this GPU has no native bf16. That is the problem we came "
              "here to escape -- check what you rented.")
else:
    print("WARNING: no CUDA device visible")
PY

cat <<'EOF'

Next steps
----------
1. export HF_TOKEN=hf_...        (Gemma 3 is gated; accept the licence first)
2. Pre-download the weights:
     4B : python -c "from huggingface_hub import snapshot_download as d; d('google/gemma-3-4b-pt')"
     27B: python -c "from huggingface_hub import snapshot_download as d; d('google/gemma-3-27b-pt')"
3. Run it:
     bash scripts/run_pipeline.sh configs/model/gemma3_4b_bf16.yaml
     bash scripts/run_pipeline.sh configs/model/gemma3_27b.yaml
EOF
