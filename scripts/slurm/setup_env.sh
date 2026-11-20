#!/bin/bash
# One-time environment setup on Ada. Run this on the LOGIN node, not in a job.
#
#   bash scripts/slurm/setup_env.sh
#
# Two Ada-specific constraints drive the pinned versions below:
#
#  * Ada's compute nodes run Ubuntu 18.04, which ships glibc 2.27. PyTorch wheels
#    from 2.4 onwards are built manylinux_2_28 and will not load against glibc
#    2.27. torch 2.3.1 is the last manylinux2014 (glibc 2.17) build, so that is
#    the ceiling here -- newer is not better, it simply will not import.
#  * cu118 wheels are used because the installed driver on Ada predates the
#    cu121/cu124 minimums. The wheel bundles its own CUDA runtime, so the
#    `module load cuda` is belt-and-braces only.
#
# Transformers must be >= 4.50 for Gemma 3 support; 4.50-4.56 work with torch
# 2.3.1. Do not let pip resolve transformers to a version that drags in a newer
# torch -- the constraint file prevents that.
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO="$PWD"
echo "repo: $REPO"

# Running this with a venv already active makes `command -v python3` resolve to
# .venv/bin/python3, and the script would then build the venv using the venv's
# own interpreter -- producing a self-referential pyvenv.cfg whose site-packages
# never lands on sys.path. The symptom is a job that starts fine and then says
# "No module named 'torch'". Refuse outright.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "ERROR: a virtualenv is active ($VIRTUAL_ENV)."
  echo "Run 'deactivate' (or open a fresh shell) and try again."
  exit 1
fi

# Weights and the HF cache must live under /home: it is the only filesystem
# visible from the compute nodes. /share1 is master-node only, and /scratch and
# /ssd_scratch are per-node and purged after 7 days.
export HF_HOME="$HOME/.cache/huggingface"
mkdir -p "$HF_HOME"

# Module names on Ada have changed over time and the wiki list is not reliable,
# so try candidates in preference order and keep going if one is absent rather
# than aborting. `module avail` is printed if nothing usable turns up.
LOADED_MODULES=()
load_first () {
  local kind=$1; shift
  for m in "$@"; do
    if module load "$m" 2>/dev/null; then
      echo "loaded $kind module: $m"
      LOADED_MODULES+=("$m")
      return 0
    fi
  done
  echo "note: no $kind module matched any of: $*"
  return 1
}

load_first python \
  u18/python/3.11.2 u18/python/3.10.2 u18/python/3.8.3 \
  python/3.11.2 python/3.10.2 python/3.9 python || true
load_first cuda \
  u18/cuda/11.6 u18/cuda/11.8 u18/cuda/12.1 cuda/11.6 cuda/11.8 cuda || true

# Pick a base interpreter >= 3.9 (transformers >= 4.50 needs it). Candidates are
# tried newest first, and anything inside the repo is rejected so we can never
# bootstrap a venv from itself.
PYBIN=""
for c in python3.12 python3.11 python3.10 python3.9 python3 \
         /usr/bin/python3.12 /usr/bin/python3.11 /usr/bin/python3.10 /usr/bin/python3.9; do
  cand=$(command -v "$c" 2>/dev/null) || continue
  case "$cand" in "$REPO"/*) continue ;; esac
  if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,9) else 1)' 2>/dev/null; then
    PYBIN="$cand"; break
  fi
done
if [ -z "$PYBIN" ]; then
  echo
  echo "ERROR: no python >= 3.9 found outside the repo."
  echo "Candidate modules on this cluster:"
  module avail 2>&1 | tr ' ' '\n' | grep -i python | sort -u
  exit 1
fi
echo "base interpreter: $PYBIN ($("$PYBIN" -c 'import platform; print(platform.python_version())'))"

# Rebuild from scratch: an existing venv may be the broken self-referential one,
# and "already satisfied" would hide that. Wheels come from the pip cache, so
# this costs minutes, not another multi-GB download.
rm -rf .venv
"$PYBIN" -m venv .venv

# Record what we built against. The venv's bin/python is a symlink to the
# interpreter that created it, so a batch job must load the same modules or the
# venv will not work. The job scripts source this file instead of hard-coding
# module names that drift.
: > .venv/ada_modules
if [ "${#LOADED_MODULES[@]}" -gt 0 ]; then
  printf 'module load %s\n' "${LOADED_MODULES[@]}" >> .venv/ada_modules
fi
echo "recorded ${#LOADED_MODULES[@]} module(s) in .venv/ada_modules"
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel

pip install --index-url https://download.pytorch.org/whl/cu118 "torch==2.3.1+cu118"
pip install -c scripts/slurm/constraints-ada.txt -r requirements.txt

# Verify the way a batch job will see it: a clean shell with nothing inherited.
echo
echo "verifying the venv from a clean shell (this is what the job does):"
env -i HOME="$HOME" PATH=/usr/bin:/bin bash -lc "
  cd '$REPO' && source .venv/bin/activate && python -c '
import sys, torch, transformers
print(\"  python\", sys.version.split()[0], \"at\", sys.executable)
print(\"  torch\", torch.__version__, \"cuda\", torch.version.cuda)
print(\"  transformers\", transformers.__version__)
'" || { echo "FATAL: venv still broken from a clean shell"; exit 1; }

cat <<'EOF'

Next steps
----------
1. Accept the Gemma 3 licence at https://huggingface.co/google/gemma-3-4b-pt
2. Export a token with read access:
       export HF_TOKEN=hf_xxxxxxxx
   and add it to ~/.bashrc so batch jobs inherit it.
3. Pre-download the weights on the LOGIN node (compute nodes may not have
   outbound internet, and you do not want to spend GPU time on a download):
       python -c "from huggingface_hub import snapshot_download; \
         snapshot_download('google/gemma-3-4b-pt')"
   This is ~8.6 GB against a 50 GB /home quota. Check with: du -sh ~/.cache/huggingface
4. Check which SLURM accounts and QoS you have:
       sacctmgr show assoc user=$USER format=Account,QOS,DefaultQOS
   Then set ACCOUNT in the sbatch scripts accordingly (default: research).
5. Submit the sanity check first:
       sbatch scripts/slurm/00_env_check.sbatch
EOF
