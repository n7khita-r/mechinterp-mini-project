#!/bin/bash
# Setup on a RunPod H200/H100 pod. Run this on the pod.
#
#   cd /workspace/MechInterp-final-project && bash scripts/setup_runpod.sh
#
# No virtualenv: the container is already an isolated environment, and building
# one here would only risk the self-referential-venv failure we hit on Ada for
# no benefit. Packages go into the image's own Python.
#
# Two RunPod facts this handles:
#   * The container root filesystem is small and ephemeral. /workspace is the
#     persistent volume, so HF_HOME points there -- otherwise a 54 GB 27B
#     download fills the root disk and the pod wedges.
#   * The PyTorch images ship torch already, often with a transformers too old
#     for Gemma 3. We keep the existing torch (it matches the image's CUDA) and
#     only upgrade transformers if needed.
set -euo pipefail
cd "$(dirname "$0")/.."

# Pick the volume with the most free space rather than assuming /workspace.
# RunPod names a resized or recreated volume /workspace-2, /workspace-3, ... and
# the original /workspace stays mounted but small, so a hardcoded path sends a
# 24.5 GB checkpoint to the small volume and fills it mid-run (this has happened).
WS=""; WS_KB=0
for v in /workspace /workspace-*; do
  [ -d "$v" ] || continue
  KB=$(df -Pk "$v" 2>/dev/null | awk 'NR==2{print $4}')
  [ -n "$KB" ] || continue
  if [ "$KB" -gt "$WS_KB" ]; then WS_KB=$KB; WS=$v; fi
done
[ -n "$WS" ] || WS="$HOME"
echo "using volume $WS ($((WS_KB/1024/1024)) GiB free)"
export HF_HOME="${HF_HOME:-$WS/hf}"
mkdir -p "$HF_HOME"

{
  echo "export HF_HOME=$HF_HOME"
  echo "export PYTHONPATH=$PWD/src"
  echo "export PYTHONUNBUFFERED=1"
  echo "export TOKENIZERS_PARALLELISM=false"
} > "$PWD/.env.sh"
# shellcheck disable=SC1090
source "$PWD/.env.sh"

echo "repo:    $PWD"
echo "HF_HOME: $HF_HOME"
df -h "$HF_HOME" | tail -1
echo

python3 - <<'PY'
import importlib.util, subprocess, sys

def ver(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None

torch_v = ver("torch")
print(f"torch: {torch_v or 'MISSING'}")
if torch_v is None:
    print("installing torch (no CUDA-matched image torch found)")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])

# Gemma 3 landed in transformers 4.50. Anything older cannot load the model at
# all, so this is the one hard requirement.
tf = ver("transformers")
need_tf = tf is None or tuple(int(x) for x in tf.split(".")[:2]) < (4, 50)
print(f"transformers: {tf or 'MISSING'}{'  -> upgrading (Gemma 3 needs >= 4.50)' if need_tf else '  ok'}")
if need_tf:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U",
                           "transformers>=4.50", "accelerate", "tokenizers", "safetensors"])

missing = [p for p, m in [("pandas", "pandas"), ("pyyaml", "yaml"),
                          ("matplotlib", "matplotlib"), ("scipy", "scipy"),
                          ("sentencepiece", "sentencepiece"), ("pytest", "pytest"),
                          ("huggingface_hub", "huggingface_hub")]
           if importlib.util.find_spec(m) is None]
if missing:
    print("installing:", " ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
PY

echo
python3 - <<'PY'
import sys, torch, transformers
print(f"python {sys.version.split()[0]}  torch {torch.__version__} (cuda {torch.version.cuda})"
      f"  transformers {transformers.__version__}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    gib = p.total_memory / 2**30
    print(f"gpu: {p.name}  {gib:.0f} GiB  sm_{p.major}{p.minor}  "
          f"native bf16={p.major >= 8}")
    if p.major < 8:
        print("WARNING: no native bf16 -- that is the problem we left Ada to escape.")
    # Which config to use is purely a memory question, so say it outright.
    if gib >= 120:
        print("recommend: configs/model/gemma3_4b_h200.yaml / gemma3_27b_h200.yaml")
    elif gib >= 70:
        print("recommend: configs/model/gemma3_4b_bf16.yaml / gemma3_27b.yaml")
    else:
        print(f"recommend: 4B only at {gib:.0f} GiB; 27B needs ~54 GB of weights plus room")
else:
    print("WARNING: no CUDA device visible")
PY

cat <<EOF

Next
----
1. export HF_TOKEN=hf_...     # accept the licence at huggingface.co/google/gemma-3-4b-pt
2. bash scripts/fetch_weights.sh 4b      # or: 27b, or: both
3. bash scripts/run_pipeline.sh configs/model/gemma3_4b_h200.yaml

Each new shell on this pod: source $PWD/.env.sh
EOF
