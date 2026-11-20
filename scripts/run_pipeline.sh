#!/bin/bash
# Run the whole pipeline on a bf16-capable GPU. Not SLURM -- plain bash, so it
# works on any rented box.
#
#   bash scripts/run_pipeline.sh configs/model/gemma3_4b_bf16.yaml
#   bash scripts/run_pipeline.sh configs/model/gemma3_27b.yaml
#
# Expect roughly, on one H100 80GB:
#   4B   ~1-2 h      27B  ~4-6 h
# The two expensive stages are 03 (causal mediation) and 05 (the layer sweep).
# Both are head-batched: one forward per layer with all heads of that layer in
# the batch, rather than one forward per (layer, head). Without that the 27B
# mediation sweep alone is ~50 h instead of ~1.5 h.
#
# Rerunning is cheap: every stage caches by (item, config hash), so an
# interrupted run resumes instead of restarting. Nothing needs cleaning first.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-}"
EXP="${2:-configs/exp/default.yaml}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
  echo "usage: bash scripts/run_pipeline.sh <model-config> [exp-config]"
  echo
  echo "available model configs:"
  ls -1 configs/model/*.yaml | sed 's/^/  /'
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "ERROR: .venv is missing. Run: bash scripts/setup_env_gpu.sh"
  exit 1
fi
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

COMMON="--model $MODEL --exp $EXP"
LOG="logs/run-$(basename "$MODEL" .yaml)-$(date +%Y%m%d-%H%M%S).log"
mkdir -p logs

echo "model:  $MODEL"
echo "exp:    $EXP"
echo "log:    $LOG"
echo "gpu:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable')"
echo

stage () {
  local name=$1; shift
  echo "================ $name ================  $(date +%H:%M:%S)"
  # `script` is not used; PYTHONUNBUFFERED plus the interval-based progress
  # reporter in src/fvcm/progress.py means tail -f on the log is readable.
  python "$@"
  echo
}

{
  echo "started: $(date)"

  # 00 gates everything. On a bf16 GPU the fp16-vs-fp32 comparison is skipped
  # automatically (it only runs when dtype is float16), so no flag is needed.
  stage "00 env check"        scripts/00_env_check.py $COMMON
  stage "01 baselines"        scripts/01_baselines.py $COMMON
  stage "02 mean activations" scripts/02_mean_activations.py $COMMON
  stage "03 mediation"        scripts/03_mediation.py $COMMON
  stage "03 mediation pooled" scripts/03_mediation.py $COMMON --pooled
  stage "04 build fv"         scripts/04_build_fv.py $COMMON
  stage "05 intervene"        scripts/05_intervene.py $COMMON
  stage "06 transfer"         scripts/06_transfer.py $COMMON
  stage "07 logit lens"       scripts/07_logitlens.py $COMMON
  stage "08 controls"         scripts/08_controls.py $COMMON
  stage "09 report"           scripts/09_report.py $COMMON

  echo "finished: $(date)"
} 2>&1 | tee "$LOG"

cat <<EOF

done.
  figures -> paper/figures/  and presentation/images/
  tables  -> paper/tables/
  log     -> $LOG

Copy the results off the rented box before you terminate it:
  rsync -avz results paper/figures paper/tables "$LOG" <somewhere-safe>/
EOF
