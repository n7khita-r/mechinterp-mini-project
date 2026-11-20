#!/bin/bash
# Run the pipeline on a subset of tasks, to keep GPU cost down.
#
#   bash scripts/run_subset.sh configs/model/gemma3_12b_h200.yaml
#   TASKS=negation,subject_gender bash scripts/run_subset.sh <config>
#
# The default subset is chosen from the 4B results: the tasks that actually
# discriminate between conditions. Dropped are the cells that sat at ceiling in
# every condition (they cannot show a scale effect) and `tense`, whose function
# vector failed in all four conditions including English -- a limitation of the
# method on that task, not a language result.
#
# Conditions are never subsetted: the comparison between them IS the experiment.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-configs/model/gemma3_12b_h200.yaml}"
EXP="${2:-configs/exp/default.yaml}"
[ -f "$MODEL" ] || { echo "no such model config: $MODEL"; ls configs/model/; exit 1; }

# subject_gender + extract_verb are the two witnesses for the headline claim
# (Telugu script recovers the task, romanized and Tenglish do not).
TASKS="${TASKS:-subject_gender,extract_verb,negation,present_past,extract_person}"

[ -f .env.sh ] && . ./.env.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p logs

LOG="logs/subset-$(basename "$MODEL" .yaml)-$(date +%Y%m%d-%H%M%S).log"
M="--model $MODEL --exp $EXP --tasks $TASKS"

{
  echo "model: $MODEL"
  echo "tasks: $TASKS"
  echo "gpu:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)"
  echo "started: $(date)"
  for s in 00_env_check 01_baselines 02_mean_activations 03_mediation \
           04_build_fv 05_intervene 06_transfer 07_logitlens 08_controls 09_report; do
    echo; echo "================ $s ================  $(date +%H:%M:%S)"
    python3 "scripts/$s.py" $M
  done
  echo; echo "finished: $(date)"
} 2>&1 | tee "$LOG"

echo
echo "log: $LOG"
echo "pull results to your laptop with:  bash scripts/sync_results.sh"
