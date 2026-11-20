#!/bin/bash
# Stage 01 only, over every tier -- the evidence for the complexity/register
# claim in the proposal, which the headline run does not produce because it
# filters to `core`.
#
# Only stage 01 is needed: "does sanskritized or complex Telugu degrade more?"
# is a question about 10-shot accuracy, not about function vectors. ~8 min on
# an H200 for 4B.
#
#   bash scripts/run_tiers.sh configs/model/gemma3_4b_h200.yaml
#
# Writes to results/01_tiers/<model>/ via --out, so the headline baselines in
# results/01_baselines/ are never touched.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-configs/model/gemma3_4b_h200.yaml}"
EXP="configs/exp/tiers.yaml"
NAME="$(.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from fvcm.config import ModelConfig; print(ModelConfig.load('$MODEL').name)")"

export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
LOG="logs/tiers-$NAME-$(date +%Y%m%d-%H%M%S).log"

{
  echo "================ 01 baselines (all tiers) ================  $(date +%H:%M:%S)"
  python scripts/01_baselines.py --model "$MODEL" --exp "$EXP" \
      --out "$PWD/results/01_tiers/$NAME"
  echo
  echo "================ tier table ================  $(date +%H:%M:%S)"
  python scripts/tier_report.py --model "$MODEL" --exp "$EXP"
} 2>&1 | tee "$LOG"

echo
echo "tier table -> paper/tables/tier_accuracy.tex"
echo "log        -> $LOG"
