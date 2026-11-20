#!/bin/bash
# Rerun only the two stages that were applying the function vector at its raw
# norm, plus the report that consumes them. ~20 min on an H200 for 4B.
#
# Stages 06 (transfer) and 08 (controls) scaled nothing, while stage 05 scaled the
# vector to a fraction of the residual norm. In Gemma that fraction is the whole
# effect, so both stages measured the pre-fix null: every control came out equal
# to the function vector, and every row of the transfer matrix came out equal to
# its target column. Neither was a finding.
#
# Stages 01-05 and 07 are untouched -- their config hash is unchanged, so they
# replay from cache and their numbers do not move.
#
#   bash scripts/rerun_controls.sh configs/model/gemma3_4b_h200.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-configs/model/gemma3_4b_h200.yaml}"
EXP="${2:-configs/exp/default.yaml}"
NAME="$(.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from fvcm.config import ModelConfig; print(ModelConfig.load('$MODEL').name)")"

export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
LOG="logs/rerun-controls-$NAME-$(date +%Y%m%d-%H%M%S).log"

# 09 is a pure function of the stages above it and its hash did not change, so it
# would otherwise replay the stale tables it wrote last time.
rm -rf "results/09_report/$NAME"

{
  for s in 06_transfer 08_controls 09_report; do
    echo "================ $s ================  $(date +%H:%M:%S)"
    python "scripts/$s.py" --model "$MODEL" --exp "$EXP"
    echo
  done
} 2>&1 | tee "$LOG"

echo
echo "check that FV now separates from the random-vector control:"
echo "  column 'FV-rand' in the stage 08 table above, and paper/tables/controls.tex"
echo "log -> $LOG"
