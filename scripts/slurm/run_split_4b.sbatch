#!/bin/bash
# Same pipeline, split four ways by condition across four GPUs. 1-2 h instead of
# 3-5 h, at the cost of four GPUs of quota (QoS medium allows exactly 4).
#
#   JID=$(sbatch --parsable scripts/slurm/run_split_4b.sbatch)
#   sbatch --dependency=afterok:$JID scripts/slurm/finalise_4b.sbatch
#
# Each array task owns one condition and runs the stages that are independent
# across conditions. Two things cannot be split and live in finalise_4b.sbatch:
#   * stage 06 (transfer) needs the function vectors from every condition;
#   * the pooled AIE sweep and stage 09 (report) need all results.
#
# Note array task 3 (tg) does Part 2 only -- a single word cannot be code-mixed,
# so Part 1 has no Tenglish condition. cells() skips it automatically.

# Partition is deliberately NOT hard-coded: Ada's partition names have changed
# (the wiki still lists `long`, which no longer exists). The job uses the
# cluster default unless you override it on the command line:
#     sbatch -p <partition> scripts/slurm/<this file>
# Find the real names with: sinfo -s
#SBATCH --job-name=fvcm-4b-split
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

# shellcheck disable=SC1091
source scripts/slurm/activate.sh

CONDS=(en te rom tg)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
COMMON="--model configs/model/gemma3_4b.yaml --exp configs/exp/default.yaml \
        --conditions $COND --no-progress"

echo "condition: $COND   node: $(hostname)"
echo "started:   $(date)"

for s in 01_baselines 02_mean_activations 03_mediation 04_build_fv \
         05_intervene 07_logitlens 08_controls; do
  echo
  echo "================ $s [$COND] ================  $(date +%H:%M:%S)"
  python "scripts/$s.py" $COMMON
done

echo
echo "finished $COND: $(date)"
