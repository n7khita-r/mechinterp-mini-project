#!/bin/bash
# The cross-condition stages, to run after run_split_4b.sbatch finishes.
#
#   sbatch --dependency=afterok:<array-job-id> scripts/slurm/finalise_4b.sbatch
#
# These three cannot be split by condition:
#   * the pooled AIE sweep, which averages indirect effects over all conditions
#     to give the single universal head set for comparison with Todd et al.;
#   * stage 06 (transfer), which adds each condition's function vector to every
#     other condition's zero-shot prompts;
#   * stage 09 (report), which needs every result to build the figures.

# Partition is deliberately NOT hard-coded: Ada's partition names have changed
# (the wiki still lists `long`, which no longer exists). The job uses the
# cluster default unless you override it on the command line:
#     sbatch -p <partition> scripts/slurm/<this file>
# Find the real names with: sinfo -s
#SBATCH --job-name=fvcm-4b-final
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

# shellcheck disable=SC1091
source scripts/slurm/activate.sh

COMMON="--model configs/model/gemma3_4b.yaml --exp configs/exp/default.yaml --no-progress"

echo "node: $(hostname)   started: $(date)"

echo; echo "================ 03 mediation (pooled) ================"
python scripts/03_mediation.py $COMMON --pooled
echo; echo "================ 06 transfer ================"
python scripts/06_transfer.py $COMMON
echo; echo "================ 09 report ================"
python scripts/09_report.py $COMMON

echo
echo "finished: $(date)"
echo "figures -> paper/figures/ and presentation/images/"
echo "tables  -> paper/tables/"
