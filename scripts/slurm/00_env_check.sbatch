#!/bin/bash
# Stage 00 -- run this FIRST and read the output before submitting anything long.
#
#   sbatch scripts/slurm/00_env_check.sbatch
#
# It answers the three questions that decide whether the main run is viable:
#   1. Does fp16 agree with fp32 on Gemma 3? Ada has no bf16 (1080/2080 Ti predate
#      Ampere) and Gemma was trained in bf16. If this fails, stop and move to a
#      bf16-capable GPU with configs/model/gemma3_4b_bf16.yaml.
#   2. Did we land on a 2080 Ti rather than a 1080 Ti? Pascal fp16 runs at 1/64
#      rate. --exclude below should prevent it; the check confirms it.
#   3. What head_batch actually fits? Printed as a measured recommendation.
#
# Memory note: the fp16-vs-fp32 comparison loads an fp32 CPU copy of the model,
# which needs ~17 GB of host RAM. Hence -c 10 at --mem-per-cpu=3000M (30 GB).
# Pass --skip-fp16-check if you have already settled the dtype question.

# Partition is deliberately NOT hard-coded: Ada's partition names have changed
# (the wiki still lists `long`, which no longer exists). The job uses the
# cluster default unless you override it on the command line:
#     sbatch -p <partition> scripts/slurm/<this file>
# Find the real names with: sinfo -s
#SBATCH --job-name=fvcm-env
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

# shellcheck disable=SC1091
source scripts/slurm/activate.sh

echo "node:    $(hostname)"
echo "gpu:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "started: $(date)"
echo

python scripts/00_env_check.py --model configs/model/gemma3_4b.yaml \
                               --exp configs/exp/default.yaml

echo
echo "finished: $(date)"
