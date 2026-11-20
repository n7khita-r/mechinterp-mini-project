#!/bin/bash
# The whole pipeline on one GPU, stages 01-09 in order. Expect 3-5 h on an
# RTX 2080 Ti in fp16.
#
#   sbatch scripts/slurm/run_all_4b.sbatch
#
# Run 00_env_check.sbatch first and read its output. In particular, if the fp16
# check failed, do not run this -- switch to a bf16-capable GPU.
#
# Prefer run_split_4b.sbatch if you have 4 GPUs of quota free: it splits by
# condition and finishes in 1-2 h.
#
# Every stage is resumable: results are cached by (item, config hash), so if the
# job is killed, resubmitting picks up where it stopped rather than restarting.

# Partition is deliberately NOT hard-coded: Ada's partition names have changed
# (the wiki still lists `long`, which no longer exists). The job uses the
# cluster default unless you override it on the command line:
#     sbatch -p <partition> scripts/slurm/<this file>
# Find the real names with: sinfo -s
#SBATCH --job-name=fvcm-4b
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

# shellcheck disable=SC1091
source scripts/slurm/activate.sh

MODEL=configs/model/gemma3_4b.yaml
EXP=configs/exp/default.yaml
COMMON="--model $MODEL --exp $EXP --no-progress"

echo "node: $(hostname)   gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "started: $(date)"

stage () {
  echo
  echo "================ $1 ================  $(date +%H:%M:%S)"
  shift
  python "$@"
}

stage "00 env check"        scripts/00_env_check.py $COMMON --skip-fp16-check
stage "01 baselines"        scripts/01_baselines.py $COMMON
stage "02 mean activations" scripts/02_mean_activations.py $COMMON
# One AIE sweep per condition plus a pooled one. This is the expensive stage;
# head batching makes it n_layers forwards per trial instead of n_layers*n_heads.
stage "03 mediation"        scripts/03_mediation.py $COMMON
stage "03 mediation pooled" scripts/03_mediation.py $COMMON --pooled
stage "04 build fv"         scripts/04_build_fv.py $COMMON
stage "05 intervene"        scripts/05_intervene.py $COMMON
stage "06 transfer"         scripts/06_transfer.py $COMMON
stage "07 logit lens"       scripts/07_logitlens.py $COMMON
stage "08 controls"         scripts/08_controls.py $COMMON
stage "09 report"           scripts/09_report.py $COMMON

echo
echo "finished: $(date)"
echo "figures -> paper/figures/ and presentation/images/"
echo "tables  -> paper/tables/"
