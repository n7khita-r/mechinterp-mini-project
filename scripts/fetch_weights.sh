#!/bin/bash
# Download Gemma 3 weights. Separate from setup so a failed/slow download does
# not force the whole environment to be rebuilt, and so it is obvious how much
# disk is about to be consumed.
#
#   bash scripts/fetch_weights.sh 4b | 12b | 27b | both
#
# Disk needed: 4B ~8.6 GB, 12B ~24.5 GB, 27B ~54 GB. The script prints free space
# before downloading, because filling the volume mid-run breaks the pipeline's
# result writes (it has happened).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env.sh ] && { . ./.env.sh; }
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

[ -n "${HF_TOKEN:-}" ] || { echo "ERROR: export HF_TOKEN first (Gemma 3 is gated)."; exit 1; }

case "${1:-4b}" in
  4b)   MODELS=(google/gemma-3-4b-pt);  NEED=10 ;;
  12b)  MODELS=(google/gemma-3-12b-pt); NEED=26 ;;
  27b)  MODELS=(google/gemma-3-27b-pt); NEED=56 ;;
  both) MODELS=(google/gemma-3-4b-pt google/gemma-3-27b-pt); NEED=64 ;;
  *)    echo "usage: bash scripts/fetch_weights.sh 4b|12b|27b|both"; exit 1 ;;
esac

# Refuse rather than half-fill the volume: a download that runs out of space
# leaves partial blobs AND breaks whatever pipeline is writing results alongside.
AVAIL=$(df -BG --output=avail "$HF_HOME" | tail -1 | tr -dc '0-9')
if [ "${AVAIL:-0}" -lt "$NEED" ]; then
  echo "ERROR: need ~${NEED} GB free for ${MODELS[*]}, but only ${AVAIL} GB available on:"
  df -h "$HF_HOME" | tail -1
  echo
  echo "Free space, or resize the volume. Current cache contents:"
  du -sh "$HF_HOME"/hub/* 2>/dev/null | sort -h
  exit 1
fi

echo "HF_HOME: $HF_HOME"
df -h "$HF_HOME" | tail -1
echo "about to download: ${MODELS[*]}  (4B ~8.6 GB, 27B ~54 GB)"
echo

for m in "${MODELS[@]}"; do
  python3 - "$m" <<'PY'
import sys
from huggingface_hub import snapshot_download
m = sys.argv[1]
# allow_patterns skips nothing we need but avoids any stray large extras.
p = snapshot_download(m, max_workers=8)
print(f"{m} -> {p}")
PY
done

du -sh "$HF_HOME"
