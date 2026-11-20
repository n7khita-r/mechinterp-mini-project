#!/bin/bash
# Push code to the RunPod box. Run this on your LAPTOP, not the pod.
#
#   bash scripts/sync_code.sh
#
# Code and configs only -- never results. The pod is the source of truth for
# results (it computes them); the laptop is the source of truth for code. Mixing
# the two directions is how a finished run gets overwritten by an empty folder.
set -euo pipefail
cd "$(dirname "$0")/.."

POD_HOST="${POD_HOST:-root@38.80.152.148}"
POD_PORT="${POD_PORT:-33146}"
POD_KEY="${POD_KEY:-$HOME/.ssh/id_ed25519}"
POD_DIR="${POD_DIR:-/workspace/MechInterp-final-project}"

SSH="ssh -p $POD_PORT -i $POD_KEY"
RS="rsync -rlvz --no-owner --no-group --no-perms -e"

echo "pushing to $POD_HOST:$POD_DIR"
for d in src scripts configs data; do
  echo "  -> $d"
  # --delete is deliberately NOT used: a stale file on the pod is harmless,
  # but deleting one mid-run is not.
  $RS "$SSH" "$d/" "$POD_HOST:$POD_DIR/$d/"
done
echo
echo "done. on the pod:"
echo "  cd $POD_DIR && source .env.sh"
