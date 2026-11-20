#!/bin/bash
# Pull results off the RunPod box. Run this on your LAPTOP, not the pod.
#
#   bash scripts/sync_results.sh
#
# Run it before anything that resets the pod. RunPod's container filesystem is
# ephemeral; only the mounted volumes survive, and a mistake there loses a run
# you paid for.
set -euo pipefail
cd "$(dirname "$0")/.."

POD_HOST="${POD_HOST:-root@38.80.152.148}"
POD_PORT="${POD_PORT:-33146}"
POD_KEY="${POD_KEY:-$HOME/.ssh/id_ed25519}"
POD_DIR="${POD_DIR:-/workspace/MechInterp-final-project}"

SSH="ssh -p $POD_PORT -i $POD_KEY"
# --no-owner/--no-group/--no-perms: RunPod volumes reject chown for a
# non-matching uid, and -a would spam "Operation not permitted" warnings.
RS="rsync -rlvz --no-owner --no-group --no-perms --progress -e"

echo "pulling from $POD_HOST:$POD_DIR"
for d in results paper/tables paper/figures presentation/images logs; do
  echo "  -> $d"
  mkdir -p "$d"
  $RS "$SSH" "$POD_HOST:$POD_DIR/$d/" "$d/" || echo "     (skipped: not present yet)"
done

echo
echo "local summary:"
for m in results/*/; do :; done
find results -maxdepth 2 -mindepth 2 -type d | sed 's/^/  /' | sort
echo
du -sh results 2>/dev/null
