#!/bin/bash
# Bring a RunPod box back after a reset (e.g. after resizing a volume).
#
#   bash scripts/runpod_resume.sh
#
# A RunPod reset keeps the mounted volumes and wipes everything else -- including
# every pip package, since they live in the container's Python, not /workspace.
# So this reinstalls, repoints the HF cache at the larger volume, and clears the
# unrelated cached repos that filled the old one.
set -euo pipefail
cd "$(dirname "$0")/.."

# Prefer the largest mounted volume. The volume is NOT hardcoded: RunPod names a
# replacement volume /workspace-2, /workspace-3, ... each time one is resized or
# recreated, and a hardcoded name silently sends a 24.5 GB download to the small
# volume and fills it mid-run. Pick by free space instead.
BEST=""; BEST_KB=0
for v in /workspace /workspace-*; do
  [ -d "$v" ] || continue
  KB=$(df -Pk "$v" 2>/dev/null | awk 'NR==2{print $4}')
  [ -n "$KB" ] || continue
  if [ "$KB" -gt "$BEST_KB" ]; then BEST_KB=$KB; BEST=$v; fi
done
[ -n "$BEST" ] || { echo "ERROR: no /workspace volume mounted"; exit 1; }
export HF_HOME="$BEST/hf"
echo "volume with most free space: $BEST ($((BEST_KB/1024/1024)) GiB free)"
mkdir -p "$HF_HOME"
echo "HF_HOME: $HF_HOME"
df -h "$HF_HOME" | tail -1

# Old coursework repos were occupying the small volume. Remove them from BOTH
# possible cache locations -- the cache may have been written under either.
echo
echo "clearing unrelated cached repos:"
for root in /workspace/.cache/huggingface /workspace/hf /workspace-*/hf; do
  [ -d "$root/hub" ] || continue
  for repo in datasets--nikhitaravi--anlp-assn3-data models--nikhitaravi--anlp-assn3 \
              datasets--belumind--en-vi-ja-curated-500k-triplets \
              datasets--Salesforce--wikitext; do
    if [ -e "$root/hub/$repo" ]; then
      echo "  rm $root/hub/$repo"
      rm -rf "${root:?}/hub/${repo:?}"
    fi
  done
done

bash scripts/setup_runpod.sh

cat <<EOF

Resumed. Next:
  source .env.sh
  export HF_TOKEN=hf_...
  bash scripts/fetch_weights.sh 12b
  bash scripts/run_subset.sh configs/model/gemma3_12b_h200.yaml
EOF
