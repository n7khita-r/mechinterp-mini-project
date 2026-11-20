#!/bin/bash
# Thin wrapper kept for convenience. The real script is run_pipeline.sh.
#
# 27B cannot run on Ada: ~54 GB of bf16 weights against 4 x 11 GB per node with a
# one-node cap, and 54 GB also exceeds the 50 GB /home quota that is the only
# filesystem Ada's compute nodes can see. See configs/model/gemma3_27b.yaml.
exec bash "$(dirname "$0")/run_pipeline.sh" configs/model/gemma3_27b.yaml "$@"
