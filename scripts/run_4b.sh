#!/bin/bash
# Thin wrapper: Gemma 3 4B in bf16 on a rented GPU.
#
# Use this rather than the Ada config. scripts/00_env_check.py on Ada's RTX
# 2080 Ti showed Gemma 3 produces NaN in fp16 -- head patching, the residual add
# and the logit lens all came back nan -- because the model was trained in bf16
# and Turing has no native bf16. That is why these runs moved off the cluster.
exec bash "$(dirname "$0")/run_pipeline.sh" configs/model/gemma3_4b_bf16.yaml "$@"
