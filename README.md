# Function Vectors in Telugu, Romanized Telugu and Tenglish

Done by: Nikhita Ravi (2024114003), Krishna Kartheek (2025701032).

Method: Todd et al., Function Vectors in Large Language Models,
[ICLR 2024](https://arxiv.org/abs/2310.15213).

Presentation compiled in the /presentation directory, paper as well in paper/

## Layout

```
configs/
  model/          a YAML per model (gemma3_4b_h200, gemma3_12b_h200, debug_tiny, ...)
  exp/            experiment configs (default, smoke, tiers)
data/
  part1/          word-level task TSVs + categories.tsv
  part2/          lexicon, verbs, templates, generated sentences.tsv
  README.md       romanization convention and tier definitions
src/fvcm/
  config.py       YAML loading, config hashing
  data.py         item construction, splits
  prompts.py      n-shot and zero-shot prompt building
  model.py        model loading, module discovery, the three hooks
  activations.py  mean head activations
  mediation.py    indirect effect per head
  fv.py           function vector construction
  intervene.py    injection, layer/scale selection, transfer
  controls.py     random, unrelated-task and shuffled-label controls
  logitlens.py    per-layer readout
  evaluate.py     scoring, competence filter, bootstrap CIs
  cache.py        crash-resume cache, run manifests
  progress.py     log-friendly progress lines
  analysis/       figures.py, tables.py, style.py
scripts/
  00..09_*.py     one per pipeline stage
  run_*.sh        pipeline drivers
  diag_*.py       layer/scale diagnostics for a single cell
  lint_data.py    TSV checks
  majority_baseline.py, tier_report.py
paper/            ACL-format write-up, generated tables and figures
presentation/     Beamer slides (LuaLaTeX, Telugu font vendored)
results/          per-stage outputs, keyed by model and config hash
```

Pipeline stages: `00` env checks, `01` baselines and competence filter,
`02` mean head activations, `03` causal mediation, `04` build function vectors,
`05` inject into zero-shot prompts, `06` cross-condition transfer, `07` logit
lens, `08` controls, `09` figures and tables.

## Setup

Needs python ≥ 3.10 and a bf-16 supporting GPU

```bash
bash scripts/setup_env_gpu.sh
export HF_TOKEN=...
bash scripts/fetch_weights.sh 4b     # 4b | 12b | 27b
```

## Run

Whole pipeline, 15 tasks:

```bash
bash scripts/run_pipeline.sh configs/model/gemma3_4b_h200.yaml
```

Five-task subset for gemma 12B, and the difficulty-tier sweep:

```bash
bash scripts/run_subset.sh configs/model/gemma3_12b_h200.yaml
bash scripts/run_tiers.sh configs/model/gemma3_4b_h200.yaml
```

