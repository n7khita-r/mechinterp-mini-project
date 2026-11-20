# Where Does Code-Mixing Break the Task?

**Function vectors in Telugu, romanized Telugu and Tenglish.**
CL3.412 Mechanistic Interpretability of LLMs — final project.

When a language model does worse on code-mixed input, there are two very
different explanations, and accuracy alone cannot tell them apart:

* **A** — it cannot *read* the input properly, or
* **B** — it read the input but no longer *knows what task* it is being asked.

They need different fixes. This project separates them using **function
vectors** (Todd et al., [ICLR 2024](https://arxiv.org/abs/2310.15213)): a task
representation can be extracted from a model's attention heads and injected
back into a prompt that contains no examples at all. If the task survives
code-mixing, injecting it should repair the failure.

**Finding:** the task representation survives. What breaks is recovering the
answer from the surface form — and it breaks *selectively*, damaging what
Telugu encodes in grammatical endings while leaving what it encodes in
recognisable names intact.

---

## The four conditions

Every item exists in four parallel versions, aligned by item id:

| condition | example (`sen-0001`) |
|---|---|
| `en` English | next week kavita is buying upma in guntur |
| `te` Telugu script | వచ్చే వారం కవిత గుంటూరులో ఉప్మా కొంటోంది |
| `rom` Romanized | vachche varam kavita guntur lo upma kontondi |
| `tg` Tenglish | **next week** kavita guntur lo upma kontondi |

Tenglish is built **on top of** the romanized version, so `te`→`rom` isolates
*script* and `rom`→`tg` isolates *code-mixing*. Because conditions are
row-aligned, every comparison is paired: the same sentence, written four ways.

---

## Results at a glance

Gemma 3 4B, `core` tier, held-out items.

| | finding |
|---|---|
| **Same heads everywhere** | Heads `17.4` and `17.6` are top-2 in all four conditions (layer 17 of 34). English is the outlier, not Tenglish. |
| **Script, not code-mixing** | `extract_verb` + FV: `en` 0.98, `te` 0.92, but `rom` 0.21 — same sentences, different script. |
| **Names survive, endings do not** | `extract_person`/`rom` 0.88 vs `extract_verb`/`rom` 0.21. |
| **Rarity hurts, complexity does not** | `complex` tier costs −0.01 to +0.05; `sanskritized` and `rare` cost −0.22 to −0.35. |
| **Scaling is required** | Function vectors do nothing on Gemma until rescaled relative to the residual norm (35×–1841×). |

### The scaling result

This is the most portable finding here. Gemma scales embeddings by
`√d_model` and never renormalises the residual stream, so its norm grows from
~780 at layer 0 to ~10⁵ at layer 33. A function vector has norm ~16, so adding
it raw is a 0.1% nudge — real, but far too small to change an argmax over a
262k vocabulary. GPT-J, which the method was developed on, has no such gap.

The fix is to scale relative to the stream it is added to:

```
v ← v · α‖h_ℓ‖ / ‖v‖
```

with `α` chosen on a tuning slice. Without this, every result is `+0.00`.

---

## Install

Needs Python ≥ 3.10 and a bf16-capable GPU (we used an H200). Gemma 3 is
gated — accept the licence on Hugging Face first.

```bash
bash scripts/setup_env_gpu.sh          # or setup_runpod.sh on a rented box
export HF_TOKEN=hf_...
bash scripts/fetch_weights.sh 4b       # 4b | 12b | 27b
```

No GPU? The whole pipeline runs on a randomly-initialised tiny model:

```bash
make smoke
```

That exercises every stage end to end on CPU and catches shape, hook and
plumbing bugs without touching the gated weights.

---

## Run

```bash
bash scripts/run_pipeline.sh configs/model/gemma3_4b_h200.yaml   # everything
bash scripts/run_subset.sh  configs/model/gemma3_12b_h200.yaml   # 5 tasks only
bash scripts/run_tiers.sh   configs/model/gemma3_4b_h200.yaml    # difficulty tiers
```

Analyses that need no GPU, run after the pipeline:

```bash
python scripts/majority_baseline.py --model configs/model/gemma3_4b_h200.yaml
python scripts/tier_report.py       --model configs/model/gemma3_4b_h200.yaml --exp configs/exp/tiers.yaml
```

Diagnostics, when a function vector appears to do nothing:

```bash
python scripts/diag_fv.py    --model <cfg> --task present_past --condition te
python scripts/diag_joint.py --model <cfg> --task present_past --condition te
```

`diag_fv` checks whether the vector is too small to move the residual stream.
`diag_joint` searches layer and strength *together*, to check whether the
sequential search missed a working combination.

### Pipeline stages

| stage | what it does |
|---|---|
| 00 | environment and shape assertions |
| 01 | 10-shot / zero-shot baselines, competence filter |
| 02 | mean head activations over 100 prompts |
| 03 | causal mediation — indirect effect of every head |
| 04 | build the function vector, `v = Σ_h W_O^(h) ā_h` |
| 05 | inject at one layer of a zero-shot prompt |
| 06 | cross-condition transfer |
| 07 | logit lens — layer curves and vector vocabulary |
| 08 | controls |
| 09 | all tables and figures |

Stage 05 caches per cell, so an interrupted run resumes. Other stages recompute.

---

## Controls

No number is reported without them:

* **random vector, matched norm** — rules out "any large nudge at that depth works"
* **unrelated-task vector** — rules out "any well-formed vector helps"
* **shuffled-label floor** — the prompt's *form* without its content
* **majority-class floor** — on 2–3-way label tasks, always guessing the commonest answer

All injected vectors are scaled to the same norm, so only direction differs.
Cells that fail a control are reported as not demonstrated rather than quietly
dropped — see `paper/tables/controls.tex` and `majority_baseline.tex`.

---

## Extending the dataset

`data/` contains **only hand-editable TSVs — no code.** Add rows in any editor.

* `data/part1/*.tsv` — one row per word pair, columns per condition
* `data/part2/{lexicon,verbs,templates}.tsv` — sources for generated sentences
* `data/part2/sentences.tsv` — regenerate with `make sentences`

Two invariants hold automatically:

1. **Conditions stay row-aligned**, so comparisons remain paired.
2. **Splits are content-addressed** — adding rows does not reshuffle existing
   train/test assignments, so cached results stay valid.

Check your edits before running anything:

```bash
make lint     # answers single-token-decidable, conditions aligned, no leakage
make test     # unit tests for the data and model layers
```

Tiers (`core`, `complex`, `sanskritized`, `rare`, `literary_en`, `ambiguous`)
let you add deliberately harder items without disturbing headline numbers,
which use `core` only.

---

## Layout

```
configs/     model and experiment configs (YAML)
data/        hand-written TSVs, no code
src/fvcm/    the library
scripts/     one per pipeline stage, plus diagnostics
paper/       ACL-format write-up, generated tables and figures
presentation/  Beamer slides (LuaLaTeX; Telugu font vendored)
results/     per-stage outputs, keyed by model and config hash
tests/       invariants that would corrupt results silently rather than crash
```

Build the write-up and slides (**LuaLaTeX** — pdfLaTeX renders Telugu blank):

```bash
make paper
make slides
make slides-zip
```

---

## Reproducibility

Every result is stamped with a config hash covering the model config, the
experiment config, **and a code version**. Changing an algorithm without
touching a config therefore invalidates the cache instead of silently replaying
stale numbers — a failure that cost us a confusing debugging round. Per-stage
overrides (`STAGE_CODE_VERSION` in `src/fvcm/config.py`) let a fix to one stage
invalidate only that stage rather than forcing hours of recomputation.

## Known limitations

* One model family (Gemma 3), two sizes.
* At 12B the condition ordering **inverts** — Telugu script becomes weakest,
  despite having the strongest 10-shot baselines and the strongest causal
  signal. Unexplained; reported rather than smoothed over.
* Layer and strength are selected sequentially, not jointly.
* Romanized word-level cells keep as few as 2–13 items after the competence
  filter. Reported, but nothing is built on them.
* One romanization convention, documented in `data/README.md`.
* The `tense` task is excluded: its vector was flat at zero in *every*
  condition including English, so it measured nothing about code-mixing.

## Credits

Method: Todd, Li, Sharma, Mueller, Wallace and Bau, *Function Vectors in Large
Language Models*, ICLR 2024 — [code](https://github.com/ericwtodd/function_vectors).
Dataset, code-mixing design, Gemma 3 port and the scaling result are ours.
