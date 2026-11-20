# Teammate handoff — full project record

Everything needed to understand this project, present it, edit the slides, and
finish the paper, without having been in the room while it was built. Read
§1–§3 to be able to explain it; §5 for the numbers; §9 before you touch
anything.

Project: **Where Does Code-Mixing Break the Task? Function Vectors in Telugu,
Romanized Telugu and Tenglish** (CL3.412, 2024114003 / 2025701032).

---

## 1. The question, and why it needs interpretability

Models do worse on code-mixed input. Accuracy alone cannot say *why*:

* **A** — the model cannot *read* the input, or
* **B** — it read it, but no longer *knows the task*.

Different failures, different fixes: better tokenisation helps A and does
nothing for B. Separating them requires looking inside the model.

**Function vectors** (Todd et al., ICLR 2024) make that possible. Show a model
ten examples of a task and it internally forms a single vector meaning "the
task is: give the opposite". You can extract that vector and add it to a
*zero-shot* prompt — one with no examples — and the model performs the task
anyway.

That gives us a probe: **if we can extract a working task vector under
code-mixing, the task representation survived.** If injecting a good vector
*repairs* the failure, what was missing was the task. If it doesn't, the damage
is upstream, in reading the input.

**Our answer: the task survives; reading breaks — and selectively.**

---

## 2. The data

Four parallel versions of every item, aligned by id:

| cond | example (`sen-0001`) |
|---|---|
| `en` | next week kavita is buying upma in guntur |
| `te` | వచ్చే వారం కవిత గుంటూరులో ఉప్మా కొంటోంది |
| `rom` | vachche varam kavita guntur lo upma kontondi |
| `tg` | **next week** kavita guntur lo upma kontondi |

Three design decisions worth saying out loud when presenting:

1. **Tenglish is built on top of `rom`.** So `te`→`rom` changes only *script*,
   and `rom`→`tg` changes only *code-mixing*. Without that, the two are
   confounded and no result means anything.
2. **Conditions are row-aligned**, so every comparison is *paired* — the same
   sentence tested four ways. Far more power than four independent samples.
3. **`subject_gender` is a built-in sanity check.** Telugu marks subject gender
   on the verb (తిన్నాడు `tinnadu` m. / తింది `tindi` f.); English does not
   ("ate" either way). English therefore *cannot* beat chance. Any method
   reporting English success here is broken. This check fired correctly — see §5d.

15 tasks in two families: word-level (same tasks as Todd et al., so our setup
is comparable) and sentence-level (ours). Tiers — `core`, `complex`,
`sanskritized`, `rare`, `literary_en` — let us add harder items without
disturbing headline numbers, which use `core` only.

---

## 3. Method, stage by stage

| stage | what it does |
|---|---|
| 01 | baselines; builds the **competence filter** |
| 02 | mean head activations over 100 ten-shot prompts |
| 03 | **causal mediation** — indirect effect of every head, 50 trials |
| 04 | `v = Σ_h W_O^(h) ā_h` over the top-10 heads |
| 05 | inject `v` at one layer of a **zero-shot** prompt |
| 06 | cross-condition transfer |
| 07 | logit lens |
| 08 | controls |
| 09 | tables and figures |

### How the three pools work (the "dev slice" question)

Two splits, one after the other. For `extract_verb/en`:

```
298 core items
 ├─ 104  example pool   → the 10 demonstrations are sampled from here
 └─ 194  evaluation pool
     ├─  58  tuning slice → pick edit layer ℓ and strength α
     └─ 136  held-out     → the only number we report (0.98)
```

Plain-language version for the slide: **demonstrations come from one pool, we
tune on a second, we report on a third we never touched.** Both free parameters
are knobs; tuning them on the test set would be reporting luck.

### What the model actually sees at test time

```
Q: fast
A:
```

That is the *whole* prompt. Nothing says "give the opposite". If the model
answers `slow`, the only thing that told it the task was the vector added to
the residual stream.

### Choices that differ from the paper

* **Per-condition head sets**, not one universal set. Whether `te`/`rom`/`tg`
  recruit the same heads as `en` is one of our questions; a shared set would
  assume the answer. This made head overlap a result.
* **Competence filter**: an item counts only if the model already answers it
  correctly at 10 shots. We test the *vector*, not the model's vocabulary.
* **Residual-relative scaling** — see below.

### The scaling result (our most portable contribution)

The first full run showed the vector doing **nothing**: `+0.00` at every layer,
every condition, *including English*. The method hadn't failed; the arithmetic
had.

Gemma scales embeddings by `√d_model` and never renormalises the stream, so
`‖h‖` grows ~780 → ~10⁵ with depth. A function vector has norm ~16–26. Raw, it
is a **0.1% nudge** — real, but far too small to move an argmax over 262k
tokens. GPT-J, which the method was built on, has no such gap, so nothing in
the original work flags it.

Fix: `v ← v · α‖h_ℓ‖/‖v‖`, with α chosen on the tuning slice. Selected
multipliers ran **35× to 1841×**.

> **Function vectors do not transfer to Gemma-class models until rescaled
> relative to the residual norm.** This generalises beyond our question.

---

## 4. Controls — what each rules out

| control | rules out |
|---|---|
| random vector, matched norm | "any large nudge at that depth works" |
| unrelated-task vector | "any well-formed function vector helps" |
| shuffled-label floor | the prompt's *form* without its content |
| majority-class floor | always guessing the commonest label |

All injected vectors are scaled to the **same** norm, so only direction
differs. **The professor is specifically attentive to controls — every results
table in the deck shows them, and cells that fail are reported as not
demonstrated.**

---

## 5. Results (Gemma 3 4B, `core` tier, held-out)

### 5a. Baselines — degradation is real and task-dependent
Extraction survives everything (`extract_place` ≥0.97 in all four). Grammar
tasks degrade monotonically: negation `0.93 → 0.78 → 0.64 → 0.60`.
`subject_gender` **inverts**: `en` 0.50 (chance) vs `te` 0.96.

### 5b. Head sets — the same heads everywhere
`17.4` and `17.6` are top-2 in **all four** conditions, layer 17 of 34 —
mid-depth, replicating Todd et al.'s depth finding in a different architecture
*and* language.

Jaccard: `en–te` 0.43, `en–rom` 0.33, `en–tg` 0.33, **`te–tg` 0.54**.

> **English is the outlier, not Tenglish.** Code-mixing does not get routed
> somewhere else in the model.

### 5c. Intervention — with controls

| task | cond | +FV | random | unrelated | shuffled | n |
|---|---|---|---|---|---|---|
| extract verb | en | **0.98** | 0.00 | 0.00 | 0.22 | 194 |
| extract verb | te | **0.92** | 0.01 | 0.06 | 0.21 | 179 |
| extract verb | rom | **0.21** | 0.00 | 0.02 | 0.11 | 189 |
| extract verb | tg | **0.29** | 0.00 | — | 0.13 | 191 |
| subject gender | te | **0.93** | 0.00 | 0.00 | 0.63 | 27 |
| subject gender | en | 0.71 | 0.00 | 0.00 | **0.71** | 14 |
| extract person | rom | **0.88** | 0.20 | **0.83** | 0.05 | 113 |

Two honest failures to keep in: English subject gender *exactly* matches its
shuffled floor (no real effect — as predicted); `extract_person/rom` is nearly
matched by the unrelated vector, so part of that gain is prompt **format**, not
task content.

### 5d. Majority floors
Of eight label cells, **exactly one** beats its floor: `subject_gender/te`,
0.89 vs 0.58. English sits *below* its floor (0.80 vs 0.90) — reported as "not
above chance", not as 80%. Negation `en` and `rom` match their floors to two
decimals, the fingerprint of emitting one label for everything.

> Without this control, that 0.80 would have read as a strong positive result.

### 5e. Transfer — diagonal dominates

| from ↓ / to → | en | rom | te | tg |
|---|---|---|---|---|
| en | **0.62** | 0.40 | 0.50 | 0.21 |
| rom | 0.36 | **0.48** | 0.37 | 0.32 |
| te | 0.43 | 0.49 | **0.74** | 0.39 |
| tg | 0.38 | 0.28 | 0.32 | **0.40** |

Vectors transfer substantially but each condition's own is best for it.

### 5f. Tiers — rarity hurts, complexity does not
Change from `core`, pooled over all 15 tasks:

| tier | en | te | rom | n |
|---|---|---|---|---|
| complex | −0.01 | +0.04 | +0.05 | 372 |
| sanskritized | −0.22 | −0.09 | −0.25 | 37 |
| rare | −0.25 | −0.35 | −0.32 | 23 |
| rare English | −0.05 | −0.12 | −0.01 | 20 |

> **Syntactic complexity costs nothing; lexical rarity costs a lot.**

Small n on the bottom rows — report direction, not exact values. **This is a
4B, all-15-tasks result.**

We also ran tiers on 12B, restricted to the 5 subset tasks. Only the `complex`
row is usable there (n=157), and it **replicates**: +0.01 / +0.01 / +0.03 /
−0.02. The `sanskritized` (n=10), `rare` (n=3) and `literary_en` (n=8) rows all
read 1.00 and must not be reported — those tiers live almost entirely in the
Part-1 word tasks, which the subset excludes.

### 5g. Logit lens — two independent confirmations

Project the vector onto the vocabulary and read what it promotes:

| cell | top tokens |
|---|---|
| `extract_verb/en` | ` is`, ` has`, ` was`, ` will`, ` did` — **auxiliary verbs** |
| `subject_gender/en` | `they`, ` he`, ` she` — **pronouns** |
| `extract_verb/te` (12B) | `w`, `på`, `getValue` — junk |
| `subject_gender/rom` | `de`, `te`, `ta` — junk |

Where the vectors work they decode into exactly the right semantic class;
where they fail, into garbage. Separately, the layer at which the gold token
enters the top 10 is *later* for weaker conditions (`te` 46 vs `en` 35 on
`extract_verb`, 12B, 48 layers).

### 5h. 12B — the ordering inverts (unexplained)

| task | 4B te | 12B te | 12B en | 12B rom | 12B tg |
|---|---|---|---|---|---|
| extract person | 0.96 | **0.68** | 0.96 | 0.83 | 1.00 |
| extract verb | 0.93 | **0.19** | 0.78 | 0.51 | 0.66 |
| negation | 0.49 | **0.07** | 0.49 | 0.50 | 0.67 |
| present past | 1.00 | **0.25** | 0.90 | 0.88 | — |
| subject gender | 0.89 | **0.10** | 0.47 | 0.20 | 0.47 |

The method scales fine — every cell improves over zero-shot and controls still
hold. But Telugu script goes from strongest (4B) to weakest (12B) in all five
tasks. **We ruled out the easy explanations:**

* 12B reads Telugu *better* — 10-shot 0.86–1.00, its **best** condition
* Telugu has the **strongest** causal signal (AIE 0.101 vs en 0.075)
* α optimum is found correctly, not clipped by the grid
* an exhaustive layer×strength search on `extract_verb/te` finds **nothing
  better** (−0.02)

Report it as an open result. "We found something we can't explain, and here's
what we ruled out" is a stronger position than a tidy story, and it pre-empts
the obvious questions.

---

## 6. The interpretation (the punchline)

**Survives romanization** — `extract_person` 0.88, `extract_place` 0.76. These
targets are **proper nouns**: "kavita", "guntur" romanize losslessly; the
string is the same object.

**Collapses** — `extract_verb` 0.21, `subject_gender` 0.00. These targets live
in **inflectional morphology**. కొంటోంది → `kontondi` is where tense, gender and
polarity are encoded, and romanization flattens exactly that.

> Code-mixing breaks the task at the **input-reading** stage, and selectively:
> it damages what is carried by grammatical endings while leaving what is
> carried by recognisable names intact.

---

## 7. Changes from the proposal

**Dropped:** incremental ± scoring (would make numbers depend on row insertion
order); vLLM (cannot expose or patch activations — the entire experiment);
`tense` as a reported task.

**Changed:** 27B → 12B (cost); per-condition head sets; residual-relative
scaling (unplanned — we didn't know it was needed); majority-class floors
(added after an English result looked like a success).

**`tense` is excluded from all tables.** Its vector was flat at zero in *every*
condition including English, at every layer and scale. A condition effect shows
as a *gap between* conditions; an identical zero in the control language is a
failure of the method on that task. Raw numbers remain in `results/`.

---

## 8. Challenges (the war-stories slide — keep it short)

**Infrastructure:** cluster docs didn't match the cluster; a venv built inside
an active venv produced `No module named torch` only on compute nodes; fp16
produced NaNs on Gemma 3, forcing a move to a rented H200; a model download
filled the disk mid-run; volumes and SSH ports changed repeatedly.

**Debugging the science — four bugs all looked identical** ("the function
vector does nothing"):

1. layer selection collapsing to layer 0 on all-zero ties (`max()` returns the
   first index)
2. a config hash that didn't cover *code* version, so stale results replayed
3. the residual-scale problem (§3) — the only real science
4. stages 06 and 08 still applying the vector *raw* while 05 applied it scaled,
   which made every control equal the function vector and looked like "the
   effect isn't task-specific"

> **Lesson:** a result of exactly `+0.00` is a bug report, not a finding. Every
> stage now prints the quantity it depends on.

---

## 9. Editing the deck — read before touching anything

Build (**LuaLaTeX only** — pdfLaTeX renders Telugu blank):

```bash
cd presentation && lualatex main.tex     # or: make slides
make slides-zip                          # sources + PDF as presentation/slides.zip
```

* Telugu inline: `\tel{...}`. Conditions: `\cond{en}`. Fonts vendored in
  `presentation/fonts/`, so it builds anywhere.
* Macros match the group's existing decks: `\src{}`, `\tight`, `\bridge{}`,
  plus `\reading{}` — **the interpretation line under every table.**
* **Do not ship a table without a `\reading{}`.** An unfamiliar audience needs
  the interpretation, not the numbers.
* The deck is 32 frames: question → what an FV is → block-diagram overview →
  data → method → controls → results by research question → heads → logit lens
  (p19) → 12B → **Full 4B results (p22–27)** → interpretation → limitations →
  changes → challenges → future work.
* **The "Full 4B results" section (p22–27)**: all-15-task baselines,
  all-15-task intervention with CIs, controls **grouped by task kind** with a
  failure count per group, the transfer matrix, the layer sweep, and vector
  similarity (as text). Tables are `\input{}` from `presentation/tables/`
  (copies of `paper/tables/`), so regenerating the analysis updates the slides.
* The controls slide is **aggregated on purpose**. The raw 50-row table is
  unreadable on a projector; the grouped version shows the same controls plus a
  "fails" column, and the failures are: `antonym/rom`, `next_item/rom`,
  `subject_gender/rom` (too few items); all four `negation` cells,
  `present_past/en`, `subject_gender/en` and `/tg` (below the shuffled floor);
  `present_past/te` and `singular_plural/rom` (unrelated vector matches). Full
  table remains in `paper/tables/controls.tex`.
* Tables are wrapped in `\adjustbox{max width=..., max totalheight=...}` so a
  wide table scales to the frame width and a tall one to its height. Plain
  `\resizebox` can only constrain one and silently overflows the other.
* `\reading{}` starts with `\par` deliberately — without it the interpretation
  text wraps *around* a centred table instead of sitting below it.

**Traps:**

* **Do not run `09_report.py` for 12B.** It writes `paper/tables/*.tex` keyed
  by *stage*, not model, and would overwrite the 4B tables the deck cites.
  `run_subset.sh` omits it by default.
* Tier tables are now **per-model** (`tier_accuracy_gemma3_4b_h200.tex`,
  `..._12b_...`). They previously shared one filename, and the 12B run
  silently overwrote the 4B numbers — the same clobbering hazard as
  `09_report`. Regenerate with `scripts/tier_report.py`; no GPU needed.
* **The logit-lens slide is p19**, "Logit lens: an independent check on the
  vector". Both halves of it are **12B** — stage 07 never ran for 4B, so there
  is no 4B logit lens. The stale `logit_lens.pdf/png` in `paper/figures/` are
  from the smoke-test model; do not cite them. Real numbers live in
  `results/07_logitlens/gemma3_12b_h200/` (`fv_vocabulary.json`,
  `lens_curves.json`).
* Cells with tiny n (romanized word-level tasks keep 2–13 items) are shown but
  carry no weight.
* Word order confounds `en`–`te` comparisons (SVO vs SOV). `rom`–`tg` is clean.

---

## 10. Status

* **4B: complete** — all stages, controls and transfer rerun after the scaling
  fix, tiers done across all 15 tasks.
* **12B: complete** for the 5-task subset — baselines, mediation, FVs,
  intervention, transfer, logit lens, controls. No tier run (see §5f).
* **Presentation: complete**, 34 frames, builds clean, zero overfull boxes.
  `make slides-zip` produces a self-contained bundle (verified by building it
  in an empty directory).
* **Paper: not started.** ACL template (`acl.sty`, `acl_natbib.bst` already in
  `paper/`), LuaLaTeX, 4–8 pages, tables generated and `\input{}`-ready in
  `paper/tables/`.

### If asked something we didn't do

* *"Did you check layer and scale together?"* — Yes, `scripts/diag_joint.py`.
  Mixed: on one cell joint search gained +0.29, on the key cell −0.02. Selection
  is sequential in the reported numbers; this is in Limitations.
* *"Why not 27B?"* — Compute cost. 12B is the second scale point.
* *"Why no tier results at 12B?"* — We have them, but only the `complex` row
  is large enough to report (n=157), and it replicates the 4B result. The
  harder tiers have n=3–10 in the subset, because those items live in the
  Part-1 word tasks we did not run at 12B.
