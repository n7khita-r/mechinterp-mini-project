# Teammate handoff

Everything needed to understand, present, and edit this project. Written for
someone picking it up cold, to refine the slides without re-deriving the work.

Project: **Where Does Code-Mixing Break the Task? Function Vectors in Telugu,
Romanized Telugu and Tenglish** (CL3.412 mini-project, 2024114003 / 2025701032).

---

## 1. The question, in one paragraph

Multilingual LLMs get worse on code-mixed input. Accuracy alone cannot say
*why*: the model might have lost the **task representation** (it no longer knows
what it is being asked), or it might still hold the task but fail to **read the
input**. Those are different failures with different fixes. We separate them
using **function vectors** (Todd et al., ICLR 2024): extract a compact vector
that encodes "the task" from attention-head activations, then inject it into a
*zero-shot* prompt. If the task representation survives code-mixing, injecting
it should repair the failure. If it does not, the damage is upstream.

**Our answer: the task representation survives. Reading the input is what
breaks, and it breaks selectively — morphology is damaged, proper nouns are
not.**

---

## 2. The four conditions (the core design)

Every item exists in four parallel versions, row-aligned by item id:

| cond | example (`sen-0001`) |
|---|---|
| `en`  | next week kavita is buying upma in guntur |
| `te`  | వచ్చే వారం కవిత గుంటూరులో ఉప్మా కొంటోంది |
| `rom` | vachche varam kavita guntur lo upma kontondi |
| `tg`  | **next week** kavita guntur lo upma kontondi |

Two things about this matter a lot and are worth saying out loud in the talk:

1. **Tenglish is built on top of `rom`**, by swapping some slots back to
   English. So `rom`→`tg` isolates **code-mixing** with script held constant,
   and `te`→`rom` isolates **script** with language held constant. Without that
   construction the two effects are confounded.
2. **Conditions are row-aligned**, so every comparison is *paired* — the same
   sentence tested four ways. That is far more statistical power than four
   independent samples, and it is why the CIs are as tight as they are.

**`subject_gender` is the built-in sanity check.** Telugu marks subject gender
on the verb (తిన్నాడు `tinnadu` masc. vs తింది `tindi` fem.); English does not
("ate" either way). So English *cannot* be above chance on this task by
construction. Any method that reports English success here is measuring an
artefact. This check fired correctly — see §5.

---

## 3. Method, in the order the pipeline runs

| stage | what it does |
|---|---|
| 01 | 10-shot and zero-shot baselines; builds the **competence filter** |
| 02 | mean head activations over 100 ten-shot prompts |
| 03 | **causal mediation**: indirect effect of every attention head, 50 trials |
| 04 | builds the function vector: `v = Σ_h W_O^(h) · ā_h` over the top-10 heads |
| 05 | **intervention**: inject `v` at one layer of a zero-shot prompt |
| 06 | transfer: apply a vector built in one condition to another condition |
| 08 | controls (see §4) |
| 09 | all tables and figures |

Key choices, and why:

- **Competence filter**: an item counts only if the model already answers it
  correctly at 10 shots. We are testing whether the *vector* works, not whether
  the model knows the word.
- **Per-condition head sets** (a deviation from the paper, which uses one
  universal set). We need to know whether `te`/`rom`/`tg` recruit the *same*
  heads as `en`; a shared set would assume the answer. This made head overlap a
  result rather than an assumption.
- **Layer and scale are chosen on a 30% dev slice**, reported on the held-out
  remainder. Both are free parameters; tuning them on the test set would be
  cheating.
- **Head batching**: all heads of a layer in one batched forward, so mediation
  costs `n_layers` forwards per trial instead of `n_layers × n_heads`. This is
  the difference between ~1.5 h and ~50 h.

### The residual-scaling discovery (our most portable contribution)

Our first full run showed the function vector changing **nothing** — `+0.00` at
every layer, in every condition, *including English*. The method had not failed;
the arithmetic had.

Gemma scales embeddings by `√d_model` and never renormalises the residual
stream, so `‖h‖` grows from ~780 at layer 0 to ~10⁵ at layer 33. A function
vector has norm ~16–22. Adding it raw is a **0.1% nudge** — real, but far too
small to move an argmax over a 262k vocabulary. GPT-J, which the method was
built on, has no such gap, so nothing in the original work flags it.

**Fix**: scale the vector to a fraction `α` of the residual norm at the edit
layer, `v ← v · α‖h_ℓ‖/‖v‖`, with `α` chosen on the dev slice. Selected
multipliers ran **35× to 1841×**.

> **Function vectors do not transfer to Gemma-class models until they are
> rescaled relative to the residual norm.** This is the one finding here that
> generalises beyond our specific question.

---

## 4. Controls — what each one rules out

| control | rules out |
|---|---|
| random vector, matched norm | "any large vector at that depth jolts the model" |
| unrelated-task function vector | "any well-formed function vector helps" |
| shuffled-label floor | the level reached with the task's *form* but not its content |
| **majority-class floor** | on 2–3-way label tasks, one label emitted for everything |

All three injected vectors are scaled to the **same** norm, so only direction
differs. This matters: before we fixed it, stages 06 and 08 applied the vector
*raw* while stage 05 applied it scaled, which put every control in the
invisible regime and made FV ≈ random ≈ unrelated in 52 of 52 rows. That looked
like "the effect is not task-specific" but was really the same null three times.

---

## 5. Results (Gemma 3 4B, `core` tier unless stated)

### 5a. Baselines — degradation is real and task-dependent
- Copy-style tasks sit at ceiling everywhere (extract place ≥0.97 in all four).
- Reasoning-style tasks degrade monotonically `en`→`te`→`rom`→`tg`:
  negation `0.93 → 0.78 → 0.64 → 0.60`.
- `subject_gender` **inverts**: `en` 0.50 (chance — no gender marking available)
  vs `te` 0.96. Exactly as the linguistics predicts.

### 5b. Head sets — the same heads, in every condition
Top heads `17.4` and `17.6` are top-2 in **all four** conditions, at layer 17 of
34 (mid-depth — replicating the depth finding from GPT-J in a different
architecture *and* a different language).

Jaccard overlap: `en–te` 0.43, `en–rom` 0.33, `en–tg` 0.33, **`te–tg` 0.54**.

> **English is the outlier, not Tenglish.** The three Telugu-derived conditions
> resemble each other more than any resembles English. Code-mixing does not
> recruit a separate mechanism.

### 5c. Intervention — Telugu script recovers, Latin script does not

| task | en | te | rom | tg |
|---|---|---|---|---|
| extract place | 0.98 | 0.99 | 0.76 | 0.83 |
| extract person | 0.84 | 0.96 | 0.85 | 0.93 |
| extract verb | 0.99 | 0.93 | **0.21** | **0.30** |
| extract food | 0.80 | 0.81 | **0.33** | **0.39** |
| subject gender | 0.80 | 0.89 | **0.00** | **0.00** |
| present past | 0.29 | 1.00 | 0.81 | — |

### 5d. Majority floors — reading the label tasks honestly
Of **eight** label-task cells, **exactly one** clears its floor:
`subject_gender/te` at 0.89 against a 0.58 floor (**+0.32**).

`subject_gender/en` is at 0.80 against a **0.90** floor — i.e. *below* chance.
Without the floor beside it, that 0.80 reads as a strong positive result. It is
the opposite, and it is the sanity check from §2 firing correctly.
Negation `en` and `rom` match their floors *exactly* — the signature of a model
emitting one label for everything.

### 5e. Controls (after the fix) — specificity holds
FV vs random vector, on cells with n ≥ 10:

| task | cond | FV | random | unrelated |
|---|---|---|---|---|
| extract verb | en | 0.98 | 0.00 | 0.00 |
| extract place | te | 1.00 | 0.05 | 0.06 |
| **subject gender** | **te** | **0.93** | **0.00** | **0.00** |
| extract verb | te | 0.92 | 0.01 | 0.06 |
| choose last | en | 0.95 | 0.00 | 0.69 |
| extract food | te | 0.82 | 0.07 | 0.67 |

The random-vector control is crushed everywhere — the effect is **not** generic
perturbation. The unrelated-task vector is usually far below FV but sometimes
substantial (`choose_last/en` 0.69, `extract_food/te` 0.67), on tasks that share
output *format* with the unrelated task. Worth stating honestly: the vector is
clearly task-bearing, but part of what it carries on some tasks is format.
**The headline cell `subject_gender/te` is perfectly clean** (0.93 vs 0.00 vs 0.00).

### 5f. Transfer — vectors are partly condition-specific
Rows = FV source, columns = prompt condition. The **diagonal is highest**:

| from ↓ / to → | en | rom | te | tg |
|---|---|---|---|---|
| en | **0.62** | 0.40 | 0.50 | 0.21 |
| rom | 0.36 | **0.48** | 0.37 | 0.32 |
| te | 0.43 | 0.49 | **0.74** | 0.39 |
| tg | 0.38 | 0.28 | 0.32 | **0.40** |

Vectors transfer substantially but not fully — each condition's own vector is
best for it.

### 5g. Tiers — **rarity hurts, complexity does not**
Change from `core`, 10-shot accuracy:

| tier | en | te | rom | n |
|---|---|---|---|---|
| complex | −0.01 | +0.04 | +0.05 | 372 |
| sanskritized | −0.22 | −0.09 | −0.25 | 37 |
| rare | −0.25 | −0.35 | −0.32 | 23 |
| rare English | −0.05 | −0.12 | −0.01 | 20 |

> **Syntactic complexity costs essentially nothing; lexical rarity and
> sanskritized register cost a lot.** This fits the morphology story in §6.

Caveat: `sanskritized`/`rare` cells are small (n = 23–37, pooled over tasks).
Report the *direction*, not the exact value. Also, the tier run evaluates a
larger item pool than the headline run, so its `core` row is **not** numerically
comparable to the headline numbers — only compare *within* the tier table.

---

## 6. The interpretation (this is the talk's punchline)

The dissociation in 5c is the whole argument:

**Survives romanization** — `extract_person` 0.85/0.93, `extract_place` 0.76/0.83.
These targets are **proper nouns**. "kavita", "guntur" romanize essentially
losslessly; the string is the same object in either script.

**Collapses under romanization** — `extract_verb` 0.21/0.30, `subject_gender`
0.00/0.00. These targets live in **inflectional morphology**. కొంటోంది →
`kontondi` is where tense, gender and polarity are encoded, and romanization
flattens exactly that.

> Code-mixing breaks the task at the **input-reading** stage, and selectively:
> it damages what is carried by morphology while leaving what is carried by
> identifiable substrings intact.

---

## 7. What changed from the proposal

**Dropped**: incremental ± scoring (would make every number depend on insertion
order); vLLM (cannot expose or patch intermediate activations — the entire
experiment); `tense` as a reported task (see below).

**Changed**: 27B → 12B as the second model (cost); per-condition head sets
(without which head overlap could not be measured); residual-relative scaling
(not planned — we did not know it was needed); majority-class floors (added
after noticing `en` subject gender looked like a success).

**`tense` is excluded from all reported tables.** Its function vector was flat
at zero in *every* condition including English, at every layer and every scale.
A condition effect shows up as a *gap between* conditions; an identical zero in
the control language is a failure of the method on that task. It carries no
information about code-mixing, so it is excluded rather than over-read. This is
stated in the Scope section of the deck. Raw numbers remain in `results/`.

---

## 8. Challenges (the "what went wrong" slide)

**Infrastructure**: cluster docs did not match the cluster (wrong partition,
modules for the wrong OS, different node-name format); a `venv` built from
inside an active `venv` resolved its own interpreter and produced
`No module named torch` only on compute nodes; **fp16 produced NaNs** in Gemma 3
mediation, forcing a move to a rented H200 in bf16; a concurrent 27B download
filled the volume mid-run and killed the job.

**Scientific debugging** — three separate bugs all presented identically as
"the function vector does nothing":
1. layer selection collapsing to layer 0 on all-zero ties (`max()` returns the
   first index),
2. a config hash that did not cover **code** version, so stale cached results
   were silently replayed,
3. the residual-scale problem (§3) — the only one that was real science.

Later, a fourth of the same family: stages 06 and 08 still applying the vector
raw, which made every control look identical.

> **Lesson worth saying in the talk:** a result of exactly `+0.00` should be
> treated as a bug report, not a finding. Every stage now prints the quantity it
> depends on, and the config hash covers code version so a stale cache cannot be
> replayed as a result.

---

## 9. Practical notes for editing the deck

Build (**must be `lualatex`** — Telugu needs a Unicode engine; `pdflatex` builds
but renders Telugu blank):

```bash
cd presentation && lualatex main.tex
```

- Telugu inline: `\tel{...}`. Condition labels: `\cond{en}`. Fonts are vendored
  in `presentation/fonts/`, so it builds on any machine.
- Style macros match the group's existing decks: `\src{}` (grey source line),
  `\tight`, `\bridge{}` (boxed narrative bridge).
- Figures come from `presentation/images/` (PNG). The paper uses
  `paper/figures/` (PDF). Stage 09 writes both.

**Traps to avoid:**
- **Do not run `09_report.py` for 12B.** It writes `paper/tables/*.tex` keyed by
  *stage*, not by model, so it would overwrite the 4B tables the deck cites.
  `run_subset.sh` now omits it by default; 12B numbers come from
  `results/05_intervene/gemma3_12b_h200/intervention.json`.
- `paper/tables/tier.tex` is the **degenerate** one-row table from the core-only
  run. The real tier result is `paper/tables/tier_accuracy.tex`. Drop `tier.tex`
  and `tier.pdf`.
- `logit_lens.pdf/png` are **stale** — stage 07 never produced 4B results. Do
  not cite that figure.
- Cells with tiny `n` (romanized word-level tasks keep 2–13 items after the
  competence filter) are shown but carry no weight. Do not build a claim on them.
- Word order is a confound for `en`–`te` vector similarity (English SVO, Telugu
  SOV). `rom`–`tg` is the clean comparison.

## 10. Status

- **4B: complete.** All stages, controls and transfer rerun post-fix, tiers done.
- **12B: pending** — subset run (`subject_gender, extract_verb, negation,
  present_past, extract_person`), all four conditions. Slots into the deck as a
  second column to test whether scale closes the romanization gap.
- **Paper**: not started; due 23:59. ACL template, LuaLaTeX, 4–8 pages, tables
  already generated and `\input{}`-ready.
