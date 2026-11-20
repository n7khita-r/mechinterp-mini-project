# Dataset

Everything here is hand-editable TSV. No code lives in this directory. Open the files in a
spreadsheet or a text editor, append rows, save. The pipeline picks the changes up on the next run.

Tab-separated, UTF-8, one header row. Do not use commas as separators (Telugu text contains them) and
do not quote fields.

---

## 1. Conditions

Four conditions. Every file carries parallel columns for each condition that applies to it.

| Condition | Column prefix | Example |
|---|---|---|
| English | `en_` | `I was very tired yesterday` |
| Telugu, native script | `te_` | `నిన్న నేను చాలా అలసిపోయాను` |
| Telugu, romanized | `rom_` | `ninna nenu chala alasipoyanu` |
| Tenglish (code-mixed) | `tg_` | `ninna nenu chala tired ayyanu` |

Word-level tasks (Part 1) have no `tg_` column: a single word cannot be code-mixed. Code-mixing is
studied at sentence level only (Part 2).

---

## 2. Romanization convention

The `rom_` columns use **natural-usage romanization** — the way Telugu speakers actually type Telugu in
Latin script in chat and search — not a reversible scholarly scheme like ITRANS or IAST.

This is a deliberate choice: the object of study is the text people really write. It is **lossy**, and the
losses are themselves candidate explanations for any degradation we measure, so they are listed here and
discussed in the report.

### Vowels — length is **not** marked

| Telugu | Latin | | Telugu | Latin |
|---|---|---|---|---|
| అ | a | | ఆ | a |
| ఇ | i | | ఈ | i |
| ఉ | u | | ఊ | u |
| ఎ | e | | ఏ | e |
| ఒ | o | | ఓ | o |
| ఐ | ai | | ఔ | au |

So చాలా → `chala`, నేను → `nenu`. Vowel length is **lost**.

### Consonants — retroflex/dental contrast is **not** marked

| Telugu | Latin | Telugu | Latin | Telugu | Latin |
|---|---|---|---|---|---|
| క | k | ఖ | kh | గ | g |
| ఘ | gh | చ | ch | ఛ | chh |
| జ | j | ఝ | jh | ట | t |
| ఠ | th | డ | d | ఢ | dh |
| ణ | n | త | t | థ | th |
| ద | d | ధ | dh | న | n |
| ప | p | ఫ | ph | బ | b |
| భ | bh | మ | m | య | y |
| ర | r | ల | l | ళ | l |
| వ | v | శ | sh | ష | sh |
| స | s | హ | h | ఱ | r |
| క్ష | ksh | జ్ఞ | jn | ం | m / n |

So both ట and త → `t`; both ద and డ → `d`; both ల and ళ → `l`; both శ and ష → `sh`.
Retroflex/dental and the ల/ళ contrast are **lost**.

Gemination is kept by doubling: పిల్లి → `pilli`, కుక్క → `kukka`.

### Known consequences

These collapses make some distinct Telugu words romanize identically. Such rows are marked
`tier = ambiguous` with an explanation in `notes`, and are excluded from headline numbers. Example:
ఇల్లు (house) and ఇళ్ళు (houses) both romanize to `illu`, so that singular→plural pair is unanswerable in
the `rom` condition. Finding more of these is useful — add them as `ambiguous`.

---

## 3. The `tier` column

`tier` controls which rows enter which analysis. Headline results use `core` only; the other tiers give
the complexity/register degradation curve.

| Tier | Meaning |
|---|---|
| `core` | Everyday vocabulary, unambiguous. The headline set. |
| `complex` | Morphologically heavy or irregular — multi-suffix verbs, irregular plurals, sandhi. |
| `sanskritized` | తత్సమ / Sanskrit-derived learned register (సత్యం, అసాధ్యం) rather than native తద్భవ words. |
| `rare` | Low-frequency Telugu, unlikely to be well represented in pretraining. |
| `literary_en` | Low-frequency, latinate English (`ephemeral`, `laconic`). Tests whether degradation is about Telugu specifically or about rarity in general — the English-side control for the Telugu tiers. |
| `ambiguous` | Genuinely has more than one correct answer, or collapses under romanization. Excluded from headline numbers, reported separately. |

Exactly one tier per row. If unsure, use `core` and say why in `notes`.

---

## 4. Files

### Part 1 — word-level tasks

Four abstractive tasks, one file each. The answer is a new word, not a word from the input.

| File | Task |
|---|---|
| `part1/antonym.tsv` | big → small |
| `part1/present_past.tsv` | eat → ate |
| `part1/singular_plural.tsv` | cat → cats |
| `part1/next_item.tsv` | Monday → Tuesday |

Columns: `id  tier  en_in  en_out  te_in  te_out  rom_in  rom_out  notes`

The four **extractive** Part-1 tasks (`fruit_vs_animal`, `animal_vs_object`, `colour_vs_animal`,
`choose_last`) are **not** stored as item files. They are built by the loader from the word pools in
`part1/categories.tsv`, deterministically under the experiment seed. Add a word to a category and it
appears in every extractive task that uses that category — you never hand-write pairs.

`part1/categories.tsv` columns: `id  tier  category  en  te  rom  notes`
Categories in use: `fruit`, `animal`, `object`, `colour`, `vegetable`
(`vegetable` is only used to build the unrelated-task function vector control.)

### Part 2 — sentence-level tasks

Sentences carry person, place, food and verb information. All seven Part-2 tasks share the same
sentences, so differences between their function vectors come only from the task.

| File | Contents |
|---|---|
| `part2/lexicon.tsv` | slot fillers: persons, places, foods, time words |
| `part2/verbs.tsv` | verb forms: lemma × tense × polarity × subject gender |
| `part2/templates.tsv` | slot orderings, and which slots may switch to English |
| `part2/sentences.tsv` | **generated** — regenerate with `make sentences`, do not hand-edit |

`lexicon.tsv`: `id  tier  slot  en  te  rom  notes` — `slot` ∈ `person_m`, `person_f`, `place`, `food`, `time`
`verbs.tsv`: `id  tier  lemma  tense  polarity  gender  en  te  rom  notes`
 — `tense` ∈ `past`, `present`, `future`; `polarity` ∈ `aff`, `neg`; `gender` ∈ `m`, `f`
`templates.tsv`: `id  tier  slot_order  mixable  notes` — `slot_order` is a `+`-joined slot list,
`mixable` is a `+`-joined subset of those slots allowed to appear in English in the Tenglish condition.

`sentences.tsv` is regenerated from the three files above, so **add words and templates, not sentences**.

---

## 5. Rules for adding rows

1. **Never change an existing `id`, and never reuse one.** Append with a fresh id following the file's
   prefix (`ant-061`, `lex-042`, …). Ids appear in cached results and in the paper's appendix.
2. **Fill every condition column**, or leave the whole row out. A row missing `te_out` is dropped from all
   conditions, not just Telugu, so the conditions stay row-aligned and comparable.
3. **Lowercase the `rom_` and `en_` columns** for Part 2, so names cannot be identified by capitalisation.
   Part-1 proper nouns (`Monday`, `January`) keep their natural case.
4. **One correct answer.** If a second answer is defensible, mark `tier = ambiguous` and say so in `notes`.
5. **Telugu answers should be single words** where the task is word-level — the metric scores the answer's
   first token, and multi-word answers make that meaningless.
6. Run `make lint` after editing. It checks for duplicate ids, missing columns, unknown tiers, Telugu
   characters sitting in a `rom_` column, and romanization outside the convention above.

Rows whose Telugu needs a second pair of eyes are flagged `CHECK` in `notes`. Clear the flag once
verified.
