"""Dataset loading and item construction from the TSVs in `data/`.

Two invariants the pipeline relies on: conditions are row-aligned, so every
cross-condition comparison is paired; and splits are keyed by content hash, so
appending rows does not reshuffle existing train/valid assignments.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import (
    PART1_ABSTRACTIVE,
    PART1_EXTRACTIVE,
    PART2_LABEL,
    PART2_TASKS,
    conditions_for,
)
from .config import REPO_ROOT, ExpConfig

DATA_DIR = REPO_ROOT / "data"

# Which category supplies the answer, and which the distractors.
EXTRACTIVE_SPEC = {
    "fruit_vs_animal": ("fruit", "animal"),
    "animal_vs_object": ("animal", "object"),
    "colour_vs_animal": ("colour", "animal"),
}
LIST_LEN = 3        # matches the reference's *_of_3 task variants
LIST_SEP = ", "
N_VARIANTS = 3      # distractor draws per target word, to grow the item pool

# Category pair used only to build the unrelated-task function vector control.
UNRELATED_TASK = "vegetable_vs_object"

N_SENTENCES = 400   # Part 2 sentences generated (shared by all Part 2 tasks)


@dataclass(frozen=True)
class Item:
    """One input/output pair in one condition."""

    id: str
    task: str
    condition: str
    tier: str
    inp: str
    out: str

    @property
    def uid(self) -> str:
        """Content hash; editing any field makes a different item."""
        blob = f"{self.task}|{self.condition}|{self.tier}|{self.inp}|{self.out}"
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

    @property
    def pair_uid(self) -> str:
        """Condition-independent identity, for pairing items across conditions."""
        return hashlib.sha1(f"{self.task}|{self.id}".encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# TSV helpers
# --------------------------------------------------------------------------- #

def read_tsv(path: str | Path) -> pd.DataFrame:
    """Read one of the hand-edited TSVs, tolerating missing trailing columns."""
    path = Path(path)
    if not path.is_absolute():
        path = DATA_DIR / path
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    return df[df["id"] != ""].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Part 1: abstractive
# --------------------------------------------------------------------------- #

def _abstractive_items(task: str, condition: str) -> list[Item]:
    df = read_tsv(f"part1/{task}.tsv")
    in_col, out_col = f"{condition}_in", f"{condition}_out"
    items = []
    for _, r in df.iterrows():
        # A row missing any condition is dropped from all of them, to stay aligned.
        if any(r[f"{c}_{s}"] == "" for c in conditions_for(task) for s in ("in", "out")):
            continue
        items.append(Item(r["id"], task, condition, r["tier"], r[in_col], r[out_col]))
    return items


# --------------------------------------------------------------------------- #
# Part 1: extractive
# --------------------------------------------------------------------------- #

def _category_pools() -> tuple[dict[str, list[str]], dict[str, dict]]:
    """(category -> [row id]) and (row id -> row), for rows complete in all conditions."""
    df = read_tsv("part1/categories.tsv")
    pools: dict[str, list[str]] = {}
    rows: dict[str, dict] = {}
    for _, r in df.iterrows():
        if any(r[c] == "" for c in ("en", "te", "rom")):
            continue
        pools.setdefault(r["category"], []).append(r["id"])
        rows[r["id"]] = r.to_dict()
    return pools, rows


def _extractive_plan(task: str, seed: int) -> list[dict]:
    """Decide the word ids for every item, drawn once so all conditions get the same items."""
    pools, rows = _category_pools()
    rng = random.Random(f"{task}|{seed}")

    def tier_of(ids: list[str], primary: str | None = None) -> str:
        tiers = [rows[i]["tier"] for i in ids]
        if "ambiguous" in tiers:
            return "ambiguous"
        if primary and rows[primary]["tier"] != "core":
            return rows[primary]["tier"]
        return "core" if all(t == "core" for t in tiers) else "complex"

    plan = []
    if task == "choose_last":
        pool = [i for ids in pools.values() for i in ids]
        for n in range(len(pool) * N_VARIANTS // LIST_LEN):
            picks = rng.sample(pool, LIST_LEN)
            plan.append({"id": f"cl-{n + 1:04d}", "words": picks,
                         "answer": picks[-1], "tier": tier_of(picks)})
        return plan

    target_cat, distractor_cat = (
        ("vegetable", "object") if task == UNRELATED_TASK else EXTRACTIVE_SPEC[task]
    )
    for tid in pools[target_cat]:
        for v in range(N_VARIANTS):
            picks = rng.sample(pools[distractor_cat], LIST_LEN - 1)
            words = [tid] + picks
            rng.shuffle(words)
            plan.append({"id": f"{tid}-v{v + 1}", "words": words,
                         "answer": tid, "tier": tier_of(words, primary=tid)})
    return plan


def _extractive_items(task: str, condition: str, seed: int) -> list[Item]:
    _, rows = _category_pools()
    return [
        Item(p["id"], task, condition, p["tier"],
             LIST_SEP.join(rows[w][condition] for w in p["words"]),
             rows[p["answer"]][condition])
        for p in _extractive_plan(task, seed)
    ]


# --------------------------------------------------------------------------- #
# Part 2: sentence generation
# --------------------------------------------------------------------------- #

# The Telugu locative: a suffix in Telugu script, a separate word in Latin script.
LOCATIVE_TE = "లో"
LOCATIVE_LATIN = " lo"
TELUGU_RANGE = range(0x0C00, 0x0C80)

TENSE_LABELS = {"past": "past", "present": "present", "future": "future"}
GENDER_LABELS = {"m": "he", "f": "she"}
NEGATION_LABELS = {"aff": "no", "neg": "yes"}   # "yes" means the sentence IS negated


def _is_telugu(s: str) -> bool:
    return any(ord(ch) in TELUGU_RANGE for ch in s)


def _realise(order: str, fillers: dict[str, str]) -> str:
    """Turn a '+'-joined template order into a sentence.

    Tokens are slot names, `lit:<text>` literals, or `place.loc` (place + locative).
    """
    parts = []
    for tok in order.split("+"):
        tok = tok.strip()
        if tok.startswith("lit:"):
            parts.append(tok[4:])
        elif tok == "place.loc":
            place = fillers["place"]
            parts.append(place + (LOCATIVE_TE if _is_telugu(place) else LOCATIVE_LATIN))
        else:
            parts.append(fillers[tok])
    return " ".join(p for p in parts if p).replace("  ", " ").strip()


def generate_sentences(seed: int = 42, n: int = N_SENTENCES) -> pd.DataFrame:
    """Build Part 2 sentences, balanced across (template, tense, polarity, gender).

    Tenglish is built on the romanized base, so `te` vs `rom` isolates script
    and `rom` vs `tg` isolates code-mixing.
    """
    lex = read_tsv("part2/lexicon.tsv")
    verbs = read_tsv("part2/verbs.tsv")
    tpls = read_tsv("part2/templates.tsv")

    by_slot = {s: g.to_dict("records") for s, g in lex.groupby("slot")}
    rng = random.Random(f"sentences|{seed}")

    combos = [
        (t, tense, pol, g)
        for _, t in tpls.iterrows()
        for tense in ("past", "present", "future")
        for pol in ("aff", "neg")
        for g in ("m", "f")
    ]
    rng.shuffle(combos)

    rows = []
    while len(rows) < n:
        tpl, tense, pol, gender = combos[len(rows) % len(combos)]
        vforms = verbs[(verbs["tense"] == tense) & (verbs["polarity"] == pol)
                       & (verbs["gender"] == gender)]
        verb = vforms.iloc[rng.randrange(len(vforms))]
        has_person = "person" in tpl["te_order"]

        src = {
            "person": rng.choice(by_slot["person_m" if gender == "m" else "person_f"]),
            "place": rng.choice(by_slot["place"]),
            "food": rng.choice(by_slot["food"]),
            "time": rng.choice(by_slot["time"]),
        }
        mixable = [s for s in tpl["mixable"].split("+") if s]
        mix_level = (len(rows) % min(3, len(mixable))) + 1 if mixable else 0
        switched = set(mixable[:mix_level])

        row = {
            "id": f"sen-{len(rows) + 1:04d}",
            "tier": tpl["tier"],
            "template": tpl["id"],
            "mix_level": str(mix_level),
            "tense": tense,
            "polarity": pol,
            "gender": gender,
        }
        for cond in ("en", "te", "rom", "tg"):
            base = "rom" if cond == "tg" else cond
            fill = {
                slot: rec["en" if (cond == "tg" and slot in switched) else base]
                for slot, rec in src.items()
            }
            # Never switched to English: the label tasks read Telugu verb morphology.
            fill["verb"] = verb[base]
            if not has_person:
                fill["person"] = ""
            row[f"{cond}_text"] = _realise(tpl[f"{base}_order"], fill)
            for slot in ("person", "place", "food", "verb"):
                row[f"{cond}_{slot}"] = fill[slot]
        rows.append(row)

    return pd.DataFrame(rows)


def write_sentences(seed: int = 42, n: int = N_SENTENCES) -> Path:
    path = DATA_DIR / "part2" / "sentences.tsv"
    generate_sentences(seed=seed, n=n).to_csv(path, sep="\t", index=False)
    return path


def _part2_items(task: str, condition: str, seed: int) -> list[Item]:
    path = DATA_DIR / "part2" / "sentences.tsv"
    if not path.exists():
        write_sentences(seed=seed)
    df = read_tsv(path)
    tpls = read_tsv("part2/templates.tsv").set_index("id")

    items = []
    for _, r in df.iterrows():
        if task not in tpls.loc[r["template"], "tasks"].split("+"):
            continue
        # Telugu marks subject gender only on affirmative verbs, so negated
        # sentences are unanswerable here rather than wrong.
        if task == "subject_gender" and r["polarity"] != "aff":
            continue

        if task in PART2_LABEL:
            answer = {
                "tense": TENSE_LABELS[r["tense"]],
                "negation": NEGATION_LABELS[r["polarity"]],
                "subject_gender": GENDER_LABELS[r["gender"]],
            }[task]
        else:
            answer = r[f"{condition}_{task.removeprefix('extract_')}"]

        text = r[f"{condition}_text"]
        if text == "" or answer == "":
            continue
        items.append(Item(r["id"], task, condition, r["tier"], text, answer))
    return items


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def load_items(task: str, condition: str, exp: ExpConfig) -> list[Item]:
    """All items for one (task, condition) cell, after tier exclusion."""
    if task in PART1_ABSTRACTIVE:
        items = _abstractive_items(task, condition)
    elif task in PART1_EXTRACTIVE or task == UNRELATED_TASK:
        items = _extractive_items(task, condition, exp.seed)
    elif task in PART2_TASKS:
        items = _part2_items(task, condition, exp.seed)
    else:
        raise KeyError(f"unknown task: {task}")

    exclude = set(exp.data.get("exclude_tiers", []))
    items = [i for i in items if i.tier not in exclude]
    cap = exp.data.get("max_items_per_cell")
    if cap:
        items = items[: int(cap)]
    return items


@dataclass
class Split:
    """Train pool for building n-shot prompts; valid pool for evaluation."""

    train: list[Item]
    valid: list[Item]

    def __len__(self) -> int:
        return len(self.train) + len(self.valid)


def split_items(items: list[Item], exp: ExpConfig) -> Split:
    """Deterministic split, ordered by a seeded hash of each item's own identity.

    Keying on the item rather than its position keeps existing assignments
    stable when rows are appended, so cached results stay valid. The train pool
    is floored at `min_train` so a small cell still supports `n_shots` prompts.
    """
    def key(i: Item) -> str:
        return hashlib.sha1(f"{exp.seed}|{i.pair_uid}".encode()).hexdigest()

    ordered = sorted(items, key=key)

    min_train = int(exp.data.get("min_train", 2 * exp.n_shots))
    min_train = min(min_train, max(1, len(ordered) // 2))
    n_valid = int(round(len(ordered) * float(exp.data["test_split"])))
    n_valid = min(n_valid, max(0, len(ordered) - min_train))
    return Split(train=ordered[n_valid:], valid=ordered[:n_valid])


def headline(items: list[Item], exp: ExpConfig) -> list[Item]:
    """Restrict to the tiers used for headline numbers (default: core only)."""
    tiers = set(exp.data.get("headline_tiers", ["core"]))
    return [i for i in items if i.tier in tiers]
