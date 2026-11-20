#!/usr/bin/env python3
"""Check the hand-edited TSVs under data/: ids, tiers, script mixing, romanization,
row alignment across conditions, answer tokenization. Exit 1 on any error.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fvcm import PART1_ABSTRACTIVE, PART2_TASKS, TIERS  # noqa: E402
from fvcm.data import read_tsv  # noqa: E402

TELUGU = range(0x0C00, 0x0C80)
# Letters the romanization convention can produce, plus space/hyphen. `w` and
# `z` are allowed for proper nouns (vijayawada, nizamabad).
ROM_OK = set("abcdefghijklmnoprstuvwyzh -")

errors: list[str] = []
warnings: list[str] = []


def err(where: str, msg: str) -> None:
    errors.append(f"{where}: {msg}")


def warn(where: str, msg: str) -> None:
    warnings.append(f"{where}: {msg}")


def has_telugu(s: str) -> bool:
    return any(ord(c) in TELUGU for c in s)


def has_latin(s: str) -> bool:
    return any("LATIN" in unicodedata.name(c, "") for c in s)


def check_ids(df, where: str, prefix_hint: str | None = None) -> None:
    dupes = [i for i, n in Counter(df["id"]).items() if n > 1]
    if dupes:
        err(where, f"duplicate ids: {sorted(dupes)[:10]}")
    blank = int((df["id"] == "").sum())
    if blank:
        err(where, f"{blank} row(s) with a blank id")


def check_tiers(df, where: str) -> None:
    bad = sorted(set(df["tier"]) - set(TIERS))
    if bad:
        err(where, f"unknown tier(s) {bad}; allowed: {list(TIERS)}")


def check_scripts(df, where: str, cols: list[str]) -> None:
    for col in cols:
        if col not in df.columns:
            continue
        lang = col.split("_")[0]
        for _, r in df.iterrows():
            v = r[col]
            if not v:
                continue
            tag = f"{where} {r['id']}.{col}"
            if lang == "te" and not has_telugu(v):
                err(tag, f"te_ column has no Telugu characters: {v!r}")
            if lang in ("rom", "en") and has_telugu(v):
                err(tag, f"{lang}_ column contains Telugu script: {v!r}")
            if lang == "rom":
                stray = sorted(set(v.lower()) - ROM_OK)
                if stray:
                    warn(tag, f"characters outside the romanization convention: {stray}")
                if v != v.lower():
                    warn(tag, f"romanization is not lowercase: {v!r}")


def check_part1() -> None:
    for task in PART1_ABSTRACTIVE:
        path = REPO_ROOT / "data" / "part1" / f"{task}.tsv"
        if not path.exists():
            err(task, "file missing")
            continue
        df = read_tsv(path)
        where = f"part1/{task}.tsv"
        expected = ["id", "tier", "en_in", "en_out", "te_in", "te_out",
                    "rom_in", "rom_out", "notes"]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            err(where, f"missing column(s): {missing}")
            continue
        check_ids(df, where)
        check_tiers(df, where)
        check_scripts(df, where, ["en_in", "en_out", "te_in", "te_out",
                                  "rom_in", "rom_out"])

        for _, r in df.iterrows():
            tag = f"{where} {r['id']}"
            blanks = [c for c in expected[2:-1] if r[c] == ""]
            if blanks:
                err(tag, f"incomplete row, blank {blanks} -- this drops the item "
                         "from EVERY condition, not just that one")
            # Only the answer's first token is scored.
            for c in ("en_out", "te_out", "rom_out"):
                if " " in r[c]:
                    warn(tag, f"{c} is multi-word ({r[c]!r}); only its first token "
                              "is scored")
            # Identical under romanization => unanswerable there.
            if r["rom_in"] and r["rom_in"] == r["rom_out"] and r["tier"] != "ambiguous":
                err(tag, f"rom_in == rom_out ({r['rom_in']!r}) but tier is "
                         f"{r['tier']!r}; this is a romanization collapse and "
                         "must be tier 'ambiguous'")
            if r["te_in"] == r["te_out"] and r["te_in"] and r["tier"] != "ambiguous":
                warn(tag, "te_in == te_out but tier is not 'ambiguous'")

        n_core = int((df["tier"] == "core").sum())
        if n_core < 40:
            warn(where, f"only {n_core} core rows; evaluation power will be thin "
                        "(see the sizing note in data/README.md)")

    # categories
    path = REPO_ROOT / "data" / "part1" / "categories.tsv"
    df = read_tsv(path)
    where = "part1/categories.tsv"
    check_ids(df, where)
    check_tiers(df, where)
    check_scripts(df, where, ["en", "te", "rom"])
    counts = Counter(df["category"])
    for cat in ("fruit", "animal", "object", "colour", "vegetable"):
        if counts[cat] == 0:
            err(where, f"category {cat!r} is empty but tasks depend on it")
        elif counts[cat] < 10:
            warn(where, f"category {cat!r} has only {counts[cat]} words")
    seen = defaultdict(set)
    for _, r in df.iterrows():
        seen[r["en"]].add(r["category"])
    for word, cats in seen.items():
        if len(cats) > 1:
            warn(where, f"{word!r} appears in categories {sorted(cats)}; "
                        "extractive items built from these are ambiguous")


def check_part2() -> None:
    d = REPO_ROOT / "data" / "part2"
    lex, verbs, tpls = (read_tsv(d / f) for f in
                        ("lexicon.tsv", "verbs.tsv", "templates.tsv"))

    check_ids(lex, "part2/lexicon.tsv")
    check_tiers(lex, "part2/lexicon.tsv")
    check_scripts(lex, "part2/lexicon.tsv", ["en", "te", "rom"])
    slots = {"person_m", "person_f", "place", "food", "time"}
    bad = sorted(set(lex["slot"]) - slots)
    if bad:
        err("part2/lexicon.tsv", f"unknown slot(s) {bad}; allowed: {sorted(slots)}")
    for s in sorted(slots):
        n = int((lex["slot"] == s).sum())
        if n == 0:
            err("part2/lexicon.tsv", f"slot {s!r} has no entries")
        elif n < 5:
            warn("part2/lexicon.tsv", f"slot {s!r} has only {n} entries")
    for _, r in lex.iterrows():
        if r["rom"] != r["rom"].lower() or r["en"] != r["en"].lower():
            err(f"part2/lexicon.tsv {r['id']}",
                "Part 2 must be lowercase so names cannot be found by capitalisation")

    check_ids(verbs, "part2/verbs.tsv")
    check_tiers(verbs, "part2/verbs.tsv")
    check_scripts(verbs, "part2/verbs.tsv", ["en", "te", "rom"])
    for col, allowed in (("tense", {"past", "present", "future"}),
                         ("polarity", {"aff", "neg"}),
                         ("gender", {"m", "f"})):
        bad = sorted(set(verbs[col]) - allowed)
        if bad:
            err("part2/verbs.tsv", f"unknown {col} value(s) {bad}")
    # An incomplete paradigm starves a sentence-generation cell.
    for lemma, g in verbs.groupby("lemma"):
        cells = {(r["tense"], r["polarity"], r["gender"]) for _, r in g.iterrows()}
        if len(cells) != 12:
            err("part2/verbs.tsv", f"lemma {lemma!r} has {len(cells)}/12 "
                                   "tense x polarity x gender forms")

    check_ids(tpls, "part2/templates.tsv")
    check_tiers(tpls, "part2/templates.tsv")
    known_tasks = set(PART2_TASKS)
    for _, r in tpls.iterrows():
        tag = f"part2/templates.tsv {r['id']}"
        bad = sorted({t for t in r["tasks"].split("+") if t} - known_tasks)
        if bad:
            err(tag, f"unknown task(s) {bad}")
        for col in ("en_order", "te_order", "rom_order"):
            toks = [t for t in r[col].split("+") if t]
            for t in toks:
                if not (t.startswith("lit:") or t in
                        {"time", "person", "place", "food", "verb", "place.loc"}):
                    err(tag, f"{col}: unknown token {t!r}")
            if "verb" not in toks:
                err(tag, f"{col}: no verb slot")
        for s in (x for x in r["mixable"].split("+") if x):
            if s not in r["te_order"]:
                warn(tag, f"mixable lists {s!r}, which the Telugu order does not use")
            if s == "verb":
                err(tag, "verb must not be mixable: tense, negation and subject "
                         "gender are all read off Telugu verb morphology")
        tasks = set(r["tasks"].split("+"))
        if "tense" in tasks and "time" in r["te_order"]:
            err(tag, "declares the tense task but the template has a time slot, "
                     "which leaks the answer")
        if "subject_gender" in tasks and "person" in r["te_order"]:
            err(tag, "declares the subject_gender task but the template has a "
                     "person slot, whose name leaks the answer")

    sentences = d / "sentences.tsv"
    if not sentences.exists():
        warn("part2/sentences.tsv", "missing; run `make sentences`")


def main() -> int:
    check_part1()
    check_part2()

    for w in warnings:
        print(f"  warn   {w}")
    for e in errors:
        print(f"  ERROR  {e}")
    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    if not errors:
        print("data looks good" + (" (warnings are advisory)" if warnings else ""))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
