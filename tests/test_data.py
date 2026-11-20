"""Tests for the data layer.

The invariants here are the ones that would silently corrupt results rather than
crash: conditions drifting out of alignment (which would break the paired
comparison), splits reshuffling when rows are appended (which would invalidate
cached results), and the control tasks leaking their answers.
"""

from __future__ import annotations

import pytest

from fvcm import PART1_TASKS, PART2_TASKS, conditions_for
from fvcm.config import ExpConfig
from fvcm.data import Item, headline, load_items, split_items


@pytest.fixture(scope="module")
def exp():
    return ExpConfig.load("configs/exp/default.yaml")


ALL = list(PART1_TASKS) + list(PART2_TASKS)


@pytest.mark.parametrize("task", ALL)
def test_conditions_are_row_aligned(task, exp):
    """Every condition must contain exactly the same item ids, with the same tier.

    Cross-condition comparisons are paired (McNemar / paired bootstrap on the
    same items), which is where the statistical power comes from. If one
    condition silently has a different item set, the pairing is meaningless.
    """
    per_cond = {}
    for cond in conditions_for(task):
        items = load_items(task, cond, exp)
        per_cond[cond] = {i.id: i.tier for i in items}

    first = conditions_for(task)[0]
    for cond in conditions_for(task)[1:]:
        assert per_cond[cond].keys() == per_cond[first].keys(), (
            f"{task}: {cond} and {first} have different item ids")
        assert per_cond[cond] == per_cond[first], (
            f"{task}: an item has different tiers in {cond} and {first}")


@pytest.mark.parametrize("task", ALL)
def test_every_cell_has_enough_train_items(task, exp):
    """Prompt construction samples n_shots items without replacement."""
    for cond in conditions_for(task):
        sp = split_items(headline(load_items(task, cond, exp), exp), exp)
        assert len(sp.train) >= exp.n_shots, (
            f"{task}/{cond}: train pool {len(sp.train)} < n_shots {exp.n_shots}")
        assert len(sp.valid) > 0, f"{task}/{cond}: no evaluation items"


@pytest.mark.parametrize("task", ALL)
def test_no_item_answers_itself(task, exp):
    """An item whose input equals its output is unanswerable, not easy."""
    for cond in conditions_for(task):
        for i in load_items(task, cond, exp):
            assert i.inp != i.out, f"{task}/{cond}/{i.id}: input == output"
            assert i.inp and i.out, f"{task}/{cond}/{i.id}: blank field"


def test_split_is_stable_when_rows_are_appended(exp):
    """Appending to a TSV must not move existing items between train and valid.

    Items are ordered by content hash rather than file position precisely so
    that extending the dataset does not invalidate every cached result.
    """
    base = load_items("antonym", "en", exp)
    before = split_items(base, exp)

    extra = base + [Item(f"zz-{n:03d}", "antonym", "en", "core",
                         f"newword{n}", f"newanswer{n}") for n in range(20)]
    after = split_items(extra, exp)

    before_valid = {i.id for i in before.valid}
    after_valid = {i.id for i in after.valid}
    original_ids = {i.id for i in base}
    moved = {i for i in original_ids
             if (i in before_valid) != (i in after_valid)}
    # A few items shift because the split is proportional; the point is that it
    # is not a wholesale reshuffle.
    assert len(moved) / len(original_ids) < 0.25, (
        f"{len(moved)}/{len(original_ids)} existing items changed split")


def test_choose_last_answer_is_the_last_element(exp):
    """The structural control must actually be structural."""
    for cond in conditions_for("choose_last"):
        for i in load_items("choose_last", cond, exp):
            assert i.inp.split(", ")[-1] == i.out


def test_extractive_answer_is_present_in_the_input(exp):
    """Extractive tasks copy; the answer has to be one of the listed words."""
    for task in ("fruit_vs_animal", "animal_vs_object", "colour_vs_animal"):
        for cond in conditions_for(task):
            for i in load_items(task, cond, exp):
                assert i.out in i.inp.split(", "), f"{task}/{cond}/{i.id}"


def test_tense_task_has_no_time_word(exp):
    """The tense answer must come from the verb, not a lexical time cue.

    Time words are the obvious shortcut, so templates declaring the tense task
    omit the time slot. This checks the data actually honours that.
    """
    from fvcm.data import read_tsv

    lex = read_tsv("part2/lexicon.tsv")
    time_words = {r["rom"] for _, r in lex.iterrows() if r["slot"] == "time"}
    time_words |= {r["te"] for _, r in lex.iterrows() if r["slot"] == "time"}

    for cond in ("te", "rom"):
        for i in load_items("tense", cond, exp):
            tokens = set(i.inp.split())
            assert not (tokens & time_words), (
                f"tense/{cond}/{i.id} contains a time word: {tokens & time_words}")


def test_subject_gender_is_affirmative_only(exp):
    """Telugu negative verbs do not mark subject gender, so those are excluded."""
    from fvcm.data import read_tsv

    sentences = read_tsv("part2/sentences.tsv").set_index("id")
    ids = {i.id for i in load_items("subject_gender", "te", exp)}
    assert ids, "no subject_gender items"
    for sid in ids:
        assert sentences.loc[sid, "polarity"] == "aff", (
            f"{sid} is negated but is in the subject_gender set; Telugu negative "
            "past/present are gender-neutral and the negative future romanizes "
            "identically for both genders")


def test_subject_gender_has_no_person_name(exp):
    """The gender answer must come from the verb, not from the subject's name."""
    from fvcm.data import read_tsv

    lex = read_tsv("part2/lexicon.tsv")
    names = {r["rom"] for _, r in lex.iterrows() if r["slot"].startswith("person")}
    names |= {r["te"] for _, r in lex.iterrows() if r["slot"].startswith("person")}

    for cond in ("te", "rom"):
        for i in load_items("subject_gender", cond, exp):
            tokens = set(i.inp.split())
            assert not (tokens & names), (
                f"subject_gender/{cond}/{i.id} names a person: {tokens & names}")


def test_tenglish_is_latin_script_throughout(exp):
    """Tenglish is built on the romanized base, not the Telugu-script one.

    Real Tenglish is written in Latin script ("ninna nenu chala tired ayyanu").
    Holding script constant between `rom` and `tg` is also what lets `te` vs
    `rom` isolate script and `rom` vs `tg` isolate code-mixing.
    """
    telugu = range(0x0C00, 0x0C80)
    for task in PART2_TASKS:
        for i in load_items(task, "tg", exp):
            assert not any(ord(c) in telugu for c in i.inp), (
                f"{task}/tg/{i.id} contains Telugu script: {i.inp!r}")


def test_tenglish_differs_from_romanized(exp):
    """If no slot were switched, `tg` would just be a copy of `rom`."""
    rom = {i.id: i.inp for i in load_items("extract_place", "rom", exp)}
    tg = {i.id: i.inp for i in load_items("extract_place", "tg", exp)}
    differing = sum(1 for k in rom if rom[k] != tg[k])
    assert differing > 0.5 * len(rom), (
        f"only {differing}/{len(rom)} Tenglish sentences differ from romanized")
