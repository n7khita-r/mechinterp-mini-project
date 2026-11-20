"""Prompt construction.

Byte-identical in form to the reference implementation:

    Q:{input}\\nA:{output}\\n\\n   ... repeated n_shots times ...
    Q:{query}\\nA:

Note there is no space after either colon -- that is what the reference does,
and the answer string is tokenized *in context* (append to the prompt, diff the
token ids), so a spacing change would silently change which token is scored.
Kept as-is for comparability with the published numbers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import ExpConfig
from .data import Item


@dataclass
class Prompt:
    """A built n-shot prompt and the query it asks about."""

    text: str
    query: Item
    shots: list[Item]
    shuffled: bool


def build(query: Item, shots: list[Item], exp: ExpConfig,
          shuffle_labels: bool = False, rng: random.Random | None = None) -> Prompt:
    """Assemble an n-shot prompt ending in the output prefix.

    With `shuffle_labels`, the demonstration outputs are permuted so the mapping
    cannot be inferred from the examples. That is the corrupted run used by
    causal mediation: whatever ability survives it cannot have come from reading
    the demonstrations.
    """
    pin, pout = exp.prefixes["input"], exp.prefixes["output"]
    sin, sout = exp.separators["input"], exp.separators["output"]

    outs = [s.out for s in shots]
    if shuffle_labels and len(outs) > 1:
        rng = rng or random.Random(0)
        # A derangement where possible, so no demonstration keeps its own label.
        for _ in range(20):
            shuffled = outs[:]
            rng.shuffle(shuffled)
            if all(a != b for a, b in zip(outs, shuffled)):
                outs = shuffled
                break
        else:
            outs = outs[1:] + outs[:1]

    text = exp.prefixes.get("instructions", "")
    for shot, out in zip(shots, outs):
        text += f"{pin}{shot.inp}{sin}{pout}{out}{sout}"
    text += f"{pin}{query.inp}{sin}{pout}"
    return Prompt(text=text, query=query, shots=shots, shuffled=shuffle_labels)


def sample_shots(train: list[Item], n_shots: int, rng: random.Random,
                 exclude: Item | None = None) -> list[Item]:
    """Draw n_shots demonstrations without replacement, never the query itself."""
    pool = [i for i in train if exclude is None or i.id != exclude.id]
    if len(pool) < n_shots:
        raise ValueError(
            f"train pool has {len(pool)} items but n_shots={n_shots}; "
            "add rows to the TSV or lower data.test_split"
        )
    return rng.sample(pool, n_shots)


def zero_shot(query: Item, exp: ExpConfig) -> Prompt:
    """A prompt with no demonstrations -- what the function vector is added to."""
    pin, pout = exp.prefixes["input"], exp.prefixes["output"]
    sin = exp.separators["input"]
    text = exp.prefixes.get("instructions", "") + f"{pin}{query.inp}{sin}{pout}"
    return Prompt(text=text, query=query, shots=[], shuffled=False)


def answer_token_ids(prompt_text: str, answer: str, tokenizer) -> list[int]:
    """Token ids of `answer` as it is tokenized *in the context of* the prompt.

    Same contextual-diff trick as the reference's `get_answer_id`: tokenize the
    prompt, tokenize prompt+answer, and take the tail. Doing it this way matters
    for Telugu, where the tokenizer's treatment of a word can depend on what
    precedes it.
    """
    src = tokenizer(prompt_text, add_special_tokens=False).input_ids
    tgt = tokenizer(prompt_text + answer, add_special_tokens=False).input_ids
    if len(tgt) <= len(src) or tgt[: len(src)] != src:
        # Rare: the tokenizer re-segments across the boundary. Fall back to
        # encoding the answer alone, which is less faithful but never crashes.
        return tokenizer(answer, add_special_tokens=False).input_ids
    return tgt[len(src):]
