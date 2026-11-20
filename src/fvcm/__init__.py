"""Function vectors under code-mixing: Telugu, romanized Telugu and Tenglish.

Implementation follows Todd et al., "Function Vectors in Large Language Models"
(ICLR 2024) and its reference code at github.com/ericwtodd/function_vectors.
Deviations from the reference are listed in CONTEXT.md and flagged in-line.
"""

__version__ = "0.1.0"

CONDITIONS = ("en", "te", "rom", "tg")
PART1_CONDITIONS = ("en", "te", "rom")          # a single word cannot be code-mixed
PART2_CONDITIONS = ("en", "te", "rom", "tg")

PART1_ABSTRACTIVE = ("antonym", "present_past", "singular_plural", "next_item")
PART1_EXTRACTIVE = ("fruit_vs_animal", "animal_vs_object", "colour_vs_animal", "choose_last")
PART1_TASKS = PART1_ABSTRACTIVE + PART1_EXTRACTIVE

PART2_COPY = ("extract_person", "extract_place", "extract_food", "extract_verb")
PART2_LABEL = ("tense", "negation", "subject_gender")
PART2_TASKS = PART2_COPY + PART2_LABEL

ALL_TASKS = PART1_TASKS + PART2_TASKS

TIERS = ("core", "complex", "sanskritized", "rare", "literary_en", "ambiguous")


def conditions_for(task: str) -> tuple:
    """Conditions a task is defined for."""
    if task in PART1_TASKS:
        return PART1_CONDITIONS
    if task in PART2_TASKS:
        return PART2_CONDITIONS
    raise KeyError(f"unknown task: {task}")
