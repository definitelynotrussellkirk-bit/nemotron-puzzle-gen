#!/usr/bin/env python3
"""Generate Roman numeral training examples focused on subtractive forms.

The model forgets IV=4, IX=9, XL=40, XC=90 patterns. This generator
produces extra examples where the QUERY answer contains subtractive forms,
and ensures subtractive forms also appear in the provided examples.

Output: data/number_conversion/pool/generated/roman_subtractive.jsonl

Numbers whose Roman form contains subtractive forms (in range 1-100):
  IV: 4, 14, 24, 34, 44, 54, 64, 74, 84, 94
  IX: 9, 19, 29, 39, 49, 59, 69, 79, 89, 99
  XL: 40, 41, 42, 43, 44, 45, 46, 47, 48, 49
  XC: 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100

This generator also ensures at least one example in each puzzle contains
a subtractive form, reinforcing the pattern before the model has to apply it.
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

# ── Roman numeral conversion ──────────────────────────────────────────────

_ROMAN_TABLE = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]

SUBTRACTIVE_FORMS = {"IV", "IX", "XL", "XC", "CD", "CM"}


def to_roman(n: int) -> str:
    parts = []
    for value, numeral in _ROMAN_TABLE:
        while n >= value:
            parts.append(numeral)
            n -= value
    return "".join(parts)


def has_subtractive(n: int) -> bool:
    roman = to_roman(n)
    return any(sf in roman for sf in SUBTRACTIVE_FORMS)


def greedy_decompose(n: int) -> list[tuple[int, str]]:
    parts = []
    remaining = n
    for value, numeral in _ROMAN_TABLE:
        while remaining >= value:
            parts.append((value, numeral))
            remaining -= value
    return parts


# ── All subtractive numbers in [1, 100] ──────────────────────────────────

SUBTRACTIVE_NUMBERS = sorted(n for n in range(1, 101) if has_subtractive(n))
# Also identify the "hardest" ones — pure subtractive or double-subtractive
HARD_SUBTRACTIVE = [4, 9, 40, 44, 49, 90, 94, 99]
# Everything in 1-100 without subtractive forms
NON_SUBTRACTIVE = sorted(n for n in range(1, 101) if not has_subtractive(n))


# ── Trace generation (matches existing numconv trace flow) ────────────────

def make_trace(n: int, example_pairs: list[tuple[int, str]]) -> str:
    """Generate trace matching the unified numconv trace flow."""
    answer = to_roman(n)
    examples_str = ", ".join(f"{num}={rom}" for num, rom in example_pairs[:4])

    parts = greedy_decompose(n)
    # Group consecutive identical parts
    groups = []
    i = 0
    while i < len(parts):
        val, sym = parts[i]
        count = 1
        while i + count < len(parts) and parts[i + count] == (val, sym):
            count += 1
        groups.append((val * count, sym * count))
        i += count

    verify = f"Checking examples: {examples_str}\n"
    verify += "Roman numerals: I=1, V=5, X=10, L=50, C=100"

    if len(groups) <= 1:
        conversion = f"{n} = {answer}"
    else:
        vals_str = "+".join(str(v) for v, _ in groups)
        syms_str = "+".join(s for _, s in groups)
        conversion = f"{n} = {vals_str} = {syms_str} = {answer}"

    return f"<think>\n{verify}\n{conversion}\n</think>\n\\boxed{{{answer}}}"


# ── Prompt templates ──────────────────────────────────────────────────────

TEMPLATES = [
    # Competition style
    (
        "In Alice's Wonderland, numbers are secretly converted into a "
        "different numeral system. Some examples are given below:\n"
        "{examples}\n"
        "Now, write the number {query} in the Wonderland numeral system.",
        " -> ",
    ),
    # Generated style
    (
        "Below are a few examples showing how numbers are secretly "
        "converted in this world:\n"
        "{examples}\n"
        "Now, write the number {query} in its converted form.",
        " = ",
    ),
]


def generate_one(rng: random.Random, query: int) -> dict:
    """Generate one training example with the given query number."""
    template_text, sep = rng.choice(TEMPLATES)

    n_examples = rng.randint(3, 5)
    # Pool for examples: all 1-100 except the query
    pool = list(range(1, 101))
    pool.remove(query)

    # Ensure at least one example has a subtractive form
    sub_pool = [n for n in pool if has_subtractive(n)]
    non_sub_pool = [n for n in pool if not has_subtractive(n)]

    # Pick 1-2 subtractive examples, rest random
    n_sub = min(rng.randint(1, 2), n_examples, len(sub_pool))
    n_other = n_examples - n_sub

    examples = rng.sample(sub_pool, n_sub) + rng.sample(non_sub_pool, min(n_other, len(non_sub_pool)))
    rng.shuffle(examples)

    example_lines = [f"{n}{sep}{to_roman(n)}" for n in examples]
    example_pairs = [(n, to_roman(n)) for n in examples]

    prompt = template_text.format(
        examples="\n".join(example_lines),
        query=query,
    )
    trace_text = make_trace(query, example_pairs)
    answer = to_roman(query)

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": trace_text},
        ],
        "answer": answer,
        "id": None,  # set by caller
        "puzzle_type": "number_conversion",
        "mode": "generated",
        "generator": "roman_subtractive_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_dataset(
    n: int = 2000,
    seed: int = 7777,
    output: str = "data/number_conversion/pool/generated/roman_subtractive.jsonl",
) -> Path:
    """Generate n examples biased toward subtractive-form queries.

    Distribution:
    - 40% hard subtractive (4, 9, 40, 44, 49, 90, 94, 99)
    - 40% other subtractive (14, 19, 24, 29, 34, 39, etc.)
    - 20% non-subtractive queries BUT with subtractive examples shown
    """
    rng = random.Random(seed)
    outpath = Path(output)

    n_hard = int(n * 0.40)
    n_other_sub = int(n * 0.40)
    n_non_sub = n - n_hard - n_other_sub

    other_sub = [x for x in SUBTRACTIVE_NUMBERS if x not in HARD_SUBTRACTIVE]

    rows = []

    # Hard subtractive queries
    for i in range(n_hard):
        query = rng.choice(HARD_SUBTRACTIVE)
        rows.append(generate_one(rng, query))

    # Other subtractive queries
    for i in range(n_other_sub):
        query = rng.choice(other_sub)
        rows.append(generate_one(rng, query))

    # Non-subtractive queries (still get subtractive examples)
    for i in range(n_non_sub):
        query = rng.choice(NON_SUBTRACTIVE)
        rows.append(generate_one(rng, query))

    rng.shuffle(rows)

    # Assign IDs
    for i, row in enumerate(rows):
        row["id"] = f"roman_sub_{i:05d}"

    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Stats
    sub_answer_count = sum(1 for r in rows if has_subtractive(
        int(r["messages"][0]["content"].split("the number ")[1].split(" ")[0].strip("."))))
    print(f"Generated {len(rows)} examples -> {outpath}")
    print(f"  Subtractive-form answers: {sub_answer_count}/{len(rows)} ({100*sub_answer_count/len(rows):.0f}%)")
    print(f"  Hard subtractive: {n_hard}")
    print(f"  Other subtractive: {n_other_sub}")
    print(f"  Non-subtractive (with sub examples): {n_non_sub}")

    # Per-form breakdown
    from collections import Counter
    form_counts = Counter()
    for row in rows:
        answer = row["answer"]
        for sf in SUBTRACTIVE_FORMS:
            if sf in answer:
                form_counts[sf] += 1
    print(f"  Per-form in answers: {dict(form_counts)}")

    return outpath


if __name__ == "__main__":
    generate_dataset()
