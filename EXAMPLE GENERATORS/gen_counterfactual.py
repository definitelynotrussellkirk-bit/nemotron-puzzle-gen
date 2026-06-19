#!/usr/bin/env python3
"""Generate counterfactual disambiguation examples for transformation.

For each near-collision op pair (e.g., a-b vs b-a, abs vs b-a),
generate puzzles where:
1. Most examples are ambiguous between the two ops
2. One "disambiguation" example forces the correct interpretation
3. The query answer differs between the two ops

This teaches the model WHAT EVIDENCE MATTERS.

Usage:
    python3 -m generators.gen_counterfactual --n 200
"""

import argparse
import json
import random
import time
from pathlib import Path

from solvers.transformation_ops import ARITHMETIC_OPS, OP_DESCRIPTIONS
from training.data import BOXED_INSTRUCTION


# Near-collision pairs: ops that give the same result under common conditions
COLLISION_PAIRS = [
    # (op1_name, op2_name, condition_where_they_differ)
    ("sub", "bsub", "when a != b, sign flips"),
    ("absdiff", "bsub", "when a > b, bsub is negative"),
    ("absdiff", "sub", "when a < b, sub is negative"),
    ("add", "add1", "always differ by 1"),
    ("mul", "mul1", "always differ by 1"),
    ("mul", "mulsub1", "always differ by 1"),
    ("sub", "absdiff", "when a < b, sub negative but absdiff positive"),
    ("mula", "mul", "differ by a"),
    ("mulb", "mul", "differ by b"),
    # New: bit-op collisions (15% collision rate)
    ("absdiff", "bitxor", "differ when carry propagates"),
    ("add", "bitor", "differ when both have 1 in same bit position"),
    ("bitor", "bitxor", "differ when both have 1 in same bit position"),
    ("bsub", "bitxor", "differ when carry propagates in subtraction"),
    # More modifier collisions
    ("add1", "sub1", "always differ by 2"),
    ("mula", "mulb", "differ by a-b"),
    ("mulsuba", "mulsubb", "differ by b-a"),
]


def generate_counterfactual_example(rng, op1_name, op2_name, base=10):
    """Generate one puzzle where op1 and op2 are disambiguated."""
    ops_by_name = {name: fn for name, fn, _ in ARITHMETIC_OPS}
    op1_fn = ops_by_name[op1_name]
    op2_fn = ops_by_name[op2_name]

    op_char = rng.choice(list("!@#$%^&*()_+=[]{}|;:<>?"))

    # Generate examples where both ops give the SAME result
    ambiguous_examples = []
    disambiguation_example = None

    for _ in range(200):
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        r1 = op1_fn(a, b)
        r2 = op2_fn(a, b)

        if r1 == r2:
            ambiguous_examples.append((a, b, str(r1)))
        elif disambiguation_example is None:
            disambiguation_example = (a, b, str(r1), str(r2))

    if len(ambiguous_examples) < 3 or disambiguation_example is None:
        return None

    # Pick 3-4 ambiguous examples + the disambiguation one
    rng.shuffle(ambiguous_examples)
    examples = ambiguous_examples[:rng.randint(3, 4)]

    da, db, d_correct, d_wrong = disambiguation_example
    examples.append((da, db, d_correct))
    rng.shuffle(examples)

    # Query: pick values where ops diverge
    for _ in range(100):
        qa = rng.randint(10, 99)
        qb = rng.randint(10, 99)
        qr1 = op1_fn(qa, qb)
        qr2 = op2_fn(qa, qb)
        if qr1 != qr2:
            break
    else:
        return None

    answer = str(qr1)

    # Build prompt
    lines = []
    for a, b, r in examples:
        lines.append(f"{a}{op_char}{b} = {r}")

    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules "
        "is applied to equations. Below are a few examples:\n"
        + "\n".join(lines)
        + f"\nNow, determine the result for: {qa}{op_char}{qb}"
    )

    # Build trace showing the disambiguation
    desc1 = OP_DESCRIPTIONS.get(op1_name, op1_name)
    desc2 = OP_DESCRIPTIONS.get(op2_name, op2_name)

    # Pick one ambiguous example to show both ops agree on it
    amb_a, amb_b, amb_r = ambiguous_examples[0]

    trace_lines = [
        "Equation rules. Base 10.",
        "",
        f"Step 1: Two candidate operations.",
        f"  {desc1}: e.g. {amb_a}{op_char}{amb_b} → {op1_fn(amb_a, amb_b)} → MATCH",
        f"  {desc2}: e.g. {amb_a}{op_char}{amb_b} → {op2_fn(amb_a, amb_b)} → MATCH",
        f"  Both give {amb_r} — ambiguous on this example.",
        "",
        f"Step 2: Killer clue.",
        f"  {da}{op_char}{db}={d_correct}:",
        f"  {desc1}: {op1_fn(da, db)} {'→ MATCH' if str(op1_fn(da, db)) == d_correct else '→ MISMATCH'}",
        f"  {desc2}: {op2_fn(da, db)} {'→ MATCH' if str(op2_fn(da, db)) == d_correct else '→ MISMATCH'}",
        f"  → {op_char} = {desc1}",
        "",
        f"Step 3: Compute query.",
        f"  {qa}{op_char}{qb}: {desc1} → {op1_fn(qa, qb)}",
        f"",
    ]
    trace = "<think>\n" + "\n".join(trace_lines) + "\n</think>\n\\boxed{" + answer + "}"

    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": trace},
        ],
        "answer": answer,
        "id": f"gen_counterfactual_{op1_name}_{op2_name}_{hash((qa,qb))%10000:04d}",
        "puzzle_type": "transformation",
        "counterfactual": True,
        "generator": "gen_counterfactual",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--output", type=str, default="data/sft_transformation.jsonl")
    args = parser.parse_args()

    rng = random.Random(int(time.time()))
    output = Path(args.output)
    count = 0

    with open(output, "a") as out:
        while count < args.n:
            op1, op2, _ = rng.choice(COLLISION_PAIRS)
            result = generate_counterfactual_example(rng, op1, op2)
            if result is None:
                continue
            out.write(json.dumps(result) + "\n")
            count += 1
            if count % 50 == 0:
                print(f"  {count}/{args.n}", flush=True)

    total = sum(1 for _ in open(output))
    print(f"Done: +{count} counterfactual examples. Total in file: {total}")


if __name__ == "__main__":
    main()
