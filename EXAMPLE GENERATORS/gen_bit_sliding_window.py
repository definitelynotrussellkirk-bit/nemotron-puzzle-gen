#!/usr/bin/env python3
"""Generate bit manipulation training traces using sliding window per-bit format.

Instead of asking the model to compute bitwise operations (which it can't),
decompose into 8 independent 1-bit prediction tasks. For each output bit,
show a window of input bits and the corresponding output bit from examples.
The model learns pattern matching on local windows, not arithmetic.

The window wraps circularly to handle rotations up to 7 positions.

Trace format:
    <think>
    Examples (window around each bit position):
    Pos 0: 10110|1|00 → 0,  01001|0|10 → 1,  11011|1|00 → 0
    Pos 1: 01101|0|01 → 1,  10010|1|01 → 0,  10111|0|00 → 1
    ...
    Pos 7: 00101|1|01 → 0,  10011|0|01 → 1,  01011|0|11 → 1

    Query: 11010110
    Pos 0: 01101|1|01 → 0
    Pos 1: 11011|0|10 → 1
    ...
    </think>
    \\boxed{01111010}
"""
import argparse
import csv
import json
import random
import re
import time
from datetime import datetime, timezone

from training.data import BOXED_INSTRUCTION


BYTE = 0xFF


def _wrap_bits(x_int, window_half=3):
    """Create a circularly-wrapped bit string for windowed access.

    For 8-bit input with window_half=3, produces a 14-char string:
    bits[5:8] + bits[0:8] + bits[0:3]  (wrap around both ends)
    """
    bits = format(x_int & BYTE, '08b')
    return bits[-window_half:] + bits + bits[:window_half]


def _extract_window(wrapped, pos, window_half=3):
    """Extract a window centered at position pos from a wrapped bit string.

    Returns a string like '101|1|010' with the center bit marked.
    """
    center = window_half + pos
    left = wrapped[center - window_half:center]
    mid = wrapped[center]
    right = wrapped[center + 1:center + 1 + window_half]
    return f"{left}{mid}{right}"


def build_sliding_window_trace(examples, query_str, answer_str,
                                window_half=3, max_examples=4):
    """Build a sliding window trace from examples and query.

    Args:
        examples: list of (input_str, output_str) 8-bit binary strings
        query_str: 8-bit binary query input
        answer_str: 8-bit binary expected output
        window_half: number of bits on each side of center (3 = 7-bit window)
        max_examples: max examples to show per position (keep trace short)

    Returns: trace string
    """
    # Select a diverse subset of examples (spread across input space)
    if len(examples) > max_examples:
        # Pick examples that are maximally spread in Hamming distance
        selected = [examples[0]]
        remaining = list(examples[1:])
        while len(selected) < max_examples and remaining:
            # Pick the example most distant from all selected
            best = max(remaining, key=lambda e: min(
                bin(int(e[0], 2) ^ int(s[0], 2)).count('1') for s in selected))
            selected.append(best)
            remaining.remove(best)
        ex_subset = selected
    else:
        ex_subset = list(examples)

    lines = []

    # Show examples in windowed form, per bit position
    lines.append("Examples per bit:")
    for pos in range(8):
        parts = []
        for inp, out in ex_subset:
            x = int(inp, 2)
            wrapped = _wrap_bits(x, window_half)
            window = _extract_window(wrapped, pos, window_half)
            out_bit = out[7 - pos]  # bit 0 is rightmost
            parts.append(f"{window}→{out_bit}")
        lines.append(f"  b{pos}: {', '.join(parts)}")

    # Query: show window for each bit, predict
    lines.append("")
    lines.append(f"Query: {query_str}")
    q_int = int(query_str, 2)
    q_wrapped = _wrap_bits(q_int, window_half)

    answer_bits = []
    for pos in range(8):
        window = _extract_window(q_wrapped, pos, window_half)
        ans_bit = answer_str[7 - pos]
        lines.append(f"  b{pos}: {window} → {ans_bit}")
        answer_bits.append(ans_bit)

    # Assembly
    assembled = ''.join(reversed(answer_bits))
    lines.append(f"Output: {assembled}")

    return '\n'.join(lines)


def generate_from_competition(train_csv, output_path, seed=42):
    """Generate sliding window traces from competition data."""
    from solvers.bit_manipulation import solve_details

    rng = random.Random(seed)
    results = []

    with open(train_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'bit manipulation' not in row['prompt'][:120]:
                continue

            details = solve_details(row['prompt'])
            if not details:
                continue

            answer = details['answer']
            if answer != row['answer']:
                continue  # skip mismatches

            examples = details['examples']
            query = details['query']

            trace = build_sliding_window_trace(
                examples, query, answer,
                window_half=3, max_examples=4)

            prompt = row['prompt']

            results.append({
                "messages": [
                    {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": f"<think>\n{trace}\n</think>\n\\boxed{{{answer}}}"},
                ],
                "answer": answer,
                "id": row[list(row.keys())[0]],
                "puzzle_type": "bit_manipulation",
                "mode": "sliding_window",
                "generator": "gen_bit_sliding_window",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })

    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f"Generated {len(results)} sliding window traces → {output_path}")
    return results


def generate_synthetic(n, output_path, seed=42, window_half=3):
    """Generate synthetic sliding window bit puzzles."""
    rng = random.Random(seed)

    # Import generator components
    from generators.gen_bit_3stream import (
        FAMILIES_3, FAMILIES_2, PERMS_3, BYTE,
        _make_sources, _make_sources_2, rol, ror
    )

    results = []
    for i in range(n):
        # Pick family and sources
        use_2input = rng.random() < 0.10
        if use_2input:
            fam_name, fam_fn, _ = rng.choices(FAMILIES_2, weights=[w for _,_,w in FAMILIES_2])[0]
            sources = _make_sources_2(rng)
            perm = None
        else:
            fam_name, fam_fn, _ = rng.choices(FAMILIES_3, weights=[w for _,_,w in FAMILIES_3])[0]
            sources = _make_sources(rng)
            perm = rng.choice(PERMS_3)

        # Generate examples + query
        n_examples = rng.randint(7, 9)
        inputs = [rng.randint(0, 255) for _ in range(n_examples + 1)]
        query_input = inputs[-1]
        example_inputs = inputs[:-1]

        def compute(x):
            vals = [fn(x) for _, fn in sources]
            if use_2input:
                return fam_fn(vals[0], vals[1])
            else:
                return fam_fn(vals[perm[0]], vals[perm[1]], vals[perm[2]])

        examples = [(format(x, "08b"), format(compute(x), "08b")) for x in example_inputs]
        query_str = format(query_input, "08b")
        answer_str = format(compute(query_input), "08b")

        # Build prompt
        prompt_lines = [
            "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. "
            "The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, "
            "and possibly majority or choice functions.",
            "",
            "Here are some examples of input -> output:",
        ]
        for inp, out in examples:
            prompt_lines.append(f"{inp} -> {out}")
        prompt_lines.append(f"\nNow, determine the output for: {query_str}")
        prompt = "\n".join(prompt_lines)

        trace = build_sliding_window_trace(
            examples, query_str, answer_str,
            window_half=window_half, max_examples=4)

        results.append({
            "messages": [
                {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                {"role": "assistant", "content": f"<think>\n{trace}\n</think>\n\\boxed{{{answer_str}}}"},
            ],
            "answer": answer_str,
            "id": f"gen_bit_sw_{i:06d}",
            "puzzle_type": "bit_manipulation",
            "mode": "sliding_window",
            "n_examples": n_examples,
            "family": fam_name,
            "generator": "gen_bit_sliding_window",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })

    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f"Generated {len(results)} synthetic sliding window traces → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["competition", "synthetic", "both"], default="both")
    parser.add_argument("--n", type=int, default=10000, help="Number of synthetic examples")
    parser.add_argument("--window", type=int, default=3, help="Window half-size (3=7bit, 4=9bit)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode in ("competition", "both"):
        generate_from_competition(
            "data/competition/train.csv",
            "data/bit_manipulation/pool/competition/sliding_window.jsonl",
            seed=args.seed)

    if args.mode in ("synthetic", "both"):
        generate_synthetic(
            args.n,
            "data/bit_manipulation/pool/generated/sliding_window.jsonl",
            seed=args.seed,
            window_half=args.window)


if __name__ == "__main__":
    main()
