#!/usr/bin/env python3
"""Generate digit-permuted transformation puzzles with structural anchors.

These are numeric puzzles where '0'-'9' don't have their standard values.
Each puzzle includes structural anchors that help discover the mapping:
- Zero-producing examples (force which digit = 0)
- Concat examples (expose digit order without carry)
- Sign examples (constrain relative ordering)

Usage:
    python3 -m generators.gen_digit_permuted --n 200
"""

import argparse
import json
import random
import time
from pathlib import Path

from solvers.transformation_ops import ARITHMETIC_OPS, OP_DESCRIPTIONS, SYMBOL_POOL
from training.data import BOXED_INSTRUCTION


def generate_permuted_puzzle(rng):
    """Generate a numeric puzzle with a hidden digit permutation."""
    base = 10

    # Random digit permutation: perm[i] = value assigned to character str(i)
    perm = list(range(10))
    rng.shuffle(perm)
    # digit_symbols[value] = character
    digit_symbols = [str(perm.index(v)) for v in range(10)]
    # sym_to_val: character -> value
    sym_to_val = {str(i): perm[i] for i in range(10)}

    # Pick operation with modifiers
    ops_list = [(n, f) for n, f, _ in ARITHMETIC_OPS
                if n in ('add', 'sub', 'mul', 'absdiff', 'concat')]
    # Use 2 operators
    op_chars = rng.sample(SYMBOL_POOL[:15], 2)

    op_configs = {}
    for oc in op_chars:
        name, fn = rng.choice(ops_list)
        rev_in = rng.choice([True, False])
        rev_out = rng.choice([True, False])
        # For permuted puzzles, keep modifiers simple to not make it impossible
        if name == 'concat':
            rev_in = False
            rev_out = False
        op_configs[oc] = (name, fn, rev_in, rev_out)

    def encode_operand(val):
        high = val // base
        low = val % base
        return digit_symbols[high] + digit_symbols[low]

    def decode(s, rev):
        if rev:
            s = s[::-1]
        return sym_to_val[s[0]] * base + sym_to_val[s[1]]

    def encode_result(val, rev_out):
        neg = val < 0
        val = abs(val)
        if val == 0:
            s = digit_symbols[0]
        else:
            chars = []
            while val > 0:
                chars.append(digit_symbols[val % base])
                val //= base
            s = ''.join(reversed(chars))
        if rev_out:
            s = s[::-1]
        if neg:
            s = '-' + s
        return s

    # Generate examples with structural anchors
    lines = []
    example_data = []

    # Anchor 1: try to include a zero-producing example
    for oc in op_chars:
        name, fn, rev_in, rev_out = op_configs[oc]
        if name in ('sub', 'absdiff'):
            # Same operand → zero result
            a_val = rng.randint(10, 99)
            a_str = encode_operand(a_val)
            a_dec = decode(a_str, rev_in)
            result = fn(a_dec, a_dec)
            result_str = encode_result(result, rev_out)
            lines.append(f"{a_str}{oc}{a_str} = {result_str}")
            example_data.append((oc, a_dec, a_dec, result))
            break

    # Anchor 2: regular examples (3-4 more)
    for _ in range(rng.randint(3, 4)):
        oc = rng.choice(op_chars)
        name, fn, rev_in, rev_out = op_configs[oc]
        a_val = rng.randint(10, 99)
        b_val = rng.randint(10, 99)
        a_str = encode_operand(a_val)
        b_str = encode_operand(b_val)
        a_dec = decode(a_str, rev_in)
        b_dec = decode(b_str, rev_in)
        result = fn(a_dec, b_dec)
        result_str = encode_result(result, rev_out)
        lines.append(f"{a_str}{oc}{b_str} = {result_str}")
        example_data.append((oc, a_dec, b_dec, result))

    # Query
    query_oc = rng.choice(op_chars)
    qname, qfn, qrev_in, qrev_out = op_configs[query_oc]
    qa_val = rng.randint(10, 99)
    qb_val = rng.randint(10, 99)
    qa_str = encode_operand(qa_val)
    qb_str = encode_operand(qb_val)
    qa_dec = decode(qa_str, qrev_in)
    qb_dec = decode(qb_str, qrev_in)
    qresult = qfn(qa_dec, qb_dec)
    answer = encode_result(qresult, qrev_out)

    query_str = f"{qa_str}{query_oc}{qb_str}"

    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules "
        "is applied to equations. Below are a few examples:\n"
        + "\n".join(lines)
        + f"\nNow, determine the result for: {query_str}"
    )

    # Trace: show the CSP-style reasoning
    trace_lines = [
        "Equation rules. Base 10 with hidden digit values.",
        "",
        f"Digit mapping: {', '.join(f'{i}→{perm[i]}' for i in range(10))}",
        "",
        "Ops:",
    ]
    for oc in op_chars:
        name, fn, rev_in, rev_out = op_configs[oc]
        desc = OP_DESCRIPTIONS.get(name, name)
        mods = []
        if rev_in: mods.append("rev_input")
        if rev_out: mods.append("rev_output")
        mod_str = ", ".join(mods) if mods else "plain"
        trace_lines.append(f"  {oc} = {desc}, {mod_str}")

    trace_lines.append("")
    trace_lines.append(f"Query: {query_str}")
    trace_lines.append(f"decode: {qa_str}→{qa_dec}, {qb_str}→{qb_dec}")
    trace_lines.append(f"compute: {qa_dec}{OP_DESCRIPTIONS.get(qname,'?')}{qb_dec}={qresult}")
    trace_lines.append(f"encode: {qresult}→{answer}")
    trace_lines.append(f"\n\\boxed{{{answer}}}")

    trace = "\n".join(trace_lines)

    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": trace},
        ],
        "id": f"gen_permuted_{rng.randint(0,999999):06d}",
        "puzzle_type": "transformation",
        "mode": "digit_permutation",
        "digit_permutation": {str(i): perm[i] for i in range(10)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--output", type=str, default="data/sft_transformation_permuted.jsonl")
    args = parser.parse_args()

    rng = random.Random(int(time.time()))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as out:
        for i in range(args.n):
            result = generate_permuted_puzzle(rng)
            out.write(json.dumps(result) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{args.n}", flush=True)

    print(f"Done: {args.n} digit-permuted examples → {output}")


if __name__ == "__main__":
    main()
