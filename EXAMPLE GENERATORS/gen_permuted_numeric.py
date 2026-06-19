#!/usr/bin/env python3
"""Generate digit-permuted numeric transformation training data.

About 7.3% of competition numeric puzzles use a non-identity digit mapping
(digits 0-9 are shuffled). This generator creates training examples specifically
for this sub-type, with traces that explicitly show the permuted mapping.

Usage:
    python3 -m generators.gen_permuted_numeric --n 1000 --output data/transformation/permuted.jsonl
"""

import argparse
import json
import random
from training.data import BOXED_INSTRUCTION
from solvers.transformation_ops import ARITHMETIC_OPS, SYMBOL_POOL


def _digits_of(val, base=10):
    if val == 0:
        return [0]
    ds = []
    v = abs(val)
    while v > 0:
        ds.append(v % base)
        v //= base
    ds.reverse()
    return ds


def generate_permuted_puzzle(rng):
    """Generate a numeric transformation puzzle with permuted digit mapping."""
    base = 10

    # Create a non-trivial digit permutation
    perm = list(range(10))
    while perm == list(range(10)):  # ensure it's actually permuted
        rng.shuffle(perm)

    # perm[i] = what digit i looks like in the puzzle
    # So if perm = [3,7,1,...], then the value 0 is written as "3", value 1 as "7", etc.
    digit_to_sym = {i: str(perm[i]) for i in range(10)}
    sym_to_digit = {str(perm[i]): i for i in range(10)}

    # Pick operator symbols
    op_syms = rng.sample(SYMBOL_POOL, rng.choice([2, 3]))

    # Pick operations
    all_ops = [(n, f) for n, f, w in ARITHMETIC_OPS]
    op_assignments = {}
    for op_sym in op_syms:
        op_name, op_fn = rng.choice(all_ops)
        op_assignments[op_sym] = (op_name, op_fn)

    # Shared modifiers
    use_rev_input = rng.random() < 0.15  # less common for permuted
    use_rev_output = rng.random() < 0.10
    use_opsign = rng.random() < 0.20

    # Generate examples
    examples = []
    n_examples = rng.randint(3, 6)
    for _ in range(n_examples * 3):
        if len(examples) >= n_examples:
            break

        op_sym = rng.choice(op_syms)
        op_name, op_fn = op_assignments[op_sym]

        a_hi, a_lo = rng.randrange(10), rng.randrange(10)
        b_hi, b_lo = rng.randrange(10), rng.randrange(10)
        a_val = a_hi * 10 + a_lo
        b_val = b_hi * 10 + b_lo

        compute_a = a_val
        compute_b = b_val
        if use_rev_input:
            compute_a = a_lo * 10 + a_hi
            compute_b = b_lo * 10 + b_hi

        try:
            result = op_fn(compute_a, compute_b)
        except:
            continue

        neg = result < 0
        abs_result = abs(result)
        digits = _digits_of(abs_result)

        if use_rev_output and abs_result != 0:
            digits = list(reversed(digits))

        if any(d >= 10 for d in digits):
            continue

        # Encode with permuted digits
        inp = digit_to_sym[a_hi] + digit_to_sym[a_lo] + op_sym + digit_to_sym[b_hi] + digit_to_sym[b_lo]
        out_chars = ''.join(digit_to_sym[d] for d in digits)
        if neg:
            sign_char = op_sym if use_opsign else '-'
            out_chars = sign_char + out_chars

        examples.append((inp, out_chars, op_sym, op_name, a_val, b_val, compute_a, compute_b, result))

    if len(examples) < 3:
        return None

    # Generate query
    q_op_sym = rng.choice(op_syms)
    q_op_name, q_op_fn = op_assignments[q_op_sym]
    qa_hi, qa_lo = rng.randrange(10), rng.randrange(10)
    qb_hi, qb_lo = rng.randrange(10), rng.randrange(10)
    qa_val = qa_hi * 10 + qa_lo
    qb_val = qb_hi * 10 + qb_lo
    compute_qa = qa_val
    compute_qb = qb_val
    if use_rev_input:
        compute_qa = qa_lo * 10 + qa_hi
        compute_qb = qb_lo * 10 + qb_hi

    try:
        q_result = q_op_fn(compute_qa, compute_qb)
    except:
        return None

    q_neg = q_result < 0
    q_abs = abs(q_result)
    q_digits = _digits_of(q_abs)
    if use_rev_output and q_abs != 0:
        q_digits = list(reversed(q_digits))
    if any(d >= 10 for d in q_digits):
        return None

    query = digit_to_sym[qa_hi] + digit_to_sym[qa_lo] + q_op_sym + digit_to_sym[qb_hi] + digit_to_sym[qb_lo]
    answer_chars = ''.join(digit_to_sym[d] for d in q_digits)
    if q_neg:
        sign_char = q_op_sym if use_opsign else '-'
        answer_chars = sign_char + answer_chars

    # Build prompt
    rng.shuffle(examples)
    lines = [f"{inp} = {out}" for inp, out, *_ in examples]
    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules "
        "is applied to equations. Below are a few examples:\n"
        + "\n".join(lines)
        + f"\nNow, determine the result for: {query}"
    )

    # Build trace — explicitly shows permuted mapping discovery
    trace_lines = ["Equation rules. Base 10, permuted digits."]

    # Step 1: Detect permutation by checking examples
    trace_lines.append("")
    trace_lines.append("Step 1: Detect digit permutation.")
    trace_lines.append("If digits were identity-mapped, outputs wouldn't match. Testing permutations.")

    # Step 2: Recover mapping from examples — show 3-4 actual recoveries
    trace_lines.append("")
    trace_lines.append("Step 2: Recover digit mapping from examples.")
    shown = 0
    recovered = set()
    for inp, out, op_sym, op_name, a_val, b_val, ca, cb, res in examples:
        if shown >= 4:
            break
        # Show how input digits reveal the mapping
        # inp is like "05" where '0' maps to digit_to_sym^-1[0], '5' maps to digit_to_sym^-1[5]
        d1_sym, d2_sym = inp[0], inp[1]
        d1_val = sym_to_digit[d1_sym]
        d2_val = sym_to_digit[d2_sym]
        new_recoveries = []
        if d1_sym not in recovered:
            new_recoveries.append(f"'{d1_sym}'={d1_val}")
            recovered.add(d1_sym)
        if d2_sym not in recovered:
            new_recoveries.append(f"'{d2_sym}'={d2_val}")
            recovered.add(d2_sym)
        if new_recoveries:
            trace_lines.append(f"  From {inp}{op_sym}...={out}: {', '.join(new_recoveries)}")
            shown += 1

    # Show full mapping
    mapping_str = ", ".join(f"'{digit_to_sym[i]}'={i}" for i in range(10))
    trace_lines.append(f"  Full mapping: {mapping_str}")

    # Step 3: Identify operations
    trace_lines.append("")
    trace_lines.append("Step 3: Identify operators.")
    for op_sym in op_syms:
        on, _ = op_assignments[op_sym]
        desc = on
        if use_rev_input:
            desc = f"rev_input, {desc}"
        if use_rev_output:
            desc = f"{desc}, rev_output"
        if use_opsign:
            desc = f"{desc}, opsign"
        trace_lines.append(f"  {op_sym} = {desc}")

    # Step 4: Verify examples
    trace_lines.append("")
    trace_lines.append("Step 4: Verify examples.")
    for inp, out, op_sym, op_name, a_val, b_val, ca, cb, res in examples:
        trace_lines.append(f"  {inp}={out}: decode→{a_val:02d},{b_val:02d} {op_name}→{res} → MATCH")

    # Step 5: Compute query
    trace_lines.append("")
    trace_lines.append(f"Step 5: Compute {query}")
    trace_lines.append(f"  decode: {digit_to_sym[qa_hi]}={qa_hi},{digit_to_sym[qa_lo]}={qa_lo} → {qa_val:02d}")
    trace_lines.append(f"          {digit_to_sym[qb_hi]}={qb_hi},{digit_to_sym[qb_lo]}={qb_lo} → {qb_val:02d}")
    if use_rev_input:
        trace_lines.append(f"  rev_input: {qa_val}→{compute_qa}, {qb_val}→{compute_qb}")
    trace_lines.append(f"  compute: {compute_qa} {q_op_name} {compute_qb} = {q_result}")
    if use_rev_output and q_result != 0:
        trace_lines.append(f"  rev_output: digits reversed")
    trace_lines.append(f"  encode: {q_abs} → {answer_chars}")

    trace = "\n".join(trace_lines) + f"\n\n\\boxed{{{answer_chars}}}"

    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"<think>\n{trace}\n</think>"},
        ],
        "id": f"gen_permuted_{rng.randint(0,999999):06d}",
        "puzzle_type": "transformation",
        "mode": "permuted_numeric",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--output", type=str, default="data/transformation/permuted.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    count = 0
    with open(args.output, "w") as out:
        for _ in range(args.n * 3):
            if count >= args.n:
                break
            result = generate_permuted_puzzle(rng)
            if result is None:
                continue
            out.write(json.dumps(result) + "\n")
            count += 1
            if count % 200 == 0:
                print(f"  {count}/{args.n}")

    print(f"Done: {count} permuted numeric examples → {args.output}")


if __name__ == "__main__":
    main()
