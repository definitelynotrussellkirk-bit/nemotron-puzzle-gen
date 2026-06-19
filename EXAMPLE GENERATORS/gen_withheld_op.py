#!/usr/bin/env python3
"""Generate withheld-query-op transformation training data (v2 — interpreter format).

Puzzles where one operator is deliberately absent from examples.
Uses Use/Check/Query interpreter contract matching other trace types.

Usage:
    python3 -m generators.gen_withheld_op --n 2000
"""

import argparse
import json
import random
import time
from pathlib import Path

from solvers.transformation_ops import ARITHMETIC_OPS, OP_DESCRIPTIONS
from training.data import BOXED_INSTRUCTION


# Exclude { and } — they break \boxed{} extraction
SYMBOL_POOL = list("!#$%&'()*+,-/:;<=>?@[\\]^`|~")


def generate_withheld_puzzle(rng):
    """Generate one puzzle with a deliberately withheld query operator."""
    base = 10

    # Pick modifier regime (shared across all ops for clean signal)
    rev_in = rng.choice([True, False])
    rev_out = rng.choice([True, False])

    # Pick 3 operators: 2 shown, 1 withheld for query
    op_chars = rng.sample(SYMBOL_POOL[:20], 3)
    shown_ops = op_chars[:2]
    query_op = op_chars[2]

    # Pick base operations for each
    ops_list = [(name, fn) for name, fn, _ in ARITHMETIC_OPS
                if name not in ('bitor', 'bitxor')]
    op_assignments = {}
    for oc in op_chars:
        name, fn = rng.choice(ops_list)
        op_assignments[oc] = (name, fn)

    def encode_operand(val):
        return str(val // base) + str(val % base)

    def decode_raw(s):
        """Decode without reversal — raw digit value."""
        return int(s[0]) * base + int(s[1])

    def decode_for_compute(s):
        """Decode with rev_in if applicable."""
        if rev_in:
            return int(s[1]) * base + int(s[0])
        return int(s[0]) * base + int(s[1])

    def compute_result(a_str, b_str, op_fn):
        a = decode_for_compute(a_str)
        b = decode_for_compute(b_str)
        return op_fn(a, b), a, b

    def format_output(result):
        neg = result < 0
        r_str = str(abs(result))
        if rev_out:
            r_str = r_str[::-1]
        if neg:
            r_str = "-" + r_str
        return r_str

    # Generate examples using ONLY the shown operators
    lines = []
    example_data = []
    for _ in range(rng.choices([3, 4, 5], weights=[65, 43, 28])[0]):
        oc = rng.choice(shown_ops)
        name, fn = op_assignments[oc]
        a_val = rng.randint(10, 99)
        b_val = rng.randint(10, 99)
        a_str = encode_operand(a_val)
        b_str = encode_operand(b_val)
        result, comp_a, comp_b = compute_result(a_str, b_str, fn)
        output = format_output(result)
        lines.append(f"{a_str}{oc}{b_str} = {output}")
        example_data.append({
            'op': oc, 'name': name, 'a_str': a_str, 'b_str': b_str,
            'a_raw': a_val, 'b_raw': b_val,
            'comp_a': comp_a, 'comp_b': comp_b,
            'result': result, 'output': output,
        })

    # Query using the withheld operator
    q_name, q_fn = op_assignments[query_op]
    qa = rng.randint(10, 99)
    qb = rng.randint(10, 99)
    qa_str = encode_operand(qa)
    qb_str = encode_operand(qb)
    q_result, q_comp_a, q_comp_b = compute_result(qa_str, qb_str, q_fn)
    answer = format_output(q_result)
    query_str = f"{qa_str}{query_op}{qb_str}"

    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules "
        "is applied to equations. Below are a few examples:\n"
        + "\n".join(lines)
        + f"\nNow, determine the result for: {query_str}"
    )

    # Build regime description
    mod_parts = []
    if rev_in: mod_parts.append("reverse digit strings")
    if rev_out: mod_parts.append("reverse output digits")
    input_desc = "reverse digit strings" if rev_in else "plain"
    output_desc = "reverse output digits" if rev_out else "plain"

    q_desc = OP_DESCRIPTIONS.get(q_name, q_name)

    # === Build honest unseen-op trace ===
    # Unseen-op is a ranking problem, not deduction. Be explicit about it.
    # Lock visible operators from support, then Choose for unseen op.

    trace_lines = ["Numeric unseen-op.", ""]

    # Lock each visible operator using support examples
    ops_locked = {}
    for ex in example_data:
        op_sym = ex['op']
        if op_sym in ops_locked:
            continue
        ex_desc = OP_DESCRIPTIONS.get(ex['name'], ex['name'])
        order_name = "BA,DC" if rev_in else "AB,CD"
        style = "rev" if rev_out else "raw"
        L_val, R_val = ex['comp_a'], ex['comp_b']
        trace_lines.append(f"Lock[{op_sym}]: order={order_name} op={ex_desc} style={style}")
        trace_lines.append(f"  Ex: {ex['a_str']}{op_sym}{ex['b_str']} → L={L_val} R={R_val}"
                          f" → {ex_desc}({L_val},{R_val})={ex['result']}"
                          f" {style}={ex['output']} → MATCH")
        ops_locked[op_sym] = ex_desc

    # Shared regime summary
    order_name = "BA,DC" if rev_in else "AB,CD"
    style = "rev" if rev_out else "raw"
    trace_lines.append("")
    trace_lines.append(f"Regime: order={order_name} style={style}")

    # Choose for unseen operator — honest about it being a choice
    trace_lines.append("")
    trace_lines.append(f"Query operator '{query_op}' not in support.")
    trace_lines.append(f"Choose: op={q_desc}")
    trace_lines.append(f"Reason: matches regime (order={order_name}, style={style})")

    # Apply
    trace_lines.append("")
    trace_lines.append(f"Query: {qa_str}{query_op}{qb_str}")
    trace_lines.append(f"  Order {order_name}: L={q_comp_a} R={q_comp_b}")
    trace_lines.append(f"  {q_desc}({q_comp_a},{q_comp_b})={q_result}")
    if rev_out and q_result != 0:
        trace_lines.append(f"  style {style}: {abs(q_result)} → {str(abs(q_result))[::-1]}")
    trace_lines.append(f"  = {answer}")

    trace = "<think>\n" + "\n".join(trace_lines) + "\n</think>\n\\boxed{" + answer + "}"

    # Validate with actual solver
    from solvers.transformation import solve
    pred = solve(prompt + BOXED_INSTRUCTION.split("Please")[0])
    if pred is not None and pred != answer:
        return None

    from datetime import datetime, timezone
    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": trace},
        ],
        "answer": answer,
        "id": f"gen_withheld_{hash((qa, qb, query_op)) % 100000:05d}",
        "puzzle_type": "transformation",
        "mode": "withheld_query_op",
        "generator": "gen_withheld_op",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--output", type=str, default="data/transformation/pool/specialized/withheld.jsonl")
    args = parser.parse_args()

    rng = random.Random(int(time.time()))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    attempts = 0
    with open(output, "w") as out:
        for i in range(args.n * 30):  # high oversample for solver rejection
            attempts += 1
            if count >= args.n:
                break
            result = generate_withheld_puzzle(rng)
            if result is None:
                continue
            out.write(json.dumps(result) + "\n")
            count += 1
            if count % 200 == 0:
                print(f"  {count}/{args.n} ({attempts} attempts)", flush=True)

    print(f"Done: {count} withheld-op examples → {output} ({attempts} attempts)")


if __name__ == "__main__":
    main()
