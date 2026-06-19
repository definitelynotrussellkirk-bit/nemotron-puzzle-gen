#!/usr/bin/env python3
"""Generate synthetic cipher-digit transformation puzzles.

Creates puzzles where digits are encrypted with a random bijective cipher.
Traces show: CRACK mapping → SCAN → LOCK → APPLY → ENCODE.

Usage:
    python3 -m generators.gen_transform_cipher --n 5000
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone

from generators.trace_transform import (
    _make_operands, _calc, _fmt, COMBO_DISPLAY, build_cipher_trace,
    build_cipher_missing_symbol_trace,
)
from training.data import BOXED_INSTRUCTION

# Safe symbols (no {, }, \, which break \boxed{})
SAFE_SYMBOLS = list('!@#$%^&*()-_=+[]:,.<>?/~`')


def _format_final_answer(answer: str) -> str:
    """Match the dataset contract for brace-unsafe symbol answers."""
    if "{" in answer or "}" in answer:
        return f"The final answer is: {answer}"
    return f"\\boxed{{{answer}}}"


def _prompt_symbols(examples, query):
    symbols = set(query)
    for lhs, rhs in examples:
        symbols.update(lhs)
        symbols.update(rhs)
    return symbols


def generate_one(rng):
    """Generate one cipher-digit puzzle with crack-scan-lock trace."""
    # Variable base — match competition distribution (base 7 and 10 dominant)
    if rng.random() < 0.20:
        # Competition: base 4-9, weighted toward 7
        base = rng.choices([4, 5, 6, 7, 8, 9], weights=[7, 7, 12, 26, 3, 10])[0]
    else:
        base = 10
    if len(SAFE_SYMBOLS) < base:
        return None
    symbols = rng.sample(SAFE_SYMBOLS, base)
    mapping = dict(zip(symbols, range(base)))
    rev_map = {v: k for k, v in mapping.items()}

    # Pick operation per operator symbol (1-2 operators)
    orders = ["BA_DC", "AB_CD", "AB_DC", "BA_CD"]
    ops = ["mul", "add", "sub", "absdiff", "cat", "rcat",
           "add1", "addm1", "muladd1", "mulsub1"]
    fmts = ["rev", "raw", "abs", "opprefix", "opprefix_rev"]

    # Match competition distribution: 43/406/374 across 1/2/3 operators
    n_ops = rng.choices([1, 2, 3], weights=[0.05, 0.50, 0.45])[0]
    available_op_syms = [s for s in SAFE_SYMBOLS if s not in symbols]
    if len(available_op_syms) < n_ops:
        return None
    op_syms = rng.sample(available_op_syms, n_ops)

    combos = {}
    for op_sym in op_syms:
        combos[op_sym] = (rng.choice(orders), rng.choice(ops), rng.choice(fmts))

    # Generate 3-5 examples
    n_examples = rng.randint(3, 5)
    examples = []
    op_pos = 2  # operator always at position 2

    for _ in range(n_examples):
        a, b, c, d = [rng.randint(0, base - 1) for _ in range(4)]
        op_sym = rng.choice(op_syms)
        order, op, fmt = combos[op_sym]

        L, R = _make_operands(a, b, c, d, order)
        val = _calc(L, R, op)
        if val is None:
            continue
        fval = _fmt(val, fmt, op_char=op_sym)
        if fval is None or len(fval) > 5:
            continue

        # Encode — skip if result has digits >= base
        lhs = rev_map[a] + rev_map[b] + op_sym + rev_map[c] + rev_map[d]
        rhs_digits = str(fval) if not str(fval).startswith('-') else str(fval)
        if any(ch.isdigit() and int(ch) >= base for ch in rhs_digits):
            continue
        rhs = ''.join(rev_map[int(ch)] if ch.isdigit() else ch for ch in rhs_digits)
        if '?' in rhs:
            continue
        examples.append((lhs, rhs))

    if len(examples) < 2:
        return None

    # Generate query. Keep this pool execution-only: the query operator is
    # present in support examples, so the trace can lock it directly.
    qa, qb, qc, qd = [rng.randint(0, base - 1) for _ in range(4)]
    used_op_syms = sorted({lhs[op_pos] for lhs, _ in examples if len(lhs) > op_pos})
    if not used_op_syms:
        return None
    unseen_op = False
    if unseen_op:
        # Query uses an operator not in support examples
        unseen_syms = [s for s in available_op_syms if s not in op_syms]
        if unseen_syms:
            q_op_sym = rng.choice(unseen_syms)
            # Assign it the same combo as a random seen operator (alias)
            q_combo = combos[rng.choice(op_syms)]
            combos[q_op_sym] = q_combo
        else:
            q_op_sym = rng.choice(op_syms)
            q_combo = combos[q_op_sym]
    else:
        q_op_sym = rng.choice(used_op_syms)
        q_combo = combos[q_op_sym]
    query = rev_map[qa] + rev_map[qb] + q_op_sym + rev_map[qc] + rev_map[qd]

    qL, qR = _make_operands(qa, qb, qc, qd, q_combo[0])
    qval = _calc(qL, qR, q_combo[1])
    if qval is None:
        return None
    qfval = _fmt(qval, q_combo[2], op_char=q_op_sym)
    if qfval is None or len(str(qfval)) > 5:
        return None

    qfval_str = str(qfval)
    if any(ch.isdigit() and int(ch) >= base for ch in qfval_str):
        return None
    answer = ''.join(rev_map[int(ch)] if ch.isdigit() else ch for ch in qfval_str)
    if '?' in answer:
        return None

    # Build prompt
    prompt_lines = [
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples:",
    ]
    shuffled_examples = list(examples)
    rng.shuffle(shuffled_examples)
    for lhs, rhs in shuffled_examples:
        prompt_lines.append(f"{lhs} = {rhs}")
    prompt_lines.append(f"Now, determine the result for: {query}")
    prompt = "\n".join(prompt_lines)

    # Build trace. If the answer uses a symbol absent from the visible prompt,
    # use the same missing-symbol prior trace used for train.csv rows.
    maj_op = max(combos.keys(), key=lambda k: sum(1 for l, _ in examples if len(l) > 2 and l[2] == k))
    dpos = [0, 1, 3, 4]
    # MUST use shuffled_examples — same order as prompt!
    maj_op_shuffled = max(combos.keys(), key=lambda k: sum(1 for l, _ in shuffled_examples if len(l) > 2 and l[2] == k))
    fresh_answer = any(ch != '-' and ch not in _prompt_symbols(shuffled_examples, query) for ch in answer)
    if fresh_answer:
        trace_result = build_cipher_missing_symbol_trace(
            shuffled_examples, query, answer, mapping, combos, op_pos
        )
        mode = "cipher_missing_symbol_synthetic"
    else:
        trace_result = build_cipher_trace(shuffled_examples, query, answer, mapping, combos, maj_op_shuffled, op_pos)
        mode = "cipher_digit_synthetic"
    if trace_result is None:
        return None
    trace_text, pred = trace_result

    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"<think>\n{trace_text}\n</think>\n" + _format_final_answer(answer)},
        ],
        "answer": answer,
        "id": f"gen_trans_cipher_{rng.randint(0, 999999):06d}",
        "puzzle_type": "transformation",
        "mode": mode,
        "generator": "gen_transform_cipher",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--output", type=str,
                        default="data/transformation/pool/generated/cipher_scan.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    t0 = time.time()
    count = 0

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        for attempt in range(args.n * 8):
            if count >= args.n:
                break
            row = generate_one(rng)
            if row:
                f.write(json.dumps(row) + "\n")
                count += 1
                if count % 1000 == 0:
                    print(f"  {count}/{args.n}", flush=True)

    dt = time.time() - t0
    print(f"Generated {count} cipher-digit traces in {dt:.1f}s → {args.output}")


if __name__ == "__main__":
    main()
