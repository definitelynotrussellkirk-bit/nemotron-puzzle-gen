#!/usr/bin/env python3
"""Generate synthetic numeric transformation puzzles with scan-reject-lock traces.

Produces AB⊕CD = result puzzles with visible digits and operator symbols.
Traces show the scan process: test combos, reject wrong ones, lock winner.

Usage:
    python3 -m generators.gen_transform_numeric --n 5000
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone

from generators.trace_transform import (
    _make_operands, _calc, _fmt, SCAN_ORDER, COMBO_DISPLAY,
    build_numeric_trace,
)
from training.data import BOXED_INSTRUCTION


def generate_one(rng):
    """Generate one numeric transformation puzzle with scan-reject-lock trace."""
    # Pick real operation
    orders = ["BA_DC", "AB_CD", "AB_DC", "BA_CD"]
    ops = ["mul", "add", "sub", "absdiff", "cat", "rcat", "rsub",
           "add1", "addm1", "muladd1", "mulsub1"]
    # Weight toward common formats, include rare competition formats
    fmts = ["rev", "raw", "abs", "dsum",
            "opprefix", "opprefix_rev",
            "opsign", "opsign_always", "tailsign", "tailsign_always"]

    real_order = rng.choice(orders)
    # Weight ops toward competition distribution (add/sub/mul dominant, ±1 variants less common)
    real_op = rng.choices(ops, weights=[
        15, 15, 10, 8, 8, 5, 5,   # mul, add, sub, absdiff, cat, rcat, rsub
        8, 8, 8, 8,                # add1, addm1, muladd1, mulsub1
    ])[0]
    # Weight formats (rev/raw/abs dominant, opsign variants less common)
    real_fmt = rng.choices(fmts, weights=[
        25, 25, 15, 5,             # rev, raw, abs, dsum
        5, 3, 5, 3, 5, 3,          # opprefix, opprefix_rev, opsign, opsign_always, tailsign, tailsign_always
    ])[0]

    # Pick operator symbols (1-3, matching competition distribution)
    n_op_syms = rng.choices([1, 2, 3], weights=[0.30, 0.50, 0.20])[0]
    op_symbols = rng.sample(['+', '-', '*', '/', '|', '^', '&', '@', '#', '!'],
                            n_op_syms)

    # Generate 3-8 examples (competition uses 3-8, varied for diversity)
    n_examples = rng.choices([3, 4, 5, 6, 7, 8], weights=[15, 30, 25, 15, 10, 5])[0]
    examples = []
    for _ in range(n_examples):
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        L, R = _make_operands(a // 10, a % 10, b // 10, b % 10, real_order)
        val = _calc(L, R, real_op)
        if val is None:
            continue
        sym = rng.choice(op_symbols)
        fval = _fmt(val, real_fmt, op_char=sym)
        if fval is None:
            continue
        examples.append((f"{a}{sym}{b}", fval))

    if len(examples) < 2:
        return None

    # Generate query. Unseen operators need a separate ranking prior; keep this
    # pool execution-only and use gen_withheld_op for unseen-op training.
    qa = rng.randint(10, 99)
    qb = rng.randint(10, 99)
    all_syms = ['+', '-', '*', '/', '|', '^', '&', '@', '#', '!']
    used_op_symbols = sorted({lhs[2] for lhs, _ in examples if len(lhs) > 2})
    if not used_op_symbols:
        return None
    unseen_op = False
    if unseen_op:
        unseen_syms = [s for s in all_syms if s not in op_symbols]
        qsym = rng.choice(unseen_syms) if unseen_syms else rng.choice(op_symbols)
    else:
        qsym = rng.choice(used_op_symbols)
    query = f"{qa}{qsym}{qb}"
    qL, qR = _make_operands(qa // 10, qa % 10, qb // 10, qb % 10, real_order)
    qval = _calc(qL, qR, real_op)
    if qval is None:
        return None
    answer = _fmt(qval, real_fmt, op_char=qsym)
    if answer is None:
        return None

    # Build prompt (Alice's Wonderland wrapper)
    prompt_lines = [
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples:",
    ]
    # Shuffle example order for diversity (same puzzle, different presentation)
    shuffled_examples = list(examples)
    rng.shuffle(shuffled_examples)
    for lhs, rhs in shuffled_examples:
        prompt_lines.append(f"{lhs} = {rhs}")
    prompt_lines.append(f"Now, determine the result for: {query}")
    prompt = "\n".join(prompt_lines)

    # Build trace — MUST use shuffled_examples (same order as prompt!)
    result = build_numeric_trace(shuffled_examples, query, answer, rng)
    if result is None:
        return None

    trace, pred = result
    if pred != answer:
        return None

    mode = f"numeric_{real_order}_{real_op}_{real_fmt}"

    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"<think>\n{trace}\n</think>\n\\boxed{{{answer}}}"},
        ],
        "answer": answer,
        "id": f"gen_trans_num_{rng.randint(0, 999999):06d}",
        "puzzle_type": "transformation",
        "mode": mode,
        "generator": "gen_transform_numeric",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_contrastive_pair(rng):
    """Generate two rows with same operands but different correct combos.

    The pair shares the same query and most examples, but one decisive
    example flips the gold from combo_A to combo_B. Teaches decision boundaries:
    sub vs absdiff, raw vs rev, AB_CD vs BA_DC, etc.
    """
    orders = ["BA_DC", "AB_CD", "AB_DC", "BA_CD"]
    # Confusable pairs (the real error sources)
    confusable = [
        # Order confusion (biggest gap — 85% of failures involve wrong op or order)
        (("BA_DC", "mul", "raw"), ("AB_CD", "mul", "raw")),       # operand order
        (("BA_DC", "add", "raw"), ("AB_CD", "add", "raw")),       # operand order
        (("BA_DC", "sub", "raw"), ("AB_CD", "sub", "raw")),       # operand order
        (("BA_DC", "mul", "rev"), ("AB_CD", "mul", "rev")),       # order + style
        # Op confusion
        (("AB_CD", "sub", "raw"), ("AB_CD", "absdiff", "raw")),  # sign kills sub
        (("AB_CD", "add", "raw"), ("AB_CD", "mul", "raw")),       # add vs mul
        (("AB_CD", "add", "raw"), ("AB_CD", "add1", "raw")),      # +1 modifier
        (("AB_CD", "mul", "raw"), ("AB_CD", "muladd1", "raw")),   # *+1 modifier
        (("AB_CD", "sub", "raw"), ("AB_CD", "sub", "abs")),       # style: raw vs abs
        # Style confusion (51% of seen-op failures)
        (("AB_CD", "add", "raw"), ("AB_CD", "add", "rev")),       # raw vs rev
        (("BA_DC", "mul", "raw"), ("BA_DC", "mul", "rev")),       # raw vs rev
        (("AB_CD", "sub", "raw"), ("AB_CD", "sub", "rev")),       # raw vs rev
        (("AB_CD", "mul", "abs"), ("AB_CD", "mul", "raw")),       # abs vs raw
    ]

    combo_a, combo_b = rng.choice(confusable)

    op_char = rng.choice(['+', '-', '*', '/', '|', '^'])

    # For order-confusion pairs (different order, same op), shared agreement is rare.
    # Strategy: generate each combo's examples independently, share just the query.
    same_order = combo_a[0] == combo_b[0]

    # Find examples where combo_a and combo_b agree on first 2 examples
    # (only enforced when same order — different orders rarely agree)
    for _ in range(100):
        shared_examples = []
        for _ in range(2):
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            La, Ra = _make_operands(a//10, a%10, b//10, b%10, combo_a[0])
            va = _calc(La, Ra, combo_a[1])
            fa = _fmt(va, combo_a[2], op_char=op_char) if va is not None else None

            Lb, Rb = _make_operands(a//10, a%10, b//10, b%10, combo_b[0])
            vb = _calc(Lb, Rb, combo_b[1])
            fb = _fmt(vb, combo_b[2], op_char=op_char) if vb is not None else None

            if fa is None or fb is None:
                break
            if same_order and str(fa) != str(fb):
                break  # need agreement when same order
            shared_examples.append((a, b, str(fa), str(fb)))

        if len(shared_examples) < 2:
            continue

        # Find decisive witness: combo_a and combo_b disagree
        for _ in range(20):
            wa, wb = rng.randint(10, 99), rng.randint(10, 99)
            La, Ra = _make_operands(wa//10, wa%10, wb//10, wb%10, combo_a[0])
            va = _calc(La, Ra, combo_a[1])
            fa = _fmt(va, combo_a[2], op_char=op_char) if va is not None else None

            Lb, Rb = _make_operands(wa//10, wa%10, wb//10, wb%10, combo_b[0])
            vb = _calc(Lb, Rb, combo_b[1])
            fb = _fmt(vb, combo_b[2], op_char=op_char) if vb is not None else None

            if fa and fb and str(fa) != str(fb):
                break
        else:
            continue

        # Query
        qa, qb = rng.randint(10, 99), rng.randint(10, 99)

        # Build row for combo_a (witness uses combo_a's answer)
        examples_a = [(f"{a}{op_char}{b}", ans_a) for a, b, ans_a, _ in shared_examples]
        examples_a.append((f"{wa}{op_char}{wb}", str(fa)))
        query = f"{qa}{op_char}{qb}"
        La_q, Ra_q = _make_operands(qa//10, qa%10, qb//10, qb%10, combo_a[0])
        va_q = _calc(La_q, Ra_q, combo_a[1])
        answer_a = _fmt(va_q, combo_a[2], op_char=op_char) if va_q is not None else None

        # Build row for combo_b (witness uses combo_b's answer)
        examples_b = [(f"{a}{op_char}{b}", ans_b) for a, b, _, ans_b in shared_examples]
        examples_b.append((f"{wa}{op_char}{wb}", str(fb)))
        Lb_q, Rb_q = _make_operands(qa//10, qa%10, qb//10, qb%10, combo_b[0])
        vb_q = _calc(Lb_q, Rb_q, combo_b[1])
        answer_b = _fmt(vb_q, combo_b[2], op_char=op_char) if vb_q is not None else None

        if answer_a is None or answer_b is None or str(answer_a) == str(answer_b):
            continue

        rows = []
        for ex_list, answer, combo, tag in [
            (examples_a, str(answer_a), combo_a, "contrastive_A"),
            (examples_b, str(answer_b), combo_b, "contrastive_B"),
        ]:
            prompt_lines = [
                "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
                "Below are a few examples:",
            ]
            for lhs, rhs in ex_list:
                prompt_lines.append(f"{lhs} = {rhs}")
            prompt_lines.append(f"Now, determine the result for: {query}")
            prompt = "\n".join(prompt_lines)

            result = build_numeric_trace(ex_list, query, answer, rng)
            if result is None:
                continue
            trace, pred = result
            if pred != answer:
                continue

            rows.append({
                "messages": [
                    {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": f"<think>\n{trace}\n</think>\n\\boxed{{{answer}}}"},
                ],
                "answer": answer,
                "id": f"gen_trans_contrast_{rng.randint(0, 999999):06d}",
                "puzzle_type": "transformation",
                "mode": f"contrastive_{combo[0]}_{combo[1]}_{combo[2]}",
                "generator": "gen_transform_numeric",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "contrast_group": f"cg_{rng.randint(0, 999999):06d}",
            })

        if len(rows) == 2:
            return rows
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--output", type=str,
                        default="data/transformation/pool/generated/numeric_scan.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    t0 = time.time()
    count = 0

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    contrastive_count = 0
    with open(args.output, "w") as f:
        for attempt in range(args.n * 8):
            if count >= args.n:
                break
            # 10% contrastive pairs
            if rng.random() < 0.45:  # 45% contrastive pairs — order+style confusion is 85% of trans failures
                pair = generate_contrastive_pair(rng)
                if pair:
                    for row in pair:
                        f.write(json.dumps(row) + "\n")
                        count += 1
                        contrastive_count += 1
            else:
                row = generate_one(rng)
                if row:
                    f.write(json.dumps(row) + "\n")
                    count += 1
            if count % 1000 == 0:
                print(f"  {count}/{args.n}", flush=True)

    dt = time.time() - t0
    print(f"Generated {count} numeric traces ({contrastive_count} contrastive) in {dt:.1f}s → {args.output}")


if __name__ == "__main__":
    main()
