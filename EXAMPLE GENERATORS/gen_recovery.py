#!/usr/bin/env python3
"""Recovery trace generator — teach the model to spot and fix mistakes.

Format: The model sees a COMPLETE wrong solution, identifies the FIRST mistake,
then produces the correct solution from that point forward.

Structure:
  [Complete wrong solution as a block]

  First error: [identify exactly where it goes wrong]

  [Correct solution from scratch]
  \boxed{answer}

30% of rows show a CORRECT solution — model confirms "no errors" and proceeds.

Usage:
    python3 -m generators.gen_recovery --n 1000
"""
import argparse
import json
import random
import time
from datetime import datetime, timezone
from collections import defaultdict

from generators.trace_transform import (
    _make_operands, _calc, _fmt, COMBO_DISPLAY,
)
from generators.trace_compact import apply_shift, apply_gate, fmt, _xor_bits
from training.data import BOXED_INSTRUCTION

_SHIFTS = ["shr1", "shr2", "shr3", "shl1", "shl2", "shl3",
           "rol1", "rol2", "rol3", "ror1", "ror2", "ror3"]
_GATES = ["xor", "and", "or", "xnor", "and_not", "or_not"]

CORRECT_RATE = 0.30


# ============================================================
# BIT RECOVERY
# ============================================================

def gen_bit_recovery(rng):
    """Show a complete wrong bit solution, identify first error, produce correct solution.
    Error types: wrong gate (70%), wrong shift source (30%)."""
    src_names = rng.sample(_SHIFTS, 2)
    real_gate = rng.choice(_GATES)

    # Choose error type
    error_type = rng.choices(['wrong_gate', 'wrong_shift'], weights=[70, 30])[0]
    if error_type == 'wrong_shift':
        wrong_gate = real_gate  # gate is correct
        # Use a wrong source for the first input
        wrong_src = rng.choice([s for s in _SHIFTS if s != src_names[0]])
        wrong_src_names = [wrong_src, src_names[1]]
    else:
        wrong_gate = rng.choice([g for g in _GATES if g != real_gate])
        wrong_src_names = src_names

    examples = []
    for _ in range(4):
        x = rng.randint(0, 255)
        x_str = format(x, '08b')
        a = fmt(apply_shift(x, src_names[0]))
        b = fmt(apply_shift(x, src_names[1]))
        out = apply_gate(a, b, real_gate)
        examples.append((x_str, out))

    # Find first example where wrong solution fails
    witness_idx = None
    for i, (ex_x, ex_out) in enumerate(examples):
        wa = fmt(apply_shift(int(ex_x, 2), wrong_src_names[0]))
        wb = fmt(apply_shift(int(ex_x, 2), wrong_src_names[1]))
        wrong_out = apply_gate(wa, wb, wrong_gate)
        if wrong_out != ex_out:
            witness_idx = i
            break
    if witness_idx is None:
        return None

    # Query
    query_x = rng.randint(0, 255)
    query_str = format(query_x, '08b')
    qa = fmt(apply_shift(query_x, src_names[0]))
    qb = fmt(apply_shift(query_x, src_names[1]))
    real_answer = apply_gate(qa, qb, real_gate)
    # Wrong answer uses wrong sources/gate
    wqa = fmt(apply_shift(query_x, wrong_src_names[0]))
    wqb = fmt(apply_shift(query_x, wrong_src_names[1]))
    wrong_answer = apply_gate(wqa, wqb, wrong_gate)

    # Build prompt
    prompt_lines = ["Below are some examples of binary transformations:"]
    for inp, out in examples:
        prompt_lines.append(f"{inp} -> {out}")
    prompt_lines.append(f"What is the output for: {query_str}")
    prompt = "\n".join(prompt_lines)

    # === WRONG SOLUTION (FULL trace — Scan + Candidate + 2 Witnesses with GRID + Query) ===
    wrong_lines = ["Bit rule.", ""]
    wrong_lines.append("Scan:")
    for i, (inp, out) in enumerate(examples[:3]):
        wrong_lines.append(f"Ex{i+1}: {inp}→{out} ones:{inp.count('1')}→{out.count('1')}")
    wrong_lines.append("")
    wrong_lines.append("Try[1]:")
    wrong_lines.append(f"  A = {wrong_src_names[0]}(x)")
    wrong_lines.append(f"  B = {wrong_src_names[1]}(x)")
    wrong_lines.append(f"  output = {wrong_gate}(A,B)")
    wrong_lines.append("")

    # Witness 1 with full GRID — uses WRONG sources/gate
    c1x, c1out = examples[0]
    c1a = fmt(apply_shift(int(c1x, 2), wrong_src_names[0]))
    c1b = fmt(apply_shift(int(c1x, 2), wrong_src_names[1]))
    c1result = apply_gate(c1a, c1b, wrong_gate)
    wrong_lines.append(f"Witness 1 (Ex1): x={c1x}")
    wrong_lines.append(f"A={wrong_src_names[0]}({c1x})={c1a}")
    wrong_lines.append(f"B={wrong_src_names[1]}({c1x})={c1b}")
    wrong_lines.append(f"GRID(A,B,{wrong_gate}):")
    wrong_lines.append(f"{' '.join(c1a)}")
    wrong_lines.append(f"{' '.join(c1b)}")
    wrong_lines.append(f"{' '.join(c1result)}")
    c1_diff = _xor_bits(c1result, c1out)
    if c1_diff == '00000000':
        wrong_lines.append(f"  diff={c1_diff} → PASS")
    else:
        wrong_lines.append(f"  expected={c1out} diff={c1_diff} → FAIL")
    wrong_lines.append("")

    # Witness 2 with full GRID
    c2x, c2out = examples[1]
    c2a = fmt(apply_shift(int(c2x, 2), wrong_src_names[0]))
    c2b = fmt(apply_shift(int(c2x, 2), wrong_src_names[1]))
    c2result = apply_gate(c2a, c2b, wrong_gate)
    c2_diff = _xor_bits(c2result, c2out)
    wrong_lines.append(f"Witness 2 (Ex2): x={c2x}")
    wrong_lines.append(f"A={c2a}")
    wrong_lines.append(f"B={c2b}")
    if c2_diff == '00000000':
        wrong_lines.append(f"  diff={c2_diff} → PASS")
    else:
        wrong_lines.append(f"  expected={c2out} diff={c2_diff} → FAIL")
    wrong_lines.append("")

    # STOP at FAIL — never show Query after failed witnesses (R14: "never FAIL→Query")
    wrong_lines.append(f"Decision[1]: REJECT")

    # === IDENTIFY FIRST ERROR ===
    wx, wout = examples[witness_idx]
    wa = fmt(apply_shift(int(wx, 2), wrong_src_names[0]))
    wb = fmt(apply_shift(int(wx, 2), wrong_src_names[1]))
    wrong_check = apply_gate(wa, wb, wrong_gate)

    # === CORRECT SOLUTION (full trace: Try[2] + 2 Witnesses + Decision + Query) ===
    correct_lines = [
        "Try[2]:",
        f"  A = {src_names[0]}(x)",
        f"  B = {src_names[1]}(x)",
        f"  output = {real_gate}(A,B)",
        "",
    ]
    for ci in range(min(2, len(examples))):
        vx, vout = examples[ci]
        va = fmt(apply_shift(int(vx, 2), src_names[0]))
        vb = fmt(apply_shift(int(vx, 2), src_names[1]))
        vc = apply_gate(va, vb, real_gate)
        vdiff = _xor_bits(vc, vout)
        correct_lines.append(f"Witness {ci+1} (Ex{ci+1}): x={vx}")
        correct_lines.append(f"A={va} B={vb}")
        correct_lines.append(f"  diff={vdiff} → PASS")
        correct_lines.append("")
    correct_lines.append("Decision[2]: LOCK")
    correct_lines.append("")
    correct_lines.append(f"Query (using LOCK Try[2]): x={query_str}")
    correct_lines.append(f"A={qa} B={qb}")
    correct_lines.append(f"{real_gate}(A,B)={real_answer}")

    # === ASSEMBLE TRACE ===
    # 50% full restart, 50% patch (just fix from error point)
    is_patch = rng.random() < 0.5

    lines = [
        "--- Solution to review ---",
        *wrong_lines,
        "",
        f"First error: Check is wrong. {wrong_gate}({wa},{wb})={wrong_check} but Ex{witness_idx+1} output is {wout}. {'Gate' if error_type == 'wrong_gate' else 'Source'} is wrong — reject this rule.",
        "",
    ]

    if is_patch:
        # Patch: fix the gate, verify with witnesses, lock, query
        lines.append(f"Fix: replace {wrong_gate} with {real_gate}")
        lines.append("")
        lines.append("Try[1]:")
        lines.append(f"  A = {src_names[0]}(x)")
        lines.append(f"  B = {src_names[1]}(x)")
        lines.append(f"  output = {real_gate}(A,B)")
        lines.append("")
        for ci in range(min(2, len(examples))):
            vx, vout = examples[ci]
            va = fmt(apply_shift(int(vx, 2), src_names[0]))
            vb = fmt(apply_shift(int(vx, 2), src_names[1]))
            vc = apply_gate(va, vb, real_gate)
            vdiff = _xor_bits(vc, vout)
            lines.append(f"Witness {ci+1}: {real_gate}({va},{vb})={vc}")
            lines.append(f"  diff={vdiff} → PASS")
        lines.append("Decision[2]: LOCK")
        lines.append("")
        lines.append(f"Query (using LOCK Try[2]): {real_gate}({qa},{qb})={real_answer}")
        mode = "recovery_bit_patch"
    else:
        # Full restart
        lines.append("--- Correct solution ---")
        lines.extend(correct_lines)
        mode = "recovery_bit"

    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"<think>\n{chr(10).join(lines)}\n</think>\n\\boxed{{{real_answer}}}"},
        ],
        "answer": real_answer,
        "id": f"recovery_bit_{rng.randint(0, 999999):06d}",
        "puzzle_type": "bit_manipulation",
        "mode": mode,
        "generator": "gen_recovery",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def gen_bit_correct(rng):
    """Show a correct bit solution — model confirms no errors."""
    src_names = rng.sample(_SHIFTS, 2)
    real_gate = rng.choice(_GATES)

    examples = []
    for _ in range(4):
        x = rng.randint(0, 255)
        x_str = format(x, '08b')
        a = fmt(apply_shift(x, src_names[0]))
        b = fmt(apply_shift(x, src_names[1]))
        out = apply_gate(a, b, real_gate)
        examples.append((x_str, out))

    query_x = rng.randint(0, 255)
    query_str = format(query_x, '08b')
    qa = fmt(apply_shift(query_x, src_names[0]))
    qb = fmt(apply_shift(query_x, src_names[1]))
    answer = apply_gate(qa, qb, real_gate)

    prompt_lines = ["Below are some examples of binary transformations:"]
    for inp, out in examples:
        prompt_lines.append(f"{inp} -> {out}")
    prompt_lines.append(f"What is the output for: {query_str}")
    prompt = "\n".join(prompt_lines)

    # Show the correct solution — same Candidate/Witness/LOCK program
    sol_lines = [
        "Try[1]:",
        f"  A = {src_names[0]}(x)",
        f"  B = {src_names[1]}(x)",
        f"  output = {real_gate}(A,B)",
        "",
    ]
    for ci in range(min(2, len(examples))):
        vx, vout = examples[ci]
        va = fmt(apply_shift(int(vx, 2), src_names[0]))
        vb = fmt(apply_shift(int(vx, 2), src_names[1]))
        vc = apply_gate(va, vb, real_gate)
        if vc != vout:
            return None
        vdiff = _xor_bits(vc, vout)
        sol_lines.append(f"Witness {ci+1}: {real_gate}({va},{vb})={vc}")
        sol_lines.append(f"  diff={vdiff} → PASS")
    sol_lines.append("Decision[1]: LOCK")
    sol_lines.append("")
    sol_lines.append(f"Query: {real_gate}({qa},{qb})={answer}")

    lines = [
        "--- Solution to review ---",
        *sol_lines,
        "",
        "No errors found. Solution is correct.",
    ]

    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"<think>\n{chr(10).join(lines)}\n</think>\n\\boxed{{{answer}}}"},
        ],
        "answer": answer,
        "id": f"recovery_bit_ok_{rng.randint(0, 999999):06d}",
        "puzzle_type": "bit_manipulation",
        "mode": "recovery_bit_correct",
        "generator": "gen_recovery",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# TRANSFORMATION RECOVERY
# ============================================================

def gen_trans_recovery(rng):
    """Show a complete wrong trans solution, identify first error, produce correct.
    Error types: wrong combo at scan (70%), correct lock but wrong order at query (30%)."""
    orders = ["BA_DC", "AB_CD", "AB_DC", "BA_CD"]
    ops = ["mul", "add", "sub", "absdiff", "add1", "addm1", "muladd1", "mulsub1"]
    fmts_list = ["rev", "raw", "abs"]
    op_char = rng.choice(['+', '-', '*', '/', '|'])

    real_combo = (rng.choice(orders), rng.choice(ops), rng.choice(fmts_list))

    # Close-miss wrong combo
    wrong_candidates = []
    for o in orders:
        for op in ops:
            for f in fmts_list:
                if (o, op, f) == real_combo: continue
                diff = (o != real_combo[0]) + (op != real_combo[1]) + (f != real_combo[2])
                if diff == 1:
                    wrong_candidates.append((o, op, f))
    if not wrong_candidates:
        return None
    wrong_combo = rng.choice(wrong_candidates)

    for _ in range(50):
        examples = []
        for _ in range(3):
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            L, R = _make_operands(a//10, a%10, b//10, b%10, real_combo[0])
            val = _calc(L, R, real_combo[1])
            fval = _fmt(val, real_combo[2], op_char=op_char) if val is not None else None
            if fval is None or len(str(fval)) > 5:
                break
            examples.append((a, b, str(fval)))
        if len(examples) < 3:
            continue

        # Find FIRST example where wrong combo fails
        witness = None
        for ei, (a, b, exp) in enumerate(examples):
            wL, wR = _make_operands(a//10, a%10, b//10, b%10, wrong_combo[0])
            wval = _calc(wL, wR, wrong_combo[1])
            wfval = _fmt(wval, wrong_combo[2], op_char=op_char) if wval is not None else None
            if wfval is not None and str(wfval) != exp:
                witness = (ei, a, b, exp, str(wfval), wL, wR)
                break
        if witness is None:
            continue

        wi, wa, wb, wexp, wgot, wL, wR = witness

        # Query
        qa, qb = rng.randint(10, 99), rng.randint(10, 99)
        qL, qR = _make_operands(qa//10, qa%10, qb//10, qb%10, real_combo[0])
        qval = _calc(qL, qR, real_combo[1])
        qfval = _fmt(qval, real_combo[2], op_char=op_char) if qval is not None else None
        if qfval is None:
            continue

        # Wrong query answer
        wqL, wqR = _make_operands(qa//10, qa%10, qb//10, qb%10, wrong_combo[0])
        wqval = _calc(wqL, wqR, wrong_combo[1])
        wqfval = _fmt(wqval, wrong_combo[2], op_char=op_char) if wqval is not None else None

        wd = COMBO_DISPLAY.get(wrong_combo[0], wrong_combo[0])
        rd = COMBO_DISPLAY.get(real_combo[0], real_combo[0])

        prompt_lines = [
            "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
            "Below are a few examples:",
        ]
        for a, b, fv in examples:
            prompt_lines.append(f"{a}{op_char}{b} = {fv}")
        prompt_lines.append(f"Now, determine the result for: {qa}{op_char}{qb}")
        prompt = "\n".join(prompt_lines)

        # Diagnose what axis is wrong
        if wrong_combo[0] != real_combo[0]:
            diagnosis = f"Wrong operand order. {wd} gives {wgot}, expected {wexp}."
        elif wrong_combo[1] != real_combo[1]:
            diagnosis = f"Wrong operation. {wrong_combo[1]} gives {wgot}, expected {wexp}."
        else:
            diagnosis = f"Wrong format. {wrong_combo[2]} gives {wgot}, expected {wexp}."

        # Wrong solution — uses canonical format
        # Wrong solution — STOP at MISMATCH, never Query (R14: never FAIL→Query)
        wrong_lines = [
            "Numeric visible.",
            "",
            f"Try: order={wd} op={wrong_combo[1]} style={wrong_combo[2]}",
            f"Ex1: {examples[0][2]} → MATCH" if wi > 0 else f"Ex1: {wgot} vs {wexp} → MISMATCH",
            f"Decision: REJECT",
        ]

        # Correct solution — uses canonical format (same headers as main traces)
        correct_lines = [
            "Numeric visible.",
            "",
            f"Lock[{op_char}]: order={rd} op={real_combo[1]} style={real_combo[2]}",
        ]
        for vi in range(2):
            va, vb, vexp = examples[vi]
            vL, vR = _make_operands(va//10, va%10, vb//10, vb%10, real_combo[0])
            vval = _calc(vL, vR, real_combo[1])
            vfval = _fmt(vval, real_combo[2], op_char=op_char)
            correct_lines.append(f"  Ex{vi+1}: {vfval} vs {vexp} → {'MATCH' if str(vfval) == str(vexp) else 'MISMATCH'}")
        # Explicit operand assembly
        a_str = str(qa).zfill(2) if qa < 10 else str(qa)
        b_str = str(qb).zfill(2) if qb < 10 else str(qb)
        correct_lines.append(f"")
        correct_lines.append(f"Query: {qa}{op_char}{qb}")
        correct_lines.append(f"  Order {rd}: L={qL} R={qR}")
        correct_lines.append(f"  {real_combo[1]}({qL},{qR})={qval} {real_combo[2]}={qfval}")

        lines = [
            "--- Solution to review ---",
            *wrong_lines,
            "",
            f"First error: Ex{wi+1} verification. {diagnosis}",
            "",
            "--- Correct solution ---",
            *correct_lines,
        ]

        return {
            "messages": [
                {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                {"role": "assistant", "content": f"<think>\n{chr(10).join(lines)}\n</think>\n\\boxed{{{qfval}}}"},
            ],
            "answer": str(qfval),
            "id": f"recovery_trans_{rng.randint(0, 999999):06d}",
            "puzzle_type": "transformation",
            "mode": "recovery_trans",
            "generator": "gen_recovery",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    return None


def gen_trans_correct(rng):
    """Show a correct trans solution — model confirms no errors."""
    orders = ["BA_DC", "AB_CD", "AB_DC", "BA_CD"]
    ops = ["mul", "add", "sub", "absdiff", "add1", "addm1"]
    fmts_list = ["rev", "raw", "abs"]
    op_char = rng.choice(['+', '-', '*', '/', '|'])

    real_combo = (rng.choice(orders), rng.choice(ops), rng.choice(fmts_list))

    examples = []
    for _ in range(3):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        L, R = _make_operands(a//10, a%10, b//10, b%10, real_combo[0])
        val = _calc(L, R, real_combo[1])
        fval = _fmt(val, real_combo[2], op_char=op_char) if val is not None else None
        if fval is None or len(str(fval)) > 5:
            return None
        examples.append((a, b, str(fval)))

    qa, qb = rng.randint(10, 99), rng.randint(10, 99)
    qL, qR = _make_operands(qa//10, qa%10, qb//10, qb%10, real_combo[0])
    qval = _calc(qL, qR, real_combo[1])
    qfval = _fmt(qval, real_combo[2], op_char=op_char) if qval is not None else None
    if qfval is None:
        return None

    rd = COMBO_DISPLAY.get(real_combo[0], real_combo[0])

    prompt_lines = [
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples:",
    ]
    for a, b, fv in examples:
        prompt_lines.append(f"{a}{op_char}{b} = {fv}")
    prompt_lines.append(f"Now, determine the result for: {qa}{op_char}{qb}")
    prompt = "\n".join(prompt_lines)

    a_str = str(qa).zfill(2) if qa < 10 else str(qa)
    b_str = str(qb).zfill(2) if qb < 10 else str(qb)

    sol_lines = [
        "Numeric visible.",
        "",
        f"Lock[{op_char}]: order={rd} op={real_combo[1]} style={real_combo[2]}",
        f"  Ex1: {examples[0][2]} → MATCH",
        f"  Ex2: {examples[1][2]} → MATCH",
        "",
        f"Query: {qa}{op_char}{qb}",
        f"  Order {rd}: L={qL} R={qR}",
        f"  {real_combo[1]}({qL},{qR})={qval} {real_combo[2]}={qfval}",
    ]

    lines = [
        "--- Solution to review ---",
        *sol_lines,
        "",
        "No errors found. Solution is correct.",
    ]

    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"<think>\n{chr(10).join(lines)}\n</think>\n\\boxed{{{qfval}}}"},
        ],
        "answer": str(qfval),
        "id": f"recovery_trans_ok_{rng.randint(0, 999999):06d}",
        "puzzle_type": "transformation",
        "mode": "recovery_trans_correct",
        "generator": "gen_recovery",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# ENCRYPTION RECOVERY
# ============================================================

def gen_enc_recovery(rng):
    """Show a complete wrong enc solution (swapped mapping), identify first error, fix."""
    from generators.microskill_framework import load_vocab
    vocab = load_vocab()
    if not vocab:
        return None
    by_len = defaultdict(list)
    for w in vocab:
        by_len[len(w)].append(w)

    words = []
    for _ in range(30):
        w = rng.choice([v for v in vocab if len(v) >= 4])
        if w not in words:
            words.append(w)
        if len(words) == 3:
            break
    if len(words) < 3:
        return None

    alpha = list('abcdefghijklmnopqrstuvwxyz')
    plain_perm = list('abcdefghijklmnopqrstuvwxyz')
    rng.shuffle(plain_perm)
    c2p = dict(zip(alpha, plain_perm))
    p2c = {v: k for k, v in c2p.items()}

    cipher_words = [''.join(p2c[ch] for ch in w) for w in words]

    # Swap two adjacent entries
    swap_pos = rng.randint(0, 24)
    c1, c2 = alpha[swap_pos], alpha[swap_pos + 1]
    bad_c2p = dict(c2p)
    bad_c2p[c1], bad_c2p[c2] = bad_c2p[c2], bad_c2p[c1]

    # Find FIRST affected word + position
    first_bad = None
    for i, (cw, gw) in enumerate(zip(cipher_words, words)):
        bad_decode = ''.join(bad_c2p[c] for c in cw)
        if bad_decode != gw:
            for pos in range(len(cw)):
                if bad_decode[pos] != gw[pos]:
                    first_bad = (i, cw, gw, bad_decode, pos, cw[pos], bad_decode[pos], gw[pos])
                    break
            break
    if first_bad is None:
        return None

    idx, cw_bad, gold_word, bad_word, bad_pos, cipher_char, got_plain, need_plain = first_bad
    bad_in_vocab = bad_word in vocab

    # Example for context
    ex_word = rng.choice([v for v in vocab if len(v) >= 5 and v not in words])
    ex_cipher = ''.join(p2c[ch] for ch in ex_word)

    prompt_lines = [
        "The following is encrypted using a one-to-one letter substitution:",
        f"Example: {ex_cipher} -> {ex_word}",
        f"Decrypt: {' '.join(cipher_words)}",
    ]
    prompt = "\n".join(prompt_lines)

    # Wrong solution: show wrong table + wrong decode
    ranges = [('a', 'e'), ('f', 'j'), ('k', 'o'), ('p', 't'), ('u', 'z')]
    wrong_table = []
    for start, end in ranges:
        pairs = [f"{chr(i)}={bad_c2p[chr(i)]}" for i in range(ord(start), ord(end) + 1)]
        wrong_table.append(f"{start}-{end}: {' '.join(pairs)}")

    wrong_lines = list(wrong_table)
    wrong_decodes = []
    for cw, gw in zip(cipher_words, words):
        bd = ''.join(bad_c2p[c] for c in cw)
        wrong_decodes.append(bd)
        wrong_lines.append(f"{cw} → {bd}")
    wrong_answer = ' '.join(wrong_decodes)
    wrong_lines.append(f"Answer: {wrong_answer}")

    # Correct table
    correct_table = []
    for start, end in ranges:
        pairs = [f"{chr(i)}={c2p[chr(i)]}" for i in range(ord(start), ord(end) + 1)]
        correct_table.append(f"{start}-{end}: {' '.join(pairs)}")

    # First error identification
    if not bad_in_vocab:
        error_desc = f"Word '{bad_word}' is not in the 77-word vocabulary. Position {bad_pos}: cipher '{cipher_char}' maps to '{got_plain}' but should be '{need_plain}'. Swap {c1}↔{c2}."
    else:
        error_desc = f"Decode of '{cw_bad}' gives '{bad_word}' but position {bad_pos} is wrong: cipher '{cipher_char}'→'{got_plain}' should be →'{need_plain}'. Swap {c1}↔{c2}."

    lines = [
        "--- Solution to review ---",
        *wrong_lines,
        "",
        f"First error: {error_desc}",
        "",
        "--- Correct solution ---",
        *correct_table,
    ]
    for cw, gw in zip(cipher_words, words):
        lines.append(f"{cw} → {gw}")
    answer = ' '.join(words)
    lines.append(f"Answer: {answer}")

    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"<think>\n{chr(10).join(lines)}\n</think>\n\\boxed{{{answer}}}"},
        ],
        "answer": answer,
        "id": f"recovery_enc_{rng.randint(0, 999999):06d}",
        "puzzle_type": "encryption",
        "mode": "recovery_enc",
        "generator": "gen_recovery",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# MAIN
# ============================================================

def gen_bit_false_identity(rng):
    """Recovery: model wrongly claims identity when it's actually a gate.
    This targets 49% of bit failures where model says output=input but it's not."""
    from generators.trace_compact import apply_shift, apply_gate, fmt, _xor_bits

    src_names = rng.sample(_SHIFTS, 2)
    real_gate = rng.choice(_GATES)

    examples = []
    for _ in range(6):
        x = rng.randint(1, 254)
        x_str = format(x, '08b')
        a = fmt(apply_shift(x, src_names[0]))
        b = fmt(apply_shift(x, src_names[1]))
        out = apply_gate(a, b, real_gate)
        examples.append((x_str, out))

    # Make sure output != input (not actually identity)
    if all(inp == out for inp, out in examples):
        return None

    query_x = rng.randint(1, 254)
    query_str = format(query_x, '08b')
    qa = fmt(apply_shift(query_x, src_names[0]))
    qb = fmt(apply_shift(query_x, src_names[1]))
    real_answer = apply_gate(qa, qb, real_gate)

    prompt_lines = ["Below are some examples of binary transformations:"]
    for inp, out in examples:
        prompt_lines.append(f"{inp} -> {out}")
    prompt_lines.append(f"What is the output for: {query_str}")
    prompt = "\n".join(prompt_lines)

    # Wrong solution: claims identity
    wrong_answer = query_str  # identity would just copy input
    n_diff = sum(1 for a, b in zip(examples[0][0], examples[0][1]) if a != b)
    diff_ex1 = _xor_bits(examples[0][0], examples[0][1])

    wrong_lines = [
        "Bit rule.", "",
        "Step 1: all outputs same? No",
        f"Step 2: output=input? Ex1: {examples[0][0]} vs {examples[0][1]} → {n_diff} positions differ → No",
        "",
        "Try[1]: output = x (identity)",
        f"  Witness (Ex1): x={examples[0][0]}",
        f"  output={examples[0][0]}",
        f"  expected={examples[0][1]}",
        f"  XOR: {' '.join(f'{a}⊕{b}={int(a)^int(b)}' for a,b in zip(examples[0][0], examples[0][1]))}",
        f"  diff={diff_ex1} → FAIL",
        f"  Decision[1]: REJECT",
        "",
        f"First error: Step 2 already showed output≠input ({n_diff} positions differ). Identity is impossible.",
        "",
        f"Try[2]: A={src_names[0]}(x), B={src_names[1]}(x), output={real_gate}(A,B)",
    ]
    # Verify on 2 examples
    for ci in range(2):
        vx, vout = examples[ci]
        va = fmt(apply_shift(int(vx, 2), src_names[0]))
        vb = fmt(apply_shift(int(vx, 2), src_names[1]))
        vc = apply_gate(va, vb, real_gate)
        vdiff = _xor_bits(vc, vout)
        wrong_lines.append(f"  Witness {ci+1}: {real_gate}({va},{vb})={vc}")
        wrong_lines.append(f"    diff={vdiff} → PASS")
    wrong_lines.append(f"  Decision[2]: LOCK")
    wrong_lines.append("")
    wrong_lines.append(f"Query (using LOCK Try[2]): x={query_str}")
    wrong_lines.append(f"  A={qa} B={qb}")
    wrong_lines.append(f"  {real_gate}(A,B)={real_answer}")

    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"<think>\n{chr(10).join(wrong_lines)}\n</think>\n\\boxed{{{real_answer}}}"},
        ],
        "answer": real_answer,
        "id": f"recovery_bit_false_id_{rng.randint(0, 999999):06d}",
        "puzzle_type": "bit_manipulation",
        "mode": "recovery_bit_false_identity",
        "generator": "gen_recovery",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }



def gen_bit_missed_const(rng):
    """Recovery: puzzle is CONST but model tries a gate. Teaches Step 1 detection."""
    from generators.trace_compact import _xor_bits

    const_val = rng.choice([0, 255, rng.randint(1, 254)])
    const_str = format(const_val, '08b')
    
    examples = []
    for _ in range(6):
        x = rng.randint(0, 255)
        examples.append((format(x, '08b'), const_str))
    
    query_x = rng.randint(0, 255)
    query_str = format(query_x, '08b')

    prompt_lines = ['Below are some examples of binary transformations:']
    for inp, out in examples:
        prompt_lines.append(f'{inp} -> {out}')
    prompt_lines.append(f'What is the output for: {query_str}')
    prompt = chr(10).join(prompt_lines)

    # Wrong: model tries xor gate
    wrong_gate = rng.choice(['xor', 'and', 'or'])
    wrong_answer = format(rng.randint(0, 255), '08b')  # random wrong

    lines = [
        'Bit rule.', '',
        f'Step 1: all outputs same? Yes → {const_str}',
        f'All {len(examples)} outputs = {const_str}. This is CONST.',
        '',
        f'Decision: LOCK (constant output = {const_str})',
        '',
        f'Query: x={query_str}',
        f'output={const_str} (constant, independent of input)',
    ]

    return {
        'messages': [
            {'role': 'user', 'content': prompt + BOXED_INSTRUCTION},
            {'role': 'assistant', 'content': "<think>\n" + "\n".join(lines) + "\n</think>\n\\boxed{" + const_str + "}"},
        ],
        'answer': const_str,
        'id': f'recovery_bit_const_{rng.randint(0, 999999):06d}',
        'puzzle_type': 'bit_manipulation',
        'mode': 'recovery_bit_const',
        'generator': 'gen_recovery',
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }

GENERATORS = {
    "bit": gen_bit_recovery,
    "bit_false_id": gen_bit_false_identity,
    "bit_const": gen_bit_missed_const,
    "trans": gen_trans_recovery,
    "enc": gen_enc_recovery,
}

def gen_enc_correct(rng):
    """Show a correct enc solution — model confirms no errors."""
    from generators.microskill_framework import load_vocab
    vocab = load_vocab()
    if not vocab: return None

    words = []
    for _ in range(30):
        w = rng.choice([v for v in vocab if len(v) >= 4])
        if w not in words:
            words.append(w)
        if len(words) == 3: break
    if len(words) < 3: return None

    alpha = list('abcdefghijklmnopqrstuvwxyz')
    plain_perm = list('abcdefghijklmnopqrstuvwxyz')
    rng.shuffle(plain_perm)
    c2p = dict(zip(alpha, plain_perm))
    p2c = {v: k for k, v in c2p.items()}

    cipher_words = [''.join(p2c[ch] for ch in w) for w in words]
    ex_word = rng.choice([v for v in vocab if len(v) >= 5 and v not in words])
    ex_cipher = ''.join(p2c[ch] for ch in ex_word)

    prompt = "\n".join([
        "The following is encrypted using a one-to-one letter substitution:",
        f"Example: {ex_cipher} -> {ex_word}",
        f"Decrypt: {' '.join(cipher_words)}",
    ])

    ranges = [('a','e'), ('f','j'), ('k','o'), ('p','t'), ('u','z')]
    sol_lines = []
    for start, end in ranges:
        slots = [c2p[chr(i)] for i in range(ord(start), ord(end) + 1)]
        sol_lines.append(f"{start}-{end}:{','.join(slots)}")
    for cw, gw in zip(cipher_words, words):
        sol_lines.append(f"{cw} → {gw}")
    answer = ' '.join(words)
    sol_lines.append(f"Answer: {answer}")

    lines = [
        "--- Solution to review ---",
        *sol_lines,
        "",
        "No errors found. Solution is correct.",
    ]

    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"<think>\n{chr(10).join(lines)}\n</think>\n\\boxed{{{answer}}}"},
        ],
        "answer": answer,
        "id": f"recovery_enc_ok_{rng.randint(0, 999999):06d}",
        "puzzle_type": "encryption",
        "mode": "recovery_enc_correct",
        "generator": "gen_recovery",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


CORRECT_GENERATORS = {
    "bit": gen_bit_correct,
    "trans": gen_trans_correct,
    "enc": gen_enc_correct,
}

WEIGHTS = {
    "bit": 0.25,         # 35% — standard bit recovery (wrong gate/shift)
    "bit_false_id": 0.20,
    "bit_const": 0.10,# 25% — false identity recovery (49% of bit failures!)
    "enc": 0.15,         # 15% — maintenance
    "trans": 0.25,       # 25% — wrong op selection
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--output", type=str,
                        default="data/recovery/recovery_traces.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    rng = random.Random(args.seed)
    t0 = time.time()
    count = 0
    by_type = {"bit": 0, "bit_false_id": 0, "bit_const": 0, "trans": 0, "enc": 0}

    with open(args.output, "w") as f:
        for attempt in range(args.n * 5):
            if count >= args.n:
                break

            is_correct = rng.random() < CORRECT_RATE

            r = rng.random()
            cumulative = 0
            chosen = "bit"
            for typ, weight in WEIGHTS.items():
                cumulative += weight
                if r < cumulative:
                    chosen = typ
                    break

            if is_correct and chosen in CORRECT_GENERATORS:
                row = CORRECT_GENERATORS[chosen](rng)
            else:
                row = GENERATORS[chosen](rng)

            if row:
                f.write(json.dumps(row) + "\n")
                count += 1
                by_type[chosen] += 1
                if count % 200 == 0:
                    print(f"  {count}/{args.n}", flush=True)

    dt = time.time() - t0
    print(f"Generated {count} recovery traces in {dt:.1f}s → {args.output}")
    for t, c in sorted(by_type.items()):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
