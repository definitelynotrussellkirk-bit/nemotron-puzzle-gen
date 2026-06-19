#!/usr/bin/env python3
"""Generate bit manipulation micro-skill curriculum.

Three stages that teach the model to EXECUTE string-level bit operations
rather than faking intermediate values.

Stage 1: Individual operations (5000 examples)
  - 29 shift/rotate/not ops: shr1-7, shl1-7, rol1-7, ror1-7, not
  - 8 gate ops: xnor, xor, and, or, nand, nor, and_not, or_not
  Each shows the string-manipulation recipe and result.

Stage 2: Two-step composition (3000 examples)
  - A shift/rotate applied to get operand, then a gate combines two operands.

Stage 3: Three-step composition (2000 examples)
  - Three shifts produce A, B, C; two gates combine them using real family formulas.

Also generates no-jump full puzzle traces that show step-by-step execution
for the same puzzle format as competition data.

Usage:
    python3 -m generators.gen_bit_microskills --stage 1 --n 5000
    python3 -m generators.gen_bit_microskills --stage 2 --n 3000
    python3 -m generators.gen_bit_microskills --stage 3 --n 2000
    python3 -m generators.gen_bit_microskills --stage nojump --n 5000
    python3 -m generators.gen_bit_microskills --all
"""
import argparse
import json
import random
import time
from datetime import datetime, timezone

from training.data import BOXED_INSTRUCTION

TRAINING_TAG = "[Alice's Training House] "

BYTE = 0xFF


# ── Primitive operations ──────────────────────────────────────────────

def _fmt(v):
    return format(v & BYTE, '08b')


def shl(x, k):
    return (x << k) & BYTE


def shr(x, k):
    return (x >> k) & BYTE


def rol(x, k):
    return ((x << k) | (x >> (8 - k))) & BYTE


def ror(x, k):
    return ((x >> k) | (x << (8 - k))) & BYTE


def bit_not(x):
    return (~x) & BYTE


# Gate functions (two 8-bit inputs -> one 8-bit output)
def gate_xnor(a, b): return (~(a ^ b)) & BYTE
def gate_xor(a, b):  return (a ^ b) & BYTE
def gate_and(a, b):  return (a & b) & BYTE
def gate_or(a, b):   return (a | b) & BYTE
def gate_nand(a, b): return (~(a & b)) & BYTE
def gate_nor(a, b):  return (~(a | b)) & BYTE
def gate_and_not(a, b): return (a & (~b)) & BYTE  # A & ~B
def gate_or_not(a, b):  return (a | (~b)) & BYTE  # A | ~B


# ── Shift/rotate trace recipes ──────────────────────────────────────

def _trace_shr(bits_str, k):
    """shr{k}: prepend {k} zeros, drop last {k}."""
    prepended = '0' * k + bits_str
    result = prepended[:8]
    recipe = f"shr{k}: prepend {k} zero{'s' if k > 1 else ''}, drop last {k}"
    return recipe, result


def _trace_shl(bits_str, k):
    """shl{k}: drop first {k}, append {k} zeros."""
    appended = bits_str + '0' * k
    result = appended[k:k+8]
    recipe = f"shl{k}: drop first {k}, append {k} zero{'s' if k > 1 else ''}"
    return recipe, result


def _trace_rol(bits_str, k):
    """rol{k}: move first {k} to end."""
    result = bits_str[k:] + bits_str[:k]
    recipe = f"rol{k}: move first {k} to end"
    return recipe, result


def _trace_ror(bits_str, k):
    """ror{k}: move last {k} to front."""
    result = bits_str[8-k:] + bits_str[:8-k]
    recipe = f"ror{k}: move last {k} to front"
    return recipe, result


def _trace_not(bits_str):
    """not: flip each bit."""
    result = ''.join('1' if c == '0' else '0' for c in bits_str)
    recipe = "not: flip each bit"
    return recipe, result


def _verify_shift(op_type, k, x_int):
    """Verify string-based trace matches integer computation."""
    bits_str = _fmt(x_int)
    if op_type == 'shr':
        _, result = _trace_shr(bits_str, k)
        expected = _fmt(shr(x_int, k))
    elif op_type == 'shl':
        _, result = _trace_shl(bits_str, k)
        expected = _fmt(shl(x_int, k))
    elif op_type == 'rol':
        _, result = _trace_rol(bits_str, k)
        expected = _fmt(rol(x_int, k))
    elif op_type == 'ror':
        _, result = _trace_ror(bits_str, k)
        expected = _fmt(ror(x_int, k))
    elif op_type == 'not':
        _, result = _trace_not(bits_str)
        expected = _fmt(bit_not(x_int))
    else:
        raise ValueError(f"Unknown op_type: {op_type}")
    assert result == expected, f"{op_type}{k}({bits_str}): trace={result} != int={expected}"
    return result


# ── Gate trace recipes ──────────────────────────────────────────────

GATE_OPS = {
    'xnor':    (gate_xnor,    "same->1, diff->0"),
    'xor':     (gate_xor,     "diff->1, same->0"),
    'and':     (gate_and,     "both 1->1, else 0"),
    'or':      (gate_or,      "either 1->1, else 0"),
    'nand':    (gate_nand,    "both 1->0, else 1"),
    'nor':     (gate_nor,     "both 0->1, else 0"),
    'and_not': (gate_and_not, "A=1 and B=0->1, else 0"),
    'or_not':  (gate_or_not,  "A=1 or B=0->1, else 0"),
}


def _trace_gate(gate_name, a_str, b_str):
    """Produce vertical position-by-position gate trace.

    Returns (trace_lines, result_str).
    """
    fn, rule_text = GATE_OPS[gate_name]
    a_int = int(a_str, 2)
    b_int = int(b_str, 2)
    result_int = fn(a_int, b_int)
    result_str = _fmt(result_int)

    # Vertical aligned display with spaces
    a_spaced = ' '.join(a_str)
    b_spaced = ' '.join(b_str)
    r_spaced = ' '.join(result_str)

    lines = [
        f"{gate_name}: {rule_text}",
        f"  {a_spaced}",
        f"  {b_spaced}",
        f"  {r_spaced}",
    ]
    return lines, result_str


def _verify_gate(gate_name, a_int, b_int):
    """Verify string-based gate trace matches integer computation."""
    a_str = _fmt(a_int)
    b_str = _fmt(b_int)
    fn, _ = GATE_OPS[gate_name]
    _, result_str = _trace_gate(gate_name, a_str, b_str)
    expected = _fmt(fn(a_int, b_int))
    assert result_str == expected, f"{gate_name}({a_str}, {b_str}): trace={result_str} != int={expected}"
    return result_str


# ── Stage 1: Individual operations ──────────────────────────────────

def _gen_stage1_shift(rng, op_type, k):
    """Generate one Stage 1 shift/rotate/not example."""
    x = rng.randint(0, 255)
    bits = _fmt(x)

    if op_type == 'not':
        recipe, result = _trace_not(bits)
        op_str = f"not({bits})"
        _verify_shift('not', 0, x)
    else:
        if op_type == 'shr':
            recipe, result = _trace_shr(bits, k)
        elif op_type == 'shl':
            recipe, result = _trace_shl(bits, k)
        elif op_type == 'rol':
            recipe, result = _trace_rol(bits, k)
        elif op_type == 'ror':
            recipe, result = _trace_ror(bits, k)
        op_str = f"{op_type}{k}({bits})"
        _verify_shift(op_type, k, x)

    prompt = f"Compute {op_str}."
    reasoning = f"<think>\n[TRAINING DRILL]\n{recipe} \u2192 {result}\n</think>"
    answer = result

    return {
        "messages": [
            {"role": "user", "content": TRAINING_TAG + prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"{reasoning}\n\\boxed{{{answer}}}"},
        ],
        "answer": answer,
        "id": f"microskill_s1_{op_type}{k}_{rng.randint(0, 999999):06d}",
        "puzzle_type": "bit_manipulation",
        "mode": "microskill_stage1",
        "generator": "gen_bit_microskills",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _gen_stage1_gate(rng, gate_name):
    """Generate one Stage 1 gate example."""
    a = rng.randint(0, 255)
    b = rng.randint(0, 255)
    a_str = _fmt(a)
    b_str = _fmt(b)

    _verify_gate(gate_name, a, b)
    lines, result = _trace_gate(gate_name, a_str, b_str)

    prompt = f"Compute {gate_name}({a_str}, {b_str})."
    trace_body = '\n'.join(lines)
    reasoning = f"<think>\n[TRAINING DRILL]\n{trace_body}\n</think>"

    return {
        "messages": [
            {"role": "user", "content": TRAINING_TAG + prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"{reasoning}\n\\boxed{{{result}}}"},
        ],
        "answer": result,
        "id": f"microskill_s1_{gate_name}_{rng.randint(0, 999999):06d}",
        "puzzle_type": "bit_manipulation",
        "mode": "microskill_stage1",
        "generator": "gen_bit_microskills",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_stage1(rng, n=5000):
    """Generate Stage 1: individual operations."""
    rows = []

    # 29 shift/rotate/not ops: ~170 each = 4930
    shift_ops = []
    for op in ('shr', 'shl', 'rol', 'ror'):
        for k in range(1, 8):
            shift_ops.append((op, k))
    shift_ops.append(('not', 0))  # 29 total

    # Allocate ~75% to shift/rotate/not (29 ops), ~25% to gates (8 ops)
    # This gives roughly equal per-op counts across both categories
    n_shift_total = int(n * 0.75)
    n_per_shift = max(1, n_shift_total // len(shift_ops))
    # Adjust so shift + gate = n
    actual_shift = n_per_shift * len(shift_ops)
    n_gate_total = n - actual_shift

    for op_type, k in shift_ops:
        for _ in range(n_per_shift):
            rows.append(_gen_stage1_shift(rng, op_type, k))

    gate_names = list(GATE_OPS.keys())
    n_per_gate = n_gate_total // len(gate_names)
    remainder = n_gate_total - n_per_gate * len(gate_names)

    for i, gate_name in enumerate(gate_names):
        count = n_per_gate + (1 if i < remainder else 0)
        for _ in range(count):
            rows.append(_gen_stage1_gate(rng, gate_name))

    rng.shuffle(rows)
    return rows


# ── Stage 2: Two-step composition ──────────────────────────────────

# All shift/rotate ops for random selection
SHIFT_OPS = []
for _op in ('shr', 'shl', 'rol', 'ror'):
    for _k in range(1, 8):
        SHIFT_OPS.append((_op, _k))
SHIFT_OPS.append(('not', 0))


def _apply_shift(op_type, k, x_int):
    """Apply shift/rotate/not to integer, return integer result."""
    if op_type == 'shr': return shr(x_int, k)
    if op_type == 'shl': return shl(x_int, k)
    if op_type == 'rol': return rol(x_int, k)
    if op_type == 'ror': return ror(x_int, k)
    if op_type == 'not': return bit_not(x_int)
    raise ValueError(op_type)


def _shift_name(op_type, k):
    if op_type == 'not':
        return 'not'
    return f"{op_type}{k}"


def _trace_one_shift(op_type, k, bits_str):
    """Produce trace line(s) for a single shift on bits_str.

    Returns (recipe_text, result_str).
    """
    if op_type == 'shr':
        return _trace_shr(bits_str, k)
    elif op_type == 'shl':
        return _trace_shl(bits_str, k)
    elif op_type == 'rol':
        return _trace_rol(bits_str, k)
    elif op_type == 'ror':
        return _trace_ror(bits_str, k)
    elif op_type == 'not':
        return _trace_not(bits_str)
    raise ValueError(op_type)


def generate_stage2(rng, n=3000):
    """Generate Stage 2: shift + gate compositions."""
    rows = []
    gate_names = list(GATE_OPS.keys())

    for _ in range(n):
        x = rng.randint(0, 255)
        x_str = _fmt(x)

        # Pick two different shifts for A and B
        op1_type, op1_k = rng.choice(SHIFT_OPS)
        op2_type, op2_k = rng.choice(SHIFT_OPS)
        while (op1_type, op1_k) == (op2_type, op2_k):
            op2_type, op2_k = rng.choice(SHIFT_OPS)

        gate_name = rng.choice(gate_names)

        # Compute
        a_int = _apply_shift(op1_type, op1_k, x)
        b_int = _apply_shift(op2_type, op2_k, x)
        fn, _ = GATE_OPS[gate_name]
        result_int = fn(a_int, b_int)
        result_str = _fmt(result_int)

        a_str = _fmt(a_int)
        b_str = _fmt(b_int)

        # Verify
        _verify_gate(gate_name, a_int, b_int)

        name1 = _shift_name(op1_type, op1_k)
        name2 = _shift_name(op2_type, op2_k)

        # Build prompt
        prompt = f"Compute {gate_name}({name1}({x_str}), {name2}({x_str}))."

        # Build trace
        recipe1, res1 = _trace_one_shift(op1_type, op1_k, x_str)
        recipe2, res2 = _trace_one_shift(op2_type, op2_k, x_str)
        assert res1 == a_str, f"Shift mismatch: {res1} != {a_str}"
        assert res2 == b_str, f"Shift mismatch: {res2} != {b_str}"

        gate_lines, gate_result = _trace_gate(gate_name, a_str, b_str)
        assert gate_result == result_str

        trace_parts = [
            f"A = {name1}({x_str})",
            f"  {recipe1} \u2192 {res1}",
            f"B = {name2}({x_str})",
            f"  {recipe2} \u2192 {res2}",
        ]
        trace_parts.extend(gate_lines)

        trace_body = '\n'.join(trace_parts)

        rows.append({
            "messages": [
                {"role": "user", "content": TRAINING_TAG + prompt + BOXED_INSTRUCTION},
                {"role": "assistant", "content": f"<think>\n[TRAINING DRILL]\n{trace_body}\n</think>\n\\boxed{{{result_str}}}"},
            ],
            "answer": result_str,
            "id": f"microskill_s2_{rng.randint(0, 999999):06d}",
            "puzzle_type": "bit_manipulation",
            "mode": "microskill_stage2",
            "generator": "gen_bit_microskills",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })

    rng.shuffle(rows)
    return rows


# ── Stage 3: Three-step composition (family formulas) ──────────────

# Family templates matching gen_bit_3stream.py
def _family_or_xnor(a, b, c):
    """C | xnor(A, B)"""
    return (c | ((~(a ^ b)) & BYTE)) & BYTE

def _family_gated_xnor_nand(a, b, c):
    """where C=0 take xnor(A,B), where C=1 take nand(A,B)"""
    p = (~(a ^ b)) & BYTE
    q = (~(a & b)) & BYTE
    return (((~c & BYTE) & p) | (c & q)) & BYTE

def _family_ch(a, b, c):
    """where A=1 take B, where A=0 take C"""
    return ((a & b) | ((~a & BYTE) & c)) & BYTE

def _family_maj3(a, b, c):
    """majority: P=A&B, Q=A&C, R=B&C, output=P|Q|R"""
    return ((a & b) | (a & c) | (b & c)) & BYTE

def _family_tt121(a, b, c):
    """where A=0 take xnor(B,C), where A=1 take nand(B,C)"""
    p = (~(b ^ c)) & BYTE
    q = (~(b & c)) & BYTE
    return (((~a & BYTE) & p) | (a & q)) & BYTE

def _family_t1(a, b, c):
    """where A=0 take (~B)|C, where A=1 take B^C"""
    p = ((~b & BYTE) | c) & BYTE
    q = (b ^ c) & BYTE
    return (((~a & BYTE) & p) | (a & q)) & BYTE


FAMILIES_STAGE3 = {
    'C | xnor(A,B)': (_family_or_xnor, 48),
    'where C=0: xnor(A,B); where C=1: nand(A,B)': (_family_gated_xnor_nand, 27),
    'where A=1: B; where A=0: C': (_family_ch, 8),
    'P=A&B, Q=A&C, R=B&C, output=P|Q|R': (_family_maj3, 4),
    'where A=0: xnor(B,C); where A=1: nand(B,C)': (_family_tt121, 3),
    'where A=0: (~B)|C; where A=1: B^C': (_family_t1, 2),
}


def _trace_family_steps(formula_key, a_str, b_str, c_str):
    """Produce step-by-step trace for a family formula.

    Returns (trace_lines, result_str).
    Each gate application is shown position-by-position.
    """
    a_int = int(a_str, 2)
    b_int = int(b_str, 2)
    c_int = int(c_str, 2)

    lines = []

    if formula_key == 'C | xnor(A,B)':
        # Step 1: P = xnor(A, B)
        gate_lines, p_str = _trace_gate('xnor', a_str, b_str)
        lines.append("P = xnor(A, B):")
        for gl in gate_lines[1:]:  # skip the redundant header
            lines.append(gl)
        lines.append(f"  P = {p_str}")
        # Step 2: output = C | P
        lines.append("output = C | P:")
        gate_lines2, result = _trace_gate('or', c_str, p_str)
        for gl in gate_lines2[1:]:
            lines.append(gl)
        lines.append(f"  output = {result}")
        return lines, result

    elif formula_key == 'where C=0: xnor(A,B); where C=1: nand(A,B)':
        # Step 1: P = xnor(A, B)
        gate_lines, p_str = _trace_gate('xnor', a_str, b_str)
        lines.append("P = xnor(A, B):")
        for gl in gate_lines[1:]:
            lines.append(gl)
        lines.append(f"  P = {p_str}")
        # Step 2: Q = nand(A, B)
        gate_lines2, q_str = _trace_gate('nand', a_str, b_str)
        lines.append("Q = nand(A, B):")
        for gl in gate_lines2[1:]:
            lines.append(gl)
        lines.append(f"  Q = {q_str}")
        # Step 3: select by C
        result_bits = []
        for i in range(8):
            if c_str[i] == '0':
                result_bits.append(p_str[i])
            else:
                result_bits.append(q_str[i])
        result = ''.join(result_bits)
        c_sp = ' '.join(c_str)
        p_sp = ' '.join(p_str)
        q_sp = ' '.join(q_str)
        r_sp = ' '.join(result)
        lines.append(f"output = where C=0 take P, where C=1 take Q:")
        lines.append(f"  C: {c_sp}")
        lines.append(f"  P: {p_sp}")
        lines.append(f"  Q: {q_sp}")
        lines.append(f"  R: {r_sp}")
        lines.append(f"  output = {result}")
        return lines, result

    elif formula_key == 'where A=1: B; where A=0: C':
        result_bits = []
        for i in range(8):
            if a_str[i] == '1':
                result_bits.append(b_str[i])
            else:
                result_bits.append(c_str[i])
        result = ''.join(result_bits)
        a_sp = ' '.join(a_str)
        b_sp = ' '.join(b_str)
        c_sp = ' '.join(c_str)
        r_sp = ' '.join(result)
        lines.append("output = where A=1 take B, where A=0 take C:")
        lines.append(f"  A: {a_sp}")
        lines.append(f"  B: {b_sp}")
        lines.append(f"  C: {c_sp}")
        lines.append(f"  R: {r_sp}")
        lines.append(f"  output = {result}")
        return lines, result

    elif formula_key == 'P=A&B, Q=A&C, R=B&C, output=P|Q|R':
        # Step 1: P = A & B
        gate_lines, p_str = _trace_gate('and', a_str, b_str)
        lines.append("P = A & B:")
        for gl in gate_lines[1:]:
            lines.append(gl)
        lines.append(f"  P = {p_str}")
        # Step 2: Q = A & C
        gate_lines2, q_str = _trace_gate('and', a_str, c_str)
        lines.append("Q = A & C:")
        for gl in gate_lines2[1:]:
            lines.append(gl)
        lines.append(f"  Q = {q_str}")
        # Step 3: R = B & C
        gate_lines3, r_str = _trace_gate('and', b_str, c_str)
        lines.append("R = B & C:")
        for gl in gate_lines3[1:]:
            lines.append(gl)
        lines.append(f"  R = {r_str}")
        # Step 4: output = P | Q | R
        pq_lines, pq_str = _trace_gate('or', p_str, q_str)
        pqr_lines, result = _trace_gate('or', pq_str, r_str)
        lines.append("output = P | Q | R:")
        p_sp = ' '.join(p_str)
        q_sp = ' '.join(q_str)
        r2_sp = ' '.join(r_str)
        # Show three-way or step by step
        lines.append(f"  P|Q:")
        for gl in pq_lines[1:]:
            lines.append(f"  {gl}")
        lines.append(f"    = {pq_str}")
        lines.append(f"  (P|Q)|R:")
        for gl in pqr_lines[1:]:
            lines.append(f"  {gl}")
        lines.append(f"    = {result}")
        lines.append(f"  output = {result}")
        return lines, result

    elif formula_key == 'where A=0: xnor(B,C); where A=1: nand(B,C)':
        # Step 1: P = xnor(B, C)
        gate_lines, p_str = _trace_gate('xnor', b_str, c_str)
        lines.append("P = xnor(B, C):")
        for gl in gate_lines[1:]:
            lines.append(gl)
        lines.append(f"  P = {p_str}")
        # Step 2: Q = nand(B, C)
        gate_lines2, q_str = _trace_gate('nand', b_str, c_str)
        lines.append("Q = nand(B, C):")
        for gl in gate_lines2[1:]:
            lines.append(gl)
        lines.append(f"  Q = {q_str}")
        # Step 3: select by A
        result_bits = []
        for i in range(8):
            if a_str[i] == '0':
                result_bits.append(p_str[i])
            else:
                result_bits.append(q_str[i])
        result = ''.join(result_bits)
        a_sp = ' '.join(a_str)
        p_sp = ' '.join(p_str)
        q_sp = ' '.join(q_str)
        r_sp = ' '.join(result)
        lines.append(f"output = where A=0 take P, where A=1 take Q:")
        lines.append(f"  A: {a_sp}")
        lines.append(f"  P: {p_sp}")
        lines.append(f"  Q: {q_sp}")
        lines.append(f"  R: {r_sp}")
        lines.append(f"  output = {result}")
        return lines, result

    elif formula_key == 'where A=0: (~B)|C; where A=1: B^C':
        # Step 1: P = not(B)
        _, nb_str = _trace_not(b_str)
        # Step 1a: P = (~B) | C
        gate_lines, p_str = _trace_gate('or', nb_str, c_str)
        lines.append("P = (~B) | C:")
        lines.append(f"  ~B = {nb_str}")
        lines.append(f"  (~B) | C:")
        for gl in gate_lines[1:]:
            lines.append(f"  {gl}")
        lines.append(f"  P = {p_str}")
        # Step 2: Q = B ^ C
        gate_lines2, q_str = _trace_gate('xor', b_str, c_str)
        lines.append("Q = B ^ C:")
        for gl in gate_lines2[1:]:
            lines.append(gl)
        lines.append(f"  Q = {q_str}")
        # Step 3: select by A
        result_bits = []
        for i in range(8):
            if a_str[i] == '0':
                result_bits.append(p_str[i])
            else:
                result_bits.append(q_str[i])
        result = ''.join(result_bits)
        a_sp = ' '.join(a_str)
        p_sp = ' '.join(p_str)
        q_sp = ' '.join(q_str)
        r_sp = ' '.join(result)
        lines.append(f"output = where A=0 take P, where A=1 take Q:")
        lines.append(f"  A: {a_sp}")
        lines.append(f"  P: {p_sp}")
        lines.append(f"  Q: {q_sp}")
        lines.append(f"  R: {r_sp}")
        lines.append(f"  output = {result}")
        return lines, result

    raise ValueError(f"Unknown formula: {formula_key}")


def generate_stage3(rng, n=2000):
    """Generate Stage 3: three shifts + family formula."""
    rows = []
    family_items = list(FAMILIES_STAGE3.items())
    family_keys = [k for k, _ in family_items]
    family_weights = [w for _, (_, w) in family_items]

    for _ in range(n):
        x = rng.randint(0, 255)
        x_str = _fmt(x)

        # Pick 3 different shifts
        ops = rng.sample(SHIFT_OPS, 3)
        while len(set((o, k) for o, k in ops)) < 3:
            ops = rng.sample(SHIFT_OPS, 3)

        # Compute A, B, C
        a_int = _apply_shift(ops[0][0], ops[0][1], x)
        b_int = _apply_shift(ops[1][0], ops[1][1], x)
        c_int = _apply_shift(ops[2][0], ops[2][1], x)

        a_str = _fmt(a_int)
        b_str = _fmt(b_int)
        c_str = _fmt(c_int)

        # Pick family
        formula_key = rng.choices(family_keys, weights=family_weights, k=1)[0]
        fam_fn, _ = FAMILIES_STAGE3[formula_key]

        # Compute expected
        expected_int = fam_fn(a_int, b_int, c_int)
        expected_str = _fmt(expected_int)

        # Build shift names
        name_a = _shift_name(ops[0][0], ops[0][1])
        name_b = _shift_name(ops[1][0], ops[1][1])
        name_c = _shift_name(ops[2][0], ops[2][1])

        # Build prompt
        # Use a clear format showing the formula
        prompt = (
            f"Given A={name_a}(x), B={name_b}(x), C={name_c}(x), "
            f"compute {formula_key} for x={x_str}."
        )

        # Build trace
        trace_parts = []

        # Step 1: compute A
        recipe_a, res_a = _trace_one_shift(ops[0][0], ops[0][1], x_str)
        trace_parts.append(f"A = {name_a}({x_str})")
        trace_parts.append(f"  {recipe_a} \u2192 {res_a}")
        assert res_a == a_str

        # Step 2: compute B
        recipe_b, res_b = _trace_one_shift(ops[1][0], ops[1][1], x_str)
        trace_parts.append(f"B = {name_b}({x_str})")
        trace_parts.append(f"  {recipe_b} \u2192 {res_b}")
        assert res_b == b_str

        # Step 3: compute C
        recipe_c, res_c = _trace_one_shift(ops[2][0], ops[2][1], x_str)
        trace_parts.append(f"C = {name_c}({x_str})")
        trace_parts.append(f"  {recipe_c} \u2192 {res_c}")
        assert res_c == c_str

        # Step 4: apply family formula with position-by-position gates
        family_lines, result = _trace_family_steps(formula_key, a_str, b_str, c_str)
        trace_parts.extend(family_lines)

        # Verify
        assert result == expected_str, (
            f"Family mismatch: {formula_key} on A={a_str}, B={b_str}, C={c_str}: "
            f"trace={result} != expected={expected_str}"
        )

        trace_body = '\n'.join(trace_parts)

        rows.append({
            "messages": [
                {"role": "user", "content": TRAINING_TAG + prompt + BOXED_INSTRUCTION},
                {"role": "assistant", "content": f"<think>\n[TRAINING DRILL]\n{trace_body}\n</think>\n\\boxed{{{result}}}"},
            ],
            "answer": result,
            "id": f"microskill_s3_{rng.randint(0, 999999):06d}",
            "puzzle_type": "bit_manipulation",
            "mode": "microskill_stage3",
            "generator": "gen_bit_microskills",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })

    rng.shuffle(rows)
    return rows


# ── No-jump full puzzle traces ──────────────────────────────────────

# These rewrite the existing 3-stream puzzle format but with step-by-step
# execution of every intermediate value.

PERMS_3 = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]

FAMILIES_NOJUMP = [
    ("OR_XNOR", _family_or_xnor, 'C | xnor(A,B)', 48),
    ("GATED_XNOR_NAND", _family_gated_xnor_nand, 'where C=0: xnor(A,B); where C=1: nand(A,B)', 27),
    ("CH", _family_ch, 'where A=1: B; where A=0: C', 8),
    ("MAJ3", _family_maj3, 'P=A&B, Q=A&C, R=B&C, output=P|Q|R', 4),
    ("TT121", _family_tt121, 'where A=0: xnor(B,C); where A=1: nand(B,C)', 3),
    ("T1", _family_t1, 'where A=0: (~B)|C; where A=1: B^C', 2),
]

# 2-input families
FAMILIES_2INPUT_NOJUMP = [
    ("AND", gate_and, 'and', 4),
    ("OR", gate_or, 'or', 4),
    ("XOR", gate_xor, 'xor', 2),
]


def _make_sources_nojump(rng):
    """Pick 3 source transforms matching gen_bit_3stream convention."""
    rot_k = rng.randint(1, 7)
    rot_type = rng.choice(["rol", "ror"])
    rot_name = f"{rot_type}{rot_k}"

    shift_pool = [("x", 'x', 0)]
    for k in range(1, 8):
        shift_pool.append((f"shl{k}", 'shl', k))
        shift_pool.append((f"shr{k}", 'shr', k))

    s1, s2 = rng.sample(shift_pool, 2)
    sources = [(rot_name, rot_type, rot_k), s1, s2]
    rng.shuffle(sources)
    return sources


def _apply_source(src_name, src_type, src_k, x_int):
    """Apply a source transform."""
    if src_name == 'x':
        return x_int
    return _apply_shift(src_type, src_k, x_int)


def _trace_source_step(src_name, src_type, src_k, x_str):
    """Produce trace for computing one source from x.

    Returns (trace_lines, result_str).
    """
    if src_name == 'x':
        return [f"  (identity) \u2192 {x_str}"], x_str

    recipe, result = _trace_one_shift(src_type, src_k, x_str)
    return [f"  {recipe} \u2192 {result}"], result


def generate_nojump(rng, n=5000):
    """Generate no-jump full puzzle traces with step-by-step execution."""
    rows = []

    for _ in range(n):
        use_2input = rng.random() < 0.10

        if use_2input:
            fam_idx = rng.choices(range(len(FAMILIES_2INPUT_NOJUMP)),
                                  weights=[w for _, _, _, w in FAMILIES_2INPUT_NOJUMP])[0]
            fam_name, fam_fn_2, gate_name, _ = FAMILIES_2INPUT_NOJUMP[fam_idx]

            # Pick 2 sources
            all_srcs = [("x", "x", 0)]
            for k in range(1, 8):
                all_srcs.append((f"shl{k}", "shl", k))
                all_srcs.append((f"shr{k}", "shr", k))
                all_srcs.append((f"rol{k}", "rol", k))
                all_srcs.append((f"ror{k}", "ror", k))
            sources = rng.sample(all_srcs, 2)

            n_examples = rng.randint(7, 9)
            inputs = [rng.randint(0, 255) for _ in range(n_examples + 1)]
            query_input = inputs[-1]
            example_inputs = inputs[:-1]

            def compute_2(x):
                va = _apply_source(sources[0][0], sources[0][1], sources[0][2], x)
                vb = _apply_source(sources[1][0], sources[1][1], sources[1][2], x)
                return fam_fn_2(va, vb) & BYTE

            examples = [(format(x, '08b'), format(compute_2(x), '08b')) for x in example_inputs]
            query_str = format(query_input, '08b')
            answer_str = format(compute_2(query_input), '08b')

            ordered_names = [sources[0][0], sources[1][0]]

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
            prompt = '\n'.join(prompt_lines)

            # Build trace
            trace_lines = []
            trace_lines.append("Rule:")
            for i, (sname, stype, sk) in enumerate(sources):
                label = chr(65 + i)
                if sname == 'x':
                    trace_lines.append(f"  {label} = x")
                else:
                    trace_lines.append(f"  {label} = {sname}(x)")
            ops_sym = {"AND": "&", "OR": "|", "XOR": "^"}
            sym = ops_sym[fam_name]
            trace_lines.append(f"  output = A {sym} B")

            # Check on first example
            trace_lines.append("")
            trace_lines.append("Check:")
            cx = example_inputs[0]
            cx_str = format(cx, '08b')
            trace_lines.append(f"  x = {cx_str}")
            for i, (sname, stype, sk) in enumerate(sources):
                label = chr(65 + i)
                src_lines, src_result = _trace_source_step(sname, stype, sk, cx_str)
                trace_lines.append(f"  {label} = {sname}({cx_str})" if sname != 'x' else f"  {label} = x = {cx_str}")
                for sl in src_lines:
                    trace_lines.append(f"  {sl}")

            ca = _apply_source(sources[0][0], sources[0][1], sources[0][2], cx)
            cb = _apply_source(sources[1][0], sources[1][1], sources[1][2], cx)
            ca_str = _fmt(ca)
            cb_str = _fmt(cb)
            gate_lines_check, check_result = _trace_gate(gate_name, ca_str, cb_str)
            trace_lines.append(f"  output = A {sym} B:")
            for gl in gate_lines_check[1:]:
                trace_lines.append(f"  {gl}")
            expected_check = format(compute_2(cx), '08b')
            match = "\u2713" if check_result == expected_check else "\u2717"
            trace_lines.append(f"  output = {check_result} {match}")

            # Query
            trace_lines.append("")
            trace_lines.append("Query:")
            trace_lines.append(f"  x = {query_str}")
            for i, (sname, stype, sk) in enumerate(sources):
                label = chr(65 + i)
                src_lines, src_result = _trace_source_step(sname, stype, sk, query_str)
                trace_lines.append(f"  {label} = {sname}({query_str})" if sname != 'x' else f"  {label} = x = {query_str}")
                for sl in src_lines:
                    trace_lines.append(f"  {sl}")

            qa = _apply_source(sources[0][0], sources[0][1], sources[0][2], query_input)
            qb = _apply_source(sources[1][0], sources[1][1], sources[1][2], query_input)
            qa_str = _fmt(qa)
            qb_str = _fmt(qb)
            gate_lines_q, q_result = _trace_gate(gate_name, qa_str, qb_str)
            trace_lines.append(f"  output = A {sym} B:")
            for gl in gate_lines_q[1:]:
                trace_lines.append(f"  {gl}")
            trace_lines.append(f"  output = {q_result}")

            assert q_result == answer_str

            trace_body = '\n'.join(trace_lines)

            rows.append({
                "messages": [
                    {"role": "user", "content": TRAINING_TAG + prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": f"<think>\n[TRAINING DRILL]\n{trace_body}\n</think>\n\\boxed{{{answer_str}}}"},
                ],
                "answer": answer_str,
                "id": f"nojump_{rng.randint(0, 999999):06d}",
                "puzzle_type": "bit_manipulation",
                "mode": "nojump_full",
                "family": fam_name,
                "generator": "gen_bit_microskills",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
            continue

        # 3-input family
        fam_idx = rng.choices(range(len(FAMILIES_NOJUMP)),
                              weights=[w for _, _, _, w in FAMILIES_NOJUMP])[0]
        fam_name, fam_fn, formula_key, _ = FAMILIES_NOJUMP[fam_idx]

        sources = _make_sources_nojump(rng)
        perm = rng.choice(PERMS_3)

        n_examples = rng.randint(7, 9)
        inputs = [rng.randint(0, 255) for _ in range(n_examples + 1)]
        query_input = inputs[-1]
        example_inputs = inputs[:-1]

        def compute_3(x, _sources=sources, _perm=perm, _fam_fn=fam_fn):
            vals = [_apply_source(s[0], s[1], s[2], x) for s in _sources]
            return _fam_fn(vals[_perm[0]], vals[_perm[1]], vals[_perm[2]])

        examples = [(format(x, '08b'), format(compute_3(x), '08b')) for x in example_inputs]
        query_str = format(query_input, '08b')
        answer_str = format(compute_3(query_input), '08b')

        ordered_sources = [sources[perm[0]], sources[perm[1]], sources[perm[2]]]

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
        prompt = '\n'.join(prompt_lines)

        # Build trace
        trace_lines = []
        trace_lines.append("Rule:")
        for i, (sname, stype, sk) in enumerate(ordered_sources):
            label = chr(65 + i)
            if sname == 'x':
                trace_lines.append(f"  {label} = x")
            else:
                trace_lines.append(f"  {label} = {sname}(x)")
        trace_lines.append(f"  {formula_key}")

        # Check on first example
        trace_lines.append("")
        trace_lines.append("Check:")
        cx = example_inputs[0]
        cx_str = format(cx, '08b')
        trace_lines.append(f"  x = {cx_str}")

        # Compute sources step by step
        check_vals = []
        for i, (sname, stype, sk) in enumerate(ordered_sources):
            label = chr(65 + i)
            src_lines, src_result = _trace_source_step(sname, stype, sk, cx_str)
            if sname == 'x':
                trace_lines.append(f"  {label} = x = {cx_str}")
            else:
                trace_lines.append(f"  {label} = {sname}({cx_str})")
                for sl in src_lines:
                    trace_lines.append(f"  {sl}")
            check_vals.append(src_result)

        # Apply family formula step by step
        family_lines, check_result = _trace_family_steps(
            formula_key, check_vals[0], check_vals[1], check_vals[2])
        for fl in family_lines:
            trace_lines.append(f"  {fl}")

        expected_check = format(compute_3(cx), '08b')
        match = "\u2713" if check_result == expected_check else "\u2717"
        trace_lines.append(f"  = {check_result} {match}")

        # Query
        trace_lines.append("")
        trace_lines.append("Query:")
        trace_lines.append(f"  x = {query_str}")

        query_vals = []
        for i, (sname, stype, sk) in enumerate(ordered_sources):
            label = chr(65 + i)
            src_lines, src_result = _trace_source_step(sname, stype, sk, query_str)
            if sname == 'x':
                trace_lines.append(f"  {label} = x = {query_str}")
            else:
                trace_lines.append(f"  {label} = {sname}({query_str})")
                for sl in src_lines:
                    trace_lines.append(f"  {sl}")
            query_vals.append(src_result)

        # Apply family formula step by step
        family_lines_q, q_result = _trace_family_steps(
            formula_key, query_vals[0], query_vals[1], query_vals[2])
        for fl in family_lines_q:
            trace_lines.append(f"  {fl}")
        trace_lines.append(f"  = {q_result}")

        assert q_result == answer_str, (
            f"Nojump mismatch: expected {answer_str}, got {q_result}"
        )

        trace_body = '\n'.join(trace_lines)

        rows.append({
            "messages": [
                {"role": "user", "content": TRAINING_TAG + prompt + BOXED_INSTRUCTION},
                {"role": "assistant", "content": f"<think>\n[TRAINING DRILL]\n{trace_body}\n</think>\n\\boxed{{{answer_str}}}"},
            ],
            "answer": answer_str,
            "id": f"nojump_{rng.randint(0, 999999):06d}",
            "puzzle_type": "bit_manipulation",
            "mode": "nojump_full",
            "family": fam_name,
            "generator": "gen_bit_microskills",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })

    rng.shuffle(rows)
    return rows


# ── Main ────────────────────────────────────────────────────────────

def write_jsonl(rows, path):
    """Write rows to JSONL file."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate bit manipulation micro-skill curriculum")
    parser.add_argument("--stage", type=str, choices=['1', '2', '3', 'nojump', 'all'],
                        default='all', help="Which stage to generate")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of examples (overrides default per stage)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default="data/bit_manipulation/pool/generated")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    stages = [args.stage] if args.stage != 'all' else ['1', '2', '3', 'nojump']

    for stage in stages:
        t0 = time.time()
        if stage == '1':
            n = args.n or 5000
            rows = generate_stage1(rng, n)
            path = f"{args.output_dir}/microskill_stage1.jsonl"
        elif stage == '2':
            n = args.n or 3000
            rows = generate_stage2(rng, n)
            path = f"{args.output_dir}/microskill_stage2.jsonl"
        elif stage == '3':
            n = args.n or 2000
            rows = generate_stage3(rng, n)
            path = f"{args.output_dir}/microskill_stage3.jsonl"
        elif stage == 'nojump':
            n = args.n or 5000
            rows = generate_nojump(rng, n)
            path = f"{args.output_dir}/nojump_full.jsonl"

        dt = time.time() - t0
        write_jsonl(rows, path)
        print(f"  Stage {stage}: {len(rows)} rows in {dt:.1f}s")


if __name__ == "__main__":
    main()
