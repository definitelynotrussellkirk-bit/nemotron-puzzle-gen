#!/usr/bin/env python3
"""Generate bit manipulation micro-skill VARIANT examples.

Six variant types that teach operations from different angles than
the basic "compute X" examples in microskill stages 1-3.

Variant types (1000 each, 6000 total):
  1. which_op     - Identify which operation was applied from input/output
  2. counterfactual - Which input CANNOT produce the given output?
  3. properties   - What must be true given an operation's result?
  4. step_by_step - Longer step-by-step explanations of single operations
  5. reverse      - Find the input given operation + output
  6. two_step_id  - Identify which shift was used in a two-step chain

Usage:
    python3 -m generators.gen_bit_microskill_variants
    python3 -m generators.gen_bit_microskill_variants --variant which_op --n 500
    python3 -m generators.gen_bit_microskill_variants --seed 99
"""
import argparse
import json
import os
import random
import time
from datetime import datetime, timezone

from training.data import BOXED_INSTRUCTION

TRAINING_TAG = "[Alice's Training House] "

BYTE = 0xFF


# ── Primitive operations (shared with gen_bit_microskills) ───────────

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


def gate_xnor(a, b): return (~(a ^ b)) & BYTE
def gate_xor(a, b):  return (a ^ b) & BYTE
def gate_and(a, b):  return (a & b) & BYTE
def gate_or(a, b):   return (a | b) & BYTE
def gate_nand(a, b): return (~(a & b)) & BYTE
def gate_nor(a, b):  return (~(a | b)) & BYTE
def gate_and_not(a, b): return (a & (~b)) & BYTE
def gate_or_not(a, b):  return (a | (~b)) & BYTE


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

# All shift/rotate ops
SHIFT_OPS = []
for _op in ('shr', 'shl', 'rol', 'ror'):
    for _k in range(1, 8):
        SHIFT_OPS.append((_op, _k))
SHIFT_OPS.append(('not', 0))


def _apply_shift(op_type, k, x_int):
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


def _shift_description(op_type, k):
    """Human-readable description of what this shift does."""
    if op_type == 'shr':
        return f"prepend {k} zero{'s' if k > 1 else ''}, drop last {k}"
    elif op_type == 'shl':
        return f"drop first {k}, append {k} zero{'s' if k > 1 else ''}"
    elif op_type == 'rol':
        return f"move first {k} to end"
    elif op_type == 'ror':
        return f"move last {k} to front"
    elif op_type == 'not':
        return "flip each bit"
    raise ValueError(op_type)


def _compute_shift_str(op_type, k, bits_str):
    """Compute shift result using string manipulation (matching trace logic)."""
    if op_type == 'shr':
        return ('0' * k + bits_str)[:8]
    elif op_type == 'shl':
        return (bits_str + '0' * k)[k:k + 8]
    elif op_type == 'rol':
        return bits_str[k:] + bits_str[:k]
    elif op_type == 'ror':
        return bits_str[8 - k:] + bits_str[:8 - k]
    elif op_type == 'not':
        return ''.join('1' if c == '0' else '0' for c in bits_str)
    raise ValueError(op_type)


def _make_row(messages, answer, variant_id, mode):
    return {
        "messages": messages,
        "answer": answer,
        "id": variant_id,
        "puzzle_type": "bit_manipulation",
        "mode": f"microskill_variant_{mode}",
        "generator": "gen_bit_microskill_variants",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Variant 1: "Which operation was used?" ──────────────────────────

def _similar_ops(op_type, k):
    """Return list of (op_type, k) in the same family, excluding the given one."""
    if op_type == 'not':
        # Compare with single-position shifts
        return [('shr', 1), ('shl', 1), ('rol', 1), ('ror', 1)]
    family = op_type  # shr, shl, rol, ror
    siblings = [(family, kk) for kk in range(1, 8) if kk != k]
    # Also add the "confusable" family
    confuse_map = {'shr': 'ror', 'shl': 'rol', 'ror': 'shr', 'rol': 'shl'}
    conf = confuse_map[family]
    siblings += [(conf, kk) for kk in range(1, 8)]
    return siblings


def gen_which_op(rng, n=1000):
    """Variant 1: identify which operation was applied."""
    rows = []
    for i in range(n):
        x = rng.randint(0, 255)
        x_str = _fmt(x)

        correct_op, correct_k = rng.choice(SHIFT_OPS)
        result_int = _apply_shift(correct_op, correct_k, x)
        result_str = _fmt(result_int)

        # Verify string computation matches
        assert _compute_shift_str(correct_op, correct_k, x_str) == result_str

        correct_name = _shift_name(correct_op, correct_k)

        # Pick 2 wrong alternatives that give different results
        candidates = _similar_ops(correct_op, correct_k)
        rng.shuffle(candidates)
        wrong = []
        for wop, wk in candidates:
            w_result = _fmt(_apply_shift(wop, wk, x))
            if w_result != result_str:
                wrong.append((wop, wk, w_result))
            if len(wrong) == 2:
                break

        # If not enough wrong alternatives (rare), pick from all ops
        if len(wrong) < 2:
            all_ops = [o for o in SHIFT_OPS if _shift_name(o[0], o[1]) != correct_name]
            rng.shuffle(all_ops)
            for wop, wk in all_ops:
                w_result = _fmt(_apply_shift(wop, wk, x))
                if w_result != result_str and not any(
                    _shift_name(wop, wk) == _shift_name(wo, wkk) for wo, wkk, _ in wrong
                ):
                    wrong.append((wop, wk, w_result))
                if len(wrong) == 2:
                    break

        if len(wrong) < 2:
            continue  # Skip (extremely rare)

        # Build the 3 options in random order
        options = [
            (correct_name, result_str, True),
            (_shift_name(wrong[0][0], wrong[0][1]), wrong[0][2], False),
            (_shift_name(wrong[1][0], wrong[1][1]), wrong[1][2], False),
        ]
        rng.shuffle(options)

        option_names = [o[0] for o in options]
        correct_label = correct_name

        # Build trace: test each option
        trace_lines = []
        for name, res, is_correct in options:
            mark = "\u2713" if is_correct else "\u2717"
            trace_lines.append(f"{name}({x_str}) \u2192 {res} {'= ' + result_str if is_correct else '\u2260 ' + result_str} {mark}")

        prompt = (
            f"x = {x_str}, result = {result_str}. "
            f"Which operation was applied: {', '.join(option_names[:-1])}, or {option_names[-1]}?"
        )

        reasoning = "<think>\n[TRAINING DRILL]\n" + '\n'.join(trace_lines) + "\n</think>"

        messages = [
            {"role": "user", "content": TRAINING_TAG + prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"{reasoning}\n\\boxed{{{correct_label}}}"},
        ]

        rows.append(_make_row(messages, correct_label,
                              f"microskill_v1_whichop_{i:06d}", "which_op"))

    return rows


# ── Variant 2: Counterfactual ("What can't be right?") ──────────────

def gen_counterfactual(rng, n=1000):
    """Variant 2: which input CANNOT produce the given output?"""
    rows = []
    labels = ['A', 'B', 'C']

    for i in range(n):
        op_type, k = rng.choice(SHIFT_OPS)
        op_name = _shift_name(op_type, k)
        desc = _shift_description(op_type, k)

        # Generate a random input, compute the output (guarantees reachability)
        source_x = rng.randint(0, 255)
        result_int = _apply_shift(op_type, k, source_x)
        result_str = _fmt(result_int)

        # Generate 2 valid inputs that produce this result
        valid_inputs = []
        if op_type in ('shr', 'shl'):
            # Shifts lose bits, so multiple inputs map to same output.
            # shr{k}: result = 0^k | x[0..7-k]. Lost = x[8-k..7].
            #   Reverse: x = result[k..7] | (any k low bits) shifted back
            #   i.e. x = (result << k) | random_low_k_bits
            # shl{k}: result = x[k..7] | 0^k. Lost = x[0..k-1].
            #   Reverse: x = (random high k bits) | result[0..7-k]
            #   i.e. x = (random_high << (8-k)) | (result >> k)
            for _ in range(2):
                if op_type == 'shr':
                    base = (result_int << k) & BYTE
                    low_bits = rng.randint(0, (1 << k) - 1)
                    candidate = base | low_bits
                elif op_type == 'shl':
                    core = (result_int >> k) & ((1 << (8 - k)) - 1)
                    high_bits = rng.randint(0, (1 << k) - 1)
                    candidate = (high_bits << (8 - k)) | core
                    candidate &= BYTE

                assert _apply_shift(op_type, k, candidate) == result_int, \
                    f"{op_type}{k}({_fmt(candidate)}) = {_fmt(_apply_shift(op_type, k, candidate))} != {result_str}"
                valid_inputs.append(candidate)

            # Make sure the two valid inputs are different
            if valid_inputs[0] == valid_inputs[1]:
                if op_type == 'shr':
                    base = (result_int << k) & BYTE
                    valid_inputs[1] = base | ((valid_inputs[0] + 1) % (1 << k))
                else:
                    core = (result_int >> k) & ((1 << (8 - k)) - 1)
                    cur_high = valid_inputs[0] >> (8 - k)
                    alt_high = (cur_high + 1) % (1 << k)
                    valid_inputs[1] = ((alt_high << (8 - k)) | core) & BYTE
                assert _apply_shift(op_type, k, valid_inputs[1]) == result_int

        elif op_type in ('rol', 'ror'):
            # Rotations are bijective, only one input produces each output
            if op_type == 'rol':
                inv = ror(result_int, k)
            else:
                inv = rol(result_int, k)
            valid_inputs = [inv]
            # Need a second valid input - use a different k that happens to work
            # Actually for rotations there's exactly 1 inverse. Use same input twice
            # and make the invalid one the distinguisher.
            valid_inputs.append(inv)  # Both are the same valid input

        elif op_type == 'not':
            inv = bit_not(result_int)
            valid_inputs = [inv, inv]

        # Generate invalid input
        for _attempt in range(100):
            bad = rng.randint(0, 255)
            if _apply_shift(op_type, k, bad) != result_int:
                break
        else:
            continue  # Skip if we can't find a bad input

        bad_result_str = _fmt(_apply_shift(op_type, k, bad))

        # Build A/B/C with the wrong one in a random position
        wrong_pos = rng.randint(0, 2)
        items = []
        valid_idx = 0
        for pos in range(3):
            if pos == wrong_pos:
                items.append((bad, False))
            else:
                items.append((valid_inputs[valid_idx], True))
                valid_idx = min(valid_idx + 1, len(valid_inputs) - 1)

        # Build trace
        trace_lines = []
        for pos, (inp, is_valid) in enumerate(items):
            inp_str = _fmt(inp)
            out = _fmt(_apply_shift(op_type, k, inp))
            mark = "\u2713" if is_valid else "\u2717"
            trace_lines.append(
                f"{labels[pos]}) {op_name}({inp_str}) \u2192 {out} "
                f"{'= ' + result_str if is_valid else '\u2260 ' + result_str} {mark}"
            )

        wrong_label = labels[wrong_pos]

        option_strs = '  '.join(
            f"{labels[p]}) {_fmt(items[p][0])}" for p in range(3)
        )
        prompt = (
            f"If {op_name}(x) = {result_str}, which CANNOT be x?\n"
            f"{option_strs}"
        )

        reasoning = (
            f"<think>\n[TRAINING DRILL]\n"
            f"{op_name}: {desc}.\n"
            + '\n'.join(trace_lines) +
            f"\n</think>"
        )

        messages = [
            {"role": "user", "content": TRAINING_TAG + prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"{reasoning}\n\\boxed{{{wrong_label}}}"},
        ]

        rows.append(_make_row(messages, wrong_label,
                              f"microskill_v2_counter_{i:06d}", "counterfactual"))

    return rows


# ── Variant 3: "What must be true?" / Properties ────────────────────

# Property templates: (setup_fn, question, answer, trace)
# setup_fn(rng) -> dict with substitution variables

def _prop_xnor_all_ones(rng):
    a = rng.randint(0, 255)
    a_str = _fmt(a)
    result = _fmt(gate_xnor(a, a))
    assert result == '11111111'
    return {
        'question': f"If xnor(A, B) = 11111111, what must be true?",
        'trace': "xnor: same->1, diff->0. All 1s means every position matched.",
        'answer': "A equals B",
    }


def _prop_xor_all_zeros(rng):
    a = rng.randint(0, 255)
    a_str = _fmt(a)
    result = _fmt(gate_xor(a, a))
    assert result == '00000000'
    return {
        'question': f"If xor(A, B) = 00000000, what must be true?",
        'trace': "xor: diff->1, same->0. All 0s means every position matched.",
        'answer': "A equals B",
    }


def _prop_and_ones(rng):
    a = rng.randint(0, 255)
    b = rng.randint(0, 255)
    r = gate_and(a, b)
    r_str = _fmt(r)
    # Find a position where result is 1
    ones_positions = [i for i in range(8) if (r >> (7 - i)) & 1]
    if not ones_positions:
        # Force at least one
        pos = rng.randint(0, 7)
        a |= (1 << (7 - pos))
        b |= (1 << (7 - pos))
        r = gate_and(a, b)
        r_str = _fmt(r)
        ones_positions = [i for i in range(8) if (r >> (7 - i)) & 1]

    pos = rng.choice(ones_positions)
    a_str, b_str = _fmt(a), _fmt(b)

    return {
        'question': f"and({a_str}, {b_str}) = {r_str}. Position {pos} of the result is 1. What must be true about position {pos} of A and B?",
        'trace': f"and: both 1->1, else 0. Result position {pos} is 1, so both A[{pos}] and B[{pos}] must be 1.\nA[{pos}] = {a_str[pos]}, B[{pos}] = {b_str[pos]}. Both are 1. Confirmed.",
        'answer': "Both A and B have 1 at that position",
    }


def _prop_or_zeros(rng):
    a = rng.randint(0, 255)
    b = rng.randint(0, 255)
    r = gate_or(a, b)
    r_str = _fmt(r)
    zeros_positions = [i for i in range(8) if not ((r >> (7 - i)) & 1)]
    if not zeros_positions:
        pos = rng.randint(0, 7)
        a &= ~(1 << (7 - pos)) & BYTE
        b &= ~(1 << (7 - pos)) & BYTE
        r = gate_or(a, b)
        r_str = _fmt(r)
        zeros_positions = [i for i in range(8) if not ((r >> (7 - i)) & 1)]

    pos = rng.choice(zeros_positions)
    a_str, b_str = _fmt(a), _fmt(b)

    return {
        'question': f"or({a_str}, {b_str}) = {r_str}. Position {pos} of the result is 0. What must be true about position {pos} of A and B?",
        'trace': f"or: either 1->1, else 0. Result position {pos} is 0, so both A[{pos}] and B[{pos}] must be 0.\nA[{pos}] = {a_str[pos]}, B[{pos}] = {b_str[pos]}. Both are 0. Confirmed.",
        'answer': "Both A and B have 0 at that position",
    }


def _prop_shr_leading_zeros(rng):
    k = rng.randint(1, 7)
    x = rng.randint(0, 255)
    r = shr(x, k)
    r_str = _fmt(r)
    assert r_str[:k] == '0' * k
    return {
        'question': f"If shr{k}(x) = {r_str}, what must be true about the first {k} bit{'s' if k > 1 else ''} of the result?",
        'trace': f"shr{k} prepends {k} zero{'s' if k > 1 else ''} and drops last {k}. So the first {k} bit{'s' if k > 1 else ''} {'are' if k > 1 else 'is'} always 0.\nResult first {k}: {r_str[:k]} = {'0' * k}. Confirmed.",
        'answer': f"The first {k} bit{'s' if k > 1 else ''} {'are' if k > 1 else 'is'} always 0",
    }


def _prop_shl_trailing_zeros(rng):
    k = rng.randint(1, 7)
    x = rng.randint(0, 255)
    r = shl(x, k)
    r_str = _fmt(r)
    assert r_str[8 - k:] == '0' * k
    return {
        'question': f"If shl{k}(x) = {r_str}, what must be true about the last {k} bit{'s' if k > 1 else ''} of the result?",
        'trace': f"shl{k} drops first {k} and appends {k} zero{'s' if k > 1 else ''}. So the last {k} bit{'s' if k > 1 else ''} {'are' if k > 1 else 'is'} always 0.\nResult last {k}: {r_str[8-k:]} = {'0' * k}. Confirmed.",
        'answer': f"The last {k} bit{'s' if k > 1 else ''} {'are' if k > 1 else 'is'} always 0",
    }


def _prop_rotate_popcount(rng):
    op = rng.choice(['rol', 'ror'])
    k = rng.randint(1, 7)
    x = rng.randint(0, 255)
    if op == 'rol':
        r = rol(x, k)
    else:
        r = ror(x, k)
    x_str = _fmt(x)
    r_str = _fmt(r)
    x_pop = bin(x).count('1')
    r_pop = bin(r).count('1')
    assert x_pop == r_pop

    return {
        'question': f"{op}{k}({x_str}) = {r_str}. How many 1-bits does the result have compared to the input?",
        'trace': f"{op}{k} rotates bits; no bits are lost or created.\nInput 1-bits: {x_pop}. Output 1-bits: {r_pop}. Same count.",
        'answer': f"Same count: {x_pop}",
    }


PROPERTY_GENERATORS = [
    _prop_xnor_all_ones,
    _prop_xor_all_zeros,
    _prop_and_ones,
    _prop_or_zeros,
    _prop_shr_leading_zeros,
    _prop_shl_trailing_zeros,
    _prop_rotate_popcount,
]


def gen_properties(rng, n=1000):
    """Variant 3: what must be true given an operation's result?"""
    rows = []
    for i in range(n):
        gen_fn = rng.choice(PROPERTY_GENERATORS)
        prop = gen_fn(rng)

        reasoning = f"<think>\n[TRAINING DRILL]\n{prop['trace']}\n</think>"
        messages = [
            {"role": "user", "content": TRAINING_TAG + prop['question'] + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"{reasoning}\n\\boxed{{{prop['answer']}}}"},
        ]

        rows.append(_make_row(messages, prop['answer'],
                              f"microskill_v3_prop_{i:06d}", "properties"))

    return rows


# ── Variant 4: "Explain step by step" ───────────────────────────────

def _step_by_step_shift(op_type, k, x_str):
    """Generate detailed step-by-step trace for a shift/rotate/not."""
    if op_type == 'shr':
        result = ('0' * k + x_str)[:8]
        steps = [
            f"shr{k} means shift right by {k}: prepend {k} zero{'s' if k > 1 else ''}, drop last {k}.",
            f"Input: {x_str}",
            f"Prepend {k} zero{'s' if k > 1 else ''}: {'0' * k}{x_str}",
            f"Take first 8 bits: {result}",
        ]
    elif op_type == 'shl':
        result = (x_str + '0' * k)[k:k + 8]
        steps = [
            f"shl{k} means shift left by {k}: drop first {k}, append {k} zero{'s' if k > 1 else ''}.",
            f"Input: {x_str}",
            f"Drop first {k}: {x_str[k:]}",
            f"Append {k} zero{'s' if k > 1 else ''}: {x_str[k:]}{'0' * k}",
            f"Result: {result}",
        ]
    elif op_type == 'rol':
        first = x_str[:k]
        rest = x_str[k:]
        result = rest + first
        steps = [
            f"rol{k} means rotate left by {k}: move the first {k} bit{'s' if k > 1 else ''} to the end.",
            f"Input: {x_str}",
            f"First {k} bit{'s' if k > 1 else ''}: {first}",
            f"Remaining: {rest}",
            f"Result: {rest} + {first} = {result}",
        ]
    elif op_type == 'ror':
        tail = x_str[8 - k:]
        head = x_str[:8 - k]
        result = tail + head
        steps = [
            f"ror{k} means rotate right by {k}: move the last {k} bit{'s' if k > 1 else ''} to the front.",
            f"Input: {x_str}",
            f"Last {k} bit{'s' if k > 1 else ''}: {tail}",
            f"Remaining: {head}",
            f"Result: {tail} + {head} = {result}",
        ]
    elif op_type == 'not':
        result = ''.join('1' if c == '0' else '0' for c in x_str)
        # Show a few position flips
        flips = []
        for j in range(8):
            flips.append(f"{x_str[j]}->{result[j]}")
        steps = [
            f"not means flip each bit: 0->1, 1->0.",
            f"Input:  {' '.join(x_str)}",
            f"Output: {' '.join(result)}",
            f"Each position flipped: {', '.join(flips)}",
        ]
    else:
        raise ValueError(op_type)

    return steps, result


def gen_step_by_step(rng, n=1000):
    """Variant 4: longer step-by-step explanation of single operations."""
    rows = []
    for i in range(n):
        op_type, k = rng.choice(SHIFT_OPS)
        x = rng.randint(0, 255)
        x_str = _fmt(x)
        op_name = _shift_name(op_type, k)

        steps, result = _step_by_step_shift(op_type, k, x_str)

        # Verify
        expected = _fmt(_apply_shift(op_type, k, x))
        assert result == expected, f"{op_name}({x_str}): step_result={result} != expected={expected}"

        prompt = f"Show step by step how to compute {op_name}({x_str})."
        reasoning = "<think>\n[TRAINING DRILL]\n" + '\n'.join(steps) + "\n</think>"

        messages = [
            {"role": "user", "content": TRAINING_TAG + prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"{reasoning}\n\\boxed{{{result}}}"},
        ]

        rows.append(_make_row(messages, result,
                              f"microskill_v4_steps_{i:06d}", "step_by_step"))

    return rows


# ── Variant 5: "Reverse: find the input" ────────────────────────────

def gen_reverse(rng, n=1000):
    """Variant 5: given operation + output, find a valid input."""
    rows = []
    for i in range(n):
        op_type, k = rng.choice(SHIFT_OPS)
        op_name = _shift_name(op_type, k)

        # Generate a random input, compute the output, then ask to recover input
        original_x = rng.randint(0, 255)
        result_int = _apply_shift(op_type, k, original_x)
        result_str = _fmt(result_int)

        if op_type == 'shr':
            # shr{k}(x) = result. x = result << k | (any k low bits).
            # Pick one valid x (e.g., with 0s in the unknown positions)
            answer_int = (result_int << k) & BYTE
            unknown_bits = "??" if k > 1 else "?"
            trace = (
                f"shr{k} prepends {k} zero{'s' if k > 1 else ''} and drops last {k}.\n"
                f"Reverse: result = {result_str}. The top {k} bit{'s are' if k > 1 else ' is'} the prepended zero{'s' if k > 1 else ''}.\n"
                f"So x[0..{7-k}] = result[{k}..7] = {result_str[k:]}.\n"
                f"The last {k} bit{'s' if k > 1 else ''} of x {'were' if k > 1 else 'was'} dropped (unknown).\n"
                f"One valid x: {_fmt(answer_int)}.\n"
                f"Check: shr{k}({_fmt(answer_int)}) = {_fmt(_apply_shift('shr', k, answer_int))} = {result_str} \u2713"
            )
        elif op_type == 'shl':
            answer_int = (result_int >> k) & BYTE
            trace = (
                f"shl{k} drops first {k} and appends {k} zero{'s' if k > 1 else ''}.\n"
                f"Reverse: result = {result_str}. The bottom {k} bit{'s are' if k > 1 else ' is'} the appended zero{'s' if k > 1 else ''}.\n"
                f"So x[{k}..7] = result[0..{7-k}] = {result_str[:8-k]}.\n"
                f"The first {k} bit{'s' if k > 1 else ''} of x {'were' if k > 1 else 'was'} dropped (unknown).\n"
                f"One valid x: {_fmt(answer_int)}.\n"
                f"Check: shl{k}({_fmt(answer_int)}) = {_fmt(_apply_shift('shl', k, answer_int))} = {result_str} \u2713"
            )
        elif op_type == 'rol':
            # Inverse of rol{k} is ror{k}
            answer_int = ror(result_int, k)
            trace = (
                f"rol{k} moves first {k} to end. Inverse: move last {k} to front (= ror{k}).\n"
                f"result = {result_str}.\n"
                f"ror{k}({result_str}) = {_fmt(answer_int)}.\n"
                f"Check: rol{k}({_fmt(answer_int)}) = {_fmt(rol(answer_int, k))} = {result_str} \u2713"
            )
        elif op_type == 'ror':
            answer_int = rol(result_int, k)
            trace = (
                f"ror{k} moves last {k} to front. Inverse: move first {k} to end (= rol{k}).\n"
                f"result = {result_str}.\n"
                f"rol{k}({result_str}) = {_fmt(answer_int)}.\n"
                f"Check: ror{k}({_fmt(answer_int)}) = {_fmt(ror(answer_int, k))} = {result_str} \u2713"
            )
        elif op_type == 'not':
            answer_int = bit_not(result_int)
            trace = (
                f"not flips each bit. Applying not again recovers the input.\n"
                f"not({result_str}) = {_fmt(answer_int)}.\n"
                f"Check: not({_fmt(answer_int)}) = {_fmt(bit_not(answer_int))} = {result_str} \u2713"
            )
        else:
            raise ValueError(op_type)

        answer_str = _fmt(answer_int)

        # Verify the answer is valid
        assert _fmt(_apply_shift(op_type, k, answer_int)) == result_str, \
            f"Reverse verification failed: {op_name}({answer_str}) != {result_str}"

        prompt = f"If {op_name}(x) = {result_str}, what is x?"
        reasoning = f"<think>\n[TRAINING DRILL]\n{trace}\n</think>"

        messages = [
            {"role": "user", "content": TRAINING_TAG + prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"{reasoning}\n\\boxed{{{answer_str}}}"},
        ]

        rows.append(_make_row(messages, answer_str,
                              f"microskill_v5_reverse_{i:06d}", "reverse"))

    return rows


# ── Variant 6: "Two-step identification" ────────────────────────────

def gen_two_step_id(rng, n=1000):
    """Variant 6: identify which shift was used in a gate(x, shift(x)) chain."""
    rows = []
    gate_names = list(GATE_OPS.keys())

    for i in range(n):
        x = rng.randint(0, 255)
        x_str = _fmt(x)

        gate_name = rng.choice(gate_names)
        gate_fn, gate_rule = GATE_OPS[gate_name]

        # Pick the correct shift
        correct_op, correct_k = rng.choice(SHIFT_OPS)
        correct_name = _shift_name(correct_op, correct_k)
        a_int = _apply_shift(correct_op, correct_k, x)
        result_int = gate_fn(x, a_int) & BYTE
        result_str = _fmt(result_int)
        a_str = _fmt(a_int)

        # Pick one wrong alternative
        candidates = _similar_ops(correct_op, correct_k)
        rng.shuffle(candidates)
        wrong_name = None
        for wop, wk in candidates:
            w_a = _apply_shift(wop, wk, x)
            w_result = gate_fn(x, w_a) & BYTE
            if w_result != result_int:
                wrong_name = _shift_name(wop, wk)
                wrong_a_str = _fmt(w_a)
                wrong_result_str = _fmt(w_result)
                break

        if wrong_name is None:
            # Try all ops
            all_ops = [o for o in SHIFT_OPS if _shift_name(o[0], o[1]) != correct_name]
            rng.shuffle(all_ops)
            for wop, wk in all_ops:
                w_a = _apply_shift(wop, wk, x)
                w_result = gate_fn(x, w_a) & BYTE
                if w_result != result_int:
                    wrong_name = _shift_name(wop, wk)
                    wrong_a_str = _fmt(w_a)
                    wrong_result_str = _fmt(w_result)
                    break

        if wrong_name is None:
            continue  # Skip (very rare)

        # Randomly order the two options
        if rng.random() < 0.5:
            opt1, opt2 = correct_name, wrong_name
        else:
            opt1, opt2 = wrong_name, correct_name

        # Build trace: work backwards from result to find A, then test each shift
        trace_lines = [
            f"{gate_name}(x, A) = {result_str}, where A = shift(x).",
            f"{gate_name}: {gate_rule}.",
            f"",
            f"Try {correct_name}:",
            f"  A = {correct_name}({x_str}) = {a_str}",
            f"  {gate_name}({x_str}, {a_str}) = {_fmt(gate_fn(x, a_int))} = {result_str} \u2713",
            f"",
            f"Try {wrong_name}:",
            f"  A = {wrong_name}({x_str}) = {wrong_a_str}",
            f"  {gate_name}({x_str}, {wrong_a_str}) = {wrong_result_str} \u2260 {result_str} \u2717",
        ]

        prompt = (
            f"x = {x_str}, result = {result_str}. "
            f"We computed A = shift(x), then {gate_name}(x, A) = result. "
            f"Was the shift {opt1} or {opt2}?"
        )

        reasoning = "<think>\n[TRAINING DRILL]\n" + '\n'.join(trace_lines) + "\n</think>"

        messages = [
            {"role": "user", "content": TRAINING_TAG + prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"{reasoning}\n\\boxed{{{correct_name}}}"},
        ]

        rows.append(_make_row(messages, correct_name,
                              f"microskill_v6_twostep_{i:06d}", "two_step_id"))

    return rows


# ── Main ────────────────────────────────────────────────────────────

VARIANT_MAP = {
    'which_op':       (gen_which_op, 1000),
    'counterfactual': (gen_counterfactual, 1000),
    'properties':     (gen_properties, 1000),
    'step_by_step':   (gen_step_by_step, 1000),
    'reverse':        (gen_reverse, 1000),
    'two_step_id':    (gen_two_step_id, 1000),
}


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate bit manipulation micro-skill variant examples")
    parser.add_argument("--variant", type=str,
                        choices=list(VARIANT_MAP.keys()) + ['all'],
                        default='all', help="Which variant to generate")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of examples per variant (overrides default)")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--output-dir", type=str,
                        default="data/bit_manipulation/pool/generated")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.variant == 'all':
        variants = list(VARIANT_MAP.keys())
    else:
        variants = [args.variant]

    all_rows = []
    for variant in variants:
        gen_fn, default_n = VARIANT_MAP[variant]
        count = args.n or default_n
        t0 = time.time()
        rows = gen_fn(rng, count)
        dt = time.time() - t0
        print(f"  {variant}: {len(rows)} rows in {dt:.1f}s")
        all_rows.extend(rows)

    # Shuffle all rows together
    rng.shuffle(all_rows)

    path = os.path.join(args.output_dir, "microskill_variants.jsonl")
    write_jsonl(all_rows, path)

    # Summary
    from collections import Counter
    mode_counts = Counter(r['mode'] for r in all_rows)
    print(f"\nTotal: {len(all_rows)} rows")
    for mode, cnt in sorted(mode_counts.items()):
        print(f"  {mode}: {cnt}")


if __name__ == "__main__":
    main()
