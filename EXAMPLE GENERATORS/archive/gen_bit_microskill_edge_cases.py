#!/usr/bin/env python3
"""Generate bit manipulation edge case micro-skill examples.

Targets 6 categories of edge cases that test true understanding:
  1. zero_ones_boundary  (300) - All 0s/1s, bit disappearance vs wrap
  2. shift_vs_rotate     (400) - Side-by-side comparison of shift vs rotate
  3. large_shift         (300) - shr6/7, shl6/7 where almost all bits lost
  4. gate_edge           (400) - Gate identity/annihilation/complement laws
  5. popcount            (200) - Popcount preservation (rotate) vs loss (shift)
  6. composition_edge    (400) - Composition giving all-0s/all-1s, identity, coincidence

Total: 2000+ examples.

Usage:
    python3 -m generators.gen_bit_microskill_edge_cases
    python3 -m generators.gen_bit_microskill_edge_cases --category shift_vs_rotate --n 200
    python3 -m generators.gen_bit_microskill_edge_cases --seed 42
"""
import argparse
import json
import os
import random
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


def gate_xnor(a, b): return (~(a ^ b)) & BYTE
def gate_xor(a, b):  return (a ^ b) & BYTE
def gate_and(a, b):  return (a & b) & BYTE
def gate_or(a, b):   return (a | b) & BYTE
def gate_nand(a, b): return (~(a & b)) & BYTE
def gate_nor(a, b):  return (~(a | b)) & BYTE


def popcount(x):
    return bin(x).count('1')


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


def _make_row(messages, answer, row_id, mode):
    return {
        "messages": messages,
        "answer": answer,
        "id": row_id,
        "puzzle_type": "bit_manipulation",
        "mode": f"microskill_edge_{mode}",
        "generator": "gen_bit_microskill_edge_cases",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _user_msg(prompt):
    return {"role": "user", "content": TRAINING_TAG + prompt + BOXED_INSTRUCTION}


def _asst_msg(think, answer):
    return {"role": "assistant", "content": f"<think>\n[TRAINING DRILL]\n{think}\n</think>\n\\boxed{{{answer}}}"}


# ══════════════════════════════════════════════════════════════════════
# Category 1: Zero/Ones Boundaries (300 examples)
# ══════════════════════════════════════════════════════════════════════

def gen_zero_ones_boundary(rng, n=300):
    rows = []
    idx = 0

    # --- All operations on 00000000 ---
    for op_type in ('shr', 'shl', 'rol', 'ror'):
        for k in range(1, 8):
            x = 0x00
            result = _apply_shift(op_type, k, x)
            name = _shift_name(op_type, k)
            prompt = f"Compute {name}(00000000)."
            think = f"{name}(00000000): all bits are 0, so shifting/rotating zeros gives all zeros → 00000000"
            rows.append(_make_row(
                [_user_msg(prompt), _asst_msg(think, "00000000")],
                "00000000", f"edge_zob_allzero_{idx:04d}", "zero_ones"
            ))
            idx += 1

    # --- All operations on 11111111 ---
    for op_type in ('shr', 'shl'):
        for k in range(1, 8):
            x = 0xFF
            result = _apply_shift(op_type, k, x)
            name = _shift_name(op_type, k)
            result_str = _fmt(result)
            if op_type == 'shr':
                lost = k
                think = (f"{name}(11111111): shift right by {k}, prepend {k} zeros, drop last {k} bits.\n"
                         f"Lost {lost} ones from the right → {result_str}")
            else:
                lost = k
                think = (f"{name}(11111111): shift left by {k}, drop first {k} bits, append {k} zeros.\n"
                         f"Lost {lost} ones from the left → {result_str}")
            rows.append(_make_row(
                [_user_msg(f"Compute {name}(11111111)."), _asst_msg(think, result_str)],
                result_str, f"edge_zob_allones_{idx:04d}", "zero_ones"
            ))
            idx += 1

    for op_type in ('rol', 'ror'):
        for k in range(1, 8):
            name = _shift_name(op_type, k)
            think = f"{name}(11111111): rotating all 1s just gives all 1s → 11111111"
            rows.append(_make_row(
                [_user_msg(f"Compute {name}(11111111)."), _asst_msg(think, "11111111")],
                "11111111", f"edge_zob_allones_{idx:04d}", "zero_ones"
            ))
            idx += 1

    # NOT on 00000000 and 11111111
    rows.append(_make_row(
        [_user_msg("Compute not(00000000)."),
         _asst_msg("not(00000000): flip every bit. All 0s become all 1s → 11111111", "11111111")],
        "11111111", f"edge_zob_not_{idx:04d}", "zero_ones"
    ))
    idx += 1
    rows.append(_make_row(
        [_user_msg("Compute not(11111111)."),
         _asst_msg("not(11111111): flip every bit. All 1s become all 0s → 00000000", "00000000")],
        "00000000", f"edge_zob_not_{idx:04d}", "zero_ones"
    ))
    idx += 1

    # --- shr on 00000001: the 1 disappears for shr1+ ---
    for k in range(1, 8):
        result = shr(0x01, k)
        result_str = _fmt(result)
        think = (f"shr{k}(00000001): the single 1-bit is at position 0 (rightmost).\n"
                 f"Shift right by {k}: the 1 falls off the right edge → {result_str}")
        rows.append(_make_row(
            [_user_msg(f"Compute shr{k}(00000001)."), _asst_msg(think, result_str)],
            result_str, f"edge_zob_shr1_{idx:04d}", "zero_ones"
        ))
        idx += 1

    # --- shl on 10000000: the 1 disappears for shl1+ ---
    for k in range(1, 8):
        result = shl(0x80, k)
        result_str = _fmt(result)
        think = (f"shl{k}(10000000): the single 1-bit is at position 7 (leftmost).\n"
                 f"Shift left by {k}: the 1 falls off the left edge → {result_str}")
        rows.append(_make_row(
            [_user_msg(f"Compute shl{k}(10000000)."), _asst_msg(think, result_str)],
            result_str, f"edge_zob_shl1_{idx:04d}", "zero_ones"
        ))
        idx += 1

    # --- ror on 00000001: the 1 MOVES (wraps) ---
    for k in range(1, 8):
        result = ror(0x01, k)
        result_str = _fmt(result)
        think = (f"ror{k}(00000001): rotate right by {k}. The single 1-bit wraps around.\n"
                 f"It moves from position 0 to position {8 - k} → {result_str}")
        rows.append(_make_row(
            [_user_msg(f"Compute ror{k}(00000001)."), _asst_msg(think, result_str)],
            result_str, f"edge_zob_ror1_{idx:04d}", "zero_ones"
        ))
        idx += 1

    # --- rol on 10000000: the 1 MOVES (wraps) ---
    for k in range(1, 8):
        result = rol(0x80, k)
        result_str = _fmt(result)
        think = (f"rol{k}(10000000): rotate left by {k}. The single 1-bit wraps around.\n"
                 f"It moves from position 7 to position {(7 - k) % 8} → {result_str}")
        rows.append(_make_row(
            [_user_msg(f"Compute rol{k}(10000000)."), _asst_msg(think, result_str)],
            result_str, f"edge_zob_rol1_{idx:04d}", "zero_ones"
        ))
        idx += 1

    # --- Shift vs rotate contrast on single-bit inputs ---
    contrast_cases = [
        (0x01, 'shr', 'ror'),  # 00000001: shr loses, ror wraps
        (0x80, 'shl', 'rol'),  # 10000000: shl loses, rol wraps
    ]
    for x_val, shift_op, rot_op in contrast_cases:
        for k in range(1, 8):
            x_str = _fmt(x_val)
            s_result = _fmt(_apply_shift(shift_op, k, x_val))
            r_result = _fmt(_apply_shift(rot_op, k, x_val))
            s_name = f"{shift_op}{k}"
            r_name = f"{rot_op}{k}"
            prompt = f"Compare {s_name}({x_str}) vs {r_name}({x_str}). Do they give the same result?"
            if s_result == r_result:
                think = (f"{s_name}({x_str}) = {s_result}\n"
                         f"{r_name}({x_str}) = {r_result}\n"
                         f"Same result: both give {s_result}")
                answer = f"Yes, both give {s_result}"
            else:
                think = (f"{s_name}({x_str}) = {s_result} (bit lost — shifted out)\n"
                         f"{r_name}({x_str}) = {r_result} (bit preserved — wrapped around)\n"
                         f"Different: shift loses the bit, rotate preserves it")
                answer = f"No. {s_name} gives {s_result}, {r_name} gives {r_result}"
            rows.append(_make_row(
                [_user_msg(prompt), _asst_msg(think, answer)],
                answer, f"edge_zob_contrast_{idx:04d}", "zero_ones"
            ))
            idx += 1

    # --- Additional: single-bit at various positions through shr vs ror ---
    for bit_pos in range(8):
        x_val = 1 << bit_pos
        x_str = _fmt(x_val)
        for k in range(1, 8):
            s_result = shr(x_val, k)
            r_result = ror(x_val, k)
            s_str = _fmt(s_result)
            r_str = _fmt(r_result)
            if s_result != r_result:
                prompt = f"Compute shr{k}({x_str}) and ror{k}({x_str})."
                think = (f"Single 1-bit at position {bit_pos}.\n"
                         f"shr{k}: bit at position {bit_pos} → position {bit_pos - k}. "
                         f"{'Lost (negative position)' if bit_pos - k < 0 else 'Survives'} → {s_str}\n"
                         f"ror{k}: bit wraps around → position {(bit_pos - k) % 8} → {r_str}")
                answer = f"shr{k}: {s_str}, ror{k}: {r_str}"
                rows.append(_make_row(
                    [_user_msg(prompt), _asst_msg(think, answer)],
                    answer, f"edge_zob_singlebit_{idx:04d}", "zero_ones"
                ))
                idx += 1

    # --- Single-bit at various positions through shl vs rol ---
    for bit_pos in range(8):
        x_val = 1 << bit_pos
        x_str = _fmt(x_val)
        for k in range(1, 8):
            s_result = shl(x_val, k)
            r_result = rol(x_val, k)
            s_str = _fmt(s_result)
            r_str = _fmt(r_result)
            if s_result != r_result:
                prompt = f"Compute shl{k}({x_str}) and rol{k}({x_str})."
                think = (f"Single 1-bit at position {bit_pos}.\n"
                         f"shl{k}: bit at position {bit_pos} → position {bit_pos + k}. "
                         f"{'Lost (position ≥ 8)' if bit_pos + k >= 8 else 'Survives'} → {s_str}\n"
                         f"rol{k}: bit wraps around → position {(bit_pos + k) % 8} → {r_str}")
                answer = f"shl{k}: {s_str}, rol{k}: {r_str}"
                rows.append(_make_row(
                    [_user_msg(prompt), _asst_msg(think, answer)],
                    answer, f"edge_zob_singlebit_l_{idx:04d}", "zero_ones"
                ))
                idx += 1

    # --- Near-zero and near-ones: powers of 2, complements of powers ---
    near_vals = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                 0xFE, 0xFD, 0xFB, 0xF7, 0xEF, 0xDF, 0xBF, 0x7F]
    for x_val in near_vals:
        x_str = _fmt(x_val)
        for op_type in ('shr', 'shl', 'rol', 'ror'):
            k = rng.randint(1, 7)
            result = _apply_shift(op_type, k, x_val)
            result_str = _fmt(result)
            name = _shift_name(op_type, k)
            pc_in = popcount(x_val)
            pc_out = popcount(result)
            prompt = f"Compute {name}({x_str}). How many 1-bits survive?"
            if op_type in ('rol', 'ror'):
                think = (f"{x_str} has {pc_in} ones.\n"
                         f"{name} is a rotation — all bits preserved.\n"
                         f"{name}({x_str}) = {result_str} has {pc_out} ones. ✓")
            else:
                think = (f"{x_str} has {pc_in} ones.\n"
                         f"{name}({x_str}) = {result_str} has {pc_out} ones.\n"
                         f"{'All survived.' if pc_in == pc_out else f'Lost {pc_in - pc_out}.'}")
            answer = f"{result_str} ({pc_out} ones)"
            rows.append(_make_row(
                [_user_msg(prompt), _asst_msg(think, answer)],
                answer, f"edge_zob_near_{idx:04d}", "zero_ones"
            ))
            idx += 1

    # --- "Does the result become all zeros?" questions ---
    for _ in range(20):
        # Pick sparse inputs and large shifts that zero them out
        while True:
            x = rng.randint(1, 255)
            op_type = rng.choice(['shr', 'shl'])
            k = rng.randint(4, 7)
            result = _apply_shift(op_type, k, x)
            if result == 0:
                break
        x_str = _fmt(x)
        name = _shift_name(op_type, k)
        prompt = f"Does {name}({x_str}) produce all zeros?"
        think = (f"{name}({x_str}) = {_fmt(result)}\n"
                 f"Yes — all {popcount(x)} one-bits were in positions that get shifted out.")
        answer = f"Yes: {_fmt(result)}"
        rows.append(_make_row(
            [_user_msg(prompt), _asst_msg(think, answer)],
            answer, f"edge_zob_allzero_q_{idx:04d}", "zero_ones"
        ))
        idx += 1

    # --- Two-bit patterns: 00000011, 11000000, etc. ---
    two_bit_vals = [0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0xC0,
                    0x81, 0x42, 0x24, 0x18]
    for x_val in two_bit_vals:
        x_str = _fmt(x_val)
        for op_type in ('shr', 'shl'):
            for k in (1, 3, 5, 7):
                result = _apply_shift(op_type, k, x_val)
                result_str = _fmt(result)
                name = _shift_name(op_type, k)
                pc_in = popcount(x_val)
                pc_out = popcount(result)
                prompt = f"Compute {name}({x_str})."
                think = (f"{x_str} has {pc_in} ones.\n"
                         f"{name}({x_str}) = {result_str}.\n"
                         f"{'Both survived.' if pc_out == 2 else f'{pc_out} survived, lost {pc_in - pc_out}.'}")
                rows.append(_make_row(
                    [_user_msg(prompt), _asst_msg(think, result_str)],
                    result_str, f"edge_zob_twobit_{idx:04d}", "zero_ones"
                ))
                idx += 1

    rng.shuffle(rows)
    return rows[:n]


# ══════════════════════════════════════════════════════════════════════
# Category 2: Shift vs Rotate Distinction (400 examples)
# ══════════════════════════════════════════════════════════════════════

def gen_shift_vs_rotate(rng, n=400):
    rows = []
    idx = 0

    # --- Side-by-side: shr vs ror ---
    for k in range(1, 8):
        for _ in range(22):
            x = rng.randint(0, 255)
            x_str = _fmt(x)
            shr_result = _fmt(shr(x, k))
            ror_result = _fmt(ror(x, k))
            # Determine lost bits
            lost_bits = x_str[8 - k:]  # rightmost k bits
            prompt = f"Compare shr{k}({x_str}) vs ror{k}({x_str})."
            if shr_result == ror_result:
                think = (f"shr{k}({x_str}) = {shr_result}\n"
                         f"ror{k}({x_str}) = {ror_result}\n"
                         f"Same result. The rightmost {k} bit{'s' if k > 1 else ''} ({lost_bits}) "
                         f"{'are' if k > 1 else 'is'} all 0, so wrapping vs losing makes no difference.")
                answer = f"Both give {shr_result}"
            else:
                think = (f"shr{k}({x_str}) = {shr_result} (lost bits: {lost_bits})\n"
                         f"ror{k}({x_str}) = {ror_result} (wrapped bits: {lost_bits} moved to front)\n"
                         f"Different because the shifted-out bits ({lost_bits}) contain at least one 1.")
                answer = f"shr{k} gives {shr_result}, ror{k} gives {ror_result}"
            rows.append(_make_row(
                [_user_msg(prompt), _asst_msg(think, answer)],
                answer, f"edge_svr_shr_ror_{idx:04d}", "shift_vs_rotate"
            ))
            idx += 1

    # --- Side-by-side: shl vs rol ---
    for k in range(1, 8):
        for _ in range(22):
            x = rng.randint(0, 255)
            x_str = _fmt(x)
            shl_result = _fmt(shl(x, k))
            rol_result = _fmt(rol(x, k))
            lost_bits = x_str[:k]  # leftmost k bits
            prompt = f"Compare shl{k}({x_str}) vs rol{k}({x_str})."
            if shl_result == rol_result:
                think = (f"shl{k}({x_str}) = {shl_result}\n"
                         f"rol{k}({x_str}) = {rol_result}\n"
                         f"Same result. The leftmost {k} bit{'s' if k > 1 else ''} ({lost_bits}) "
                         f"{'are' if k > 1 else 'is'} all 0, so wrapping vs losing makes no difference.")
                answer = f"Both give {shl_result}"
            else:
                think = (f"shl{k}({x_str}) = {shl_result} (lost bits: {lost_bits})\n"
                         f"rol{k}({x_str}) = {rol_result} (wrapped bits: {lost_bits} moved to end)\n"
                         f"Different because the shifted-out bits ({lost_bits}) contain at least one 1.")
                answer = f"shl{k} gives {shl_result}, rol{k} gives {rol_result}"
            rows.append(_make_row(
                [_user_msg(prompt), _asst_msg(think, answer)],
                answer, f"edge_svr_shl_rol_{idx:04d}", "shift_vs_rotate"
            ))
            idx += 1

    # --- Explicit "what's lost" format ---
    for _ in range(80):
        x = rng.randint(1, 254)  # not 0 or 255
        x_str = _fmt(x)
        k = rng.randint(1, 7)

        # Pick shift/rotate pair
        if rng.random() < 0.5:
            s_op, r_op = 'shr', 'ror'
            lost = x_str[8 - k:]
            direction = "right"
            wrap_desc = "moved to front"
        else:
            s_op, r_op = 'shl', 'rol'
            lost = x_str[:k]
            direction = "left"
            wrap_desc = "moved to end"

        s_result = _fmt(_apply_shift(s_op, k, x))
        r_result = _fmt(_apply_shift(r_op, k, x))
        s_name = f"{s_op}{k}"
        r_name = f"{r_op}{k}"

        prompt = f"What bits are lost in {s_name}({x_str}) vs wrapped in {r_name}({x_str})?"
        think = (f"{s_name}({x_str}) = {s_result} (bits shifted out: {lost})\n"
                 f"{r_name}({x_str}) = {r_result} (bits wrapped: {lost} {wrap_desc})\n"
                 f"Shift loses {lost}, rotate preserves {lost}.")
        answer = f"Lost bits: {lost}. Shift result: {s_result}. Rotate result: {r_result}"
        rows.append(_make_row(
            [_user_msg(prompt), _asst_msg(think, answer)],
            answer, f"edge_svr_lost_{idx:04d}", "shift_vs_rotate"
        ))
        idx += 1

    # --- "When are they the same?" targeted examples ---
    # Find inputs where shr/ror give same result (rightmost k bits are 0)
    for k in range(1, 8):
        attempts = 0
        found = 0
        while found < 3 and attempts < 200:
            x = rng.randint(0, 255)
            if (x & ((1 << k) - 1)) == 0:  # rightmost k bits are 0
                x_str = _fmt(x)
                result = _fmt(shr(x, k))
                prompt = f"Do shr{k}({x_str}) and ror{k}({x_str}) give the same result?"
                think = (f"shr{k}({x_str}) = {result}\n"
                         f"ror{k}({x_str}) = {result}\n"
                         f"Yes — the rightmost {k} bits are all 0, so whether they're lost (shift) "
                         f"or wrapped (rotate), the result is the same.")
                answer = f"Yes, both give {result}"
                rows.append(_make_row(
                    [_user_msg(prompt), _asst_msg(think, answer)],
                    answer, f"edge_svr_same_{idx:04d}", "shift_vs_rotate"
                ))
                idx += 1
                found += 1
            attempts += 1

    for k in range(1, 8):
        attempts = 0
        found = 0
        while found < 3 and attempts < 200:
            x = rng.randint(0, 255)
            if (x >> (8 - k)) == 0:  # leftmost k bits are 0
                x_str = _fmt(x)
                result = _fmt(shl(x, k))
                prompt = f"Do shl{k}({x_str}) and rol{k}({x_str}) give the same result?"
                think = (f"shl{k}({x_str}) = {result}\n"
                         f"rol{k}({x_str}) = {result}\n"
                         f"Yes — the leftmost {k} bits are all 0, so whether they're lost (shift) "
                         f"or wrapped (rotate), the result is the same.")
                answer = f"Yes, both give {result}"
                rows.append(_make_row(
                    [_user_msg(prompt), _asst_msg(think, answer)],
                    answer, f"edge_svr_same_{idx:04d}", "shift_vs_rotate"
                ))
                idx += 1
                found += 1
            attempts += 1

    rng.shuffle(rows)
    return rows[:n]


# ══════════════════════════════════════════════════════════════════════
# Category 3: Large Shifts (300 examples)
# ══════════════════════════════════════════════════════════════════════

def gen_large_shift(rng, n=300):
    rows = []
    idx = 0

    # --- shr7: only MSB survives as LSB ---
    for _ in range(40):
        x = rng.randint(0, 255)
        x_str = _fmt(x)
        result = _fmt(shr(x, 7))
        msb = x_str[0]
        think = (f"shr7({x_str}): shift right by 7, only the leftmost bit survives.\n"
                 f"MSB = {msb} → result = 0000000{msb} → {result}")
        rows.append(_make_row(
            [_user_msg(f"Compute shr7({x_str})."), _asst_msg(think, result)],
            result, f"edge_ls_shr7_{idx:04d}", "large_shift"
        ))
        idx += 1

    # --- shl7: only LSB survives as MSB ---
    for _ in range(40):
        x = rng.randint(0, 255)
        x_str = _fmt(x)
        result = _fmt(shl(x, 7))
        lsb = x_str[7]
        think = (f"shl7({x_str}): shift left by 7, only the rightmost bit survives.\n"
                 f"LSB = {lsb} → result = {lsb}0000000 → {result}")
        rows.append(_make_row(
            [_user_msg(f"Compute shl7({x_str})."), _asst_msg(think, result)],
            result, f"edge_ls_shl7_{idx:04d}", "large_shift"
        ))
        idx += 1

    # --- shr6: only top 2 bits survive ---
    for _ in range(40):
        x = rng.randint(0, 255)
        x_str = _fmt(x)
        result = _fmt(shr(x, 6))
        top2 = x_str[:2]
        think = (f"shr6({x_str}): shift right by 6, only the top 2 bits survive.\n"
                 f"Top 2 bits: {top2} → result = 000000{top2} → {result}")
        rows.append(_make_row(
            [_user_msg(f"Compute shr6({x_str})."), _asst_msg(think, result)],
            result, f"edge_ls_shr6_{idx:04d}", "large_shift"
        ))
        idx += 1

    # --- shl6: only bottom 2 bits survive ---
    for _ in range(40):
        x = rng.randint(0, 255)
        x_str = _fmt(x)
        result = _fmt(shl(x, 6))
        bot2 = x_str[6:]
        think = (f"shl6({x_str}): shift left by 6, only the bottom 2 bits survive.\n"
                 f"Bottom 2 bits: {bot2} → result = {bot2}000000 → {result}")
        rows.append(_make_row(
            [_user_msg(f"Compute shl6({x_str})."), _asst_msg(think, result)],
            result, f"edge_ls_shl6_{idx:04d}", "large_shift"
        ))
        idx += 1

    # --- "How many bits survive?" questions ---
    for k in (5, 6, 7):
        survive = 8 - k
        for op_type in ('shr', 'shl'):
            for _ in range(15):
                x = rng.randint(0, 255)
                x_str = _fmt(x)
                result = _fmt(_apply_shift(op_type, k, x))
                result_ones = popcount(_apply_shift(op_type, k, x))
                input_ones = popcount(x)
                name = f"{op_type}{k}"
                prompt = f"How many of the 8 bits of {x_str} survive after {name}?"
                think = (f"{name}: {k} bits are shifted out, {survive} bit{'s' if survive > 1 else ''} survive.\n"
                         f"{name}({x_str}) = {result}\n"
                         f"Input had {input_ones} ones, output has {result_ones} ones.")
                answer = f"{survive}"
                rows.append(_make_row(
                    [_user_msg(prompt), _asst_msg(think, answer)],
                    answer, f"edge_ls_survive_{idx:04d}", "large_shift"
                ))
                idx += 1

    # --- Compare large shift vs rotate on 11111111 ---
    for k in (5, 6, 7):
        for op_pair in [('shr', 'ror'), ('shl', 'rol')]:
            s_op, r_op = op_pair
            s_result = _fmt(_apply_shift(s_op, k, 0xFF))
            r_result = _fmt(_apply_shift(r_op, k, 0xFF))
            s_name = f"{s_op}{k}"
            r_name = f"{r_op}{k}"
            prompt = f"Compare {s_name}(11111111) vs {r_name}(11111111)."
            think = (f"{s_name}(11111111) = {s_result} (lost {k} ones)\n"
                     f"{r_name}(11111111) = {r_result} (all ones just rotate, stay all ones)\n"
                     f"Shift changes the count, rotate preserves it.")
            answer = f"{s_name} gives {s_result}, {r_name} gives {r_result}"
            rows.append(_make_row(
                [_user_msg(prompt), _asst_msg(think, answer)],
                answer, f"edge_ls_ff_{idx:04d}", "large_shift"
            ))
            idx += 1

    # --- Step-by-step large shifts ---
    for k in (5, 6, 7):
        for _ in range(12):
            x = rng.randint(1, 254)
            x_str = _fmt(x)
            # shr
            result = _fmt(shr(x, k))
            kept = x_str[:8 - k]
            lost = x_str[8 - k:]
            think = (f"shr{k}({x_str}): prepend {k} zeros, drop last {k}.\n"
                     f"Input bits: [{kept}][{lost}]\n"
                     f"Keep [{kept}], lose [{lost}].\n"
                     f"Result: {'0' * k}{kept} → {result}")
            rows.append(_make_row(
                [_user_msg(f"Show step by step: shr{k}({x_str})."), _asst_msg(think, result)],
                result, f"edge_ls_step_{idx:04d}", "large_shift"
            ))
            idx += 1

            # shl
            result = _fmt(shl(x, k))
            lost = x_str[:k]
            kept = x_str[k:]
            think = (f"shl{k}({x_str}): drop first {k}, append {k} zeros.\n"
                     f"Input bits: [{lost}][{kept}]\n"
                     f"Lose [{lost}], keep [{kept}].\n"
                     f"Result: {kept}{'0' * k} → {result}")
            rows.append(_make_row(
                [_user_msg(f"Show step by step: shl{k}({x_str})."), _asst_msg(think, result)],
                result, f"edge_ls_step_{idx:04d}", "large_shift"
            ))
            idx += 1

    rng.shuffle(rows)
    return rows[:n]


# ══════════════════════════════════════════════════════════════════════
# Category 4: Gate Edge Cases (400 examples)
# ══════════════════════════════════════════════════════════════════════

def gen_gate_edge(rng, n=400):
    rows = []
    idx = 0

    # Gate property templates
    # Each: (prompt_template, think_template, answer_template, gate_func)

    # --- xnor(X, X) = 11111111 ---
    for _ in range(40):
        x = rng.randint(0, 255)
        x_str = _fmt(x)
        result = _fmt(gate_xnor(x, x))
        assert result == "11111111"
        prompt = f"Compute xnor({x_str}, {x_str})."
        think = (f"xnor compares bits: same → 1, different → 0.\n"
                 f"Both inputs are identical ({x_str}), so every position matches.\n"
                 f"Result: 11111111")
        rows.append(_make_row(
            [_user_msg(prompt), _asst_msg(think, "11111111")],
            "11111111", f"edge_gate_xnor_id_{idx:04d}", "gate_edge"
        ))
        idx += 1

    # --- xor(X, X) = 00000000 ---
    for _ in range(40):
        x = rng.randint(0, 255)
        x_str = _fmt(x)
        result = _fmt(gate_xor(x, x))
        assert result == "00000000"
        prompt = f"Compute xor({x_str}, {x_str})."
        think = (f"xor compares bits: different → 1, same → 0.\n"
                 f"Both inputs are identical ({x_str}), so every position matches → all 0.\n"
                 f"Result: 00000000")
        rows.append(_make_row(
            [_user_msg(prompt), _asst_msg(think, "00000000")],
            "00000000", f"edge_gate_xor_id_{idx:04d}", "gate_edge"
        ))
        idx += 1

    # --- and(X, 00000000) = 00000000 ---
    for _ in range(30):
        x = rng.randint(0, 255)
        x_str = _fmt(x)
        prompt = f"Compute and({x_str}, 00000000)."
        think = (f"and: both must be 1 to get 1.\n"
                 f"Second input is all zeros, so every position gets 0 regardless of first input.\n"
                 f"Result: 00000000")
        rows.append(_make_row(
            [_user_msg(prompt), _asst_msg(think, "00000000")],
            "00000000", f"edge_gate_and_zero_{idx:04d}", "gate_edge"
        ))
        idx += 1

    # --- or(X, 11111111) = 11111111 ---
    for _ in range(30):
        x = rng.randint(0, 255)
        x_str = _fmt(x)
        prompt = f"Compute or({x_str}, 11111111)."
        think = (f"or: either 1 gives 1.\n"
                 f"Second input is all ones, so every position gets 1 regardless of first input.\n"
                 f"Result: 11111111")
        rows.append(_make_row(
            [_user_msg(prompt), _asst_msg(think, "11111111")],
            "11111111", f"edge_gate_or_ones_{idx:04d}", "gate_edge"
        ))
        idx += 1

    # --- and(X, 11111111) = X (identity) ---
    for _ in range(30):
        x = rng.randint(0, 255)
        x_str = _fmt(x)
        prompt = f"Compute and({x_str}, 11111111)."
        think = (f"and: both must be 1 to get 1.\n"
                 f"Second input is all ones, so the result copies the first input.\n"
                 f"and({x_str}, 11111111) = {x_str}")
        rows.append(_make_row(
            [_user_msg(prompt), _asst_msg(think, x_str)],
            x_str, f"edge_gate_and_identity_{idx:04d}", "gate_edge"
        ))
        idx += 1

    # --- or(X, 00000000) = X (identity) ---
    for _ in range(30):
        x = rng.randint(0, 255)
        x_str = _fmt(x)
        prompt = f"Compute or({x_str}, 00000000)."
        think = (f"or: either 1 gives 1.\n"
                 f"Second input is all zeros, so the result copies the first input.\n"
                 f"or({x_str}, 00000000) = {x_str}")
        rows.append(_make_row(
            [_user_msg(prompt), _asst_msg(think, x_str)],
            x_str, f"edge_gate_or_identity_{idx:04d}", "gate_edge"
        ))
        idx += 1

    # --- xor(X, 11111111) = not(X) ---
    for _ in range(35):
        x = rng.randint(0, 255)
        x_str = _fmt(x)
        result = _fmt(gate_xor(x, 0xFF))
        not_x = _fmt(bit_not(x))
        assert result == not_x
        prompt = f"Compute xor({x_str}, 11111111)."
        think = (f"xor with all 1s flips every bit (same as NOT).\n"
                 f"xor({x_str}, 11111111) = {result} = not({x_str})")
        rows.append(_make_row(
            [_user_msg(prompt), _asst_msg(think, result)],
            result, f"edge_gate_xor_comp_{idx:04d}", "gate_edge"
        ))
        idx += 1

    # --- xnor(X, 00000000) = not(X) ---
    for _ in range(30):
        x = rng.randint(0, 255)
        x_str = _fmt(x)
        result = _fmt(gate_xnor(x, 0x00))
        not_x = _fmt(bit_not(x))
        assert result == not_x
        prompt = f"Compute xnor({x_str}, 00000000)."
        think = (f"xnor with all zeros: same → 1, diff → 0.\n"
                 f"Every 0-bit in X matches the 0, every 1-bit differs.\n"
                 f"xnor({x_str}, 00000000) = {result} = not({x_str})")
        rows.append(_make_row(
            [_user_msg(prompt), _asst_msg(think, result)],
            result, f"edge_gate_xnor_comp_{idx:04d}", "gate_edge"
        ))
        idx += 1

    # --- nand(X, X) = not(X) ---
    for _ in range(30):
        x = rng.randint(0, 255)
        x_str = _fmt(x)
        result = _fmt(gate_nand(x, x))
        not_x = _fmt(bit_not(x))
        assert result == not_x
        prompt = f"Compute nand({x_str}, {x_str})."
        think = (f"nand: NOT of AND. nand(X, X) = not(and(X, X)) = not(X).\n"
                 f"nand({x_str}, {x_str}) = not({x_str}) = {result}")
        rows.append(_make_row(
            [_user_msg(prompt), _asst_msg(think, result)],
            result, f"edge_gate_nand_self_{idx:04d}", "gate_edge"
        ))
        idx += 1

    # --- nor(X, X) = not(X) ---
    for _ in range(30):
        x = rng.randint(0, 255)
        x_str = _fmt(x)
        result = _fmt(gate_nor(x, x))
        not_x = _fmt(bit_not(x))
        assert result == not_x
        prompt = f"Compute nor({x_str}, {x_str})."
        think = (f"nor: NOT of OR. nor(X, X) = not(or(X, X)) = not(X).\n"
                 f"nor({x_str}, {x_str}) = not({x_str}) = {result}")
        rows.append(_make_row(
            [_user_msg(prompt), _asst_msg(think, result)],
            result, f"edge_gate_nor_self_{idx:04d}", "gate_edge"
        ))
        idx += 1

    # --- "What property does this show?" format ---
    property_examples = [
        ("xnor", "self", lambda x: gate_xnor(x, x), "11111111",
         "xnor(X, X) always gives all 1s because every bit matches itself"),
        ("xor", "self", lambda x: gate_xor(x, x), "00000000",
         "xor(X, X) always gives all 0s because no bit differs from itself"),
    ]
    for gate_name, variant, func, expected, explanation in property_examples:
        for _ in range(15):
            x = rng.randint(0, 255)
            x_str = _fmt(x)
            result = _fmt(func(x))
            assert result == expected
            prompt = (f"{gate_name}({x_str}, {x_str}) = {result}. "
                      f"Is this always true for any X, or specific to this input?")
            think = (f"Result is {expected}.\n"
                     f"{explanation}.\n"
                     f"This holds for ALL 8-bit values, not just {x_str}.")
            answer = f"Always true: {gate_name}(X, X) = {expected} for any X"
            rows.append(_make_row(
                [_user_msg(prompt), _asst_msg(think, answer)],
                answer, f"edge_gate_prop_{idx:04d}", "gate_edge"
            ))
            idx += 1

    # --- Double complement: not(not(X)) = X ---
    for _ in range(25):
        x = rng.randint(0, 255)
        x_str = _fmt(x)
        mid = _fmt(bit_not(x))
        result = _fmt(bit_not(bit_not(x)))
        assert result == x_str
        prompt = f"Compute not(not({x_str}))."
        think = (f"not({x_str}) = {mid}\n"
                 f"not({mid}) = {result}\n"
                 f"Double NOT returns to the original: {result} = {x_str} ✓")
        rows.append(_make_row(
            [_user_msg(prompt), _asst_msg(think, result)],
            result, f"edge_gate_doublenot_{idx:04d}", "gate_edge"
        ))
        idx += 1

    # --- De Morgan's laws: nand(A,B) = or(not(A), not(B)) ---
    for _ in range(20):
        a = rng.randint(0, 255)
        b = rng.randint(0, 255)
        a_str, b_str = _fmt(a), _fmt(b)
        nand_result = _fmt(gate_nand(a, b))
        demorgan = _fmt(gate_or(bit_not(a), bit_not(b)))
        assert nand_result == demorgan
        prompt = f"Compute nand({a_str}, {b_str}). Verify it equals or(not({a_str}), not({b_str}))."
        think = (f"nand({a_str}, {b_str}) = {nand_result}\n"
                 f"not({a_str}) = {_fmt(bit_not(a))}, not({b_str}) = {_fmt(bit_not(b))}\n"
                 f"or({_fmt(bit_not(a))}, {_fmt(bit_not(b))}) = {demorgan}\n"
                 f"Equal: {nand_result} = {demorgan} ✓ (De Morgan's law)")
        rows.append(_make_row(
            [_user_msg(prompt), _asst_msg(think, nand_result)],
            nand_result, f"edge_gate_demorgan_{idx:04d}", "gate_edge"
        ))
        idx += 1

    # --- and(X, not(X)) = 00000000 (always) ---
    for _ in range(15):
        x = rng.randint(0, 255)
        x_str = _fmt(x)
        not_x_str = _fmt(bit_not(x))
        prompt = f"Compute and({x_str}, {not_x_str})."
        think = (f"{not_x_str} = not({x_str}). Every position where X is 1, not(X) is 0.\n"
                 f"and: both must be 1. No position has both → 00000000")
        rows.append(_make_row(
            [_user_msg(prompt), _asst_msg(think, "00000000")],
            "00000000", f"edge_gate_and_comp_{idx:04d}", "gate_edge"
        ))
        idx += 1

    rng.shuffle(rows)
    return rows[:n]


# ══════════════════════════════════════════════════════════════════════
# Category 5: Popcount Preservation (200 examples)
# ══════════════════════════════════════════════════════════════════════

def gen_popcount(rng, n=200):
    rows = []
    idx = 0

    # --- Rotate preserves popcount ---
    for op_type in ('rol', 'ror'):
        for _ in range(40):
            x = rng.randint(1, 254)  # avoid 0 and 255 for interesting cases
            k = rng.randint(1, 7)
            x_str = _fmt(x)
            result_int = _apply_shift(op_type, k, x)
            result_str = _fmt(result_int)
            pc = popcount(x)
            name = _shift_name(op_type, k)
            prompt = f"How many 1-bits does {name}({x_str}) have?"
            think = (f"{x_str} has {pc} ones.\n"
                     f"{name} is a rotation — bits wrap around, none are lost.\n"
                     f"{name}({x_str}) = {result_str} has {pc} ones. ✓ Preserved.")
            answer = f"{pc}"
            rows.append(_make_row(
                [_user_msg(prompt), _asst_msg(think, answer)],
                answer, f"edge_pop_rot_{idx:04d}", "popcount"
            ))
            idx += 1

    # --- Shift can reduce popcount ---
    for op_type in ('shr', 'shl'):
        for _ in range(40):
            # Pick inputs where bits WILL be lost (non-zero shifted-out bits)
            while True:
                x = rng.randint(1, 254)
                k = rng.randint(1, 7)
                result_int = _apply_shift(op_type, k, x)
                pc_in = popcount(x)
                pc_out = popcount(result_int)
                if pc_out < pc_in:
                    break
            x_str = _fmt(x)
            result_str = _fmt(result_int)
            lost = pc_in - pc_out
            name = _shift_name(op_type, k)
            prompt = f"How many 1-bits does {name}({x_str}) have compared to the input?"
            think = (f"{x_str} has {pc_in} ones.\n"
                     f"{name} is a shift — bits shifted out are lost.\n"
                     f"{name}({x_str}) = {result_str} has {pc_out} ones.\n"
                     f"Lost {lost} one{'s' if lost > 1 else ''} from the shifted-out bits.")
            answer = f"{pc_out} (lost {lost})"
            rows.append(_make_row(
                [_user_msg(prompt), _asst_msg(think, answer)],
                answer, f"edge_pop_shr_{idx:04d}", "popcount"
            ))
            idx += 1

    # --- Shift that preserves popcount (shifted-out bits are all 0) ---
    for op_type in ('shr', 'shl'):
        for _ in range(20):
            k = rng.randint(1, 5)
            while True:
                x = rng.randint(1, 254)
                if op_type == 'shr':
                    # Bottom k bits must be 0
                    if (x & ((1 << k) - 1)) == 0 and popcount(x) > 0:
                        break
                else:
                    # Top k bits must be 0
                    if (x >> (8 - k)) == 0 and popcount(x) > 0:
                        break
            x_str = _fmt(x)
            result_int = _apply_shift(op_type, k, x)
            result_str = _fmt(result_int)
            pc = popcount(x)
            name = _shift_name(op_type, k)
            prompt = f"Does {name}({x_str}) have the same number of 1-bits as the input?"
            think = (f"{x_str} has {pc} ones.\n"
                     f"{name}({x_str}) = {result_str} has {popcount(result_int)} ones.\n"
                     f"Yes — the shifted-out bits were all 0, so no 1s were lost.")
            answer = f"Yes, both have {pc}"
            rows.append(_make_row(
                [_user_msg(prompt), _asst_msg(think, answer)],
                answer, f"edge_pop_preserve_{idx:04d}", "popcount"
            ))
            idx += 1

    # --- Rotate vs shift comparison on same input ---
    for _ in range(40):
        x = rng.randint(1, 254)
        k = rng.randint(1, 7)
        x_str = _fmt(x)
        if rng.random() < 0.5:
            s_op, r_op = 'shr', 'ror'
        else:
            s_op, r_op = 'shl', 'rol'
        s_result_int = _apply_shift(s_op, k, x)
        r_result_int = _apply_shift(r_op, k, x)
        s_str = _fmt(s_result_int)
        r_str = _fmt(r_result_int)
        pc_in = popcount(x)
        pc_shift = popcount(s_result_int)
        pc_rot = popcount(r_result_int)
        s_name = f"{s_op}{k}"
        r_name = f"{r_op}{k}"
        prompt = f"Compare popcount: {s_name}({x_str}) vs {r_name}({x_str}). Input has {pc_in} ones."
        think = (f"{s_name}({x_str}) = {s_str} → {pc_shift} ones\n"
                 f"{r_name}({x_str}) = {r_str} → {pc_rot} ones\n"
                 f"Input had {pc_in} ones. Rotate preserves count ({pc_rot}). "
                 f"Shift {'also preserved (shifted-out bits were 0)' if pc_shift == pc_in else f'lost {pc_in - pc_shift}'}.")
        answer = f"Shift: {pc_shift}, Rotate: {pc_rot}"
        rows.append(_make_row(
            [_user_msg(prompt), _asst_msg(think, answer)],
            answer, f"edge_pop_compare_{idx:04d}", "popcount"
        ))
        idx += 1

    rng.shuffle(rows)
    return rows[:n]


# ══════════════════════════════════════════════════════════════════════
# Category 6: Composition Edge Cases (400 examples)
# ══════════════════════════════════════════════════════════════════════

def gen_composition_edge(rng, n=400):
    rows = []
    idx = 0

    # --- xnor(f(X), f(X)) = 11111111 (same transform twice → all 1s) ---
    for op_type in ('shr', 'shl', 'rol', 'ror'):
        for k in range(1, 8):
            for _ in range(3):
                x = rng.randint(0, 255)
                x_str = _fmt(x)
                name = _shift_name(op_type, k)
                f_x = _apply_shift(op_type, k, x)
                f_x_str = _fmt(f_x)
                result = _fmt(gate_xnor(f_x, f_x))
                assert result == "11111111"
                prompt = f"Compute xnor({name}({x_str}), {name}({x_str}))."
                think = (f"{name}({x_str}) = {f_x_str}\n"
                         f"xnor({f_x_str}, {f_x_str}): both inputs are identical.\n"
                         f"xnor of any value with itself = 11111111")
                rows.append(_make_row(
                    [_user_msg(prompt), _asst_msg(think, "11111111")],
                    "11111111", f"edge_comp_xnor_same_{idx:04d}", "composition"
                ))
                idx += 1

    # --- xor(f(X), f(X)) = 00000000 ---
    for op_type in ('shr', 'shl', 'rol', 'ror'):
        for k in range(1, 8):
            for _ in range(3):
                x = rng.randint(0, 255)
                x_str = _fmt(x)
                name = _shift_name(op_type, k)
                f_x = _apply_shift(op_type, k, x)
                f_x_str = _fmt(f_x)
                result = _fmt(gate_xor(f_x, f_x))
                assert result == "00000000"
                prompt = f"Compute xor({name}({x_str}), {name}({x_str}))."
                think = (f"{name}({x_str}) = {f_x_str}\n"
                         f"xor({f_x_str}, {f_x_str}): both inputs are identical.\n"
                         f"xor of any value with itself = 00000000")
                rows.append(_make_row(
                    [_user_msg(prompt), _asst_msg(think, "00000000")],
                    "00000000", f"edge_comp_xor_same_{idx:04d}", "composition"
                ))
                idx += 1

    # --- rol(k) then ror(k) = identity ---
    for k in range(1, 8):
        for _ in range(6):
            x = rng.randint(0, 255)
            x_str = _fmt(x)
            mid = _apply_shift('rol', k, x)
            mid_str = _fmt(mid)
            result = _apply_shift('ror', k, mid)
            result_str = _fmt(result)
            assert result_str == x_str
            prompt = f"Compute ror{k}(rol{k}({x_str}))."
            think = (f"rol{k}({x_str}) = {mid_str}\n"
                     f"ror{k}({mid_str}) = {result_str}\n"
                     f"Rotating left then right by the same amount returns the original: {result_str} = {x_str} ✓")
            rows.append(_make_row(
                [_user_msg(prompt), _asst_msg(think, result_str)],
                result_str, f"edge_comp_identity_{idx:04d}", "composition"
            ))
            idx += 1

    # --- shl(k) then shr(k) ≠ identity (bits lost) ---
    for k in range(1, 6):
        for _ in range(6):
            # Ensure top k bits have at least one 1 so we see the loss
            while True:
                x = rng.randint(1, 255)
                if (x >> (8 - k)) != 0:
                    break
            x_str = _fmt(x)
            mid = shl(x, k)
            mid_str = _fmt(mid)
            result = shr(mid, k)
            result_str = _fmt(result)
            prompt = f"Compute shr{k}(shl{k}({x_str})). Does it return the original?"
            if result_str == x_str:
                think = (f"shl{k}({x_str}) = {mid_str}\n"
                         f"shr{k}({mid_str}) = {result_str}\n"
                         f"Yes — happened to get back the original (top {k} bits were 0).")
            else:
                lost = x_str[:k]
                think = (f"shl{k}({x_str}) = {mid_str} (lost top {k} bits: {lost})\n"
                         f"shr{k}({mid_str}) = {result_str}\n"
                         f"NOT the original: {result_str} ≠ {x_str}. The top {k} bits ({lost}) were lost by shl.")
            answer = result_str
            rows.append(_make_row(
                [_user_msg(prompt), _asst_msg(think, answer)],
                answer, f"edge_comp_notidentity_{idx:04d}", "composition"
            ))
            idx += 1

    # --- Two different transforms that coincidentally give the same result ---
    for _ in range(50):
        x = rng.randint(0, 255)
        x_str = _fmt(x)
        # Try many pairs to find coincidence
        ops = [(op, k) for op in ('shr', 'shl', 'rol', 'ror') for k in range(1, 8)]
        results_map = {}
        for op, k in ops:
            r = _apply_shift(op, k, x)
            r_str = _fmt(r)
            if r_str not in results_map:
                results_map[r_str] = []
            results_map[r_str].append((op, k))
        # Find any collision
        collisions = [(r_str, ops_list) for r_str, ops_list in results_map.items()
                      if len(ops_list) >= 2]
        if collisions:
            r_str, ops_list = rng.choice(collisions)
            pair = rng.sample(ops_list, 2)
            n1 = _shift_name(pair[0][0], pair[0][1])
            n2 = _shift_name(pair[1][0], pair[1][1])
            prompt = f"Both {n1}({x_str}) and {n2}({x_str}) give the same result. What is it?"
            think = (f"{n1}({x_str}) = {r_str}\n"
                     f"{n2}({x_str}) = {r_str}\n"
                     f"Same result: {r_str}. Different operations can coincide on specific inputs.")
            rows.append(_make_row(
                [_user_msg(prompt), _asst_msg(think, r_str)],
                r_str, f"edge_comp_coincide_{idx:04d}", "composition"
            ))
            idx += 1

    # --- Compositions that give all 0s ---
    for _ in range(40):
        x = rng.randint(1, 254)
        x_str = _fmt(x)
        # xor(X, X) = 0, and(X, not(X)) = 0
        not_x = bit_not(x)
        not_x_str = _fmt(not_x)

        # and(X, not(X)) = 0
        prompt = f"Compute and({x_str}, not({x_str}))."
        think = (f"not({x_str}) = {not_x_str}\n"
                 f"and({x_str}, {not_x_str}): wherever X is 1, not(X) is 0, and vice versa.\n"
                 f"No position has both 1, so result = 00000000")
        rows.append(_make_row(
            [_user_msg(prompt), _asst_msg(think, "00000000")],
            "00000000", f"edge_comp_and_not_{idx:04d}", "composition"
        ))
        idx += 1

    # --- Compositions that give all 1s ---
    for _ in range(40):
        x = rng.randint(1, 254)
        x_str = _fmt(x)
        not_x_str = _fmt(bit_not(x))

        # or(X, not(X)) = 1
        prompt = f"Compute or({x_str}, not({x_str}))."
        think = (f"not({x_str}) = {not_x_str}\n"
                 f"or({x_str}, {not_x_str}): wherever X is 0, not(X) is 1, and vice versa.\n"
                 f"Every position has at least one 1, so result = 11111111")
        rows.append(_make_row(
            [_user_msg(prompt), _asst_msg(think, "11111111")],
            "11111111", f"edge_comp_or_not_{idx:04d}", "composition"
        ))
        idx += 1

    # --- not(not(X)) = X ---
    for _ in range(30):
        x = rng.randint(0, 255)
        x_str = _fmt(x)
        mid = _fmt(bit_not(x))
        prompt = f"Compute not(not({x_str}))."
        think = (f"not({x_str}) = {mid}\n"
                 f"not({mid}) = {x_str}\n"
                 f"Double NOT returns to original.")
        rows.append(_make_row(
            [_user_msg(prompt), _asst_msg(think, x_str)],
            x_str, f"edge_comp_doublenot_{idx:04d}", "composition"
        ))
        idx += 1

    # --- xor(shr(X, k), shl(X, 8-k)) for rotate equivalence ---
    for _ in range(30):
        x = rng.randint(1, 254)
        k = rng.randint(1, 7)
        x_str = _fmt(x)
        a = shr(x, k)
        b = shl(x, 8 - k)
        result = gate_or(a, b)
        result_str = _fmt(result)
        rol_result = _fmt(ror(x, k))
        assert result_str == rol_result, f"or(shr{k}, shl{8-k}) should equal ror{k}"
        prompt = f"Compute or(shr{k}({x_str}), shl{8 - k}({x_str}))."
        a_str = _fmt(a)
        b_str = _fmt(b)
        think = (f"shr{k}({x_str}) = {a_str}\n"
                 f"shl{8 - k}({x_str}) = {b_str}\n"
                 f"or({a_str}, {b_str}) = {result_str}\n"
                 f"This equals ror{k}({x_str}) = {rol_result}. "
                 f"Rotate right can be built from shift right + shift left + OR.")
        rows.append(_make_row(
            [_user_msg(prompt), _asst_msg(think, result_str)],
            result_str, f"edge_comp_rot_equiv_{idx:04d}", "composition"
        ))
        idx += 1

    rng.shuffle(rows)
    return rows[:n]


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

CATEGORIES = {
    'zero_ones_boundary': (gen_zero_ones_boundary, 300),
    'shift_vs_rotate':    (gen_shift_vs_rotate,    400),
    'large_shift':        (gen_large_shift,        300),
    'gate_edge':          (gen_gate_edge,           400),
    'popcount':           (gen_popcount,            200),
    'composition_edge':   (gen_composition_edge,    400),
}


def main():
    parser = argparse.ArgumentParser(description='Generate bit manipulation edge case micro-skills')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--category', type=str, default=None,
                        choices=list(CATEGORIES.keys()),
                        help='Generate only one category')
    parser.add_argument('--n', type=int, default=None,
                        help='Override count for the category')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path')
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.category:
        gen_func, default_n = CATEGORIES[args.category]
        count = args.n or default_n
        rows = gen_func(rng, count)
    else:
        rows = []
        for cat_name, (gen_func, default_n) in CATEGORIES.items():
            count = default_n
            cat_rows = gen_func(rng, count)
            rows.extend(cat_rows)
            print(f"  {cat_name}: {len(cat_rows)} rows")

    # Final shuffle
    rng.shuffle(rows)

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'bit_manipulation', 'pool', 'generated',
        'microskill_edge_cases.jsonl'
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')

    print(f"\nWrote {len(rows)} edge case examples to {output_path}")

    # Category breakdown
    cats = {}
    for row in rows:
        mode = row['mode']
        cats[mode] = cats.get(mode, 0) + 1
    for mode, count in sorted(cats.items()):
        print(f"  {mode}: {count}")


if __name__ == '__main__':
    main()
