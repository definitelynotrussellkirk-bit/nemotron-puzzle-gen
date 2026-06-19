#!/usr/bin/env python3
"""Regenerate bit manipulation corpora with no-jump step-by-step traces.

Two modes:
  --mode generated    Generate N synthetic puzzles and trace them (default)
  --mode competition  Trace all bit manipulation rows from train.csv (no-jump format)
  --mode both         Both of the above

The no-jump format shows every shift as a string operation and every gate
position-by-position, so the model never has to "teleport" to an answer.

Usage:
    python3 -m generators.regen_bit --mode generated --n 8000
    python3 -m generators.regen_bit --mode competition
    python3 -m generators.regen_bit --mode both --n 8000
"""
import argparse
import csv
import json
import time
from datetime import datetime, timezone

from generators.archive.gen_bit_microskills import (
    _fmt, _trace_shr, _trace_shl, _trace_rol, _trace_ror, _trace_not,
    _trace_gate, _trace_family_steps, _trace_one_shift, _trace_source_step,
    BYTE, shl, shr, rol, ror, bit_not,
    gate_xnor, gate_xor, gate_and, gate_or, gate_nand, gate_nor,
    gate_and_not, gate_or_not,
)
from generators.bit_manipulation import BitManipulationGenerator
from solvers.bit_manipulation import (
    solve_details, _parse_prompt_family_label, _find_transform_fn,
    _BYTE_MASK, _GATE2_FNS, _GATE3_FNS,
)
from solvers.bit_3stream import _build_sources
from training.data import BOXED_INSTRUCTION


def _diff_verdict(computed, expected, lines, show_result=True, prefix="  "):
    """Append diff-based verdict lines. Returns True if PASS."""
    diff = ''.join('0' if a == b else '1' for a, b in zip(computed, expected))
    if show_result:
        lines.append(f"{prefix}= {computed}")
    if diff == '00000000':
        lines.append(f"{prefix}diff={diff} → PASS")
        return True
    else:
        lines.append(f"{prefix}expected={expected} diff={diff} → FAIL")
        return False


def _git_short_hash():
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ── Source name mapping ─────────────────────────────────────────────
# prompt_family uses "rotl(5)", "shl(3)", "shr(2)"
# 3stream/nojump uses "shl3", "shr2", "rol5", "ror3", "x"

def _pf_label_to_nojump_src(label):
    """Convert prompt_family transform label to (src_name, src_type, src_k).

    'rotl(5)' -> ('rol5', 'rol', 5)
    'shl(3)' -> ('shl3', 'shl', 3)
    'shr(2)' -> ('shr2', 'shr', 2)
    'rotl(0)' -> ('x', 'x', 0)  (identity)
    """
    import re
    m = re.match(r'^(rotl|rotr|shl|shr)\((\d+)\)$', label)
    if not m:
        return None
    op = m.group(1)
    k = int(m.group(2))
    if op == 'rotl':
        if k == 0:
            return ('x', 'x', 0)
        return (f'rol{k}', 'rol', k)
    elif op == 'rotr':
        if k == 0:
            return ('x', 'x', 0)
        return (f'ror{k}', 'ror', k)
    elif op == 'shl':
        return (f'shl{k}', 'shl', k)
    elif op == 'shr':
        return (f'shr{k}', 'shr', k)
    return None


def _3stream_src_to_nojump(src_name):
    """Convert 3stream source name to (src_name, src_type, src_k, is_complement).

    'shl3' -> ('shl3', 'shl', 3, False)
    '~shr1' -> ('shr1', 'shr', 1, True)
    'x' -> ('x', 'x', 0, False)
    '~x' -> ('x', 'x', 0, True)
    """
    import re
    complement = src_name.startswith('~')
    base = src_name.lstrip('~')

    if base == 'x':
        return ('x', 'x', 0, complement)

    m = re.match(r'^(shl|shr|rol|ror)(\d+)$', base)
    if m:
        return (base, m.group(1), int(m.group(2)), complement)
    return None


def _apply_source_int(src_type, src_k, x_int):
    """Apply a source transform to integer x."""
    if src_type == 'x':
        return x_int
    elif src_type == 'shl':
        return shl(x_int, src_k)
    elif src_type == 'shr':
        return shr(x_int, src_k)
    elif src_type == 'rol':
        return rol(x_int, src_k)
    elif src_type == 'ror':
        return ror(x_int, src_k)
    raise ValueError(f"Unknown src_type: {src_type}")


# ── Family mapping ──────────────────────────────────────────────────
# Map solver family names to _trace_family_steps formula keys

_FAMILY_TO_FORMULA = {
    'OR_XNOR': 'C | xnor(A,B)',
    'GATED_XNOR_NAND': 'where C=0: xnor(A,B); where C=1: nand(A,B)',
    'CH': 'where A=1: B; where A=0: C',
    'MAJ3': 'P=A&B, Q=A&C, R=B&C, output=P|Q|R',
    'TT121': 'where A=0: xnor(B,C); where A=1: nand(B,C)',
    'T1': 'where A=0: (~B)|C; where A=1: B^C',
}

# Map SIMPLE gate names to _trace_gate names (only truly simple gates)
# XNOR/NAND/NOR use decomposition (XOR+NOT, AND+NOT, OR+NOT)
_GATE2_TO_TRACE = {
    'XOR': 'xor',
    'AND': 'and',
    'OR': 'or',
}


# ── Trace builders ──────────────────────────────────────────────────

def _trace_source_with_complement(src_name, src_type, src_k, is_complement, x_str):
    """Trace a source computation, possibly with complement.

    Returns (trace_lines, result_str).
    """
    if is_complement and src_type == 'x':
        _, not_result = _trace_not(x_str)
        return [f"  not: flip each bit -> {not_result}"], not_result
    elif is_complement:
        # First apply the shift, then NOT
        shift_lines, shift_result = _trace_source_step(src_name, src_type, src_k, x_str)
        _, not_result = _trace_not(shift_result)
        shift_lines.append(f"  not: flip each bit -> {not_result}")
        return shift_lines, not_result
    else:
        return _trace_source_step(src_name, src_type, src_k, x_str)


def _build_nojump_trace_3input(sources_info, formula_key, examples, query_str, query_int):
    """Build no-jump trace for a 3-input family.

    sources_info: list of (src_name, src_type, src_k) for A, B, C
    formula_key: key into _trace_family_steps
    examples: [(inp_str, out_str), ...]
    query_str: 8-bit binary string
    query_int: integer value of query
    """
    lines = ["Bit rule.", ""]

    # Popcount
    pc_in = bin(int(examples[0][0], 2)).count('1')
    pc_out = bin(int(examples[0][1], 2)).count('1')
    pc_delta = pc_out - pc_in
    pc_sign = f"+{pc_delta}" if pc_delta > 0 else str(pc_delta)
    lines.append(f"Ones: input {pc_in} ones, output {pc_out} ones ({pc_sign}).")
    lines.append("")

    # Rule declaration
    lines.append("Try[1]:")
    for i, (sname, stype, sk) in enumerate(sources_info):
        label = chr(65 + i)
        if sname == 'x':
            lines.append(f"  {label} = x")
        else:
            lines.append(f"  {label} = {sname}(x)")
    lines.append(f"  {formula_key}")

    # Check on 2 random examples (teaches model to verify against multiple)
    import random as _rng
    _check_indices = list(range(len(examples)))
    _rng.Random(hash(query_str)).shuffle(_check_indices)
    _check_indices = _check_indices[:2]
    for _ci, check_idx in enumerate(_check_indices):
        lines.append("")
        lines.append(f"Witness {_ci + 1}:")
        cx_str = examples[check_idx][0]
        cx_int = int(cx_str, 2)
        lines.append(f"  x = {cx_str}")

        check_vals = []
        for i, (sname, stype, sk) in enumerate(sources_info):
            label = chr(65 + i)
            src_lines, src_result = _trace_source_step(sname, stype, sk, cx_str)
            if sname == 'x':
                lines.append(f"  {label} = x = {cx_str}")
            else:
                lines.append(f"  {label} = {sname}({cx_str})")
                for sl in src_lines:
                    lines.append(f"  {sl}")
            check_vals.append(src_result)

        family_lines, check_result = _trace_family_steps(
            formula_key, check_vals[0], check_vals[1], check_vals[2])
        for fl in family_lines:
            lines.append(f"  {fl}")

        expected_check = examples[check_idx][1]
        _diff_verdict(check_result, expected_check, lines)

    # LOCK + Query
    lines.append("")
    lines.append("Decision[1]: LOCK")
    lines.append("")
    lines.append("Query (using LOCK Try[1]):")
    lines.append(f"  x = {query_str}")

    query_vals = []
    for i, (sname, stype, sk) in enumerate(sources_info):
        label = chr(65 + i)
        src_lines, src_result = _trace_source_step(sname, stype, sk, query_str)
        if sname == 'x':
            lines.append(f"  {label} = x = {query_str}")
        else:
            lines.append(f"  {label} = {sname}({query_str})")
            for sl in src_lines:
                lines.append(f"  {sl}")
        query_vals.append(src_result)

    family_lines_q, q_result = _trace_family_steps(
        formula_key, query_vals[0], query_vals[1], query_vals[2])
    for fl in family_lines_q:
        lines.append(f"  {fl}")
    lines.append(f"  = {q_result}")

    return '\n'.join(lines), q_result


def _build_nojump_trace_2input(sources_info, gate_trace_name, gate_fn,
                                examples, query_str, query_int,
                                complement_flags=None):
    """Build no-jump trace for a 2-input gate.

    sources_info: list of (src_name, src_type, src_k) for A, B
    gate_trace_name: name for _trace_gate (e.g. 'xor', 'and', 'or')
    gate_fn: function(a_int, b_int) -> int for validation
    complement_flags: list of bools for whether each source is complemented
    """
    if complement_flags is None:
        complement_flags = [False, False]

    lines = ["Bit rule.", ""]

    # Popcount
    pc_in = bin(int(examples[0][0], 2)).count('1')
    pc_out = bin(int(examples[0][1], 2)).count('1')
    pc_delta = pc_out - pc_in
    pc_sign = f"+{pc_delta}" if pc_delta > 0 else str(pc_delta)
    lines.append(f"Ones: input {pc_in} ones, output {pc_out} ones ({pc_sign}).")
    lines.append("")

    # Rule declaration
    lines.append("Try[1]:")
    src_labels = []
    for i, ((sname, stype, sk), comp) in enumerate(zip(sources_info, complement_flags)):
        label = chr(65 + i)
        if comp:
            if sname == 'x':
                lines.append(f"  {label} = ~x")
            else:
                lines.append(f"  {label} = ~{sname}(x)")
        else:
            if sname == 'x':
                lines.append(f"  {label} = x")
            else:
                lines.append(f"  {label} = {sname}(x)")

    # Gate symbol for display
    gate_syms = {'xor': '^', 'and': '&', 'or': '|', 'xnor': 'XNOR',
                 'nand': 'NAND', 'nor': 'NOR', 'and_not': 'AND_NOT', 'or_not': 'OR_NOT'}
    sym = gate_syms.get(gate_trace_name, gate_trace_name)
    if sym in ('^', '&', '|'):
        lines.append(f"  output = A {sym} B")
    else:
        lines.append(f"  output = {sym}(A, B)")

    # Check on 2 random examples
    import random as _rng
    _check_indices = list(range(len(examples)))
    _rng.Random(hash(query_str)).shuffle(_check_indices)
    _check_indices = _check_indices[:2]
    for _ci, check_idx in enumerate(_check_indices):
        lines.append("")
        lines.append(f"Witness {_ci + 1}:")
        cx_str = examples[check_idx][0]
        cx_int = int(cx_str, 2)
        lines.append(f"  x = {cx_str}")

        check_vals = []
        for i, ((sname, stype, sk), comp) in enumerate(zip(sources_info, complement_flags)):
            label = chr(65 + i)
            src_lines, src_result = _trace_source_with_complement(sname, stype, sk, comp, cx_str)
            if comp:
                if sname == 'x':
                    lines.append(f"  {label} = ~x")
                else:
                    lines.append(f"  {label} = ~{sname}({cx_str})")
            else:
                if sname == 'x':
                    lines.append(f"  {label} = x = {cx_str}")
                else:
                    lines.append(f"  {label} = {sname}({cx_str})")
            for sl in src_lines:
                lines.append(f"  {sl}")
            check_vals.append(src_result)

        gate_lines, check_result = _trace_gate(gate_trace_name, check_vals[0], check_vals[1])
        if sym in ('^', '&', '|'):
            lines.append(f"  output = A {sym} B:")
        else:
            lines.append(f"  output = {sym}(A, B):")
        for gl in gate_lines[1:]:
            lines.append(f"  {gl}")

        expected_check = examples[check_idx][1]
        lines.append(f"  output = {check_result}")
        _diff_verdict(check_result, expected_check, lines, show_result=False)

    # LOCK + Query
    lines.append("")
    lines.append("Decision[1]: LOCK")
    lines.append("")
    lines.append("Query (using LOCK Try[1]):")
    lines.append(f"  x = {query_str}")

    query_vals = []
    for i, ((sname, stype, sk), comp) in enumerate(zip(sources_info, complement_flags)):
        label = chr(65 + i)
        src_lines, src_result = _trace_source_with_complement(sname, stype, sk, comp, query_str)
        if comp:
            if sname == 'x':
                lines.append(f"  {label} = ~x")
            else:
                lines.append(f"  {label} = ~{sname}({query_str})")
        else:
            if sname == 'x':
                lines.append(f"  {label} = x = {query_str}")
            else:
                lines.append(f"  {label} = {sname}({query_str})")
        for sl in src_lines:
            lines.append(f"  {sl}")
        query_vals.append(src_result)

    gate_lines_q, q_result = _trace_gate(gate_trace_name, query_vals[0], query_vals[1])
    if sym in ('^', '&', '|'):
        lines.append(f"  output = A {sym} B:")
    else:
        lines.append(f"  output = {sym}(A, B):")
    for gl in gate_lines_q[1:]:
        lines.append(f"  {gl}")
    lines.append(f"  output = {q_result}")

    return '\n'.join(lines), q_result


def _build_nojump_trace_single(src_info, is_not, examples, query_str, query_int):
    """Build no-jump trace for a single-transform rule (with optional NOT).

    src_info: (src_name, src_type, src_k)
    is_not: True if output = ~transform(x)
    """
    sname, stype, sk = src_info
    lines = ["Bit rule.", ""]

    # Popcount
    pc_in = bin(int(examples[0][0], 2)).count('1')
    pc_out = bin(int(examples[0][1], 2)).count('1')
    pc_delta = pc_out - pc_in
    pc_sign = f"+{pc_delta}" if pc_delta > 0 else str(pc_delta)
    lines.append(f"Ones: input {pc_in} ones, output {pc_out} ones ({pc_sign}).")
    lines.append("")

    # Rule declaration
    lines.append("Try[1]:")
    if sname == 'x':
        if is_not:
            lines.append("  output = ~x")
        else:
            lines.append("  output = x")
    else:
        if is_not:
            lines.append(f"  A = {sname}(x)")
            lines.append("  output = ~A")
        else:
            lines.append(f"  A = {sname}(x)")
            lines.append("  output = A")

    # Check on 2 random examples
    import random as _rng
    _check_indices = list(range(len(examples)))
    _rng.Random(hash(query_str)).shuffle(_check_indices)
    _check_indices = _check_indices[:2]
    for _ci, check_idx in enumerate(_check_indices):
        lines.append("")
        lines.append(f"Witness {_ci + 1}:")
        cx_str = examples[check_idx][0]
        lines.append(f"  x = {cx_str}")

        if sname == 'x':
            if is_not:
                _, not_result = _trace_not(cx_str)
                lines.append(f"  output = ~x:")
                lines.append(f"  not: flip each bit -> {not_result}")
                check_result = not_result
            else:
                check_result = cx_str
                lines.append(f"  output = {cx_str}")
        else:
            src_lines, src_result = _trace_source_step(sname, stype, sk, cx_str)
            lines.append(f"  A = {sname}({cx_str})")
            for sl in src_lines:
                lines.append(f"  {sl}")
            if is_not:
                _, not_result = _trace_not(src_result)
                lines.append(f"  output = ~A:")
                lines.append(f"  not: flip each bit -> {not_result}")
                check_result = not_result
            else:
                check_result = src_result
                lines.append(f"  output = {src_result}")

        expected_check = examples[check_idx][1]
        _diff_verdict(check_result, expected_check, lines, show_result=False)

    # LOCK + Query
    lines.append("")
    lines.append("Decision[1]: LOCK")
    lines.append("")
    lines.append("Query (using LOCK Try[1]):")
    lines.append(f"  x = {query_str}")

    if sname == 'x':
        if is_not:
            _, not_result = _trace_not(query_str)
            lines.append(f"  output = ~x:")
            lines.append(f"  not: flip each bit -> {not_result}")
            q_result = not_result
        else:
            q_result = query_str
            lines.append(f"  output = {query_str}")
    else:
        src_lines, src_result = _trace_source_step(sname, stype, sk, query_str)
        lines.append(f"  A = {sname}({query_str})")
        for sl in src_lines:
            lines.append(f"  {sl}")
        if is_not:
            _, not_result = _trace_not(src_result)
            lines.append(f"  output = ~A:")
            lines.append(f"  not: flip each bit -> {not_result}")
            q_result = not_result
        else:
            q_result = src_result
            lines.append(f"  output = {src_result}")

    return '\n'.join(lines), q_result


def _build_nojump_trace_residual(transform_label, residual_val, transform_fn,
                                  is_not, examples, query_str, query_int):
    """Build no-jump trace for a residual solver (transform XOR constant).

    transform_label: e.g. 'shl(3)' or 'NOT(rotl(5))'
    residual_val: integer XOR constant
    transform_fn: function(x_int) -> int
    is_not: True if transform is NOT(base)
    """
    lines = ["Bit rule.", ""]

    # Popcount
    pc_in = bin(int(examples[0][0], 2)).count('1')
    pc_out = bin(int(examples[0][1], 2)).count('1')
    pc_delta = pc_out - pc_in
    pc_sign = f"+{pc_delta}" if pc_delta > 0 else str(pc_delta)
    lines.append(f"Ones: input {pc_in} ones, output {pc_out} ones ({pc_sign}).")
    lines.append("")

    residual_str = _fmt(residual_val)

    # Parse the base transform
    base_label = transform_label
    if is_not and base_label.startswith("NOT(") and base_label.endswith(")"):
        base_label = base_label[4:-1]

    src_info = _pf_label_to_nojump_src(base_label)
    if src_info is None:
        # Fallback: just use the label
        lines.append("Try[1]:")
        lines.append(f"  T = {transform_label}(x)")
        lines.append(f"  output = T ^ {residual_str}")
        lines.append("")
        lines.append("Check:")
        cx_str = examples[0][0]
        cx_int = int(cx_str, 2)
        lines.append(f"  x = {cx_str}")
        t_val = ((~transform_fn(cx_int)) & _BYTE_MASK) if is_not else (transform_fn(cx_int) & _BYTE_MASK)
        xored = (t_val ^ residual_val) & _BYTE_MASK
        lines.append(f"  T = {_fmt(t_val)}")
        lines.append(f"  output = T ^ {residual_str} = {_fmt(xored)}")
        expected = examples[0][1]
        _diff_verdict(_fmt(xored), expected, lines, show_result=False)
        lines.append("")
        lines.append("Decision[1]: LOCK")
        lines.append("")
        lines.append("Query (using LOCK Try[1]):")
        lines.append(f"  x = {query_str}")
        qt_val = ((~transform_fn(query_int)) & _BYTE_MASK) if is_not else (transform_fn(query_int) & _BYTE_MASK)
        q_xored = (qt_val ^ residual_val) & _BYTE_MASK
        lines.append(f"  T = {_fmt(qt_val)}")
        lines.append(f"  output = T ^ {residual_str} = {_fmt(q_xored)}")
        return '\n'.join(lines), _fmt(q_xored)

    sname, stype, sk = src_info

    # Rule declaration
    lines.append("Try[1]:")
    if sname == 'x':
        if is_not:
            lines.append("  T = ~x")
        else:
            lines.append("  T = x")
    else:
        if is_not:
            lines.append(f"  T = ~{sname}(x)")
        else:
            lines.append(f"  T = {sname}(x)")
    lines.append(f"  output = T ^ {residual_str}")

    # Check
    lines.append("")
    lines.append("Check:")
    cx_str = examples[0][0]
    cx_int = int(cx_str, 2)
    lines.append(f"  x = {cx_str}")

    # Compute T step by step
    if sname == 'x':
        t_str = cx_str
        if is_not:
            _, t_str = _trace_not(cx_str)
            lines.append(f"  T = ~x:")
            lines.append(f"  not: flip each bit -> {t_str}")
        else:
            lines.append(f"  T = x = {cx_str}")
    else:
        src_lines, src_result = _trace_source_step(sname, stype, sk, cx_str)
        lines.append(f"  T = {sname}({cx_str})")
        for sl in src_lines:
            lines.append(f"  {sl}")
        if is_not:
            _, t_str = _trace_not(src_result)
            lines.append(f"  ~T: not: flip each bit -> {t_str}")
        else:
            t_str = src_result

    # XOR with residual
    gate_lines, check_result = _trace_gate('xor', t_str, residual_str)
    lines.append(f"  output = T ^ {residual_str}:")
    for gl in gate_lines[1:]:
        lines.append(f"  {gl}")

    expected_check = examples[0][1]
    lines.append(f"  output = {check_result}")
    _diff_verdict(check_result, expected_check, lines, show_result=False)

    # LOCK + Query
    lines.append("")
    lines.append("Decision[1]: LOCK")
    lines.append("")
    lines.append("Query (using LOCK Try[1]):")
    lines.append(f"  x = {query_str}")

    if sname == 'x':
        qt_str = query_str
        if is_not:
            _, qt_str = _trace_not(query_str)
            lines.append(f"  T = ~x:")
            lines.append(f"  not: flip each bit -> {qt_str}")
        else:
            lines.append(f"  T = x = {query_str}")
    else:
        src_lines, src_result = _trace_source_step(sname, stype, sk, query_str)
        lines.append(f"  T = {sname}({query_str})")
        for sl in src_lines:
            lines.append(f"  {sl}")
        if is_not:
            _, qt_str = _trace_not(src_result)
            lines.append(f"  ~T: not: flip each bit -> {qt_str}")
        else:
            qt_str = src_result

    # XOR with residual
    gate_lines_q, q_result = _trace_gate('xor', qt_str, residual_str)
    lines.append(f"  output = T ^ {residual_str}:")
    for gl in gate_lines_q[1:]:
        lines.append(f"  {gl}")
    lines.append(f"  output = {q_result}")

    return '\n'.join(lines), q_result


def _build_nojump_trace_2input_complex(gate_name, sources_info,
                                        complement_flags, gate_fn,
                                        examples, query_str, query_int):
    """Build no-jump trace for 2-input gates that need decomposition.

    Handles: XNOR (xnor), NAND (nand), NOR (nor),
             a_AND_NOTb, NOTa_AND_b, a_OR_NOTb, NOTa_OR_b.
    """
    lines = ["Bit rule.", ""]

    # Popcount
    pc_in = bin(int(examples[0][0], 2)).count('1')
    pc_out = bin(int(examples[0][1], 2)).count('1')
    pc_delta = pc_out - pc_in
    pc_sign = f"+{pc_delta}" if pc_delta > 0 else str(pc_delta)
    lines.append(f"Ones: input {pc_in} ones, output {pc_out} ones ({pc_sign}).")
    lines.append("")

    # Determine the decomposition steps
    # a_AND_NOTb: NOT B, then AND(A, ~B)
    # NOTa_AND_b: NOT A, then AND(~A, B)
    # a_OR_NOTb: NOT B, then OR(A, ~B)
    # NOTa_OR_b: NOT A, then OR(~A, B)
    # XNOR: XOR(A,B), then NOT result
    # NAND: AND(A,B), then NOT result
    # NOR: OR(A,B), then NOT result

    # Rule declaration
    lines.append("Try[1]:")
    for i, ((sname, stype, sk), comp) in enumerate(zip(sources_info, complement_flags)):
        label = chr(65 + i)
        if comp:
            if sname == 'x':
                lines.append(f"  {label} = ~x")
            else:
                lines.append(f"  {label} = ~{sname}(x)")
        else:
            if sname == 'x':
                lines.append(f"  {label} = x")
            else:
                lines.append(f"  {label} = {sname}(x)")

    # Formula description
    formula_map = {
        'XNOR': '~(A ^ B)',
        'NAND': '~(A & B)',
        'NOR': '~(A | B)',
        'a_AND_NOTb': 'A & ~B',
        'NOTa_AND_b': '~A & B',
        'a_OR_NOTb': 'A | ~B',
        'NOTa_OR_b': '~A | B',
    }
    formula = formula_map.get(gate_name, f"{gate_name}(A, B)")
    lines.append(f"  output = {formula}")

    def _compute_complex(a_str, b_str, gate_name):
        """Compute complex gate step by step, return (trace_lines, result_str)."""
        step_lines = []
        if gate_name == 'XNOR':
            gl, xor_result = _trace_gate('xor', a_str, b_str)
            step_lines.append(f"  P = A ^ B:")
            for g in gl[1:]:
                step_lines.append(f"  {g}")
            step_lines.append(f"  P = {xor_result}")
            _, result = _trace_not(xor_result)
            step_lines.append(f"  output = ~P:")
            step_lines.append(f"  not: flip each bit -> {result}")
            return step_lines, result
        elif gate_name == 'NAND':
            gl, and_result = _trace_gate('and', a_str, b_str)
            step_lines.append(f"  P = A & B:")
            for g in gl[1:]:
                step_lines.append(f"  {g}")
            step_lines.append(f"  P = {and_result}")
            _, result = _trace_not(and_result)
            step_lines.append(f"  output = ~P:")
            step_lines.append(f"  not: flip each bit -> {result}")
            return step_lines, result
        elif gate_name == 'NOR':
            gl, or_result = _trace_gate('or', a_str, b_str)
            step_lines.append(f"  P = A | B:")
            for g in gl[1:]:
                step_lines.append(f"  {g}")
            step_lines.append(f"  P = {or_result}")
            _, result = _trace_not(or_result)
            step_lines.append(f"  output = ~P:")
            step_lines.append(f"  not: flip each bit -> {result}")
            return step_lines, result
        elif gate_name == 'a_AND_NOTb':
            _, nb = _trace_not(b_str)
            step_lines.append(f"  ~B: not: flip each bit -> {nb}")
            gl, result = _trace_gate('and', a_str, nb)
            step_lines.append(f"  output = A & ~B:")
            for g in gl[1:]:
                step_lines.append(f"  {g}")
            return step_lines, result
        elif gate_name == 'NOTa_AND_b':
            _, na = _trace_not(a_str)
            step_lines.append(f"  ~A: not: flip each bit -> {na}")
            gl, result = _trace_gate('and', na, b_str)
            step_lines.append(f"  output = ~A & B:")
            for g in gl[1:]:
                step_lines.append(f"  {g}")
            return step_lines, result
        elif gate_name == 'a_OR_NOTb':
            _, nb = _trace_not(b_str)
            step_lines.append(f"  ~B: not: flip each bit -> {nb}")
            gl, result = _trace_gate('or', a_str, nb)
            step_lines.append(f"  output = A | ~B:")
            for g in gl[1:]:
                step_lines.append(f"  {g}")
            return step_lines, result
        elif gate_name == 'NOTa_OR_b':
            _, na = _trace_not(a_str)
            step_lines.append(f"  ~A: not: flip each bit -> {na}")
            gl, result = _trace_gate('or', na, b_str)
            step_lines.append(f"  output = ~A | B:")
            for g in gl[1:]:
                step_lines.append(f"  {g}")
            return step_lines, result
        else:
            raise ValueError(f"Unknown complex gate: {gate_name}")

    # Check on 2 random examples
    import random as _rng
    _check_indices = list(range(len(examples)))
    _rng.Random(hash(query_str)).shuffle(_check_indices)
    _check_indices = _check_indices[:2]
    for _ci, check_idx in enumerate(_check_indices):
        lines.append("")
        lines.append(f"Witness {_ci + 1}:")
        cx_str = examples[check_idx][0]
        cx_int = int(cx_str, 2)
        lines.append(f"  x = {cx_str}")

        check_vals = []
        for i, ((sname, stype, sk), comp) in enumerate(zip(sources_info, complement_flags)):
            label = chr(65 + i)
            src_lines, src_result = _trace_source_with_complement(sname, stype, sk, comp, cx_str)
            if comp:
                if sname == 'x':
                    lines.append(f"  {label} = ~x")
                else:
                    lines.append(f"  {label} = ~{sname}({cx_str})")
            else:
                if sname == 'x':
                    lines.append(f"  {label} = x = {cx_str}")
                else:
                    lines.append(f"  {label} = {sname}({cx_str})")
            for sl in src_lines:
                lines.append(f"  {sl}")
            check_vals.append(src_result)

        complex_lines, check_result = _compute_complex(check_vals[0], check_vals[1], gate_name)
        lines.extend(complex_lines)

        expected_check = examples[check_idx][1]
        lines.append(f"  output = {check_result}")
        _diff_verdict(check_result, expected_check, lines, show_result=False)

    # LOCK + Query
    lines.append("")
    lines.append("Decision[1]: LOCK")
    lines.append("")
    lines.append("Query (using LOCK Try[1]):")
    lines.append(f"  x = {query_str}")

    query_vals = []
    for i, ((sname, stype, sk), comp) in enumerate(zip(sources_info, complement_flags)):
        label = chr(65 + i)
        src_lines, src_result = _trace_source_with_complement(sname, stype, sk, comp, query_str)
        if comp:
            if sname == 'x':
                lines.append(f"  {label} = ~x")
            else:
                lines.append(f"  {label} = ~{sname}({query_str})")
        else:
            if sname == 'x':
                lines.append(f"  {label} = x = {query_str}")
            else:
                lines.append(f"  {label} = {sname}({query_str})")
        for sl in src_lines:
            lines.append(f"  {sl}")
        query_vals.append(src_result)

    complex_lines_q, q_result = _compute_complex(query_vals[0], query_vals[1], gate_name)
    lines.extend(complex_lines_q)
    lines.append(f"  output = {q_result}")

    return '\n'.join(lines), q_result


def _build_nojump_trace_3input_not(gate_name, sources_info, gate3_fn,
                                    examples, query_str, query_int):
    """Build no-jump trace for NOT_CH or NOT_MAJ3.

    These are: output = ~CH(A,B,C) or output = ~MAJ3(A,B,C).
    We compute the base family step-by-step, then NOT the result.
    """
    # Determine base family
    base_gate = gate_name.replace("NOT_", "")  # "CH" or "MAJ3"
    formula_key = _FAMILY_TO_FORMULA.get(base_gate)
    if formula_key is None:
        return None, None

    lines = ["Bit rule.", ""]

    # Popcount
    pc_in = bin(int(examples[0][0], 2)).count('1')
    pc_out = bin(int(examples[0][1], 2)).count('1')
    pc_delta = pc_out - pc_in
    pc_sign = f"+{pc_delta}" if pc_delta > 0 else str(pc_delta)
    lines.append(f"Ones: input {pc_in} ones, output {pc_out} ones ({pc_sign}).")
    lines.append("")

    # Rule declaration
    lines.append("Try[1]:")
    for i, (sname, stype, sk) in enumerate(sources_info):
        label = chr(65 + i)
        if sname == 'x':
            lines.append(f"  {label} = x")
        else:
            lines.append(f"  {label} = {sname}(x)")
    lines.append(f"  P = {formula_key}")
    lines.append("  output = ~P")

    # Check on 2 random examples
    import random as _rng
    _check_indices = list(range(len(examples)))
    _rng.Random(hash(query_str)).shuffle(_check_indices)
    _check_indices = _check_indices[:2]
    for _ci, check_idx in enumerate(_check_indices):
        lines.append("")
        lines.append(f"Witness {_ci + 1}:")
        cx_str = examples[check_idx][0]
        cx_int = int(cx_str, 2)
        lines.append(f"  x = {cx_str}")

        check_vals = []
        for i, (sname, stype, sk) in enumerate(sources_info):
            label = chr(65 + i)
            src_lines, src_result = _trace_source_step(sname, stype, sk, cx_str)
            if sname == 'x':
                lines.append(f"  {label} = x = {cx_str}")
            else:
                lines.append(f"  {label} = {sname}({cx_str})")
                for sl in src_lines:
                    lines.append(f"  {sl}")
            check_vals.append(src_result)

        family_lines, base_result = _trace_family_steps(
            formula_key, check_vals[0], check_vals[1], check_vals[2])
        for fl in family_lines:
            lines.append(f"  {fl}")
        lines.append(f"  P = {base_result}")

        _, not_result = _trace_not(base_result)
        lines.append(f"  output = ~P:")
        lines.append(f"  not: flip each bit -> {not_result}")

        expected_check = examples[check_idx][1]
        _diff_verdict(not_result, expected_check, lines, show_result=False)

    # LOCK + Query
    lines.append("")
    lines.append("Decision[1]: LOCK")
    lines.append("")
    lines.append("Query (using LOCK Try[1]):")
    lines.append(f"  x = {query_str}")

    query_vals = []
    for i, (sname, stype, sk) in enumerate(sources_info):
        label = chr(65 + i)
        src_lines, src_result = _trace_source_step(sname, stype, sk, query_str)
        if sname == 'x':
            lines.append(f"  {label} = x = {query_str}")
        else:
            lines.append(f"  {label} = {sname}({query_str})")
            for sl in src_lines:
                lines.append(f"  {sl}")
        query_vals.append(src_result)

    family_lines_q, q_base = _trace_family_steps(
        formula_key, query_vals[0], query_vals[1], query_vals[2])
    for fl in family_lines_q:
        lines.append(f"  {fl}")
    lines.append(f"  P = {q_base}")

    _, q_result = _trace_not(q_base)
    lines.append(f"  output = ~P:")
    lines.append(f"  not: flip each bit -> {q_result}")

    return '\n'.join(lines), q_result


# ── Main trace dispatcher ──────────────────────────────────────────

def build_compact_competition_trace(prompt, expected_answer):
    """Build a compact trace for a competition bit manipulation row.

    Uses trace_compact format: Scan + GRID + bookend verification.
    Returns (reasoning, answer) or None if the row can't be traced.
    """
    from generators.trace_compact import build_trace_from_solver

    details = solve_details(prompt)
    if details is None:
        return None

    answer = details["answer"]
    if answer != expected_answer:
        return None

    query_str = details["query"]
    examples = details["examples"]
    solver = details.get("solver", "local_dsl")

    # Extract source names, complements, and formula from solver details
    src_names = None
    complements = None
    formula = None

    if solver == "3stream":
        meta = details.get("stream_meta", {})
        family = meta.get("family", "?")
        sources = meta.get("sources", [])
        perm = meta.get("perm")

        FAMILY_FORMULAS = {
            "OR_XNOR":          {"chain": [{"gate": "xnor", "inputs": ["A", "B"], "out": "P"},
                                            {"gate": "or", "inputs": ["C", "P"]}]},
            "GATED_XNOR_NAND":  {"family": "gated_xnor_nand", "inputs": ["A", "B", "C"]},
            "CH":               {"family": "ch", "inputs": ["A", "B", "C"]},
            "MAJ3":             {"family": "maj", "inputs": ["A", "B", "C"]},
            "TT121":            {"family": "tt121", "inputs": ["A", "B", "C"]},
            "T1":               {"family": "t1", "inputs": ["A", "B", "C"]},
            "AND":              {"gate": "and", "inputs": ["A", "B"]},
            "OR":               {"gate": "or", "inputs": ["A", "B"]},
            "XOR":              {"gate": "xor", "inputs": ["A", "B"]},
        }

        formula = FAMILY_FORMULAS.get(family)
        if formula is None:
            return _build_nojump_fallback(prompt, expected_answer)

        if family in ("AND", "OR", "XOR"):
            ordered = sources[:2]
        elif perm:
            ordered = [sources[perm[0]], sources[perm[1]], sources[perm[2]]]
        else:
            ordered = sources[:3]

        src_names = []
        complements = []
        for sn in ordered:
            parsed = _3stream_src_to_nojump(sn)
            if parsed is None:
                return _build_nojump_fallback(prompt, expected_answer)
            name, stype, sk, is_comp = parsed
            src_names.append(name)
            complements.append(is_comp)

    elif solver == "prompt_family":
        pf = details.get("prompt_family", {})
        label = pf.get("labels", [None])[0]
        if label is None:
            return _build_nojump_fallback(prompt, expected_answer)
        gate_name, transform_labels = _parse_prompt_family_label(label)

        src_names = []
        complements = []
        for tl in transform_labels:
            si = _pf_label_to_nojump_src(tl)
            if si is None:
                return _build_nojump_fallback(prompt, expected_answer)
            name, stype, sk = si
            src_names.append(name if name != 'x' else 'x')
            complements.append(False)

        # Map gate to formula
        GATE_MAP = {
            'XOR': {"gate": "xor", "inputs": ["A", "B"]},
            'AND': {"gate": "and", "inputs": ["A", "B"]},
            'OR': {"gate": "or", "inputs": ["A", "B"]},
            'XNOR': {"gate": "xnor", "inputs": ["A", "B"]},
            'NAND': {"gate": "nand", "inputs": ["A", "B"]},
            'NOR': {"gate": "nor", "inputs": ["A", "B"]},
            'a_AND_NOTb': {"gate": "and", "inputs": ["A", "B"], "_complement_b": True},
            'NOTa_AND_b': {"gate": "and", "inputs": ["A", "B"], "_complement_a": True},
            'a_OR_NOTb': {"gate": "or", "inputs": ["A", "B"], "_complement_b": True},
            'NOTa_OR_b': {"gate": "or", "inputs": ["A", "B"], "_complement_a": True},
        }
        FAMILY_MAP = {
            'OR_XNOR': {"chain": [{"gate": "xnor", "inputs": ["A", "B"], "out": "P"},
                                   {"gate": "or", "inputs": ["C", "P"]}]},
            'GATED_XNOR_NAND': {"family": "gated_xnor_nand", "inputs": ["A", "B", "C"]},
            'CH': {"family": "ch", "inputs": ["A", "B", "C"]},
            'MAJ3': {"family": "maj", "inputs": ["A", "B", "C"]},
            'TT121': {"family": "tt121", "inputs": ["A", "B", "C"]},
            'T1': {"family": "t1", "inputs": ["A", "B", "C"]},
            'NOT_CH': {"not_of": {"family": "ch", "inputs": ["A", "B", "C"]}},
            'NOT_MAJ3': {"not_of": {"family": "maj", "inputs": ["A", "B", "C"]}},
        }

        if len(transform_labels) == 1:
            # 1-source: T(x) or ~T(x)
            # Build compact trace without GRID — just Scan + shift + bookend
            from generators.trace_compact import build_scan, scan_line, shift_line as _cshift, ones as _ones, fmt as _cfmt, apply_shift as _cshift_fn
            trace_lines = []
            trace_lines.extend(build_scan(examples))
            trace_lines.append("")
            if gate_name == 'NOT':
                trace_lines.append("Try[1]:")
                trace_lines.append(f"  A = {src_names[0]}(x)")
                trace_lines.append(f"  output = not(A)")
            else:
                trace_lines.append("Try[1]:")
                trace_lines.append(f"  output = {src_names[0]}(x)")
            trace_lines.append("")

            import random as _rng
            ci_list = list(range(len(examples)))
            _rng.Random(hash(query_str)).shuffle(ci_list)
            for ci in range(min(2, len(examples))):
                idx = ci_list[ci]
                inp, expected = examples[idx]
                x_int = int(inp, 2)
                shift_result = _cfmt(_cshift_fn(x_int, src_names[0]))
                if gate_name == 'NOT':
                    final = ''.join('1' if c == '0' else '0' for c in shift_result)
                else:
                    final = shift_result
                out_ones = _ones(final)
                _d = ''.join('0' if a==b else '1' for a,b in zip(final, expected))
                _v = "PASS" if _d == '00000000' else "FAIL"
                trace_lines.append(f"Witness {ci+1}: x={inp}")
                trace_lines.append(f"A={src_names[0]}({inp})={shift_result}")
                if gate_name == 'NOT':
                    trace_lines.append(f"output=not({shift_result})={final}")
                else:
                    trace_lines.append(f"output={final}")
                if _v == "PASS":
                    trace_lines.append(f"  diff={_d} → PASS")
                else:
                    trace_lines.append(f"  expected={expected} diff={_d} → FAIL")
                trace_lines.append("")

            q_ones = _ones(query_str)
            x_int = int(query_str, 2)
            shift_result = _cfmt(_cshift_fn(x_int, src_names[0]))
            if gate_name == 'NOT':
                final = ''.join('1' if c == '0' else '0' for c in shift_result)
            else:
                final = shift_result
            a_ones = _ones(final)
            delta = a_ones - q_ones
            match_n = sum(1 for a, b in zip(final, query_str) if a == b)
            delta_s = f"+{delta}" if delta >= 0 else str(delta)
            trace_lines.append(f"Query: x={query_str} ones={q_ones}")
            trace_lines.append(f"A={src_names[0]}({query_str})={shift_result}")
            if gate_name == 'NOT':
                trace_lines.append(f"output=not({shift_result})={final} ones={a_ones} delta={delta_s} match={match_n}/8")
            else:
                trace_lines.append(f"output={final} ones={a_ones} delta={delta_s} match={match_n}/8")

            reasoning = '\n'.join(trace_lines)
            if final != answer:
                return _build_nojump_fallback(prompt, expected_answer)
            return reasoning, final

        elif len(transform_labels) == 2:
            formula = GATE_MAP.get(gate_name)
            if formula and formula.get("_complement_a"):
                complements[0] = True
            if formula and formula.get("_complement_b"):
                complements[1] = True
            # Clean formula of internal flags
            if formula:
                formula = {k: v for k, v in formula.items() if not k.startswith("_")}
        elif len(transform_labels) == 3:
            formula = FAMILY_MAP.get(gate_name)

        if formula is None:
            return _build_nojump_fallback(prompt, expected_answer)

    elif solver == "residual":
        # T(x) XOR constant — compact format
        from generators.trace_compact import build_scan, ones as _ones, fmt as _cfmt, apply_shift as _cshift_fn
        transform_label = details.get("residual_transform", "?")
        residual_val = details.get("residual_value", 0)
        residual_str = format(residual_val, '08b')

        # Parse transform name
        import re as _re
        m = _re.match(r'(shl|shr|rotl|rotr)\((\d+)\)', transform_label)
        if m:
            op_map = {'shl': 'shl', 'shr': 'shr', 'rotl': 'rol', 'rotr': 'ror'}
            src_name = f"{op_map[m.group(1)]}{m.group(2)}"
        else:
            return _build_nojump_fallback(prompt, expected_answer)

        trace_lines = []
        trace_lines.extend(build_scan(examples))
        trace_lines.append("")
        trace_lines.append("Try[1]:")
        trace_lines.append(f"  A = {src_name}(x)")
        trace_lines.append(f"  output = A xor {residual_str}")
        trace_lines.append("")

        import random as _rng
        ci_list = list(range(len(examples)))
        _rng.Random(hash(query_str)).shuffle(ci_list)
        for ci in range(min(2, len(examples))):
            idx = ci_list[ci]
            inp, expected = examples[idx]
            x_int = int(inp, 2)
            shift_result = _cfmt(_cshift_fn(x_int, src_name))
            final = format(int(shift_result, 2) ^ residual_val, '08b')
            out_ones = _ones(final)
            _d = ''.join('0' if a==b else '1' for a,b in zip(final, expected))
            _v = "PASS" if _d == '00000000' else "FAIL"
            trace_lines.append(f"Witness {ci+1}: x={inp}")
            trace_lines.append(f"A={src_name}({inp})={shift_result}")
            trace_lines.append(f"output=A xor {residual_str}={final}")
            if _v == "PASS":
                trace_lines.append(f"  diff={_d} → PASS")
            else:
                trace_lines.append(f"  expected={expected} diff={_d} → FAIL")
            trace_lines.append("")

        q_ones = _ones(query_str)
        shift_result = _cfmt(_cshift_fn(int(query_str, 2), src_name))
        final = format(int(shift_result, 2) ^ residual_val, '08b')
        a_ones = _ones(final)
        delta = a_ones - q_ones
        match_n = sum(1 for a, b in zip(final, query_str) if a == b)
        delta_s = f"+{delta}" if delta >= 0 else str(delta)
        trace_lines.append(f"Query: x={query_str} ones={q_ones}")
        trace_lines.append(f"A={src_name}({query_str})={shift_result}")
        trace_lines.append(f"output=A xor {residual_str}={final} ones={a_ones} delta={delta_s} match={match_n}/8")

        reasoning = '\n'.join(trace_lines)
        if final != answer:
            return _build_nojump_fallback(prompt, expected_answer)
        return reasoning, final

    else:
        return _build_nojump_fallback(prompt, expected_answer)

    if src_names is None or formula is None:
        return _build_nojump_fallback(prompt, expected_answer)

    trace, result = build_trace_from_solver(
        src_names, complements, formula, examples, query_str,
        seed=hash(query_str))

    if result != answer:
        return _build_nojump_fallback(prompt, expected_answer)

    return trace, result


def _build_nojump_fallback(prompt, expected_answer):
    """Fallback to old nojump trace when compact can't handle the case."""
    return build_nojump_competition_trace_old(prompt, expected_answer)


def build_nojump_competition_trace_old(prompt, expected_answer):
    """OLD format trace builder — kept as fallback.

    Returns (reasoning, answer) or None if the row can't be traced.
    """
    details = solve_details(prompt)
    if details is None:
        return None

    answer = details["answer"]
    if answer != expected_answer:
        return None

    query_str = details["query"]
    query_int = int(query_str, 2)
    examples = details["examples"]
    solver = details.get("solver", "local_dsl")

    if solver == "prompt_family":
        pf = details["prompt_family"]
        label = pf["labels"][0] if pf["labels"] else "?"
        gate_name, transform_labels = _parse_prompt_family_label(label)

        # Convert transform labels to nojump source info
        sources_info = []
        for tl in transform_labels:
            si = _pf_label_to_nojump_src(tl)
            if si is None:
                return None  # Can't parse transform
            sources_info.append(si)

        n_transforms = len(transform_labels)

        if gate_name is None and n_transforms == 1:
            # Single transform: output = transform(x)
            reasoning, result = _build_nojump_trace_single(
                sources_info[0], False, examples, query_str, query_int)
            if result != answer:
                return None
            return reasoning, result

        elif gate_name == 'NOT' and n_transforms == 1:
            # NOT single: output = ~transform(x)
            reasoning, result = _build_nojump_trace_single(
                sources_info[0], True, examples, query_str, query_int)
            if result != answer:
                return None
            return reasoning, result

        elif n_transforms == 2:
            # 2-input gate
            simple_gates = _GATE2_TO_TRACE
            if gate_name in simple_gates:
                # Simple 2-input: XOR, AND, OR -> use _trace_gate directly
                reasoning, result = _build_nojump_trace_2input(
                    sources_info, simple_gates[gate_name], None,
                    examples, query_str, query_int)
                if result != answer:
                    return None
                return reasoning, result
            elif gate_name in ('XNOR', 'NAND', 'NOR',
                              'a_AND_NOTb', 'NOTa_AND_b',
                              'a_OR_NOTb', 'NOTa_OR_b'):
                # Complex 2-input
                reasoning, result = _build_nojump_trace_2input_complex(
                    gate_name, sources_info, [False, False], None,
                    examples, query_str, query_int)
                if result != answer:
                    return None
                return reasoning, result
            else:
                return None  # Unknown gate

        elif n_transforms == 3:
            # 3-input family
            if gate_name in ('NOT_CH', 'NOT_MAJ3'):
                gate3_fn = _GATE3_FNS.get(gate_name)
                reasoning, result = _build_nojump_trace_3input_not(
                    gate_name, sources_info, gate3_fn,
                    examples, query_str, query_int)
                if reasoning is None or result != answer:
                    return None
                return reasoning, result
            elif gate_name in _FAMILY_TO_FORMULA:
                formula_key = _FAMILY_TO_FORMULA[gate_name]
                reasoning, result = _build_nojump_trace_3input(
                    sources_info, formula_key, examples, query_str, query_int)
                if result != answer:
                    return None
                return reasoning, result
            else:
                # Try to handle as 2-input with 3 transforms (first 2 matter)
                if gate_name in _GATE2_TO_TRACE:
                    reasoning, result = _build_nojump_trace_2input(
                        sources_info[:2], _GATE2_TO_TRACE[gate_name], None,
                        examples, query_str, query_int)
                    if result != answer:
                        return None
                    return reasoning, result
                elif gate_name in ('XNOR', 'NAND', 'NOR',
                                  'a_AND_NOTb', 'NOTa_AND_b',
                                  'a_OR_NOTb', 'NOTa_OR_b'):
                    reasoning, result = _build_nojump_trace_2input_complex(
                        gate_name, sources_info[:2], [False, False], None,
                        examples, query_str, query_int)
                    if result != answer:
                        return None
                    return reasoning, result
                return None

        return None  # Fallback

    elif solver == "3stream":
        meta = details.get("stream_meta", {})
        family = meta.get("family", "?")
        sources = meta.get("sources", [])
        perm = meta.get("perm")

        if family in ('AND', 'OR', 'XOR'):
            # 2-input from 3stream
            gate_trace = family.lower()
            s_infos = []
            comp_flags = []
            for sn in sources[:2]:
                parsed = _3stream_src_to_nojump(sn)
                if parsed is None:
                    return None
                name, stype, sk, is_comp = parsed
                s_infos.append((name, stype, sk))
                comp_flags.append(is_comp)

            if any(comp_flags):
                # Has complement sources - use complex gate approach
                # OR(~shr1, x) = ~shr1 | x = or_not in a sense, but with complement sources
                # We handle this by computing complement sources step-by-step
                reasoning, result = _build_nojump_trace_2input(
                    s_infos, gate_trace, None,
                    examples, query_str, query_int,
                    complement_flags=comp_flags)
            else:
                reasoning, result = _build_nojump_trace_2input(
                    s_infos, gate_trace, None,
                    examples, query_str, query_int)

            if result != answer:
                return None
            return reasoning, result

        elif family in _FAMILY_TO_FORMULA:
            formula_key = _FAMILY_TO_FORMULA[family]

            # Convert sources with permutation
            if perm is None:
                ordered_src_names = sources[:3]
            else:
                ordered_src_names = [sources[perm[0]], sources[perm[1]], sources[perm[2]]]

            s_infos = []
            has_complement = False
            for sn in ordered_src_names:
                parsed = _3stream_src_to_nojump(sn)
                if parsed is None:
                    return None
                name, stype, sk, is_comp = parsed
                if is_comp:
                    has_complement = True
                s_infos.append((name, stype, sk))

            if has_complement:
                # 3-input with complement sources - not supported by _trace_family_steps
                return None

            reasoning, result = _build_nojump_trace_3input(
                s_infos, formula_key, examples, query_str, query_int)

            if result != answer:
                return None
            return reasoning, result

        else:
            return None  # Unknown family

    elif solver == "residual":
        transform_label = details.get("residual_transform", "?")
        residual_val = details.get("residual_value", 0)

        # Find the transform function
        is_not = "NOT(" in transform_label
        base_label = transform_label
        if is_not and base_label.startswith("NOT(") and base_label.endswith(")"):
            base_label = base_label[4:-1]

        tfn = None
        from solvers.bit_manipulation import _PROMPT_TRANSFORMS
        for plabel, fn in _PROMPT_TRANSFORMS:
            if plabel == base_label:
                tfn = fn
                break

        if tfn is None:
            return None

        reasoning, result = _build_nojump_trace_residual(
            transform_label, residual_val, tfn, is_not,
            examples, query_str, query_int)

        if result != answer:
            return None
        return reasoning, result

    else:
        # whole_byte or local_dsl - per-bit fallback, skip (or handle later)
        return None


# ── CLI entry points ────────────────────────────────────────────────

def regen_generated(n: int, output: str, seed: int):
    """Generate n synthetic bit manipulation puzzles with procedural traces."""
    from solvers.bit_manipulation import trace as bm_trace
    gen = BitManipulationGenerator(seed=seed)
    now = datetime.now(timezone.utc).isoformat()
    git_hash = _git_short_hash()
    count = 0
    skipped = 0
    t0 = time.time()

    with open(output, "w") as f:
        for i in range(n * 3):
            if count >= n:
                break

            prompt, answer = gen.generate_one()
            result = bm_trace(prompt)

            if result is None:
                skipped += 1
                continue

            reasoning, traced_answer = result
            if traced_answer != answer:
                skipped += 1
                continue

            content = f"<think>\n{reasoning}\n</think>\n\\boxed{{{traced_answer}}}"

            example = {
                "messages": [
                    {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": content},
                ],
                "answer": traced_answer,
                "id": f"gen_bit_{count:06d}",
                "puzzle_type": "bit_manipulation",
                "mode": "regular",
                "trace_quality": "full",
                "generator": "regen_bit",
                "generated_at": now,
                "data_version": git_hash,
            }
            f.write(json.dumps(example) + "\n")
            count += 1

            if count % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  {count}/{n} ({elapsed:.0f}s, {skipped} skipped)")

    elapsed = time.time() - t0
    print(f"Generated: {count} examples -> {output} ({elapsed:.0f}s, {skipped} skipped)")
    return count


def regen_competition(train_csv: str, output: str):
    """Trace train.csv bit rows with the first-witness-gold BIT_PROGRAM_V1 policy."""
    from training.bit_consensus import repair_row

    now = datetime.now(timezone.utc).isoformat()
    git_hash = _git_short_hash()
    count = 0
    skipped = 0
    t0 = time.time()

    with open(train_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        rows = [(r[0], r[1], r[2]) for r in reader
                if "bit manipulation" in r[1][:100]]

    print(f"  Found {len(rows)} bit manipulation rows in {train_csv}")

    with open(output, "w") as f:
        for row_id, prompt, expected_answer in rows:
            repaired = repair_row(
                {
                    "id": row_id,
                    "prompt": prompt,
                    "answer": expected_answer,
                    "puzzle_type": "bit_manipulation",
                },
                allow_first_witness_gold=True,
                trace_format="v1",
            )
            if repaired is None:
                skipped += 1
                continue
            content = repaired["messages"][1]["content"]

            example = {
                "messages": [
                    {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": content},
                ],
                "answer": expected_answer,
                "id": f"comp_bit_{row_id}",
                "puzzle_type": "bit_manipulation",
                "mode": "competition_traced",
                "trace_quality": "full",
                "generator": "bit_first_witness_v1",
                "derivability_status": repaired.get("derivability_status"),
                "repair_meta": repaired.get("repair_meta"),
                "generated_at": now,
                "data_version": git_hash,
            }
            f.write(json.dumps(example) + "\n")
            count += 1

    elapsed = time.time() - t0
    print(f"Competition first-witness BIT: {count} examples -> {output} ({elapsed:.0f}s, {skipped} skipped)")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generated", "competition", "both"], default="both")
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--output-generated", type=str,
                        default="data/bit_manipulation/regular.jsonl")
    parser.add_argument("--output-competition", type=str,
                        default="data/bit_manipulation/pool/competition/competition_traced.jsonl")
    parser.add_argument("--train-csv", type=str, default="data/competition/train.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode in ("generated", "both"):
        print(f"\n=== Regenerating generated bit manipulation data ({args.n} examples) ===")
        regen_generated(args.n, args.output_generated, args.seed)

    if args.mode in ("competition", "both"):
        print(f"\n=== Regenerating competition bit manipulation traces (no-jump format) ===")
        regen_competition(args.train_csv, args.output_competition)


if __name__ == "__main__":
    main()
