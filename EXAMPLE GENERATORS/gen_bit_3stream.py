#!/usr/bin/env python3
"""Generate bit manipulation puzzles using 3-stream families.

Uses compact trace format (trace_compact.py):
  Scan preamble → Rule → Check 1 → Check 2 → Query with bookend verification.
  GRID() markers for position-by-position gate computation.
    P=xnor(A,B)=00010001
    output=C | P=11110001 → MATCH

  Query:
    x=10111110
    A=11000000
    B=00101111
    C=11110101
    P=xnor(A,B)=11110101
    output=C | P=11110101
  </think>
  \\boxed{11110101}

When fingerprint_rate > 0, a fraction of traces include 2-3 "Scan:" lines
before the Use: section, showing HOW sources were identified from examples.
This teaches the model to derive source assignments rather than accept them as given.

Usage:
    python3 -m generators.gen_bit_3stream --n 20000
    python3 -m generators.gen_bit_3stream --n 20000 --fingerprint-rate 0.2
"""
import argparse
import json
import random
import time
from datetime import datetime, timezone

from training.data import BOXED_INSTRUCTION

BYTE = 0xFF
FINGERPRINT_INPUTS = [
    0x00, 0xFF,
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    0xAA, 0x55, 0x0F, 0xF0, 0x33, 0xCC,
]


def _append_unique(dst, seen, values, limit):
    for x in values:
        if len(dst) >= limit:
            break
        if x in seen:
            continue
        dst.append(x)
        seen.add(x)


def _fmt(v):
    return format(v & BYTE, '08b')


def rol(x, k):
    return ((x << k) | (x >> (8 - k))) & BYTE


def ror(x, k):
    return ((x >> k) | (x << (8 - k))) & BYTE


# Boolean templates (internal only — never exposed in traces)
def f1(a, b, c): return (c | (a & b) | (~a & ~b)) & BYTE      # OR_XNOR
def f2(a, b, c): return f1(a, b, c) & (~(a & b & c)) & BYTE   # GATED_XNOR_NAND
def ch(a, b, c): return ((a & b) | ((~a) & c)) & BYTE
def maj(a, b, c): return ((a & b) | (a & c) | (b & c)) & BYTE
def tt121(a, b, c): return (((~a) & ((b & c) | ((~b) & (~c)))) | (a & (~(b & c)))) & BYTE
def t1(a, b, c): return (~(a ^ b ^ c) | ((~a) & (~b) & c)) & BYTE

# Level 3 composites (added after eval revealed they were missing from training)
def ao(a, b, c): return ((a & b) | c) & BYTE         # AND-then-OR
def oa(a, b, c): return ((a | b) & c) & BYTE         # OR-then-AND
def ax(a, b, c): return ((a & b) ^ c) & BYTE         # AND-then-XOR
def ox(a, b, c): return ((a | b) ^ c) & BYTE         # OR-then-XOR
def xa(a, b, c): return ((a ^ b) & c) & BYTE         # XOR-then-AND
def xo(a, b, c): return ((a ^ b) | c) & BYTE         # XOR-then-OR
def par3(a, b, c): return (a ^ b ^ c) & BYTE         # 3-input parity

# Flattened weights — old weights caused XNOR dominance (model picked XNOR 65% of wrong answers)
FAMILIES_3 = [
    ("OR_XNOR", f1, 10),
    ("GATED_XNOR_NAND", f2, 10),
    ("CH", ch, 10),
    ("MAJ3", maj, 10),
    ("TT121", tt121, 8),
    ("T1", t1, 8),
    # Level 3 composites (new — covers ~5% of competition per-bit functions)
    ("AO", ao, 6),     # AND-then-OR
    ("OA", oa, 6),     # OR-then-AND
    ("AX", ax, 6),     # AND-then-XOR
    ("OX", ox, 6),     # OR-then-XOR
    ("XA", xa, 6),     # XOR-then-AND
    ("XO", xo, 6),     # XOR-then-OR
    ("PAR3", par3, 6), # 3-input parity
]

FAMILIES_2 = [
    ("AND", lambda a, b: a & b, 4),
    ("OR", lambda a, b: (a | b) & BYTE, 4),
    ("XOR", lambda a, b: (a ^ b) & BYTE, 2),
]

PERMS_3 = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]

# Rival families for discriminative example selection.
# Each maps a family to its top-2 rivals (ordered by confusion likelihood).
FAMILY_RIVALS_3 = {
    "OR_XNOR": [("MAJ3", maj), ("CH", ch)],
    "GATED_XNOR_NAND": [("OR_XNOR", f1), ("CH", ch)],
    "CH": [("MAJ3", maj), ("OR_XNOR", f1)],
    "MAJ3": [("CH", ch), ("OR_XNOR", f1)],
    "TT121": [("CH", ch), ("MAJ3", maj)],
    "T1": [("CH", ch), ("OR_XNOR", f1)],
    # Composites: confuse with their component gates
    "AO": [("OA", oa), ("MAJ3", maj)],
    "OA": [("AO", ao), ("CH", ch)],
    "AX": [("OX", ox), ("PAR3", par3)],
    "OX": [("AX", ax), ("PAR3", par3)],
    "XA": [("XO", xo), ("AX", ax)],
    "XO": [("XA", xa), ("OX", ox)],
    "PAR3": [("AX", ax), ("OX", ox)],
}

FAMILY_RIVALS_2 = {
    "AND": [("OR", lambda a, b: (a | b) & BYTE), ("XOR", lambda a, b: (a ^ b) & BYTE)],
    "OR": [("AND", lambda a, b: a & b), ("XOR", lambda a, b: (a ^ b) & BYTE)],
    "XOR": [("AND", lambda a, b: a & b), ("OR", lambda a, b: (a | b) & BYTE)],
}


# === Trace templates per family ===
# Each returns: (use_lines, row_fn)
# use_lines: lines for the Use: section (no indent)
# row_fn(a, b, c) -> (computation_lines, output_val)
#   computation_lines show helper bytes + final output

def _trace_or_xnor():
    """C | xnor(A, B) — introduce P = xnor(A, B)"""
    use = [
        "  P = xnor(A, B)",
        "  output = C | P",
    ]
    def row(a, b, c):
        p = (~(a ^ b)) & BYTE
        out = (c | p) & BYTE
        return [
            f"  P=xnor(A,B)={_fmt(p)}",
            f"  output=C | P={_fmt(out)}",
        ], out
    return use, row

def _trace_gated_xnor_nand():
    """where C=0 take xnor(A,B), where C=1 take nand(A,B)"""
    use = [
        "  P = xnor(A, B)",
        "  Q = nand(A, B)",
        "  output = where C=0 take P, where C=1 take Q",
    ]
    def row(a, b, c):
        p = (~(a ^ b)) & BYTE
        q = (~(a & b)) & BYTE
        out = ((~c & BYTE) & p | c & q) & BYTE
        return [
            f"  P=xnor(A,B)={_fmt(p)}",
            f"  Q=nand(A,B)={_fmt(q)}",
            f"  output=select(C,P,Q)={_fmt(out)}",
        ], out
    return use, row

def _trace_ch():
    """where A=1 take B, where A=0 take C"""
    use = [
        "  output = where A=1 take B, where A=0 take C",
    ]
    def row(a, b, c):
        out = ((a & b) | ((~a & BYTE) & c)) & BYTE
        return [
            f"  output=select(A,B,C)={_fmt(out)}",
        ], out
    return use, row

def _trace_maj3():
    """P = A&B, Q = A&C, R = B&C, output = P|Q|R"""
    use = [
        "  P = A & B",
        "  Q = A & C",
        "  R = B & C",
        "  output = P | Q | R",
    ]
    def row(a, b, c):
        p = (a & b) & BYTE
        q = (a & c) & BYTE
        r = (b & c) & BYTE
        out = (p | q | r) & BYTE
        return [
            f"  P=A&B={_fmt(p)}",
            f"  Q=A&C={_fmt(q)}",
            f"  R=B&C={_fmt(r)}",
            f"  output=P|Q|R={_fmt(out)}",
        ], out
    return use, row

def _trace_tt121():
    """where A=0 take xnor(B,C), where A=1 take nand(B,C)"""
    use = [
        "  P = xnor(B, C)",
        "  Q = nand(B, C)",
        "  output = where A=0 take P, where A=1 take Q",
    ]
    def row(a, b, c):
        p = (~(b ^ c)) & BYTE
        q = (~(b & c)) & BYTE
        out = (((~a & BYTE) & p) | (a & q)) & BYTE
        return [
            f"  P=xnor(B,C)={_fmt(p)}",
            f"  Q=nand(B,C)={_fmt(q)}",
            f"  output=select(A,P,Q)={_fmt(out)}",
        ], out
    return use, row

def _trace_t1():
    """where A=0 take (~B)|C, where A=1 take B^C"""
    use = [
        "  P = (~B) | C",
        "  Q = B ^ C",
        "  output = where A=0 take P, where A=1 take Q",
    ]
    def row(a, b, c):
        p = ((~b & BYTE) | c) & BYTE
        q = (b ^ c) & BYTE
        out = (((~a & BYTE) & p) | (a & q)) & BYTE
        return [
            f"  P=(~B)|C={_fmt(p)}",
            f"  Q=B^C={_fmt(q)}",
            f"  output=select(A,P,Q)={_fmt(out)}",
        ], out
    return use, row

def _trace_2input(op_name, op_fn):
    """A op B"""
    ops = {"AND": "&", "OR": "|", "XOR": "^"}
    sym = ops.get(op_name, op_name)
    use = [f"  output = A {sym} B"]
    def row(a, b, _c):
        out = op_fn(a, b) & BYTE
        return [f"  output=A {sym} B={_fmt(out)}"], out
    return use, row


def _trace_composite(inner_op, inner_sym, outer_op, outer_sym, fn):
    """Generic trace for composite gates: outer(inner(A,B), C)."""
    use = [f"  P = A {inner_sym} B", f"  output = P {outer_sym} C"]
    def row(a, b, c):
        p = {"&": lambda x,y: x&y, "|": lambda x,y: (x|y)&BYTE, "^": lambda x,y: (x^y)&BYTE}[inner_sym](a, b)
        out = fn(a, b, c)
        return [f"  P=A{inner_sym}B={_fmt(p)}", f"  output=P{outer_sym}C={_fmt(out)}"], out
    return use, row


TRACE_BUILDERS = {
    "OR_XNOR": _trace_or_xnor,
    "GATED_XNOR_NAND": _trace_gated_xnor_nand,
    "CH": _trace_ch,
    "MAJ3": _trace_maj3,
    "TT121": _trace_tt121,
    "T1": _trace_t1,
    # Level 3 composites: op1(A,B) op2 C
    "AO": lambda: _trace_composite("AND", "&", "OR", "|", ao),
    "OA": lambda: _trace_composite("OR", "|", "AND", "&", oa),
    "AX": lambda: _trace_composite("AND", "&", "XOR", "^", ax),
    "OX": lambda: _trace_composite("OR", "|", "XOR", "^", ox),
    "XA": lambda: _trace_composite("XOR", "^", "AND", "&", xa),
    "XO": lambda: _trace_composite("XOR", "^", "OR", "|", xo),
    "PAR3": lambda: (["  output = A ^ B ^ C"],
                      lambda a, b, c: ([f"  output=A^B^C={_fmt(par3(a,b,c))}"], par3(a,b,c))),
}


def _make_sources(rng):
    """Pick 3 source transforms: one rotation + two from {x, shl, shr}."""
    rot_k = rng.randint(1, 7)
    rot_type = rng.choice(["rol", "ror"])
    rot_name = f"{rot_type}{rot_k}"
    rot_fn = (lambda x, k=rot_k: rol(x, k)) if rot_type == "rol" else (lambda x, k=rot_k: ror(x, k))

    shift_pool = [("x", lambda x: x)]
    for k in range(1, 8):
        shift_pool.append((f"shl{k}", lambda x, k=k: (x << k) & BYTE))
        shift_pool.append((f"shr{k}", lambda x, k=k: (x >> k) & BYTE))

    s1, s2 = rng.sample(shift_pool, 2)
    sources = [(rot_name, rot_fn), s1, s2]
    # 15% chance: complement one source (teaches ~source handling)
    if rng.random() < 0.15:
        idx = rng.randrange(len(sources))
        name, fn = sources[idx]
        sources[idx] = (f"~{name}", lambda x, fn=fn: (~fn(x)) & BYTE)
    rng.shuffle(sources)
    return sources


def _make_sources_2(rng):
    """Pick 2 source transforms for 2-input families."""
    pool = [("x", lambda x: x)]
    for k in range(1, 8):
        pool.append((f"shl{k}", lambda x, k=k: (x << k) & BYTE))
        pool.append((f"shr{k}", lambda x, k=k: (x >> k) & BYTE))
        pool.append((f"rol{k}", lambda x, k=k: rol(x, k)))
        pool.append((f"ror{k}", lambda x, k=k: ror(x, k)))
        pool.append((f"~shl{k}", lambda x, k=k: (~(x << k)) & BYTE))
        pool.append((f"~shr{k}", lambda x, k=k: (~(x >> k)) & BYTE))
    return rng.sample(pool, 2)


def _is_diagnostic(fam_name, a, b, c, out):
    """Check if a support row exposes the rule (both branches active, non-trivial)."""
    if fam_name in ("CH", "GATED_XNOR_NAND", "TT121", "T1"):
        # The selector should have both 0s and 1s
        if fam_name == "CH":
            selector = a
        elif fam_name in ("GATED_XNOR_NAND",):
            selector = c
        else:  # TT121, T1
            selector = a
        ones = bin(selector).count('1')
        if ones == 0 or ones == 8:
            return False
        # Output should not be identical to any single input
        if out == a or out == b or (c is not None and out == c):
            return False
        return True
    elif fam_name == "MAJ3":
        # Output should not equal any single input
        if out == a or out == b or out == c:
            return False
        return True
    elif fam_name == "OR_XNOR":
        # C should not be all 1s, and xnor(A,B) should not equal C
        xnor_ab = (~(a ^ b)) & BYTE
        if c == 0xFF:
            return False
        if xnor_ab == c:
            return False
        return True
    # 2-input: output should not be trivially one input
    if out == a or out == b:
        return False
    if out == 0 or out == 0xFF:
        return False
    return True


def _find_discriminative_indices(fam_name, fam_fn, use_2input, example_inputs, get_abc, compute):
    """Find example indices where the correct family disagrees with rivals.

    Returns a list of (index, rival_index) tuples, ordered so that the first
    entry disagrees with rival #0, the second with rival #1, etc.
    Falls back to diagnostic examples if no disagreement exists.
    """
    if use_2input:
        rivals = FAMILY_RIVALS_2.get(fam_name, [])
    else:
        rivals = FAMILY_RIVALS_3.get(fam_name, [])

    if not rivals:
        return []

    result = []
    used_indices = set()

    for rival_name, rival_fn in rivals:
        best_idx = None
        for idx in range(len(example_inputs)):
            if idx in used_indices:
                continue
            x = example_inputs[idx]
            a, b, c = get_abc(x)
            correct_out = compute(x)
            if use_2input:
                rival_out = rival_fn(a, b) & BYTE
            else:
                rival_out = rival_fn(a, b, c) & BYTE
            if correct_out != rival_out:
                best_idx = idx
                break
        if best_idx is not None:
            result.append(best_idx)
            used_indices.add(best_idx)

    return result


def generate_one(rng, contrastive=None, fingerprint_rate=0.0):
    """Generate one bit puzzle with 1B-optimal procedural trace.

    Args:
        rng: Random instance.
        contrastive: If True, generate multi-check trace (2 Check rows).
                     If False, generate standard trace (1 Check row).
                     If None (default), randomly choose with 50% probability.
        fingerprint_rate: Fraction of 3-input traces that include Scan: evidence
                         lines showing how sources were identified. 0.0 = never,
                         1.0 = always. Only applies to 3-input families.
    """
    # Choose family — 35% 2-input to balance gate distribution
    # (was 10%, causing 76% XNOR dominance in training data)
    use_2input = rng.random() < 0.35
    if use_2input:
        fam_name, fam_fn, _ = rng.choices(FAMILIES_2, weights=[w for _,_,w in FAMILIES_2])[0]
        sources = _make_sources_2(rng)
        perm = None
    else:
        fam_name, fam_fn, _ = rng.choices(FAMILIES_3, weights=[w for _,_,w in FAMILIES_3])[0]
        sources = _make_sources(rng)
        perm = rng.choice(PERMS_3)

    # Always 2-check — model was only verifying 1 example, causing 96% of bit errors
    do_multi_check = True

    # Generate enough high-information examples that the shown prompt supports
    # the source/family choice instead of relying on hidden generator metadata.
    n_examples = 24
    query_input = rng.randrange(256)

    def compute(x):
        vals = [fn(x) for _, fn in sources]
        if use_2input:
            return fam_fn(vals[0], vals[1])
        else:
            return fam_fn(vals[perm[0]], vals[perm[1]], vals[perm[2]])

    def get_abc(x):
        vals = [fn(x) for _, fn in sources]
        if use_2input:
            return vals[0], vals[1], None
        else:
            return vals[perm[0]], vals[perm[1]], vals[perm[2]]

    shuffled = [x for x in range(256) if x != query_input]
    rng.shuffle(shuffled)
    candidate_inputs = [x for x in FINGERPRINT_INPUTS if x != query_input]
    candidate_inputs.extend(x for x in shuffled if x not in candidate_inputs)

    query_str = format(query_input, "08b")
    answer_str = format(compute(query_input), "08b")

    query_probes = [
        query_input ^ (1 << bit)
        for bit in range(8)
        if (query_input ^ (1 << bit)) != query_input
    ]

    # Reorder examples: discriminative first, then query-neighbor probes,
    # fingerprint probes, then diagnostic. This keeps the first two Verify
    # examples informative while constraining source identity near the query.
    disc_indices = _find_discriminative_indices(fam_name, fam_fn, use_2input, candidate_inputs, get_abc, compute)
    diag_indices = []
    for idx, x in enumerate(candidate_inputs):
        a, b, c = get_abc(x)
        if _is_diagnostic(fam_name, a, b, c, compute(x)):
            diag_indices.append(idx)

    example_inputs = []
    seen = set()
    for idx in disc_indices:
        _append_unique(example_inputs, seen, [candidate_inputs[idx]], 6)
    _append_unique(
        example_inputs,
        seen,
        [x for x in query_probes if x != query_input],
        n_examples,
    )
    _append_unique(
        example_inputs,
        seen,
        [x for x in FINGERPRINT_INPUTS if x != query_input],
        n_examples,
    )
    _append_unique(
        example_inputs,
        seen,
        [candidate_inputs[idx] for idx in diag_indices],
        n_examples,
    )
    _append_unique(example_inputs, seen, candidate_inputs, n_examples)

    examples = [(format(x, "08b"), format(compute(x), "08b")) for x in example_inputs]

    # Ordered source names
    if use_2input:
        ordered_names = [sources[0][0], sources[1][0]]
    else:
        ordered_names = [sources[perm[0]][0], sources[perm[1]][0], sources[perm[2]][0]]

    # Get trace builder
    if use_2input:
        use_lines, row_fn = _trace_2input(fam_name, fam_fn)
    else:
        use_lines, row_fn = TRACE_BUILDERS[fam_name]()

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
    prompt = "\n".join(prompt_lines)

    # === Build compact trace ===
    from generators.trace_compact import build_trace_from_solver_with_meta

    # Map family to compact formula
    FAMILY_FORMULAS = {
        "OR_XNOR":          {"chain": [{"gate": "xnor", "inputs": ["A", "B"], "out": "P"},
                                        {"gate": "or", "inputs": ["C", "P"]}]},
        "GATED_XNOR_NAND":  {"family": "gated_xnor_nand", "inputs": ["A", "B", "C"]},
        "CH":               {"family": "ch", "inputs": ["A", "B", "C"]},
        "MAJ3":             {"family": "maj", "inputs": ["A", "B", "C"]},
        "TT121":            {"family": "tt121", "inputs": ["A", "B", "C"]},
        "T1":               {"family": "t1", "inputs": ["A", "B", "C"]},
        "AO":               {"chain": [{"gate": "and", "inputs": ["A", "B"], "out": "P"},
                                        {"gate": "or", "inputs": ["P", "C"]}]},
        "OA":               {"chain": [{"gate": "or", "inputs": ["A", "B"], "out": "P"},
                                        {"gate": "and", "inputs": ["P", "C"]}]},
        "AX":               {"chain": [{"gate": "and", "inputs": ["A", "B"], "out": "P"},
                                        {"gate": "xor", "inputs": ["P", "C"]}]},
        "OX":               {"chain": [{"gate": "or", "inputs": ["A", "B"], "out": "P"},
                                        {"gate": "xor", "inputs": ["P", "C"]}]},
        "XA":               {"chain": [{"gate": "xor", "inputs": ["A", "B"], "out": "P"},
                                        {"gate": "and", "inputs": ["P", "C"]}]},
        "XO":               {"chain": [{"gate": "xor", "inputs": ["A", "B"], "out": "P"},
                                        {"gate": "or", "inputs": ["P", "C"]}]},
        "PAR3":             {"chain": [{"gate": "xor", "inputs": ["A", "B"], "out": "P"},
                                        {"gate": "xor", "inputs": ["P", "C"]}]},
        "AND":              {"gate": "and", "inputs": ["A", "B"]},
        "OR":               {"gate": "or", "inputs": ["A", "B"]},
        "XOR":              {"gate": "xor", "inputs": ["A", "B"]},
    }

    formula = FAMILY_FORMULAS.get(fam_name)
    if formula is None:
        return None  # unknown family

    # Split ~prefix from source names — trace_compact needs bare names + complement flags
    trace_names = []
    complements = []
    for name in ordered_names:
        if name.startswith('~'):
            trace_names.append(name[1:])
            complements.append(True)
        else:
            trace_names.append(name)
            complements.append(False)

    reasoning, trace_answer, witness_meta = build_trace_from_solver_with_meta(
        trace_names, complements, formula, examples, query_str,
        seed=rng.randint(0, 999999))

    # Post-render verify: trace must produce same answer
    if trace_answer != answer_str:
        return None

    mode_str = f"3stream_{fam_name.lower()}"

    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer_str}}}"},
        ],
        "answer": answer_str,
        "id": f"gen_bit_3stream_{rng.randint(0, 999999):06d}",
        "puzzle_type": "bit_manipulation",
        "mode": mode_str,
        "witness_strength": witness_meta.get("witness_strength", "w0"),
        "n_examples": n_examples,
        "family": fam_name,
        "sources": ordered_names,
        "generator": "gen_bit_3stream",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _retrace_perbit(row):
    """Re-trace a 3stream row with the canonical locked program format."""
    from generators.trace_bit_program import (
        UnrenderableBitProgram,
        assert_no_forbidden_bit_trace,
        render_generated_row,
    )

    try:
        reasoning, traced_answer = render_generated_row(row)
        assert_no_forbidden_bit_trace(reasoning)
    except (UnrenderableBitProgram, AssertionError):
        return None

    row["messages"][1]["content"] = f"<think>\n{reasoning}\n</think>\n\\boxed{{{traced_answer}}}"
    row["generator"] = "gen_bit_3stream_program_v1"
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--output", type=str, default="data/bit_manipulation/pool/generated/3stream.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fingerprint-rate", type=float, default=0.30,
                        help="Fraction of 3-input traces with Scan: evidence (0.0-1.0)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    t0 = time.time()
    count = 0
    skipped = 0

    with open(args.output, "w") as f:
        for i in range(args.n * 2):
            if count >= args.n:
                break
            row = generate_one(rng, fingerprint_rate=args.fingerprint_rate)
            if row is None:
                skipped += 1
                continue
            row = _retrace_perbit(row)
            if row is None:
                skipped += 1
                continue
            f.write(json.dumps(row) + "\n")
            count += 1
            if count % 5000 == 0:
                print(f"  {count}/{args.n} ({skipped} skipped)")

    dt = time.time() - t0
    print(f"Generated {count} rows in {dt:.1f}s → {args.output} ({skipped} skipped)")


if __name__ == "__main__":
    main()
