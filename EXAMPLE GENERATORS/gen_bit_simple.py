#!/usr/bin/env python3
"""Generate bit manipulation puzzles using simple 2-source and 1-source rules.

Balances the training pool against 3-stream traces. Competition distribution:
  56.5% 2-source, 9.1% 1-source, 34.3% 3-source.

Operations covered:
  2-source: A&B, A|B, A^B, ~(A^B), A&~B, ~A&B, A|~B, ~A|B, ~(A&B), ~(A|B)
  1-source: T(x), ~T(x), T(x) ^ const

Uses compact trace format (Scan + GRID + bookend verification).

Usage:
    python3 -m generators.gen_bit_simple --n 10000
    python3 -m generators.gen_bit_simple --n 10000 --output data/bit_manipulation/pool/generated/simple_gates.jsonl
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

def _add_elimination_steps(lines, examples):
    """Add mechanical elimination Steps 1-3 after scan."""
    outputs = [out for _, out in examples]
    inputs = [inp for inp, _ in examples]
    if len(set(outputs)) == 1:
        lines.append(f"Step 1: all outputs same? Yes → {outputs[0]}")
    else:
        lines.append(f"Step 1: all outputs same? No ({len(set(outputs))} distinct)")
    identity_matches = sum(1 for i, o in zip(inputs, outputs) if i == o)
    if identity_matches == len(examples):
        lines.append(f"Step 2: output=input? Yes")
    else:
        for inp, out in examples:
            if inp != out:
                n_diff = sum(1 for a, b in zip(inp, out) if a != b)
                lines.append(f"Step 2: output=input? No (Ex1 differs at {n_diff} positions)")
                break
    not_matches = sum(1 for i, o in zip(inputs, outputs)
                      if o == ''.join('1' if c == '0' else '0' for c in i))
    if not_matches == len(examples):
        lines.append(f"Step 3: output=NOT(input)? Yes")
    else:
        lines.append(f"Step 3: output=NOT(input)? No")
    lines.append("")


def _xor_line(computed, expected):
    """Per-bit XOR computation line."""
    return "  XOR: " + " ".join(f"{a}⊕{b}={int(a)^int(b)}" for a, b in zip(computed, expected))




def _fmt(v):
    return format(v & BYTE, '08b')


def rol(x, k):
    return ((x << k) | (x >> (8 - k))) & BYTE


def ror(x, k):
    return ((x >> k) | (x << (8 - k))) & BYTE


# ── Source pool ──────────────────────────────────────────────────────────────

def _build_source_pool():
    """All single-transform sources: x, shl1-7, shr1-7, rol1-7, ror1-7."""
    pool = [("x", lambda x: x)]
    for k in range(1, 8):
        pool.append((f"shl{k}", lambda x, k=k: (x << k) & BYTE))
        pool.append((f"shr{k}", lambda x, k=k: (x >> k) & BYTE))
        pool.append((f"rol{k}", lambda x, k=k: rol(x, k)))
        pool.append((f"ror{k}", lambda x, k=k: ror(x, k)))
    return pool


SOURCE_POOL = _build_source_pool()


# ── 2-source operations ─────────────────────────────────────────────────────

# Each entry: (name, fn(a,b)->out, weight, trace_builder)
# trace_builder(a,b) -> (check_lines, output_val)
# Weight reflects competition frequency.

def _op_and():
    def trace(a, b):
        out = (a & b) & BYTE
        return [f"  output=A & B={_fmt(out)}"], out
    return ("A & B", lambda a, b: (a & b) & BYTE, 180,
            "  output = A & B", trace)


def _op_or():
    def trace(a, b):
        out = (a | b) & BYTE
        return [f"  output=A | B={_fmt(out)}"], out
    return ("A | B", lambda a, b: (a | b) & BYTE, 180,
            "  output = A | B", trace)


def _op_xor():
    def trace(a, b):
        out = (a ^ b) & BYTE
        return [f"  output=A ^ B={_fmt(out)}"], out
    return ("A ^ B", lambda a, b: (a ^ b) & BYTE, 200,
            "  output = A ^ B", trace)


def _op_xnor():
    def trace(a, b):
        xor_val = (a ^ b) & BYTE
        out = (~xor_val) & BYTE
        return [
            f"  A ^ B={_fmt(xor_val)}",
            f"  output=~(A ^ B)={_fmt(out)}",
        ], out
    return ("~(A ^ B)", lambda a, b: (~(a ^ b)) & BYTE, 60,
            "  output = ~(A ^ B)", trace)


def _op_and_not():
    def trace(a, b):
        nb = (~b) & BYTE
        out = (a & nb) & BYTE
        return [
            f"  ~B={_fmt(nb)}",
            f"  output=A & ~B={_fmt(out)}",
        ], out
    return ("A & ~B", lambda a, b: (a & (~b)) & BYTE, 80,
            "  output = A & ~B", trace)


def _op_not_and():
    def trace(a, b):
        na = (~a) & BYTE
        out = (na & b) & BYTE
        return [
            f"  ~A={_fmt(na)}",
            f"  output=~A & B={_fmt(out)}",
        ], out
    return ("~A & B", lambda a, b: ((~a) & b) & BYTE, 20,
            "  output = ~A & B", trace)


def _op_or_not():
    def trace(a, b):
        nb = (~b) & BYTE
        out = (a | nb) & BYTE
        return [
            f"  ~B={_fmt(nb)}",
            f"  output=A | ~B={_fmt(out)}",
        ], out
    return ("A | ~B", lambda a, b: (a | (~b)) & BYTE, 40,
            "  output = A | ~B", trace)


def _op_not_or():
    def trace(a, b):
        na = (~a) & BYTE
        out = (na | b) & BYTE
        return [
            f"  ~A={_fmt(na)}",
            f"  output=~A | B={_fmt(out)}",
        ], out
    return ("~A | B", lambda a, b: ((~a) | b) & BYTE, 40,
            "  output = ~A | B", trace)


def _op_nand():
    """~(A & B) — not in competition but good for diversity."""
    def trace(a, b):
        and_val = (a & b) & BYTE
        out = (~and_val) & BYTE
        return [
            f"  A & B={_fmt(and_val)}",
            f"  output=~(A & B)={_fmt(out)}",
        ], out
    return ("~(A & B)", lambda a, b: (~(a & b)) & BYTE, 30,
            "  output = ~(A & B)", trace)


def _op_nor():
    """~(A | B) — not in competition but good for diversity."""
    def trace(a, b):
        or_val = (a | b) & BYTE
        out = (~or_val) & BYTE
        return [
            f"  A | B={_fmt(or_val)}",
            f"  output=~(A | B)={_fmt(out)}",
        ], out
    return ("~(A | B)", lambda a, b: (~(a | b)) & BYTE, 30,
            "  output = ~(A | B)", trace)


# Build the ops list
_OPS_2SRC_RAW = [
    _op_xor(),    # 200 — flattened from 468. Model over-picks XOR already.
    _op_and(),    # 180 — boosted. Model under-predicts AND (9% wrong vs 15% correct).
    _op_or(),     # 180 — boosted. Model under-predicts OR (7% wrong vs 12% correct).
    _op_and_not(),  # 80
    _op_xnor(),   # 60
    _op_not_or(),  # 40
    _op_or_not(),  # 40
    _op_nand(),    # 30
    _op_nor(),     # 30
    _op_not_and(), # 20
]

OPS_2SRC = []
# Canonical gate metadata — no string surgery needed downstream
_CANONICAL = {
    "A ^ B":    {"gate": "xor",     "comp": [False, False]},
    "A & B":    {"gate": "and",     "comp": [False, False]},
    "A | B":    {"gate": "or",      "comp": [False, False]},
    "A & ~B":   {"gate": "and",     "comp": [False, True]},
    "~(A ^ B)": {"gate": "xnor",    "comp": [False, False]},
    "~A | B":   {"gate": "or",      "comp": [True, False]},
    "A | ~B":   {"gate": "or",      "comp": [False, True]},
    "~(A & B)": {"gate": "nand",    "comp": [False, False]},
    "~(A | B)": {"gate": "nor",     "comp": [False, False]},
    "~A & B":   {"gate": "and",     "comp": [True, False]},
}
for name, fn, weight, use_line, trace_fn in _OPS_2SRC_RAW:
    canon = _CANONICAL.get(name, {"gate": "xor", "comp": [False, False]})
    OPS_2SRC.append({
        "name": name,
        "fn": fn,
        "weight": weight,
        "use_line": use_line,
        "trace_fn": trace_fn,
        "canonical_gate": canon["gate"],
        "complements": canon["comp"],
    })

OPS_2SRC_NAMES = [op["name"] for op in OPS_2SRC]
OPS_2SRC_WEIGHTS = [op["weight"] for op in OPS_2SRC]


# ── 1-source operations ─────────────────────────────────────────────────────

# T(x) — just a transform
# ~T(x) — NOT of transform
# T(x) ^ const — XOR with a constant byte


# ── Rival ops for discriminative checks ──────────────────────────────────────

RIVALS_2SRC = {
    "A & B":    ["A | B", "A ^ B"],
    "A | B":    ["A & B", "A ^ B"],
    "A ^ B":    ["A & B", "A | B"],
    "~(A ^ B)": ["A & B", "A | B"],
    "A & ~B":   ["A ^ B", "~A & B"],
    "~A & B":   ["A ^ B", "A & ~B"],
    "A | ~B":   ["~A | B", "~(A ^ B)"],
    "~A | B":   ["A | ~B", "~(A ^ B)"],
    "~(A & B)": ["~(A ^ B)", "A | B"],
    "~(A | B)": ["~(A ^ B)", "A & B"],
}


def _get_rival_fn(name):
    """Get the compute function for a named operation."""
    for op in OPS_2SRC:
        if op["name"] == name:
            return op["fn"]
    return None


def _is_discriminative_2src(op_name, op_fn, a, b, example_x, src_fns):
    """Check if this example discriminates against at least one rival."""
    rivals = RIVALS_2SRC.get(op_name, [])
    correct = op_fn(a, b) & BYTE
    for rival_name in rivals:
        rival_fn = _get_rival_fn(rival_name)
        if rival_fn is not None:
            rival_out = rival_fn(a, b) & BYTE
            if rival_out != correct:
                return True
    return False


def _is_diagnostic_2src(a, b, out):
    """Basic quality: output is not trivially one input or all-0/all-1."""
    if out == a or out == b:
        return False
    if out == 0 or out == BYTE:
        return False
    return True


# ── Generation ───────────────────────────────────────────────────────────────

def generate_2src(rng):
    """Generate a 2-source bit puzzle."""
    # Pick operation
    op = rng.choices(OPS_2SRC, weights=OPS_2SRC_WEIGHTS)[0]
    op_name = op["name"]
    op_fn = op["fn"]
    trace_fn = op["trace_fn"]
    use_line = op["use_line"]

    # Pick 2 sources (distinct)
    s1, s2 = rng.sample(SOURCE_POOL, 2)
    src_names = [s1[0], s2[0]]
    src_fns = [s1[1], s2[1]]

    # Use enough high-information examples that the shown prompt supports the
    # program choice, instead of relying on hidden generator metadata.
    n_examples = 16
    query_input = rng.randrange(256)

    def compute(x):
        a = src_fns[0](x)
        b = src_fns[1](x)
        return op_fn(a, b) & BYTE

    # Score candidates: prefer ones that separate this op from rivals
    disc_inputs = []
    diag_inputs = []
    other_inputs = []
    candidate_inputs = [x for x in range(256) if x != query_input]
    rng.shuffle(candidate_inputs)
    for x in candidate_inputs:
        if x == query_input:
            continue
        a = src_fns[0](x)
        b = src_fns[1](x)
        out = compute(x)
        if _is_discriminative_2src(op_name, op_fn, a, b, x, src_fns):
            disc_inputs.append(x)
        elif _is_diagnostic_2src(a, b, out):
            diag_inputs.append(x)
        else:
            other_inputs.append(x)

    # Take discriminative first, then source-fingerprint probes, then diagnostic.
    selected = set()
    example_inputs = []
    _append_unique(example_inputs, selected, disc_inputs, 6)
    _append_unique(
        example_inputs,
        selected,
        [x for x in FINGERPRINT_INPUTS if x != query_input],
        n_examples,
    )
    _append_unique(example_inputs, selected, diag_inputs, n_examples)
    all_remaining = [x for x in (disc_inputs + diag_inputs + other_inputs) if x not in selected]
    _append_unique(example_inputs, selected, all_remaining, n_examples)

    examples_raw = [(format(x, "08b"), format(compute(x), "08b")) for x in example_inputs]

    # Reorder: discriminative first (they'll be used in Scan preamble + Checks)
    examples = examples_raw  # already ordered disc→diag→other
    query_str = format(query_input, "08b")
    answer_str = format(compute(query_input), "08b")

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

    # Build compact trace
    from generators.trace_compact import build_trace_from_solver_with_meta

    # Use canonical metadata directly — no string surgery
    gate_final = op["canonical_gate"]
    complements = op["complements"]

    reasoning, trace_answer, witness_meta = build_trace_from_solver_with_meta(
        src_names, complements, gate_final, examples, query_str,
        seed=rng.randint(0, 999999))

    # Post-render verify: trace must produce same answer as direct computation
    if trace_answer != answer_str:
        return None  # reject — trace computation disagrees with direct computation

    mode_str = f"simple_{op_name.replace(' ', '').replace('~', 'not')}"

    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer_str}}}"},
        ],
        "answer": answer_str,
        "id": f"gen_bit_simple_{rng.randint(0, 999999):06d}",
        "puzzle_type": "bit_manipulation",
        "mode": mode_str,
        "witness_strength": witness_meta.get("witness_strength", "w0"),
        "n_examples": n_examples,
        "family": op_name,
        "sources": src_names,
        "generator": "gen_bit_simple",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_1src(rng):
    """Generate a 1-source bit puzzle: T(x), ~T(x), or T(x) ^ const."""
    # Pick source (exclude 'x' — identity is trivial)
    source_pool_no_x = [(n, f) for n, f in SOURCE_POOL if n != 'x']
    src_name, src_fn = rng.choice(source_pool_no_x)

    # Pick operation variant
    variant = rng.choices(
        ["T", "~T", "T^const"],
        weights=[109, 15, 36],  # from competition: T=109, T^const=36, ~T≈15
    )[0]

    if variant == "T^const":
        # Exclude 0x00 (identity) and 0xFF (just NOT) — degenerate
        const_byte = rng.choice([b for b in range(1, 255)])
        const_str = _fmt(const_byte)

        def compute(x):
            return (src_fn(x) ^ const_byte) & BYTE

        use_line = f"  output = T ^ {const_str}"

        def trace_fn(t_val):
            out = (t_val ^ const_byte) & BYTE
            return [f"  output=T ^ {const_str}={_fmt(out)}"], out

    elif variant == "~T":
        def compute(x):
            return (~src_fn(x)) & BYTE

        use_line = "  output = ~T"

        def trace_fn(t_val):
            out = (~t_val) & BYTE
            return [f"  output=~T={_fmt(out)}"], out

    else:  # "T"
        def compute(x):
            return src_fn(x)

        use_line = "  output = T"

        def trace_fn(t_val):
            return [f"  output={_fmt(t_val)}"], t_val

    # Fingerprint probes make source identity mechanically visible.
    n_examples = 16
    query_input = rng.randrange(256)
    example_inputs = [x for x in FINGERPRINT_INPUTS if x != query_input]
    if len(example_inputs) < n_examples:
        fill = [x for x in range(256) if x != query_input and x not in example_inputs]
        rng.shuffle(fill)
        example_inputs.extend(fill[:n_examples - len(example_inputs)])

    examples = [(format(x, "08b"), format(compute(x), "08b")) for x in example_inputs]
    query_str = format(query_input, "08b")
    answer_str = format(compute(query_input), "08b")

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

    # Build compact trace
    from generators.trace_compact import build_scan, scan_line, shift_line as compact_shift, fmt as compact_fmt, ones as count_ones

    trace_lines = ["Bit rule.", ""]

    # Scan preamble
    scan = build_scan(examples)
    trace_lines.extend(scan)
    trace_lines.append("")
    _add_elimination_steps(trace_lines, examples)

    from generators.trace_compact import _xor_bits

    # Mandatory backtracking: try a wrong source first
    wrong_srcs = [s for s in SOURCE_POOL if s[0] != src_name]
    cand_num = 1
    if wrong_srcs:
        wrong_src = rng.choice(wrong_srcs)
        wx, wexp = examples[0]
        wrong_out = format(wrong_src[1](int(wx, 2)) & BYTE, '08b')
        if wrong_out != wexp:
            diff_r = _xor_bits(wrong_out, wexp)
            trace_lines.append(f"Try[1]: T={wrong_src[0]}(x), output=T")
            trace_lines.append(f"  Witness (Ex1): x={wx}")
            trace_lines.append(f"  output={wrong_out}")
            trace_lines.append(f"  expected={wexp}")
            trace_lines.append(f"  diff={diff_r} → FAIL")
            trace_lines.append(f"  Decision[1]: REJECT")
            trace_lines.append("")
            cand_num = 2

    # Correct candidate
    trace_lines.append(f"Try[{cand_num}]:")
    trace_lines.append(f"  T = {src_name}(x)")
    trace_lines.append(use_line)
    trace_lines.append("")

    # Witnesses
    check_indices = list(range(len(examples)))
    rng.shuffle(check_indices)
    for ci in range(min(2, len(examples))):
        idx = check_indices[ci]
        inp, expected = examples[idx]
        sx = int(inp, 2)
        t_val = src_fn(sx)
        t_str = _fmt(t_val)
        result = compute(sx)
        result_str = format(result, '08b')
        diff = _xor_bits(result_str, expected)

        trace_lines.append(f"Witness {ci+1}: x={inp}")
        trace_lines.append(f"T={src_name}({inp})={t_str}")
        if variant == "T^const":
            trace_lines.append(f"output=T^{const_str}={result_str}")
        elif variant == "~T":
            trace_lines.append(f"output=not({t_str})={result_str}")
        else:
            trace_lines.append(f"output={t_str}")
        trace_lines.append(f"  output={result_str}")
        trace_lines.append(f"  expected={expected}")
        trace_lines.append(_xor_line(result_str, expected))
        trace_lines.append(f"  diff={diff} → {'PASS' if diff == '00000000' else 'FAIL'}")
        trace_lines.append("")

    trace_lines.append(f"Decision[{cand_num}]: LOCK")
    trace_lines.append("")

    # Query — clean
    t_query = src_fn(query_input)
    t_str = _fmt(t_query)

    trace_lines.append(f"Query: x={query_str}")
    trace_lines.append(f"T={src_name}({query_str})={t_str}")
    if variant == "T^const":
        trace_lines.append(f"output=T^{const_str}={answer_str}")
    elif variant == "~T":
        trace_lines.append(f"output=not({t_str})={answer_str}")
    else:
        trace_lines.append(f"output={t_str}")

    reasoning = "\n".join(trace_lines)
    mode_str = f"simple_1src_{variant.replace('^', 'xor').replace('~', 'not')}"

    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer_str}}}"},
        ],
        "answer": answer_str,
        "id": f"gen_bit_simple_{rng.randint(0, 999999):06d}",
        "puzzle_type": "bit_manipulation",
        "mode": mode_str,
        "n_examples": n_examples,
        "family": variant,
        "sources": [src_name],
        **({"xor_const": const_byte} if variant == "T^const" else {}),
        "generator": "gen_bit_simple",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_identity(rng):
    """Generate an IDENTITY puzzle: output = shifted/rotated copy of input.
    41.2% of competition bits are identity — uses SAME trace program as 2src."""
    src_name, src_fn = rng.choice(SOURCE_POOL)

    n_examples = 16
    query_x = rng.randrange(256)
    example_inputs = [x for x in FINGERPRINT_INPUTS if x != query_x]
    if len(example_inputs) < n_examples:
        fill = [x for x in range(256) if x != query_x and x not in example_inputs]
        rng.shuffle(fill)
        example_inputs.extend(fill[:n_examples - len(example_inputs)])

    examples = [(format(x, '08b'), format(src_fn(x) & BYTE, '08b')) for x in example_inputs]
    query_str = format(query_x, '08b')
    answer = format(src_fn(query_x) & BYTE, '08b')

    # Build prompt — SAME format as competition
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

    # Build trace using SAME compact format — Scan + EXCLUDE + Rule + Check (GRID) + Query
    from generators.trace_compact import build_scan, scan_line, compact_check, compact_query, fmt, apply_shift

    lines = ["Bit rule.", ""]

    # Scan (same as 2src)
    scan = build_scan(examples)
    lines.extend(scan)
    lines.append("")
    _add_elimination_steps(lines, examples)

    # Mandatory backtracking: show a wrong rival failing before the correct candidate
    from generators.trace_compact import _xor_bits
    rival_src = rng.choice([s for s in SOURCE_POOL if s[0] != src_name])
    rival_name = rival_src[0]
    wx, wout = examples[0]
    rival_out = format(rival_src[1](int(wx, 2)) & BYTE, '08b')
    if rival_out != wout:
        diff_r = _xor_bits(rival_out, wout)
        lines.append(f"Try[1]: A={rival_name}(x), output=A")
        lines.append(f"  Witness (Ex1): x={wx}")
        lines.append(f"  A={rival_name}({wx})={rival_out}")
        lines.append(f"  output={rival_out}")
        lines.append(f"  expected={wout}")
        lines.append(f"  diff={diff_r} → FAIL")
        lines.append(f"  Decision[1]: REJECT")
        lines.append("")
        cand_num = 2
    else:
        cand_num = 1

    # Correct candidate
    lines.append(f"Try[{cand_num}]: A={src_name}(x), output=A")
    lines.append("")

    # Witness 1 and 2 — two-pass verify
    for ci in range(min(2, len(examples))):
        inp_str, exp_str = examples[ci]
        x_val = int(inp_str, 2)
        a_val = format(src_fn(x_val) & BYTE, '08b')
        diff = _xor_bits(a_val, exp_str)
        lines.append(f"Witness {ci+1} (Ex{ci+1}): x={inp_str}")
        lines.append(f"A={src_name}({inp_str})={a_val}")
        lines.append(f"  output={a_val}")
        lines.append(f"  expected={exp_str}")
        lines.append(_xor_line(a_val, exp_str))
        lines.append(f"  diff={diff} → {'PASS' if diff == '00000000' else 'FAIL'}")
        lines.append("")

    lines.append(f"Decision[{cand_num}]: LOCK")
    lines.append("")

    # Query — clean, no decorative stats
    qa = format(src_fn(query_x) & BYTE, '08b')
    lines.append(f"Query: x={query_str}")
    lines.append(f"A={src_name}({query_str})={qa}")

    content = f"<think>\n{chr(10).join(lines)}\n</think>\n\\boxed{{{answer}}}"

    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": content},
        ],
        "answer": answer,
        "id": f"identity_{rng.randint(0, 999999):06d}",
        "puzzle_type": "bit_manipulation",
        "mode": f"simple_identity_{src_name}",
        "family": "T",
        "sources": [src_name],
        "generator": "gen_bit_simple",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_const(rng):
    """Generate a CONST puzzle: all outputs are the same constant byte.
    14.8% of competition bits are CONST — uses SAME trace program as 2src."""
    const_type = rng.choice(['all_zero', 'all_one', 'fixed_byte'])
    if const_type == 'all_zero':
        const_val = 0
    elif const_type == 'all_one':
        const_val = BYTE
    else:
        const_val = rng.randint(1, 254)

    const_str = format(const_val, '08b')

    n_examples = 16
    query_x = rng.randrange(256)
    example_inputs = [x for x in FINGERPRINT_INPUTS if x != query_x]
    if len(example_inputs) < n_examples:
        fill = [x for x in range(256) if x != query_x and x not in example_inputs]
        rng.shuffle(fill)
        example_inputs.extend(fill[:n_examples - len(example_inputs)])

    examples = [(format(x, '08b'), const_str) for x in example_inputs]
    query_str = format(query_x, '08b')

    # Build prompt — SAME format as competition
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

    # Build trace — SAME structure: Scan + observation + Rule + Check + Query
    from generators.trace_compact import build_scan

    lines = ["Bit rule.", ""]

    # Scan (same as 2src)
    scan = build_scan(examples)
    lines.extend(scan)
    lines.append("")
    _add_elimination_steps(lines, examples)

    from generators.trace_compact import _xor_bits

    # Observation: all outputs identical
    lines.append(f"All outputs = {const_str}. No input dependence.")
    lines.append("")

    # Mandatory backtracking: try a wrong rule first
    # For CONST, a good rival is identity (output=input)
    wx, wout = examples[0]
    if wx != const_str:  # identity would give wrong answer
        diff_r = _xor_bits(wx, const_str)
        lines.append(f"Try[1]: output = x (identity)")
        lines.append(f"  Witness (Ex1): x={wx}")
        lines.append(f"  output={wx}")
        lines.append(f"  expected={const_str}")
        lines.append(f"  diff={diff_r} → FAIL")
        lines.append(f"  Decision[1]: REJECT")
        lines.append("")
        cand_num = 2
    else:
        cand_num = 1

    # Correct candidate
    lines.append(f"Try[{cand_num}]: output = {const_str} (constant)")
    lines.append("")

    # Witness 1 and 2 — two-pass verify
    for ci in range(min(2, len(examples))):
        inp_str, exp_str = examples[ci]
        diff = _xor_bits(const_str, exp_str)
        lines.append(f"Witness {ci+1} (Ex{ci+1}): x={inp_str}")
        lines.append(f"  output={const_str}")
        lines.append(f"  expected={exp_str}")
        lines.append(_xor_line(const_str, exp_str))
        lines.append(f"  diff={diff} → {'PASS' if diff == '00000000' else 'FAIL'}")
        lines.append("")

    lines.append(f"Decision[{cand_num}]: LOCK")
    lines.append("")

    # Query — clean
    lines.append(f"Query: x={query_str}")
    lines.append(f"output={const_str}")

    content = f"<think>\n{chr(10).join(lines)}\n</think>\n\\boxed{{{const_str}}}"

    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": content},
        ],
        "answer": const_str,
        "id": f"const_{rng.randint(0, 999999):06d}",
        "puzzle_type": "bit_manipulation",
        "mode": f"simple_const_{const_type}",
        "family": "CONST",
        "const": const_str,
        "generator": "gen_bit_simple",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_contrastive_bit(rng):
    """Generate a CONTRASTIVE pair: same sources, two confusable gates, shared examples.
    One decisive example flips the gold between the two gates.
    Returns TWO rows (or None if can't find confusable pair)."""
    CONFUSABLE = [
        ('xor', 'or'), ('xor', 'and'), ('and', 'or'),
        ('xnor', 'xor'), ('nand', 'nor'), ('xnor', 'or'),
        ('and', 'nand'), ('or', 'nor'),  # complement confusion
        ('xor', 'xnor'),                 # complement confusion
    ]
    gate_a_name, gate_b_name = rng.choice(CONFUSABLE)

    # Find gates
    gate_a_fn = gate_b_fn = None
    for op in OPS_2SRC:
        if op['canonical_gate'] == gate_a_name and not any(op['complements']):
            gate_a_fn = op['fn']
        if op['canonical_gate'] == gate_b_name and not any(op['complements']):
            gate_b_fn = op['fn']
    if not gate_a_fn or not gate_b_fn:
        return None

    s1, s2 = rng.sample(SOURCE_POOL, 2)
    src_fns = [s1[1], s2[1]]

    def compute_a(x):
        return gate_a_fn(src_fns[0](x), src_fns[1](x)) & BYTE
    def compute_b(x):
        return gate_b_fn(src_fns[0](x), src_fns[1](x)) & BYTE

    # Find inputs where gates AGREE and one where they DISAGREE
    shuffled = list(range(256))
    rng.shuffle(shuffled)
    all_inputs = FINGERPRINT_INPUTS + [x for x in shuffled if x not in FINGERPRINT_INPUTS]
    agree = []
    disagree = []
    for x in all_inputs:
        if compute_a(x) == compute_b(x):
            agree.append(x)
        else:
            disagree.append(x)
        if len(agree) >= 8 and len(disagree) >= 2:
            break

    if len(agree) < 6 or len(disagree) < 1:
        return None

    # Build shared examples (where gates agree) + one decisive
    n_shared = 12
    shared = agree[:n_shared]
    decisive = disagree[0]
    query_x = disagree[1] if len(disagree) > 1 else rng.choice([x for x in all_inputs if x not in shared and x != decisive])

    # Row A: uses gate_a (decisive example shows gate_a's output)
    examples_a = [(format(x, '08b'), format(compute_a(x), '08b')) for x in shared]
    examples_a.append((format(decisive, '08b'), format(compute_a(decisive), '08b')))
    rng.shuffle(examples_a)

    # Row B: uses gate_b (decisive example shows gate_b's output)
    examples_b = [(format(x, '08b'), format(compute_b(x), '08b')) for x in shared]
    examples_b.append((format(decisive, '08b'), format(compute_b(decisive), '08b')))
    rng.shuffle(examples_b)

    query_str = format(query_x, '08b')
    answer_a = format(compute_a(query_x), '08b')
    answer_b = format(compute_b(query_x), '08b')

    if answer_a == answer_b:
        return None  # Not useful if query gives same answer

    # Build both rows (prompt + trace)
    from generators.trace_compact import build_trace_from_solver_with_meta
    rows = []
    for gate_name, examples, answer, gate_label in [
        (gate_a_name, examples_a, answer_a, 'contrastive_A'),
        (gate_b_name, examples_b, answer_b, 'contrastive_B'),
    ]:
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

        try:
            result = build_trace_from_solver_with_meta(
                [s1[0], s2[0]], [False, False], gate_name, examples, query_str,
                seed=rng.randint(0, 999999))
        except Exception:
            return None

        if result is None:
            return None
        reasoning, trace_answer, meta = result
        if trace_answer != answer:
            continue

        rows.append({
            "messages": [
                {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"},
            ],
            "answer": answer,
            "id": f"bit_contrastive_{rng.randint(0, 999999):06d}",
            "puzzle_type": "bit_manipulation",
            "mode": f"contrastive_{gate_name}",
            "family": gate_name,
            "sources": [s1[0], s2[0]],
            "generator": "gen_bit_simple",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })

    return rows if len(rows) == 2 else None


def generate_one(rng):
    """Generate one bit puzzle matching competition per-bit distribution.

    Competition distribution (verified on 12816 bits):
      Identity: 41.2%, CONST: 14.8%, NOT: 5.7%, 2-input: 36.5%, 3-input: 1.8%

    Since our puzzles use whole-byte ops (not per-bit), we approximate:
      ~35% identity/shift puzzles
      ~15% constant output puzzles
      ~35% 2-input gate puzzles
      ~5% 1-source NOT/XOR puzzles
      ~20% contrastive pairs (boosted from 10% — user: "VERY important")
    """
    # Distribution: match competition (identity+CONST+NOT=61.7%)
    # R17 feedback: "If model nails Steps 1-3 at 90%, that alone gets 55% bit"
    r = rng.random()
    if r < 0.15:
        # 15% contrastive pairs — gate discrimination
        pair = generate_contrastive_bit(rng)
        if pair:
            return pair[rng.randint(0, 1)]
        return generate_2src(rng)
    elif r < 0.55:
        # 40% identity — competition is 41.2%, this is guaranteed points
        return generate_identity(rng)
    elif r < 0.70:
        # 15% CONST — competition is 14.8%
        return generate_const(rng)
    elif r < 0.77:
        # 7% 1-source NOT — competition is 5.7%
        return generate_1src(rng)
    else:
        # 23% 2-input gates — competition is 36.5% but harder, lower training share
        return generate_2src(rng)


def _retrace_perbit(row):
    """Re-trace an existing row with the canonical locked program format."""
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
    row["generator"] = "gen_bit_simple_program_v1"
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--output", type=str,
                        default="data/bit_manipulation/pool/generated/simple_gates.jsonl")
    parser.add_argument("--seed", type=int, default=77)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    t0 = time.time()
    count = 0
    skipped = 0

    with open(args.output, "w") as f:
        for i in range(args.n * 2):  # oversample for retrace failures
            if count >= args.n:
                break
            row = generate_one(rng)
            if row is None:
                skipped += 1
                continue
            # Re-trace with per-bit format
            row = _retrace_perbit(row)
            if row is None:
                skipped += 1
                continue
            f.write(json.dumps(row) + "\n")
            count += 1
            if (count) % 5000 == 0:
                print(f"  {count}/{args.n} ({skipped} skipped)")

    dt = time.time() - t0
    print(f"Generated {count} rows in {dt:.1f}s -> {args.output} ({skipped} skipped)")


if __name__ == "__main__":
    main()
