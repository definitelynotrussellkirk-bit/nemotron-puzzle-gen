#!/usr/bin/env python3
"""Per-bit column decomposition trace generator for bit manipulation puzzles.

Replaces the Try/REJECT/LOCK state machine with a fixed mechanical procedure:
1. Route: classify puzzle tier (CONST / shift / simple / complex)
2. Solve: for each output bit, scan candidates in fixed order
3. Apply: collect 8 bit results into the answer

Every token is mechanically derivable from previous tokens. The model follows
a checklist — no creative search, no state machine.
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Gate evaluation helpers
# ---------------------------------------------------------------------------

def _gate(a: str, b: str, op: str) -> str:
    """Evaluate a 2-input gate on single bits."""
    ai, bi = int(a), int(b)
    if op == "AND":    return str(ai & bi)
    if op == "OR":     return str(ai | bi)
    if op == "XOR":    return str(ai ^ bi)
    if op == "NAND":   return str(1 - (ai & bi))
    if op == "NOR":    return str(1 - (ai | bi))
    if op == "XNOR":   return str(1 - (ai ^ bi))
    raise ValueError(f"Unknown gate: {op}")


def _gate_col(a_col: str, b_col: str, op: str) -> str:
    """Evaluate a gate on two columns (strings of 0/1)."""
    return "".join(_gate(a, b, op) for a, b in zip(a_col, b_col))


def _not_col(col: str) -> str:
    return "".join("1" if c == "0" else "0" for c in col)


def _col_str(col: str) -> str:
    """Format a column as comma-separated bits: '1,0,1,0'."""
    return ",".join(col)


# ---------------------------------------------------------------------------
# 3-input gate helpers
# ---------------------------------------------------------------------------

_3INPUT_GATES = {
    "MAJ": lambda a, b, c: str(int((int(a) + int(b) + int(c)) >= 2)),
    "CHO": lambda a, b, c: b if a == "1" else c,  # a chooses b or c
    "PAR3": lambda a, b, c: str(int(a) ^ int(b) ^ int(c)),
    # AND-OR composites
    "AO": lambda a, b, c: str((int(a) & int(b)) | int(c)),
    "OA": lambda a, b, c: str((int(a) | int(b)) & int(c)),
    "AX": lambda a, b, c: str((int(a) & int(b)) ^ int(c)),
    "OX": lambda a, b, c: str((int(a) | int(b)) ^ int(c)),
    "XA": lambda a, b, c: str((int(a) ^ int(b)) & int(c)),
    "XO": lambda a, b, c: str((int(a) ^ int(b)) | int(c)),
}


def _3gate_col(a_col: str, b_col: str, c_col: str, op: str) -> str:
    fn = _3INPUT_GATES[op]
    return "".join(fn(a, b, c) for a, b, c in zip(a_col, b_col, c_col))


# ---------------------------------------------------------------------------
# Parse examples from prompt
# ---------------------------------------------------------------------------

def parse_bit_examples(prompt: str):
    """Extract (input, output) pairs and query from a bit manipulation prompt.

    Returns (examples: list[(str,str)], query: str) or (None, None).
    """
    examples = []
    for line in prompt.split("\n"):
        m = re.match(r".*?([01]{8})\s*(?:->|→|=>)\s*([01]{8})", line)
        if m:
            examples.append((m.group(1), m.group(2)))
    # Find query
    query = None
    for line in prompt.split("\n"):
        line_s = line.strip()
        if re.fullmatch(r"[01]{8}", line_s):
            # standalone 8-bit string that isn't part of an example
            if not any(line_s == ex[0] or line_s == ex[1] for ex in examples):
                query = line_s
        m2 = re.search(r"(?:apply|find|what|compute|determine|output for|result for)\s*.*?([01]{8})", line, re.IGNORECASE)
        if m2:
            query = m2.group(1)
    # Fallback: last 8-bit string in prompt that's not in examples
    if query is None:
        all_bins = re.findall(r"[01]{8}", prompt)
        example_strs = set()
        for inp, out in examples:
            example_strs.add(inp)
            example_strs.add(out)
        for b in reversed(all_bins):
            if b not in example_strs:
                query = b
                break
    return examples, query


# ---------------------------------------------------------------------------
# Per-bit column analysis
# ---------------------------------------------------------------------------

def _build_columns(examples):
    """Build input and output bit columns from examples.

    Returns (input_cols: list[str], output_cols: list[str]) where each
    column is a string of bits (one per example).
    """
    inputs = [ex[0] for ex in examples]
    outputs = [ex[1] for ex in examples]
    input_cols = ["".join(inp[b] for inp in inputs) for b in range(8)]
    output_cols = ["".join(out[b] for out in outputs) for b in range(8)]
    return input_cols, output_cols


# Rule representation: (type, *params)
# ("CONST", val)
# ("COPY", input_bit)
# ("NOT", input_bit)
# ("GATE2", gate_name, input_a, input_b)
# ("GATE2N", gate_name, input_a, input_b)  -- NOT applied to b
# ("GATE2NA", gate_name, input_a, input_b) -- NOT applied to a
# ("GATE3", gate_name, input_a, input_b, input_c)
# No answer-conditioned fallback is allowed. If no rule is found, rendering
# fails closed and the caller must use a richer locked-program renderer.


def _scan_bit(target_col: str, input_cols: list[str],
              answer_bit: str | None = None,
              show_rejects: int = 2) -> tuple[tuple, list[str]]:
    """Scan candidates for one output bit in fixed order.

    Returns (rule_tuple, trace_lines).
    trace_lines contains the scan steps (tested candidates with match/reject).
    """
    n = len(target_col)
    lines = []

    # 1. CONST
    if all(c == "0" for c in target_col):
        lines.append("CONST(0)")
        return ("CONST", "0"), lines
    if all(c == "1" for c in target_col):
        lines.append("CONST(1)")
        return ("CONST", "1"), lines

    # 2. COPY — test each input column (show up to show_rejects failures)
    copy_rejects = 0
    for i in range(8):
        if input_cols[i] == target_col:
            lines.append(f"i[{i}]={_col_str(input_cols[i])} match → COPY({i})")
            return ("COPY", i), lines
        elif copy_rejects < show_rejects:
            for pos in range(n):
                if input_cols[i][pos] != target_col[pos]:
                    lines.append(f"i[{i}]={_col_str(input_cols[i])} no (pos {pos})")
                    break
            copy_rejects += 1
    if copy_rejects >= show_rejects:
        lines.append("copy? no match")

    # 3. NOT — test NOT of each input column
    not_rejects = 0
    for i in range(8):
        inv = _not_col(input_cols[i])
        if inv == target_col:
            lines.append(f"NOT({i})={_col_str(inv)} match → NOT({i})")
            return ("NOT", i), lines
        elif not_rejects < show_rejects:
            for pos in range(n):
                if inv[pos] != target_col[pos]:
                    lines.append(f"NOT({i})={_col_str(inv)} no (pos {pos})")
                    break
            not_rejects += 1
    if not_rejects >= show_rejects:
        lines.append("NOT? no match")

    # 4. 2-input symmetric gates: AND, OR, XOR, NAND, NOR, XNOR
    sym_gates = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR"]
    gate_rejects = 0
    for a in range(8):
        for b in range(a + 1, 8):
            for gname in sym_gates:
                result = _gate_col(input_cols[a], input_cols[b], gname)
                if result == target_col:
                    lines.append(f"{gname}({a},{b})={_col_str(result)} match → {gname}({a},{b})")
                    return ("GATE2", gname, a, b), lines
                elif gate_rejects < show_rejects:
                    for pos in range(n):
                        if result[pos] != target_col[pos]:
                            lines.append(f"{gname}({a},{b})={_col_str(result)} no (pos {pos})")
                            break
                    gate_rejects += 1

    # 5. AND-NOT variants: a AND NOT(b), NOT(a) AND b, a OR NOT(b), etc.
    asym_gates = [
        ("AND", "b"),   # a AND NOT(b)
        ("AND", "a"),   # NOT(a) AND b
        ("OR", "b"),    # a OR NOT(b)
        ("OR", "a"),    # NOT(a) OR b
        ("XOR", "b"),   # a XOR NOT(b) = XNOR(a,b) — already covered, skip
    ]
    for base_gate, negate_which in asym_gates:
        if base_gate == "XOR":
            continue  # XOR(a, NOT(b)) = XNOR(a,b), already tested
        for a in range(8):
            for b in range(8):
                if a == b:
                    continue
                if negate_which == "b":
                    col_a, col_b = input_cols[a], _not_col(input_cols[b])
                    label = f"{base_gate}({a},~{b})"
                    rule = ("GATE2N", base_gate, a, b)
                else:
                    col_a, col_b = _not_col(input_cols[a]), input_cols[b]
                    label = f"{base_gate}(~{a},{b})"
                    rule = ("GATE2NA", base_gate, a, b)
                result = _gate_col(col_a, col_b, base_gate)
                if result == target_col:
                    lines.append(f"{label}={_col_str(result)} match → {label}")
                    return rule, lines

    # 6. 3-input gates
    for gname, fn in _3INPUT_GATES.items():
        for a in range(8):
            for b in range(8):
                if b == a:
                    continue
                for c in range(8):
                    if c == a or c == b:
                        continue
                    # Limit search: for symmetric gates, enforce a < b < c
                    if gname in ("MAJ", "PAR3") and not (a < b < c):
                        continue
                    result = _3gate_col(input_cols[a], input_cols[b], input_cols[c], gname)
                    if result == target_col:
                        lines.append(f"{gname}({a},{b},{c})={_col_str(result)} match → {gname}({a},{b},{c})")
                        return ("GATE3", gname, a, b, c), lines

    lines.append("no representable rule found")
    return ("UNRESOLVED",), lines


# ---------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------

def _apply_rule(rule: tuple, query_bits: str) -> str:
    """Apply a per-bit rule to the query input, return single bit."""
    rtype = rule[0]
    if rtype == "CONST":
        return rule[1]
    if rtype == "COPY":
        return query_bits[rule[1]]
    if rtype == "NOT":
        return "1" if query_bits[rule[1]] == "0" else "0"
    if rtype == "GATE2":
        _, gname, a, b = rule
        return _gate(query_bits[a], query_bits[b], gname)
    if rtype == "GATE2N":
        _, gname, a, b = rule
        nb = "1" if query_bits[b] == "0" else "0"
        return _gate(query_bits[a], nb, gname)
    if rtype == "GATE2NA":
        _, gname, a, b = rule
        na = "1" if query_bits[a] == "0" else "0"
        return _gate(na, query_bits[b], gname)
    if rtype == "GATE3":
        _, gname, a, b, c = rule
        fn = _3INPUT_GATES[gname]
        return fn(query_bits[a], query_bits[b], query_bits[c])
    if rtype == "UNRESOLVED":
        raise ValueError("cannot apply unresolved bit rule")
    return "0"


def _rule_label(rule: tuple) -> str:
    """Human-readable label for a rule."""
    rtype = rule[0]
    if rtype == "CONST":
        return f"CONST({rule[1]})"
    if rtype == "COPY":
        return f"COPY({rule[1]})"
    if rtype == "NOT":
        return f"NOT({rule[1]})"
    if rtype == "GATE2":
        return f"{rule[1]}({rule[2]},{rule[3]})"
    if rtype == "GATE2N":
        return f"{rule[1]}({rule[2]},~{rule[3]})"
    if rtype == "GATE2NA":
        return f"{rule[1]}(~{rule[2]},{rule[3]})"
    if rtype == "GATE3":
        return f"{rule[1]}({rule[2]},{rule[3]},{rule[4]})"
    return "UNRESOLVED"


def _apply_label(rule: tuple, query_bits: str) -> str:
    """Show the application step: rule with values substituted."""
    rtype = rule[0]
    result = _apply_rule(rule, query_bits)
    if rtype == "CONST":
        return result
    if rtype == "COPY":
        return f"i[{rule[1]}]={result}"
    if rtype == "NOT":
        return f"NOT({query_bits[rule[1]]})={result}"
    if rtype == "GATE2":
        _, gname, a, b = rule
        return f"{gname}({query_bits[a]},{query_bits[b]})={result}"
    if rtype == "GATE2N":
        _, gname, a, b = rule
        nb = "1" if query_bits[b] == "0" else "0"
        return f"{gname}({query_bits[a]},~{query_bits[b]})={gname}({query_bits[a]},{nb})={result}"
    if rtype == "GATE2NA":
        _, gname, a, b = rule
        na = "1" if query_bits[a] == "0" else "0"
        return f"{gname}(~{query_bits[a]},{query_bits[b]})={gname}({na},{query_bits[b]})={result}"
    if rtype == "GATE3":
        _, gname, a, b, c = rule
        return f"{gname}({query_bits[a]},{query_bits[b]},{query_bits[c]})={result}"
    return result


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

def _classify_tier(rules: list[tuple], examples) -> int:
    """Classify puzzle into tier 1-5 based on the per-bit rules found."""
    rule_types = set(r[0] for r in rules)

    # Tier 1: all outputs identical (all CONST with same value)
    outputs = [ex[1] for ex in examples]
    if len(set(outputs)) == 1:
        return 1

    # Tier 2: all COPY with consistent stride (shift/rotate)
    if rule_types <= {"COPY"}:
        sources = [r[1] for r in rules]
        # Check stride: sources[i] = (sources[0] + i * stride) % 8 for some stride
        for stride in range(1, 8):
            if all(sources[i] == (sources[0] + i * stride) % 8 for i in range(8)):
                return 2
        # Still all-copy but no stride — just a permutation
        return 3

    # Tier 3: only CONST and COPY
    if rule_types <= {"CONST", "COPY"}:
        return 3

    # Tier 4: no 3-input gates or unresolved bits
    if "GATE3" not in rule_types and "UNRESOLVED" not in rule_types:
        return 4

    # Tier 5: has complex/unresolved bits
    return 5


def _detect_shift(rules: list[tuple]) -> Optional[tuple[str, int]]:
    """If all rules are COPY with consistent stride, return (op_name, amount).

    Returns ("shl", k), ("shr", k), ("rol", k), ("ror", k), or None.
    """
    if not all(r[0] == "COPY" for r in rules):
        return None
    sources = [r[1] for r in rules]
    # stride = sources[1] - sources[0] mod 8
    stride = (sources[1] - sources[0]) % 8
    if not all((sources[i] - sources[0]) % 8 == (i * stride) % 8 for i in range(8)):
        return None

    # Determine which operation this is
    # rol(k): output[i] = input[(i+k)%8] → source[i] = (i+k)%8 → stride = 1, offset = k
    offset = sources[0] % 8
    if stride == 1:
        # Check if it's a rotate (all 8 input bits used) or shift (some zeros)
        if set(sources) == set(range(8)):
            return ("rol", offset)
        else:
            return ("shl", offset)
    # ror(k): output[i] = input[(i-k)%8] → source[i] = (i-k)%8 → stride = 1, offset = -k
    # Actually stride is always 1 for standard shifts. Let's just check offset.
    # For shr(k): output[i] = input[i-k] if i>=k else 0 → sources[i] = i-k
    # This means sources[0] = -k (which wraps around). But shr fills with 0, so
    # the "CONST(0)" bits wouldn't be COPY. So if all bits are COPY, it's a rotate.
    return ("rol", offset)


# ---------------------------------------------------------------------------
# Trace builders per tier
# ---------------------------------------------------------------------------

def _trace_tier1(examples, query_str, answer_str) -> str:
    """Tier 1: all outputs identical."""
    const_val = examples[0][1]
    lines = [f"Bit rule. {len(examples)} examples.", ""]
    lines.append(f"All outputs identical: {const_val}")
    lines.append(f"Answer: {const_val}")
    return "\n".join(lines)


def _trace_tier2(examples, query_str, answer_str, rules, shift_info) -> str:
    """Tier 2: whole-byte shift/rotate."""
    op_name, amount = shift_info
    n = len(examples)
    lines = [f"Bit rule. {n} examples.", ""]

    # Show the stride detection
    sources = [r[1] for r in rules]
    lines.append(f"Check: each output bit copies from shifted input position")
    for i in range(8):
        lines.append(f"  o[{i}] = i[{sources[i]}]")

    lines.append(f"Stride: +{(sources[1] - sources[0]) % 8} mod 8 → {op_name}({amount})")
    lines.append("")

    # Verify on examples
    lines.append("Verify:")
    for idx, (inp, out) in enumerate(examples[:3]):
        computed = "".join(inp[sources[i]] for i in range(8))
        match = "match" if computed == out else "MISMATCH"
        lines.append(f"  Ex{idx+1}: {op_name}({amount})({inp})={computed} vs {out} → {match}")

    lines.append("")
    computed_answer = "".join(query_str[sources[i]] for i in range(8))
    lines.append(f"Query: {op_name}({amount})({query_str})={computed_answer}")
    lines.append(f"Answer: {computed_answer}")
    return "\n".join(lines)


def _trace_perbit(examples, query_str, answer_str, rules, scan_results, tier_label: str) -> str:
    """General per-bit trace for tiers 3-5."""
    n = len(examples)
    input_cols, output_cols = _build_columns(examples)

    lines = [f"Bit rule. {n} examples.", ""]

    # Show per-bit scan
    for i in range(8):
        rule, scan_lines = scan_results[i]
        target = _col_str(output_cols[i])

        if rule[0] == "CONST":
            # Compact: one line
            lines.append(f"o[{i}]: {target} → {_rule_label(rule)}")
        elif rule[0] == "COPY" and len(scan_lines) <= 2:
            # Compact: few checks needed
            lines.append(f"o[{i}]: {target}")
            for sl in scan_lines:
                lines.append(f"  {sl}")
        else:
            # Full scan trace
            lines.append(f"o[{i}]: {target}")
            for sl in scan_lines:
                lines.append(f"  {sl}")

    # Summary line
    lines.append("")
    rule_strs = [_rule_label(rules[i]) for i in range(8)]
    lines.append(f"Rule: {','.join(rule_strs)}")

    # Apply to query
    lines.append("")
    lines.append(f"Query: {query_str}")
    answer_bits = []
    apply_parts = []
    for i in range(8):
        bit_val = _apply_rule(rules[i], query_str)
        answer_bits.append(bit_val)
        apply_parts.append(f"o[{i}]={_apply_label(rules[i], query_str)}")
    lines.append("  " + "  ".join(apply_parts[:4]))
    lines.append("  " + "  ".join(apply_parts[4:]))

    computed = "".join(answer_bits)
    lines.append(f"Answer: {computed}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_perbit_trace(examples: list[tuple[str, str]],
                       query_str: str,
                       answer_str: str) -> tuple[str, str]:
    """Build the per-bit column decomposition trace.

    Args:
        examples: list of (input_8bit, output_8bit) pairs
        query_str: 8-bit query input
        answer_str: known correct 8-bit answer (from solver)

    Returns:
        (reasoning_text, answer_str)
    """
    if not examples or not query_str:
        return None

    input_cols, output_cols = _build_columns(examples)

    # Scan each output bit
    rules = []
    scan_results = []
    for i in range(8):
        answer_bit = answer_str[i] if answer_str else None
        rule, scan_lines = _scan_bit(output_cols[i], input_cols,
                                      answer_bit=answer_bit)
        if rule[0] == "UNRESOLVED":
            return None
        rules.append(rule)
        scan_results.append((rule, scan_lines))

    # Classify tier
    tier = _classify_tier(rules, examples)

    # Compute answer from rules
    computed_answer = "".join(_apply_rule(rules[i], query_str) for i in range(8))

    if computed_answer != answer_str and answer_str:
        return None

    # Build trace based on tier
    if tier == 1:
        trace = _trace_tier1(examples, query_str, computed_answer)
    elif tier == 2:
        shift_info = _detect_shift(rules)
        if shift_info:
            trace = _trace_tier2(examples, query_str, computed_answer, rules, shift_info)
        else:
            trace = _trace_perbit(examples, query_str, computed_answer, rules, scan_results, "permute")
    else:
        label = {3: "const+copy", 4: "simple", 5: "complex"}[tier]
        trace = _trace_perbit(examples, query_str, computed_answer, rules, scan_results, label)

    return trace, computed_answer
