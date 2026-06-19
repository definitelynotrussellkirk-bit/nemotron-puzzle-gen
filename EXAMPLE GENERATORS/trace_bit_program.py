#!/usr/bin/env python3
"""Canonical locked-program trace renderer for bit manipulation.

BIT_PROGRAM_V1 has one job: teach the model to execute a program that the
solver/generator has already locked. It does not search and it never sees the
gold answer as an input to fill missing bits.
"""
from __future__ import annotations

import re
from typing import Any

BYTE = 0xFF


class UnrenderableBitProgram(ValueError):
    """Raised when a solver result has no mechanical program to render."""


def parse_bit_examples(prompt: str) -> tuple[list[tuple[str, str]], str | None]:
    """Extract bit examples and query from a competition-style prompt."""
    examples: list[tuple[str, str]] = []
    for line in prompt.splitlines():
        m = re.match(r".*?([01]{8})\s*(?:->|=>|→)\s*([01]{8})", line)
        if m:
            examples.append((m.group(1), m.group(2)))

    query = None
    for line in prompt.splitlines():
        m = re.search(
            r"(?:output for|result for|determine the output for:|determine.*?:)\s*([01]{8})",
            line,
            re.IGNORECASE,
        )
        if m:
            query = m.group(1)

    if query is None:
        all_bins = re.findall(r"[01]{8}", prompt)
        example_values = {v for pair in examples for v in pair}
        for bits in reversed(all_bins):
            if bits not in example_values:
                query = bits
                break

    return examples, query


def _fmt(value: int) -> str:
    return format(value & BYTE, "08b")


def _diff(a: str, b: str) -> str:
    return "".join("0" if x == y else "1" for x, y in zip(a, b))


def _rol(value: int, amount: int) -> int:
    amount &= 7
    if amount == 0:
        return value & BYTE
    return ((value << amount) | (value >> (8 - amount))) & BYTE


def _ror(value: int, amount: int) -> int:
    amount &= 7
    if amount == 0:
        return value & BYTE
    return ((value >> amount) | (value << (8 - amount))) & BYTE


def _normalize_source_name(name: str) -> str:
    """Convert solver labels to renderer source names."""
    name = name.strip()
    m = re.fullmatch(r"NOT\((.+)\)", name)
    if m:
        return "~" + _normalize_source_name(m.group(1))
    name = name.replace("rotl(", "rol").replace("rotr(", "ror")
    name = name.replace("shl(", "shl").replace("shr(", "shr")
    name = name.replace(")", "")
    name = name.replace(",", "")
    return name


def _source_expr(name: str) -> str:
    name = _normalize_source_name(name)
    complement = name.startswith("~")
    base = name[1:] if complement else name
    if base == "x":
        expr = "x"
    else:
        expr = f"{base}(x)"
    return f"not({expr})" if complement else expr


def _eval_source(name: str, x: int) -> int:
    name = _normalize_source_name(name)
    complement = name.startswith("~")
    base = name[1:] if complement else name

    if base == "x":
        value = x & BYTE
    else:
        m = re.fullmatch(r"(shl|shr|rol|ror)(\d+)", base)
        if not m:
            raise UnrenderableBitProgram(f"unsupported source: {name}")
        op, amount_s = m.group(1), m.group(2)
        amount = int(amount_s)
        if op == "shl":
            value = (x << amount) & BYTE
        elif op == "shr":
            value = (x >> amount) & BYTE
        elif op == "rol":
            value = _rol(x, amount)
        else:
            value = _ror(x, amount)

    return (~value) & BYTE if complement else value


def _source_value_line(label: str, source: str, value: int) -> str:
    return f"{label} = {_source_expr(source)} = {_fmt(value)}"


def _select_verify_examples(examples: list[tuple[str, str]], max_verify: int) -> list[tuple[int, str, str]]:
    if max_verify <= 0 or max_verify >= len(examples):
        return [(idx, inp, expected) for idx, (inp, expected) in enumerate(examples, 1)]
    # Keep the visible trace compact while the renderer still verifies all rows.
    return [(idx, inp, expected) for idx, (inp, expected) in enumerate(examples[:max_verify], 1)]


def _canonical_combiner(name: str) -> str:
    mapping = {
        "A & B": "AND",
        "A | B": "OR",
        "A ^ B": "XOR",
        "~(A ^ B)": "XNOR",
        "~(A & B)": "NAND",
        "~(A | B)": "NOR",
        "A & ~B": "a_AND_NOTb",
        "~A & B": "NOTa_AND_b",
        "A | ~B": "a_OR_NOTb",
        "~A | B": "NOTa_OR_b",
        "and": "AND",
        "or": "OR",
        "xor": "XOR",
        "xnor": "XNOR",
        "nand": "NAND",
        "nor": "NOR",
        "and_not": "a_AND_NOTb",
        "or_not": "a_OR_NOTb",
        "T": "SOURCE",
        "~T": "NOT_SOURCE",
    }
    return mapping.get(name, name)


def _eval_combiner(name: str, values: list[int], mask: int | None = None) -> tuple[int, list[str], str]:
    """Return output value, mechanical lines, and program formula."""
    name = _canonical_combiner(name)
    a = values[0] if values else 0
    b = values[1] if len(values) > 1 else 0
    c = values[2] if len(values) > 2 else 0

    if name == "SOURCE":
        return a, [f"output = A = {_fmt(a)}"], "output = A"
    if name == "NOT_SOURCE":
        out = (~a) & BYTE
        return out, [f"output = not(A) = {_fmt(out)}"], "output = not(A)"
    if name == "XOR_MASK":
        if mask is None:
            raise UnrenderableBitProgram("XOR_MASK requires mask")
        out = (a ^ mask) & BYTE
        return out, [f"output = A XOR {_fmt(mask)} = {_fmt(out)}"], f"output = A XOR {_fmt(mask)}"

    if name == "AND":
        out = a & b
        return out, [f"output = A AND B = {_fmt(out)}"], "output = A AND B"
    if name == "OR":
        out = (a | b) & BYTE
        return out, [f"output = A OR B = {_fmt(out)}"], "output = A OR B"
    if name == "XOR":
        out = (a ^ b) & BYTE
        return out, [f"output = A XOR B = {_fmt(out)}"], "output = A XOR B"
    if name == "XNOR":
        p = (a ^ b) & BYTE
        out = (~p) & BYTE
        return out, [f"P = A XOR B = {_fmt(p)}", f"output = not(P) = {_fmt(out)}"], "output = XNOR(A,B)"
    if name == "NAND":
        p = a & b
        out = (~p) & BYTE
        return out, [f"P = A AND B = {_fmt(p)}", f"output = not(P) = {_fmt(out)}"], "output = NAND(A,B)"
    if name == "NOR":
        p = (a | b) & BYTE
        out = (~p) & BYTE
        return out, [f"P = A OR B = {_fmt(p)}", f"output = not(P) = {_fmt(out)}"], "output = NOR(A,B)"
    if name == "a_AND_NOTb":
        nb = (~b) & BYTE
        out = a & nb
        return out, [f"not(B) = {_fmt(nb)}", f"output = A AND not(B) = {_fmt(out)}"], "output = A AND not(B)"
    if name == "NOTa_AND_b":
        na = (~a) & BYTE
        out = na & b
        return out, [f"not(A) = {_fmt(na)}", f"output = not(A) AND B = {_fmt(out)}"], "output = not(A) AND B"
    if name == "a_OR_NOTb":
        nb = (~b) & BYTE
        out = (a | nb) & BYTE
        return out, [f"not(B) = {_fmt(nb)}", f"output = A OR not(B) = {_fmt(out)}"], "output = A OR not(B)"
    if name == "NOTa_OR_b":
        na = (~a) & BYTE
        out = (na | b) & BYTE
        return out, [f"not(A) = {_fmt(na)}", f"output = not(A) OR B = {_fmt(out)}"], "output = not(A) OR B"

    if name == "OR_XNOR":
        p = (~(a ^ b)) & BYTE
        out = (c | p) & BYTE
        return out, [f"P = XNOR(A,B) = {_fmt(p)}", f"output = C OR P = {_fmt(out)}"], "output = C OR XNOR(A,B)"
    if name == "GATED_XNOR_NAND":
        p = (~(a ^ b)) & BYTE
        q = (~(a & b)) & BYTE
        out = (((~c) & BYTE) & p | c & q) & BYTE
        return out, [
            f"P = XNOR(A,B) = {_fmt(p)}",
            f"Q = NAND(A,B) = {_fmt(q)}",
            f"output = where C=0 take P, C=1 take Q = {_fmt(out)}",
        ], "output = select(C,P,Q)"
    if name == "CH":
        out = ((a & b) | (((~a) & BYTE) & c)) & BYTE
        return out, [f"output = where A=1 take B, A=0 take C = {_fmt(out)}"], "output = CH(A,B,C)"
    if name == "NOT_CH":
        ch = ((a & b) | (((~a) & BYTE) & c)) & BYTE
        out = (~ch) & BYTE
        return out, [f"P = CH(A,B,C) = {_fmt(ch)}", f"output = not(P) = {_fmt(out)}"], "output = not(CH(A,B,C))"
    if name == "MAJ3":
        p, q, r = a & b, a & c, b & c
        out = (p | q | r) & BYTE
        return out, [f"P=A AND B = {_fmt(p)}", f"Q=A AND C = {_fmt(q)}", f"R=B AND C = {_fmt(r)}", f"output = P OR Q OR R = {_fmt(out)}"], "output = MAJ3(A,B,C)"
    if name == "NOT_MAJ3":
        p, q, r = a & b, a & c, b & c
        maj = (p | q | r) & BYTE
        out = (~maj) & BYTE
        return out, [f"P=MAJ3(A,B,C) = {_fmt(maj)}", f"output = not(P) = {_fmt(out)}"], "output = not(MAJ3(A,B,C))"
    if name == "TT121":
        p = (~(b ^ c)) & BYTE
        q = (~(b & c)) & BYTE
        out = (((~a) & BYTE) & p | a & q) & BYTE
        return out, [f"P = XNOR(B,C) = {_fmt(p)}", f"Q = NAND(B,C) = {_fmt(q)}", f"output = where A=0 take P, A=1 take Q = {_fmt(out)}"], "output = TT121(A,B,C)"
    if name == "T1":
        p = (~(a ^ b ^ c)) & BYTE
        q = (((~a) & BYTE) & ((~b) & BYTE) & c) & BYTE
        out = (p | q) & BYTE
        return out, [f"P = not(A XOR B XOR C) = {_fmt(p)}", f"Q = not(A) AND not(B) AND C = {_fmt(q)}", f"output = P OR Q = {_fmt(out)}"], "output = T1(A,B,C)"
    if name == "T2":
        p = (~(a ^ b ^ c)) & BYTE
        q = (((~a) & BYTE) & b & ((~c) & BYTE)) & BYTE
        out = (p | q) & BYTE
        return out, [f"P = not(A XOR B XOR C) = {_fmt(p)}", f"Q = not(A) AND B AND not(C) = {_fmt(q)}", f"output = P OR Q = {_fmt(out)}"], "output = T2(A,B,C)"
    if name == "T3":
        p = (~(a ^ b ^ c)) & BYTE
        q = (a & ((~b) & BYTE) & ((~c) & BYTE)) & BYTE
        out = (p | q) & BYTE
        return out, [f"P = not(A XOR B XOR C) = {_fmt(p)}", f"Q = A AND not(B) AND not(C) = {_fmt(q)}", f"output = P OR Q = {_fmt(out)}"], "output = T3(A,B,C)"
    if name == "OR2_3":
        out = (a | c) & BYTE
        return out, [f"output = A OR C = {_fmt(out)}"], "output = A OR C"
    if name == "AND2_3":
        out = a & b
        return out, [f"output = A AND B = {_fmt(out)}"], "output = A AND B"
    if name == "AND_NOT":
        na = (~a) & BYTE
        out = c & na
        return out, [f"not(A) = {_fmt(na)}", f"output = C AND not(A) = {_fmt(out)}"], "output = C AND not(A)"
    if name == "OR_NOT":
        nb = (~b) & BYTE
        out = (c | nb) & BYTE
        return out, [f"not(B) = {_fmt(nb)}", f"output = C OR not(B) = {_fmt(out)}"], "output = C OR not(B)"
    if name == "AO":
        p = a & b
        out = (p | c) & BYTE
        return out, [f"P = A AND B = {_fmt(p)}", f"output = P OR C = {_fmt(out)}"], "output = (A AND B) OR C"
    if name == "OA":
        p = (a | b) & BYTE
        out = p & c
        return out, [f"P = A OR B = {_fmt(p)}", f"output = P AND C = {_fmt(out)}"], "output = (A OR B) AND C"
    if name == "AX":
        p = a & b
        out = (p ^ c) & BYTE
        return out, [f"P = A AND B = {_fmt(p)}", f"output = P XOR C = {_fmt(out)}"], "output = (A AND B) XOR C"
    if name == "OX":
        p = (a | b) & BYTE
        out = (p ^ c) & BYTE
        return out, [f"P = A OR B = {_fmt(p)}", f"output = P XOR C = {_fmt(out)}"], "output = (A OR B) XOR C"
    if name == "XA":
        p = (a ^ b) & BYTE
        out = p & c
        return out, [f"P = A XOR B = {_fmt(p)}", f"output = P AND C = {_fmt(out)}"], "output = (A XOR B) AND C"
    if name == "XO":
        p = (a ^ b) & BYTE
        out = (p | c) & BYTE
        return out, [f"P = A XOR B = {_fmt(p)}", f"output = P OR C = {_fmt(out)}"], "output = (A XOR B) OR C"
    if name == "PAR3":
        out = (a ^ b ^ c) & BYTE
        return out, [f"output = A XOR B XOR C = {_fmt(out)}"], "output = A XOR B XOR C"

    raise UnrenderableBitProgram(f"unsupported combiner: {name}")


def _program_lines_for_combiner(name: str, mask: int | None = None) -> list[str]:
    """Return left-to-right symbolic program lines for a combiner."""
    name = _canonical_combiner(name)
    if name == "SOURCE":
        return ["output = A"]
    if name == "NOT_SOURCE":
        return ["output = not(A)"]
    if name == "XOR_MASK":
        if mask is None:
            raise UnrenderableBitProgram("XOR_MASK requires mask")
        return [f"output = A XOR {_fmt(mask)}"]
    if name in {"AND", "OR", "XOR"}:
        return [f"output = A {name} B"]
    if name in {"XNOR", "NAND", "NOR"}:
        op = {"XNOR": "XOR", "NAND": "AND", "NOR": "OR"}[name]
        return [f"P = A {op} B", "output = not(P)"]
    if name == "a_AND_NOTb":
        return ["not(B) = not(B)", "output = A AND not(B)"]
    if name == "NOTa_AND_b":
        return ["not(A) = not(A)", "output = not(A) AND B"]
    if name == "a_OR_NOTb":
        return ["not(B) = not(B)", "output = A OR not(B)"]
    if name == "NOTa_OR_b":
        return ["not(A) = not(A)", "output = not(A) OR B"]
    if name == "OR_XNOR":
        return ["P = XNOR(A,B)", "output = C OR P"]
    if name == "GATED_XNOR_NAND":
        return ["P = XNOR(A,B)", "Q = NAND(A,B)", "output = select(C,P,Q)"]
    if name == "CH":
        return ["P = A AND B", "Q = not(A) AND C", "output = P OR Q"]
    if name == "NOT_CH":
        return ["P = A AND B", "Q = not(A) AND C", "R = P OR Q", "output = not(R)"]
    if name == "MAJ3":
        return ["P = A AND B", "Q = A AND C", "R = B AND C", "output = P OR Q OR R"]
    if name == "NOT_MAJ3":
        return ["P = A AND B", "Q = A AND C", "R = B AND C", "S = P OR Q OR R", "output = not(S)"]
    if name == "TT121":
        return ["P = XNOR(B,C)", "Q = NAND(B,C)", "output = select(A,P,Q)"]
    if name == "T1":
        return ["P = not(A XOR B XOR C)", "Q = not(A) AND not(B) AND C", "output = P OR Q"]
    if name == "T2":
        return ["P = not(A XOR B XOR C)", "Q = not(A) AND B AND not(C)", "output = P OR Q"]
    if name == "T3":
        return ["P = not(A XOR B XOR C)", "Q = A AND not(B) AND not(C)", "output = P OR Q"]
    if name == "OR2_3":
        return ["output = A OR C"]
    if name == "AND2_3":
        return ["output = A AND B"]
    if name == "AND_NOT":
        return ["not(A) = not(A)", "output = C AND not(A)"]
    if name == "OR_NOT":
        return ["not(B) = not(B)", "output = C OR not(B)"]
    if name == "AO":
        return ["P = A AND B", "output = P OR C"]
    if name == "OA":
        return ["P = A OR B", "output = P AND C"]
    if name == "AX":
        return ["P = A AND B", "output = P XOR C"]
    if name == "OX":
        return ["P = A OR B", "output = P XOR C"]
    if name == "XA":
        return ["P = A XOR B", "output = P AND C"]
    if name == "XO":
        return ["P = A XOR B", "output = P OR C"]
    if name == "PAR3":
        return ["output = A XOR B XOR C"]
    raise UnrenderableBitProgram(f"unsupported combiner: {name}")


def render_byte_program(
    examples: list[tuple[str, str]],
    query: str,
    sources: list[str],
    combiner: str,
    *,
    mask: int | None = None,
    expected_answer: str | None = None,
    max_verify: int = 2,
) -> tuple[str, str]:
    """Render a locked whole-byte program trace."""
    if not sources:
        raise UnrenderableBitProgram("byte program needs at least one source")

    labels = ["A", "B", "C"][: len(sources)]

    verified_outputs: list[tuple[str, str, list[int], int, list[str], str]] = []
    for inp, expected in examples:
        x = int(inp, 2)
        values = [_eval_source(source, x) for source in sources]
        output, calc_lines, _ = _eval_combiner(combiner, values, mask)
        out_s = _fmt(output)
        diff = _diff(out_s, expected)
        if diff != "00000000":
            raise UnrenderableBitProgram("locked byte program does not verify all support rows")
        verified_outputs.append((inp, expected, values, output, calc_lines, diff))

    lines = ["BIT_PROGRAM_V1", "", "Program:"]
    for label, source in zip(labels, sources):
        lines.append(f"  {label} = {_source_expr(source)}")

    sample_vals = [_eval_source(source, int(query, 2)) for source in sources]
    _eval_combiner(combiner, sample_vals, mask)
    for formula in _program_lines_for_combiner(combiner, mask):
        lines.append(f"  {formula}")
    lines.append("")
    lines.append("Verify:")
    lines.append(f"  support_pass = {len(examples)}/{len(examples)}")

    by_input = {inp: (expected, values, output, calc_lines, diff) for inp, expected, values, output, calc_lines, diff in verified_outputs}
    for idx, inp, expected in _select_verify_examples(examples, max_verify):
        expected, values, _output, calc_lines, diff = by_input[inp]
        lines.append(f"  Ex{idx}:")
        lines.append(f"    x = {inp}")
        for label, source, value in zip(labels, sources, values):
            lines.append(f"    {_source_value_line(label, source, value)}")
        for calc_line in calc_lines:
            lines.append(f"    {calc_line}")
        lines.append(f"    expected = {expected}")
        lines.append(f"    diff = {diff} {'PASS' if diff == '00000000' else 'FAIL'}")

    q = int(query, 2)
    q_values = [_eval_source(source, q) for source in sources]
    q_output, q_lines, _ = _eval_combiner(combiner, q_values, mask)
    answer = _fmt(q_output)
    if expected_answer is not None and answer != expected_answer:
        raise UnrenderableBitProgram(f"program answer {answer} != expected {expected_answer}")

    lines.append("")
    lines.append("Apply:")
    lines.append(f"  x = {query}")
    for label, source, value in zip(labels, sources, q_values):
        lines.append(f"  {_source_value_line(label, source, value)}")
    for calc_line in q_lines:
        lines.append(f"  {calc_line}")
    lines.append(f"Answer: {answer}")
    return "\n".join(lines), answer


def render_const_program(
    examples: list[tuple[str, str]],
    query: str,
    const_bits: str,
    *,
    expected_answer: str | None = None,
    max_verify: int = 2,
) -> tuple[str, str]:
    for _inp, expected in examples:
        if _diff(const_bits, expected) != "00000000":
            raise UnrenderableBitProgram("constant program does not verify all support rows")
    lines = [
        "BIT_PROGRAM_V1",
        "",
        "Program:",
        f"  output = CONST({const_bits})",
        "",
        "Verify:",
        f"  support_pass = {len(examples)}/{len(examples)}",
    ]
    for idx, inp, expected in _select_verify_examples(examples, max_verify):
        diff = _diff(const_bits, expected)
        lines.append(f"  Ex{idx}:")
        lines.append(f"    x = {inp}")
        lines.append(f"    output = {const_bits}")
        lines.append(f"    expected = {expected}")
        lines.append(f"    diff = {diff} {'PASS' if diff == '00000000' else 'FAIL'}")
    if expected_answer is not None and const_bits != expected_answer:
        raise UnrenderableBitProgram(f"program answer {const_bits} != expected {expected_answer}")
    lines.extend(["", "Apply:", f"  x = {query}", f"  output = {const_bits}", f"Answer: {const_bits}"])
    return "\n".join(lines), const_bits


def _bit_rule_label(rule: dict[str, Any]) -> str:
    family = rule["family"]
    inputs = tuple(rule.get("inputs", ()))
    if family == "CONST_0":
        return "CONST(0)"
    if family == "CONST_1":
        return "CONST(1)"
    if family == "COPY":
        return f"COPY(i{inputs[0]})"
    if family == "NOT":
        return f"NOT(i{inputs[0]})"
    if family == "a_AND_NOTb":
        return f"AND(i{inputs[0]},NOT(i{inputs[1]}))"
    if family == "NOTa_AND_b":
        return f"AND(NOT(i{inputs[0]}),i{inputs[1]})"
    if family == "a_OR_NOTb":
        return f"OR(i{inputs[0]},NOT(i{inputs[1]}))"
    if family == "NOTa_OR_b":
        return f"OR(NOT(i{inputs[0]}),i{inputs[1]})"
    m = re.fullmatch(r"TT3_(\d+)", family)
    if m:
        return f"TABLE3(t{m.group(1)},i{inputs[0]},i{inputs[1]},i{inputs[2]})"
    args = ",".join(f"i{i}" for i in inputs)
    return f"{family}({args})"


def _eval_bit_rule(rule: dict[str, Any], bits: list[int]) -> int:
    family = rule["family"]
    inputs = tuple(rule.get("inputs", ()))
    if family == "CONST_0":
        return 0
    if family == "CONST_1":
        return 1
    if family == "COPY" and len(inputs) == 1:
        return bits[inputs[0]]
    if family == "NOT" and len(inputs) == 1:
        return 1 - bits[inputs[0]]
    if len(inputs) == 2:
        a, b = bits[inputs[0]], bits[inputs[1]]
        if family == "AND":
            return a & b
        if family == "OR":
            return a | b
        if family == "XOR":
            return a ^ b
        if family == "XNOR":
            return 1 - (a ^ b)
        if family == "NAND":
            return 1 - (a & b)
        if family == "NOR":
            return 1 - (a | b)
        if family == "a_AND_NOTb":
            return a & (1 - b)
        if family == "NOTa_AND_b":
            return (1 - a) & b
        if family == "a_OR_NOTb":
            return a | (1 - b)
        if family == "NOTa_OR_b":
            return (1 - a) | b
    if len(inputs) == 3:
        a, b, c = bits[inputs[0]], bits[inputs[1]], bits[inputs[2]]
        if family == "MAJ3":
            return int(a + b + c >= 2)
        if family == "NOT_MAJ3":
            return 1 - int(a + b + c >= 2)
        if family == "CH":
            return b if a else c
        if family == "NOT_CH":
            return 1 - (b if a else c)
        m = re.fullmatch(r"TT3_(\d+)", family)
        if m:
            mask = int(m.group(1))
            index = (a << 2) | (b << 1) | c
            return (mask >> index) & 1
    raise UnrenderableBitProgram(f"unsupported bit rule: {family}{inputs}")


def _rule_calc_line(out_pos: int, rule: dict[str, Any], bits: list[int]) -> str:
    value = _eval_bit_rule(rule, bits)
    return f"o{out_pos} = {_bit_rule_label(rule)} = {value}"


def render_per_bit_program(
    examples: list[tuple[str, str]],
    query: str,
    rules: list[dict[str, Any]],
    *,
    expected_answer: str | None = None,
    max_verify: int = 2,
) -> tuple[str, str]:
    if len(rules) != 8:
        raise UnrenderableBitProgram("per-bit program needs 8 rules")

    table_ids = sorted(
        {
            int(m.group(1))
            for rule in rules
            for m in [re.fullmatch(r"TT3_(\d+)", rule["family"])]
            if m
        }
    )

    verified_outputs: list[tuple[str, str, list[int], str, str]] = []
    for inp, expected in examples:
        bits = [int(ch) for ch in inp]
        output = "".join(str(_eval_bit_rule(rule, bits)) for rule in rules)
        diff = _diff(output, expected)
        if diff != "00000000":
            raise UnrenderableBitProgram("per-bit program does not verify all support rows")
        verified_outputs.append((inp, expected, bits, output, diff))

    lines = ["BIT_PROGRAM_V1", "", "Program:"]
    for table_id in table_ids:
        rows = []
        for idx in range(8):
            key = format(idx, "03b")
            rows.append(f"{key}->{(table_id >> idx) & 1}")
        lines.append(f"  t{table_id}: " + " ".join(rows))
    for out_pos, rule in enumerate(rules):
        lines.append(f"  o{out_pos} = {_bit_rule_label(rule)}")

    lines.append("")
    lines.append("Verify:")
    lines.append(f"  support_pass = {len(examples)}/{len(examples)}")
    by_input = {inp: (expected, bits, output, diff) for inp, expected, bits, output, diff in verified_outputs}
    for idx, inp, expected in _select_verify_examples(examples, max_verify):
        expected, bits, output, diff = by_input[inp]
        lines.append(f"  Ex{idx}:")
        lines.append(f"    x = {inp}")
        lines.append("    inputs = " + " ".join(f"i{i}={bit}" for i, bit in enumerate(bits)))
        for out_pos, rule in enumerate(rules):
            lines.append(f"    {_rule_calc_line(out_pos, rule, bits)}")
        lines.append(f"    output = {output}")
        lines.append(f"    expected = {expected}")
        lines.append(f"    diff = {diff} {'PASS' if diff == '00000000' else 'FAIL'}")

    q_bits = [int(ch) for ch in query]
    answer = "".join(str(_eval_bit_rule(rule, q_bits)) for rule in rules)
    if expected_answer is not None and answer != expected_answer:
        raise UnrenderableBitProgram(f"program answer {answer} != expected {expected_answer}")

    lines.append("")
    lines.append("Apply:")
    lines.append(f"  x = {query}")
    lines.append("  inputs = " + " ".join(f"i{i}={bit}" for i, bit in enumerate(q_bits)))
    for out_pos, rule in enumerate(rules):
        lines.append(f"  {_rule_calc_line(out_pos, rule, q_bits)}")
    lines.append(f"  output = {answer}")
    lines.append(f"Answer: {answer}")
    return "\n".join(lines), answer


def _parse_prompt_family_label(label: str) -> tuple[str | None, list[str]]:
    m = re.fullmatch(r"NOT\((.+)\)", label)
    if m:
        return "NOT_SOURCE", [m.group(1)]

    known = {
        "XOR", "AND", "OR", "XNOR", "NAND", "NOR",
        "a_OR_NOTb", "NOTa_AND_b", "NOTa_OR_b", "a_AND_NOTb",
        "MAJ3", "CH", "NOT_CH", "NOT_MAJ3",
        "OR_XNOR", "GATED_XNOR_NAND", "TT121", "T1",
        "AO", "OA", "AX", "OX", "XA", "XO", "PAR3",
    }
    m = re.fullmatch(r"(\w+)\((.+)\)", label)
    if m and m.group(1) in known:
        gate = m.group(1)
        inner = m.group(2)
        parts: list[str] = []
        depth = 0
        buf: list[str] = []
        for ch in inner:
            if ch == "(":
                depth += 1
                buf.append(ch)
            elif ch == ")":
                depth -= 1
                buf.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)
        if buf:
            parts.append("".join(buf).strip())
        return gate, parts
    return "SOURCE", [label]


def _rules_from_details(details: dict[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for pos, bit_detail in enumerate(details.get("bit_details", [])):
        chosen = bit_detail.get("chosen")
        if not chosen:
            raise UnrenderableBitProgram(f"bit {pos} has no chosen rule")
        family = chosen.get("family")
        inputs = tuple(chosen.get("inputs", ()))
        if family in {"TT3_random", "whole_byte"}:
            raise UnrenderableBitProgram(f"bit {pos} has unrenderable family {family}")
        if family is None:
            raise UnrenderableBitProgram(f"bit {pos} has missing family")
        rules.append({"family": family, "inputs": inputs})
    return rules


def render_from_solver_details(
    details: dict[str, Any],
    *,
    expected_answer: str | None = None,
    max_verify: int = 2,
) -> tuple[str, str]:
    """Render a BIT_PROGRAM_V1 trace from solve_details() metadata."""
    examples = details["examples"]
    query = details["query"]
    solver = details.get("solver")

    if solver == "prompt_family":
        labels = details.get("prompt_family", {}).get("labels", [])
        if not labels:
            raise UnrenderableBitProgram("prompt_family has no label")
        combiner, sources = _parse_prompt_family_label(labels[0])
        return render_byte_program(
            examples,
            query,
            [_normalize_source_name(s) for s in sources],
            combiner or "SOURCE",
            expected_answer=expected_answer,
            max_verify=max_verify,
        )

    if solver == "residual":
        transform = details.get("residual_transform")
        if not transform:
            raise UnrenderableBitProgram("residual solver missing transform")
        source = _normalize_source_name(transform)
        return render_byte_program(
            examples,
            query,
            [source],
            "XOR_MASK",
            mask=int(details.get("residual_value", 0)),
            expected_answer=expected_answer,
            max_verify=max_verify,
        )

    if solver == "3stream":
        meta = details.get("stream_meta", {})
        sources = list(meta.get("sources", []))
        perm = meta.get("perm")
        if perm is not None and len(sources) >= 3:
            sources = [sources[i] for i in perm]
        if not sources:
            raise UnrenderableBitProgram("3stream solver missing sources")
        return render_byte_program(
            examples,
            query,
            [_normalize_source_name(s) for s in sources],
            meta.get("family", ""),
            expected_answer=expected_answer,
            max_verify=max_verify,
        )

    if solver == "local_dsl":
        return render_per_bit_program(
            examples,
            query,
            _rules_from_details(details),
            expected_answer=expected_answer,
            max_verify=max_verify,
        )

    raise UnrenderableBitProgram(f"unsupported solver: {solver}")


def render_generated_row(row: dict[str, Any]) -> tuple[str, str]:
    """Render a generated bit row using generator metadata, no solver search."""
    prompt = row["messages"][0]["content"]
    examples, query = parse_bit_examples(prompt)
    if not examples or not query:
        raise UnrenderableBitProgram("could not parse generated row prompt")
    answer = row["answer"]
    mode = row.get("mode", "")
    family = row.get("family")
    sources = list(row.get("program_sources") or row.get("sources") or [])

    if mode.startswith("simple_const") or family == "CONST":
        const_bits = row.get("const") or answer
        return render_const_program(examples, query, const_bits, expected_answer=answer)

    if mode.startswith("simple_identity_") and not sources:
        sources = [mode.replace("simple_identity_", "", 1)]

    if family in {"T", "SOURCE"} or mode.startswith("simple_identity_"):
        return render_byte_program(examples, query, sources, "SOURCE", expected_answer=answer)
    if family == "~T":
        return render_byte_program(examples, query, sources, "NOT_SOURCE", expected_answer=answer)
    if family == "T^const":
        if "xor_const" not in row:
            raise UnrenderableBitProgram("T^const row missing xor_const")
        return render_byte_program(
            examples,
            query,
            sources,
            "XOR_MASK",
            mask=int(row["xor_const"]),
            expected_answer=answer,
        )

    if sources and family:
        return render_byte_program(examples, query, sources, str(family), expected_answer=answer)

    raise UnrenderableBitProgram(f"generated row lacks program metadata: {row.get('id')}")


def assert_no_forbidden_bit_trace(trace: str) -> None:
    """Fail if a trace contains the old answer-conditioned bit dialect."""
    forbidden = ["HARD", "complex rule", "answer bit"]
    for token in forbidden:
        if token in trace:
            raise AssertionError(f"forbidden bit trace token found: {token}")
