#!/usr/bin/env python3
"""Oversolve transformation trace generator (R14).

Three explicit phases per operator:
  Phase 1 — ORDERING: AB,CD or BA,DC?
  Phase 2 — OPERATION: add, sub, mul, absdiff, ...? (exhaustive scan with arithmetic shown)
  Phase 3 — STYLE: plain, rev, opsign, tailsign? (explicit decision tree)

Cipher-digit adds a Phase 0 — DECODE: map symbols to digits.

Every token is mechanically derivable. The model follows a checklist.
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Operations (frequency order — common first)
# ---------------------------------------------------------------------------

COMMON_OPS = [
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("bsub", lambda a, b: b - a),
    ("mul", lambda a, b: a * b),
    ("absdiff", lambda a, b: abs(a - b)),
    ("concat", lambda a, b: int(str(abs(a)) + str(abs(b)))),
    ("bconcat", lambda a, b: int(str(abs(b)) + str(abs(a)))),
    ("negabsdiff", lambda a, b: -abs(a - b)),
]

RARE_OPS = [
    ("muladd1", lambda a, b: a * b + 1),
    ("mulsub1", lambda a, b: a * b - 1),
    ("addp1", lambda a, b: a + b + 1),
    ("addm1", lambda a, b: a + b - 1),
    ("subp1", lambda a, b: a - b + 1),
    ("subm1", lambda a, b: a - b - 1),
    ("floordiv", lambda a, b: a // b if b != 0 else 99999),
    ("bfloordiv", lambda a, b: b // a if a != 0 else 99999),
    ("mod", lambda a, b: a % b if b != 0 else 99999),
    ("bmod", lambda a, b: b % a if a != 0 else 99999),
    ("maxmod", lambda a, b: max(a, b) % min(a, b) if min(a, b) != 0 else 99999),
]

ALL_OPS = COMMON_OPS + RARE_OPS

STYLE_SCAN_ORDER = [
    "plain",
    "rev",
    "abs",
    "abs_rev",
    "dsum",
    "opprefix",
    "opprefix_rev",
    "opsign",
    "opsign_always",
    "tailsign",
    "tailsign_always",
    "rev_opsign",
    "rev_opsign_always",
    "rev_tailsign",
    "rev_tailsign_always",
]


def _rev_str(s: str) -> str:
    """Reverse digits, preserving negative sign."""
    if s.startswith("-"):
        return "-" + s[1:][::-1]
    return s[::-1]


def _digit_sum_str(value: int) -> str:
    """Return the sum of decimal digits of abs(value), as a string."""
    return str(sum(int(ch) for ch in str(abs(value))))


def _compute_op(op_name: str, a: int, b: int) -> Optional[int]:
    for name, fn in ALL_OPS:
        if name == op_name:
            try:
                return fn(a, b)
            except:
                return None
    return None


def _show_mul(a: int, b: int) -> str:
    """Multiplication with place-value decomposition for 2+ digit operands."""
    result = a * b
    sa = str(abs(a))
    if len(sa) >= 2 and abs(a) >= 10:
        places = []
        for i, d in enumerate(sa):
            pv = int(d) * (10 ** (len(sa) - 1 - i))
            if pv != 0:
                places.append(pv)
        if len(places) > 1:
            products = [p * abs(b) for p in places]
            total = sum(products)
            sign = -1 if (a < 0) != (b < 0) else 1
            decomp = "+".join(str(p) for p in places)
            prod_str = " + ".join(f"{p}*{abs(b)}={p*abs(b)}" for p in places)
            return f"({decomp})*{abs(b)} = {' + '.join(str(x) for x in products)} = {sign * total}"
    return f"{a}*{b}={result}"


def _show_op(op_name: str, a: int, b: int) -> str:
    """Show arithmetic for an operation."""
    result = _compute_op(op_name, a, b)
    if result is None:
        return f"{op_name}({a},{b})=ERR"
    if op_name == "add": return f"{a}+{b}={result}"
    if op_name == "sub": return f"{a}-{b}={result}"
    if op_name == "bsub": return f"{b}-{a}={result}"
    if op_name == "mul": return _show_mul(a, b)
    if op_name == "muladd1": return f"{a}*{b}+1={a*b}+1={result}"
    if op_name == "mulsub1": return f"{a}*{b}-1={a*b}-1={result}"
    if op_name == "addp1": return f"{a}+{b}+1={result}"
    if op_name == "addm1": return f"{a}+{b}-1={result}"
    if op_name == "subp1": return f"{a}-{b}+1={result}"
    if op_name == "subm1": return f"{a}-{b}-1={result}"
    if op_name == "absdiff": return f"|{a}-{b}|={result}"
    if op_name == "negabsdiff": return f"-|{a}-{b}|={result}"
    if op_name == "concat": return f"{abs(a)}||{abs(b)}={result}"
    if op_name == "bconcat": return f"{abs(b)}||{abs(a)}={result}"
    if op_name == "floordiv": return f"{a}//{b}={result}"
    if op_name == "bfloordiv": return f"{b}//{a}={result}"
    if op_name == "mod": return f"{a}%{b}={result}"
    if op_name == "bmod": return f"{b}%{a}={result}"
    if op_name == "maxmod":
        big, small = max(abs(a), abs(b)), min(abs(a), abs(b))
        return f"max%min={big}%{small}={result}"
    return f"{op_name}({a},{b})={result}"


# ---------------------------------------------------------------------------
# Style detection and application
# ---------------------------------------------------------------------------

def _detect_style(raw: int, expected: str, op_char: str) -> Optional[str]:
    """Detect what style transforms raw result into expected string."""
    styles = _detect_styles(raw, expected, op_char)
    return styles[0] if styles else None


def _detect_styles(raw: int, expected: str, op_char: str) -> list[str]:
    """Return all output styles that transform raw into expected."""
    return [
        style
        for style in STYLE_SCAN_ORDER
        if _apply_style(raw, style, op_char) == expected
    ]


def _apply_style(raw: int, style: str, op_char: str) -> str:
    raw_s = str(raw)
    abs_s = str(abs(raw))
    abs_rev = _rev_str(abs_s)
    if style == "plain": return raw_s
    if style == "rev": return _rev_str(raw_s)
    if style == "abs": return abs_s
    if style == "abs_rev": return abs_rev
    if style == "dsum": return _digit_sum_str(raw)
    if style == "opprefix":
        return op_char + abs_s
    if style == "opprefix_rev":
        return op_char + abs_rev
    if style == "opsign":
        return (op_char + abs_s) if raw < 0 else raw_s
    if style == "opsign_always":
        return (op_char + abs_s) if raw < 0 else (op_char + raw_s)
    if style == "tailsign":
        return (abs_s + op_char) if raw < 0 else raw_s
    if style == "tailsign_always":
        return (abs_s + op_char) if raw < 0 else (raw_s + op_char)
    if style == "rev_opsign":
        return (op_char + abs_rev) if raw < 0 else abs_rev
    if style == "rev_opsign_always":
        return (op_char + abs_rev) if raw < 0 else (op_char + abs_rev)
    if style == "rev_tailsign":
        return (abs_rev + op_char) if raw < 0 else abs_rev
    if style == "rev_tailsign_always":
        return (abs_rev + op_char) if raw < 0 else (abs_rev + op_char)
    return raw_s


def _show_style_step(raw: int, style: str, op_char: str) -> list[str]:
    """Show style application as explicit decision tree."""
    lines = []
    raw_s = str(raw)
    final = _apply_style(raw, style, op_char)

    if style == "plain":
        lines.append(f"  Step 3 — style: plain → {final}")
    elif style == "rev":
        lines.append(f"  Step 3 — style: rev → reverse {raw_s} → {final}")
    elif style == "abs":
        lines.append(f"  Step 3 — style: abs → |{raw_s}| → {final}")
    elif style == "abs_rev":
        lines.append(f"  Step 3 — style: abs+rev → |{raw_s}|={abs(raw)} rev → {final}")
    elif style == "dsum":
        lines.append(f"  Step 3 — style: dsum → digit_sum(|{raw_s}|) → {final}")
    elif style == "opprefix":
        lines.append(f"  Step 3 — style: opprefix → '{op_char}'+{abs(raw)} → {final}")
    elif style == "opprefix_rev":
        lines.append(f"  Step 3 — style: opprefix_rev → rev |{raw_s}|={abs(raw)} → {_rev_str(str(abs(raw)))} → '{op_char}'+{_rev_str(str(abs(raw)))} → {final}")
    elif style == "opsign":
        if raw < 0:
            lines.append(f"  Step 3 — style: opsign, {raw_s} is negative → '{op_char}'+{abs(raw)} → {final}")
        else:
            lines.append(f"  Step 3 — style: opsign, {raw_s} non-negative → {final}")
    elif style == "opsign_always":
        lines.append(f"  Step 3 — style: opsign_always → '{op_char}'+{abs(raw)} → {final}")
    elif style == "tailsign":
        if raw < 0:
            lines.append(f"  Step 3 — style: tailsign, {raw_s} is negative → {abs(raw)}+'{op_char}' → {final}")
        else:
            lines.append(f"  Step 3 — style: tailsign, {raw_s} non-negative → {final}")
    elif style == "tailsign_always":
        lines.append(f"  Step 3 — style: tailsign_always → {abs(raw)}+'{op_char}' → {final}")
    elif style == "rev_opsign":
        if raw < 0:
            lines.append(f"  Step 3 — style: rev+opsign, {raw_s} is negative → rev |{raw_s}|={abs(raw)} → {_rev_str(str(abs(raw)))} → '{op_char}'+{_rev_str(str(abs(raw)))} → {final}")
        else:
            lines.append(f"  Step 3 — style: rev+opsign → rev {raw_s} → {final}")
    elif style == "rev_opsign_always":
        lines.append(f"  Step 3 — style: rev+opsign_always → rev |{raw_s}|={abs(raw)} → {_rev_str(str(abs(raw)))} → '{op_char}'+{_rev_str(str(abs(raw)))} → {final}")
    elif style == "rev_tailsign":
        if raw < 0:
            lines.append(f"  Step 3 — style: rev+tailsign, {raw_s} is negative → rev {abs(raw)} → {_rev_str(str(abs(raw)))}+'{op_char}' → {final}")
        else:
            lines.append(f"  Step 3 — style: rev+tailsign → rev {raw_s} → {final}")
    elif style == "rev_tailsign_always":
        lines.append(f"  Step 3 — style: rev+tailsign_always → rev {abs(raw)} → {_rev_str(str(abs(raw)))}+'{op_char}' → {final}")
    return lines


# ---------------------------------------------------------------------------
# Core scan: find ordering × operation × style for one operator
# ---------------------------------------------------------------------------

ORDERINGS = [
    ("AB,CD", "AB_CD"),
    ("BA,DC", "BA_DC"),
    ("AB,DC", "AB_DC"),
    ("BA,CD", "BA_CD"),
]

MAX_SCAN_REJECT_LINES = 18


def _normalize_scan_line(line: str) -> str:
    """Use compact lint-friendly verdict marks in scan traces."""
    line = line.replace("Ex1 match", "Ex1 candidate")
    line = line.replace("→ match! verify:", "→ ✓; verify:")
    line = line.replace("→ match!", "→ ✓")
    line = line.replace("→ match", "→ ✓")
    line = line.replace("→ no", "→ ✗")
    return line


def _compact_scan_lines(raw_lines: list[str]) -> list[str]:
    """Keep the scan executable but bound the dead-branch token cost."""
    compact = []
    reject_count = 0
    hidden_rejects = 0

    for line in raw_lines:
        is_reject = "→ no" in line
        normalized = _normalize_scan_line(line)
        if not is_reject:
            if hidden_rejects and "→ ✓" in normalized:
                compact.append(
                    f"    ... {hidden_rejects} rejected candidates omitted; continued same scan order."
                )
                hidden_rejects = 0
            compact.append(normalized)
            continue

        reject_count += 1
        if reject_count <= MAX_SCAN_REJECT_LINES:
            compact.append(normalized)
        else:
            hidden_rejects += 1

    if hidden_rejects:
        compact.append(
            f"    ... {hidden_rejects} rejected candidates omitted; no lock in this scan."
        )
    return compact


def _ordered_strings(a_s: str, b_s: str, order_key: str) -> tuple[str, str]:
    """Return operand strings in one of AB_CD, BA_DC, AB_DC, BA_CD."""
    if order_key == "BA_DC":
        return a_s[::-1], b_s[::-1]
    if order_key == "AB_DC":
        return a_s, b_s[::-1]
    if order_key == "BA_CD":
        return a_s[::-1], b_s
    return a_s, b_s


def _ordered_ints(a_s: str, b_s: str, order_key: str) -> tuple[int, int]:
    left, right = _ordered_strings(a_s, b_s, order_key)
    return int(left), int(right)


def _evaluate_rule(a_s: str, b_s: str, op_char: str, rule: tuple) -> tuple[int, int, int, str] | None:
    """Apply a locked rule to one visible row."""
    _ord_label, order_key, op_name, style = rule
    left, right = _ordered_ints(a_s, b_s, order_key)
    raw = _compute_op(op_name, left, right)
    if raw is None:
        return None
    formatted = _apply_style(raw, style, op_char)
    return left, right, raw, formatted


def _first_full_support_rule(group: list[tuple[str, str, str]], op_char: str,
                             prefer_order: Optional[str] = None) -> Optional[tuple]:
    """Return the first rule in scan order that replays every support row.

    This is the stable-program rule: do not lock from Ex1/Ex2 only, and do
    not search for a later gold-matching rule if the first full witness is
    not the intended query answer.
    """
    if not group:
        return None

    orderings = list(ORDERINGS)
    if prefer_order:
        orderings.sort(key=lambda item: item[1] != prefer_order)

    for ord_label, order_key in orderings:
        for ops_list in (COMMON_OPS, RARE_OPS):
            for op_name, _op_fn in ops_list:
                first_left, first_right = _ordered_ints(group[0][0], group[0][1], order_key)
                first_raw = _compute_op(op_name, first_left, first_right)
                if first_raw is None:
                    continue
                styles = _detect_styles(first_raw, group[0][2], op_char)
                if not styles:
                    continue
                for style in styles:
                    rule = (ord_label, order_key, op_name, style)
                    ok = True
                    for a_s, b_s, expected in group:
                        evaluated = _evaluate_rule(a_s, b_s, op_char, rule)
                        if evaluated is None or evaluated[3] != expected:
                            ok = False
                            break
                    if ok:
                        return rule
    return None


def _render_rule_card(group: list[tuple[str, str, str]], op_char: str, rule: tuple,
                      max_examples: int = 2) -> list[str]:
    """Render a compact executable support replay for a locked rule."""
    ord_label, _order_key, op_name, style = rule
    lines = [
        f"Scan[{op_char}]:",
        f"  locked = {ord_label}|{op_name}|{style}",
        f"Lock[{op_char}]: {ord_label}|{op_name}|{style}",
        f"Format[{op_char}]: {style}",
        "Verify:",
        f"  support_pass = {len(group)}/{len(group)}",
    ]
    for idx, (a_s, b_s, expected) in enumerate(group[:max_examples], start=1):
        evaluated = _evaluate_rule(a_s, b_s, op_char, rule)
        if evaluated is None:
            continue
        left, right, raw, formatted = evaluated
        lines.extend([
            f"  Ex{idx}:",
            f"    input = {a_s}{op_char}{b_s} -> {expected}",
            f"    L = {left}",
            f"    R = {right}",
            f"    value = {_show_op(op_name, left, right)}",
            f"    formatted = {formatted}",
            f"    expected = {expected}",
            "    match = PASS",
        ])
    return lines


def _choose_context_rule(by_op: dict[str, list[tuple[str, str, str]]],
                         query_op: str) -> tuple[str, list[tuple[str, str, str]], tuple] | None:
    """Pick a visible non-query operator to provide order/style context."""
    candidates = []
    for op_char, group in by_op.items():
        if op_char == query_op or len(group) < 2:
            continue
        rule = _first_full_support_rule(group, op_char)
        if rule is not None:
            candidates.append((op_char, group, rule))
    if not candidates:
        return None
    # Most support first, then deterministic operator order.
    candidates.sort(key=lambda item: (-len(item[1]), item[0]))
    return candidates[0]


def _select_one_shot_operation(single_group: list[tuple[str, str, str]],
                               op_char: str,
                               order_key: str,
                               style: str) -> tuple[str, int, int, int, str] | None:
    """Choose the first operation matching the single query-op support row."""
    if len(single_group) != 1:
        return None
    a_s, b_s, expected = single_group[0]
    left, right = _ordered_ints(a_s, b_s, order_key)
    for op_name, _fn in ALL_OPS:
        raw = _compute_op(op_name, left, right)
        if raw is None:
            continue
        formatted = _apply_style(raw, style, op_char)
        if formatted == expected:
            return op_name, left, right, raw, formatted
    return None


def _render_one_shot_trace(header: str,
                           detect_line: str,
                           query_str: str,
                           query_decoded: str,
                           query_op: str,
                           support_ops: dict[str, int],
                           context_op: str,
                           context_group: list[tuple[str, str, str]],
                           context_rule: tuple,
                           witness_group: list[tuple[str, str, str]],
                           chosen_op: str,
                           query_left: int,
                           query_right: int,
                           query_raw: int,
                           final_value: str,
                           answer_str: str,
                           encode_line: str | None = None,
                           surface_kind: str = "numeric_visible",
                           surface_extra: list[str] | None = None,
                           pre_program_lines: list[str] | None = None) -> str:
    """Render the one-shot program body."""
    ord_label, order_key, _context_op_name, style = context_rule
    w_a, w_b, w_expected = witness_group[0]
    w_left, w_right = _ordered_ints(w_a, w_b, order_key)
    w_raw = _compute_op(chosen_op, w_left, w_right)
    support_ops_text = ", ".join(f"{op}:{count}" for op, count in sorted(support_ops.items()))
    lines = [
        "TRANS_ROUTE_V1",
        "",
        "Flow:",
        "  parse support examples and query",
        "  classify surface",
    ]
    if surface_kind == "cipher_digit":
        lines.extend([
            "  mark cipher digit slots and operator slot",
            "  run CIPHER_MAP_V1 before numeric execution",
        ])
    lines.extend([
        "  extract query_op",
        "  count support examples using query_op",
        "  exit to the first matching route",
        "",
        "Surface:",
        f"  kind = {surface_kind}",
        f"  query = {query_str}",
    ])
    if surface_extra:
        lines.extend(surface_extra)
    lines.extend([
        f"  query_op = {query_op}",
        f"  support_ops = {support_ops_text}",
        "  query_op_support = 1",
        "",
        "Route:",
        "  program = TRANS_ONE_SHOT_V1",
        "  reason = query operator has exactly one direct support row",
        "  route_check = query_op_support:1 -> TRANS_ONE_SHOT_V1",
        "",
    ])
    if pre_program_lines:
        lines.extend(pre_program_lines)
        lines.append("")
    lines.extend([
        header,
        "TRANS_ONE_SHOT_V1",
        detect_line,
        "",
        "ProgramOrder:",
        "  1. from Route.Surface, choose the non-query context operator with the most support",
        "  2. from Step 1 context_op, lock context order and format by full support replay",
        "  3. from Step 2 Lock, keep order/format fixed",
        "  4. from Step 3 fixed order/format, choose operation from the single query-op witness",
        "  5. from Step 4 chosen operation, apply fixed order/operation/format to query",
        "",
        "Context:",
        f"  context_op = {context_op}",
        f"  context_support = {len(context_group)}",
    ])
    lines.extend("  " + line for line in _render_rule_card(context_group, context_op, context_rule))
    lines.extend([
        "",
        "QueryOpWitness:",
        f"  input = {w_a}{query_op}{w_b} -> {w_expected}",
        f"  fixed_order = {ord_label}",
        f"  fixed_format = {style}",
        f"  L = {w_left}",
        f"  R = {w_right}",
        f"  operation = {_show_op(chosen_op, w_left, w_right)}",
        f"  formatted = {_apply_style(w_raw, style, query_op) if w_raw is not None else 'ERR'}",
        "  match = PASS",
        "",
        "Apply:",
        f"  query = {query_str}" + (f" -> {query_decoded}" if query_decoded != query_str else ""),
        f"  ordering = {ord_label} -> L={query_left} R={query_right}",
        f"  operation = {_show_op(chosen_op, query_left, query_right)}",
    ])
    lines.extend(_show_style_step(query_raw, style, query_op))
    lines.append(f"  formatted = {final_value}")
    if encode_line:
        lines.append(f"  {encode_line}")
    lines.append(f"Answer: {answer_str}")
    return "\n".join(lines)


def _scan_operator(group: list[tuple[str, str, str]], op_char: str,
                   prefer_order: Optional[str] = None) -> tuple[Optional[tuple], list[str]]:
    """Scan all combos for one operator.

    group: [(a_str, b_str, output_str), ...]
    Returns: (found_rule, trace_lines)
    found_rule = (ord_label, order_key, op_name, style) or None
    """
    lines = []
    found = None
    ex1_a, ex1_b, ex1_out = group[0]
    ex2 = group[1] if len(group) > 1 else None

    ex_strs = [f"{a}{op_char}{b}={out}" for a, b, out in group]
    lines.append(f"  Examples: {', '.join(ex_strs)}")

    orderings = list(ORDERINGS)
    if prefer_order:
        orderings.sort(key=lambda item: item[1] != prefer_order)

    # String-based concat (preserves leading zeros like "08").
    # Test before int-based ops since concat doesn't need int conversion.
    for ord_label, order_key in orderings:
        if found:
            break
        a_s, b_s = _ordered_strings(ex1_a, ex1_b, order_key)
        for cname, cfn in [("concat", lambda a, b: a + b), ("bconcat", lambda a, b: b + a)]:
            result_s = cfn(a_s, b_s)
            if result_s == ex1_out:
                if ex2:
                    a2_s, b2_s = _ordered_strings(ex2[0], ex2[1], order_key)
                    r2 = cfn(a2_s, b2_s)
                    if r2 == ex2[2]:
                        lines.append(f"  {ord_label} {cname}: {a_s}||{b_s}={result_s} → match! verify: {a2_s}||{b2_s}={r2} → match!")
                        found = (ord_label, order_key, cname, "plain")
                        break
                else:
                    lines.append(f"  {ord_label} {cname}: {a_s}||{b_s}={result_s} → match!")
                    found = (ord_label, order_key, cname, "plain")
                    break

    for ord_label, order_key in orderings:
        if found:
            break

        a_int, b_int = _ordered_ints(ex1_a, ex1_b, order_key)
        lines.append(f"  {ord_label}: L={a_int} R={b_int}")

        for ops_list in [COMMON_OPS, RARE_OPS]:
            if found:
                break
            for op_name, op_fn in ops_list:
                if found:
                    break
                try:
                    raw = op_fn(a_int, b_int)
                except:
                    continue
                raw_s = str(raw)
                rev_s = _rev_str(raw_s)
                arith = _show_op(op_name, a_int, b_int)

                # Test all styles against expected
                for style_name, test_val in [("plain", raw_s), ("rev", rev_s)]:
                    if test_val == ex1_out:
                        if ex2:
                            a2, b2 = _ordered_ints(ex2[0], ex2[1], order_key)
                            try:
                                raw2 = op_fn(a2, b2)
                            except:
                                continue
                            test_val2 = str(raw2) if style_name == "plain" else _rev_str(str(raw2))
                            if test_val2 == ex2[2]:
                                sfx = f", rev {rev_s}" if style_name == "rev" else ""
                                lines.append(f"    {arith}{sfx} = {ex1_out} → match! verify: {test_val2} = {ex2[2]} → match!")
                                found = (ord_label, order_key, op_name, style_name)
                                break
                            else:
                                sfx = f", rev {rev_s}" if style_name == "rev" else ""
                                lines.append(f"    {arith}{sfx} = {ex1_out} → Ex1 match, Ex2: {test_val2} ≠ {ex2[2]} → no")
                        else:
                            sfx = f", rev {rev_s}" if style_name == "rev" else ""
                            lines.append(f"    {arith}{sfx} = {ex1_out} → match!")
                            found = (ord_label, order_key, op_name, style_name)
                            break
                        continue

                # Check other styles (opsign, tailsign, etc.)
                style = _detect_style(raw, ex1_out, op_char)
                if style and style not in ("plain", "rev") and not found:
                    styled = _apply_style(raw, style, op_char)
                    if styled == ex1_out:
                        if ex2:
                            a2, b2 = _ordered_ints(ex2[0], ex2[1], order_key)
                            try:
                                raw2 = op_fn(a2, b2)
                            except:
                                continue
                            styled2 = _apply_style(raw2, style, op_char)
                            if styled2 == ex2[2]:
                                lines.append(f"    {arith}, {style} → {styled} = {ex1_out} → match! verify: {styled2} = {ex2[2]} → match!")
                                found = (ord_label, order_key, op_name, style)
                            else:
                                lines.append(f"    {arith}, {style} → {styled} = {ex1_out} → Ex1 match, Ex2: {styled2} ≠ {ex2[2]} → no")
                        else:
                            lines.append(f"    {arith}, {style} → {styled} = {ex1_out} → match!")
                            found = (ord_label, order_key, op_name, style)
                        continue

                # Rejection line (compact)
                if not found:
                    if rev_s != raw_s:
                        lines.append(f"    {arith}, rev {rev_s} → no")
                    else:
                        lines.append(f"    {arith} → no")

    return found, _compact_scan_lines(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_transform_trace(examples: list[tuple[str, str]],
                          query_str: str,
                          answer_str: str) -> tuple[str, str] | None:
    """Build the oversolve transformation trace.

    examples: list of (input_expr, output_value) — e.g. ("34+56", "201")
    query_str: the query expression — e.g. "85/77"
    answer_str: known correct answer from solver

    Returns (reasoning_text, answer_str) or None.
    """
    if not examples or not query_str:
        return None

    # Group examples by operator
    by_op = {}
    for inp, out in examples:
        m = re.match(r"^(\d+)(\D)(\d+)$", inp)
        if not m:
            continue
        a_s, op_char, b_s = m.group(1), m.group(2), m.group(3)
        by_op.setdefault(op_char, []).append((a_s, b_s, out))

    # Parse query
    qm = re.match(r"^(\d+)(\D)(\d+)$", query_str)
    if not qm:
        return None
    q_a, q_op, q_b = qm.group(1), qm.group(2), qm.group(3)

    ops = sorted(by_op.keys())
    support_ops_text = ", ".join(f"{op}:{len(by_op[op])}" for op in ops) or "none"
    q_support = len(by_op.get(q_op, []))
    lines = [
        "TRANS_ROUTE_V1",
        "",
        "Flow:",
        "  parse support examples and query",
        "  classify surface",
        "  extract query_op",
        "  count support examples using query_op",
        "  exit to the first matching route",
        "",
        "Surface:",
        "  kind = numeric_visible",
        f"  query = {query_str}",
        f"  query_op = {q_op}",
        f"  support_ops = {support_ops_text}",
        f"  query_op_support = {q_support}",
        "",
        "Route:",
        "  program = TRANS_MULTI_SUPPORT_V1",
        "  reason = query operator has at least two direct support rows",
        f"  route_check = query_op_support:{q_support} -> TRANS_MULTI_SUPPORT_V1",
        "",
        "Numeric visible.",
        f"Detect: query_op=[{q_op}], operators {ops}, {len(examples)} examples.",
    ]
    if q_op not in by_op:
        return None
    if len(ops) > 1:
        lines.append("Plan: scan the query operator first; skip support-only operators.")
    lines.append("")

    group = by_op[q_op]
    if len(group) < 2:
        return None
    rule = _first_full_support_rule(group, q_op)
    if rule is None:
        return None
    lines.extend(_render_rule_card(group, q_op, rule))
    lines.append("")

    # Query application
    lines.append("Apply:")
    lines.append(f"  query = {query_str}")
    computed_answer = None

    if rule:
        ord_label, order_key, op_name, style = rule
        q_a_int, q_b_int = _ordered_ints(q_a, q_b, order_key)

        lines.append(f"  ordering = {ord_label} -> L={q_a_int} R={q_b_int}")
        raw = _compute_op(op_name, q_a_int, q_b_int)
        if raw is None:
            return None
        lines.append(f"  operation = {_show_op(op_name, q_a_int, q_b_int)}")
        style_lines = _show_style_step(raw, style, q_op)
        lines.extend(style_lines)
        computed_answer = _apply_style(raw, style, q_op)
        lines.append(f"  formatted = {computed_answer}")

    if computed_answer != answer_str:
        return None

    lines.append(f"Answer: {answer_str}")

    reasoning = "\n".join(lines)
    return reasoning, answer_str


# ---------------------------------------------------------------------------
# Compatibility stubs (old micro-skills import these)
# ---------------------------------------------------------------------------

COMBO_DISPLAY = {
    "AB_CD": "AB,CD", "BA_DC": "BA,DC", "AB,CD": "AB,CD", "BA,DC": "BA,DC",
    "AB_DC": "AB,DC", "BA_CD": "BA,CD",
}

SCAN_ORDER = [key for _, key in ORDERINGS]


def build_numeric_trace(examples: list[tuple[str, str]],
                        query_str: str,
                        answer_str: str,
                        rng=None) -> tuple[str, str] | None:
    """Compatibility wrapper; rng is accepted for old generator call sites."""
    return build_transform_trace(examples, query_str, answer_str)


def build_numeric_one_shot_trace(examples: list[tuple[str, str]],
                                 query_str: str,
                                 answer_str: str,
                                 rng=None) -> tuple[str, str] | None:
    """Build a visible one-shot trace for numeric transformation rows.

    Preconditions:
      - query operator appears exactly once in support
      - another visible operator has >=2 support rows to provide order/style
      - the one-shot program computes the stored answer
    """
    if not examples or not query_str or answer_str is None:
        return None

    by_op = {}
    for inp, out in examples:
        m = re.match(r"^(\d+)(\D)(\d+)$", inp)
        if not m:
            return None
        a_s, op_char, b_s = m.group(1), m.group(2), m.group(3)
        by_op.setdefault(op_char, []).append((a_s, b_s, out))

    qm = re.match(r"^(\d+)(\D)(\d+)$", query_str)
    if not qm:
        return None
    q_a, q_op, q_b = qm.group(1), qm.group(2), qm.group(3)
    witness_group = by_op.get(q_op, [])
    if len(witness_group) != 1:
        return None

    context = _choose_context_rule(by_op, q_op)
    if context is None:
        return None
    context_op, context_group, context_rule = context
    ord_label, order_key, _context_op_name, style = context_rule

    chosen = _select_one_shot_operation(witness_group, q_op, order_key, style)
    if chosen is None:
        return None
    chosen_op, _w_left, _w_right, _w_raw, _w_formatted = chosen

    q_left, q_right = _ordered_ints(q_a, q_b, order_key)
    q_raw = _compute_op(chosen_op, q_left, q_right)
    if q_raw is None:
        return None
    final_value = _apply_style(q_raw, style, q_op)
    if final_value != answer_str:
        return None

    ops = sorted(by_op.keys())
    detect = f"Detect: query_op=[{q_op}], operators {ops}, {len(examples)} examples."
    trace = _render_one_shot_trace(
        "Numeric visible.",
        detect,
        query_str,
        query_str,
        q_op,
        {op: len(group) for op, group in by_op.items()},
        context_op,
        context_group,
        context_rule,
        witness_group,
        chosen_op,
        q_left,
        q_right,
        q_raw,
        final_value,
        answer_str,
    )
    return trace, answer_str


def _encode_with_inverse_map(value: str, inv_map: dict[int, str]) -> str | None:
    """Encode a formatted digit string with a visible inverse map.

    Non-digits such as '-' or operator prefixes are preserved. Missing digit
    symbols are not invented here; fresh binding would be a hidden prior.
    """
    out = []
    for ch in str(value):
        if ch.isdigit():
            sym = inv_map.get(int(ch))
            if sym is None:
                return None
            out.append(sym)
        else:
            out.append(ch)
    return "".join(out)


def build_cipher_one_shot_trace(examples_raw: list[tuple[str, str]],
                                query_raw: str,
                                answer_str: str,
                                mapping: dict[str, int],
                                op_pos: int = 2) -> tuple[str, str] | None:
    """Build a visible one-shot trace for cipher-digit transformation rows.

    The mapping must come from a visible-only solver. This function only renders
    and verifies the route -> map -> one-shot execution chain.
    """
    if not examples_raw or not query_raw or answer_str is None or not mapping:
        return None
    if len(query_raw) <= op_pos:
        return None
    digit_pos = [i for i in range(5) if i != op_pos]
    if len(digit_pos) != 4:
        return None

    inv_map = {v: k for k, v in mapping.items()}

    def decode_sym(text: str) -> str:
        return "".join(str(mapping.get(ch, ch)) for ch in text)

    by_op: dict[str, list[tuple[str, str, str]]] = {}
    decode_lines = []
    support_ops: dict[str, int] = {}
    for inp_raw, out_raw in examples_raw:
        if len(inp_raw) <= op_pos:
            return None
        op_char = inp_raw[op_pos]
        support_ops[op_char] = support_ops.get(op_char, 0) + 1
        try:
            left_digits = [mapping[inp_raw[digit_pos[0]]], mapping[inp_raw[digit_pos[1]]]]
            right_digits = [mapping[inp_raw[digit_pos[2]]], mapping[inp_raw[digit_pos[3]]]]
        except KeyError:
            return None
        a_s = str(left_digits[0]) + str(left_digits[1])
        b_s = str(right_digits[0]) + str(right_digits[1])
        out_digits = decode_sym(out_raw)
        by_op.setdefault(op_char, []).append((a_s, b_s, out_digits))
        decode_lines.append(f"  {inp_raw} = {out_raw} -> {a_s}{op_char}{b_s} = {out_digits}")

    q_op = query_raw[op_pos]
    witness_group = by_op.get(q_op, [])
    if len(witness_group) != 1:
        return None
    try:
        q_a = str(mapping[query_raw[digit_pos[0]]]) + str(mapping[query_raw[digit_pos[1]]])
        q_b = str(mapping[query_raw[digit_pos[2]]]) + str(mapping[query_raw[digit_pos[3]]])
    except KeyError:
        return None

    context = _choose_context_rule(by_op, q_op)
    if context is None:
        return None
    context_op, context_group, context_rule = context
    _ord_label, order_key, _context_op_name, style = context_rule

    chosen = _select_one_shot_operation(witness_group, q_op, order_key, style)
    if chosen is None:
        return None
    chosen_op, _w_left, _w_right, _w_raw, _w_formatted = chosen

    q_left, q_right = _ordered_ints(q_a, q_b, order_key)
    q_raw = _compute_op(chosen_op, q_left, q_right)
    if q_raw is None:
        return None
    final_digits = _apply_style(q_raw, style, q_op)
    encoded = _encode_with_inverse_map(final_digits, inv_map)
    if encoded is None or encoded != answer_str:
        return None

    ops = sorted(by_op.keys())
    sorted_map = sorted(mapping.items(), key=lambda item: item[1])
    map_str = " ".join(f"{sym}={digit}" for sym, digit in sorted_map)
    pre_program_lines = [
        "CIPHER_MAP_V1",
        "MapOrder:",
        "  1. from Route.Surface, use op_pos to separate operator symbols from digit symbols",
        "  2. from digit_slots, collect all symbols that must map to digits",
        "  3. enumerate bijective symbol->digit mappings",
        "  4. decode support rows under each candidate mapping",
        "  5. replay decoded support rows and keep only support-passing mappings",
        "  6. use the kept mapping to decode the query; do not use the stored answer",
        "",
        "Mapping:",
        f"  {map_str}",
        "",
        "Decode examples:",
        *decode_lines,
        "",
        f"Query: {query_raw} -> {q_a}{q_op}{q_b}",
    ]
    detect = f"Detect: query_op=[{q_op}], operators {ops}, {len(examples_raw)} examples."
    trace = _render_one_shot_trace(
        "Cipher-digit.",
        detect,
        query_raw,
        f"{q_a}{q_op}{q_b}",
        q_op,
        support_ops,
        context_op,
        context_group,
        context_rule,
        witness_group,
        chosen_op,
        q_left,
        q_right,
        q_raw,
        final_digits,
        answer_str,
        encode_line=f"encode = {final_digits} -> {encoded}",
        surface_kind="cipher_digit",
        surface_extra=[
            f"  op_pos = {op_pos}",
            f"  digit_slots = {digit_pos}",
        ],
        pre_program_lines=pre_program_lines,
    )
    return trace, answer_str


def _normalize_cipher_combo(combo: tuple) -> tuple[str, str, str, str] | None:
    """Convert solver cipher combo names to trace_transform rule names."""
    if not combo or len(combo) != 3:
        return None
    order_key, op_name, style = combo
    order_map = {
        "AB_CD": ("AB,CD", "AB_CD"),
        "BA_DC": ("BA,DC", "BA_DC"),
        "AB_DC": ("AB,DC", "AB_DC"),
        "BA_CD": ("BA,CD", "BA_CD"),
        "AB,CD": ("AB,CD", "AB_CD"),
        "BA,DC": ("BA,DC", "BA_DC"),
        "AB,DC": ("AB,DC", "AB_DC"),
        "BA,CD": ("BA,CD", "BA_CD"),
    }
    op_map = {
        "rsub": "bsub",
        "cat": "concat",
        "rcat": "bconcat",
        "add1": "addp1",
        "addm1": "addm1",
        "muladd1": "muladd1",
        "mulsub1": "mulsub1",
    }
    style_map = {"raw": "plain"}
    order = order_map.get(order_key)
    if order is None:
        return None
    return order[0], order[1], op_map.get(op_name, op_name), style_map.get(style, style)


def build_cipher_best_fit_trace(examples_raw: list[tuple[str, str]],
                                query_raw: str,
                                answer_str: str,
                                mapping: dict[str, int],
                                combos: dict[str, tuple],
                                op_pos: int = 2) -> tuple[str, str] | None:
    """Build a labeled best-fit trace for cipher rows.

    This is intentionally not a uniqueness proof. It renders a deterministic
    route state, a visible symbol mapping, and the selected best-fit decoded
    rule. Rows are only emitted by builders after the selected rule predicts
    the stored answer and the final symbols are prompt-visible.
    """
    if not examples_raw or not query_raw or answer_str is None or not mapping or not combos:
        return None
    if len(query_raw) <= op_pos:
        return None

    digit_pos = [i for i in range(5) if i != op_pos]
    if len(digit_pos) != 4:
        return None
    inv_map = {v: k for k, v in mapping.items()}

    def decode_sym(text: str) -> str:
        return "".join(str(mapping.get(ch, ch)) for ch in text)

    by_op: dict[str, list[tuple[str, str, str, str, str]]] = {}
    support_ops: dict[str, int] = {}
    decode_lines: list[str] = []
    for inp_raw, out_raw in examples_raw:
        if len(inp_raw) <= op_pos:
            return None
        op_char = inp_raw[op_pos]
        support_ops[op_char] = support_ops.get(op_char, 0) + 1
        try:
            a_s = str(mapping[inp_raw[digit_pos[0]]]) + str(mapping[inp_raw[digit_pos[1]]])
            b_s = str(mapping[inp_raw[digit_pos[2]]]) + str(mapping[inp_raw[digit_pos[3]]])
        except KeyError:
            return None
        out_digits = decode_sym(out_raw)
        by_op.setdefault(op_char, []).append((a_s, b_s, out_digits, inp_raw, out_raw))
        decode_lines.append(f"  {inp_raw} = {out_raw} -> {a_s}{op_char}{b_s} = {out_digits}")

    q_op = query_raw[op_pos]
    try:
        q_a = str(mapping[query_raw[digit_pos[0]]]) + str(mapping[query_raw[digit_pos[1]]])
        q_b = str(mapping[query_raw[digit_pos[2]]]) + str(mapping[query_raw[digit_pos[3]]])
    except KeyError:
        return None

    combo = _normalize_cipher_combo(combos.get(q_op))
    if combo is None:
        return None
    ord_label, order_key, op_name, style = combo
    q_left, q_right = _ordered_ints(q_a, q_b, order_key)
    q_raw = _compute_op(op_name, q_left, q_right)
    if q_raw is None:
        return None
    final_digits = _apply_style(q_raw, style, q_op)
    encoded = _encode_with_inverse_map(final_digits, inv_map)
    if encoded is None or encoded != answer_str:
        return None

    compatible = 0
    checked = 0
    first_examples: list[str] = []
    for op_char, group in sorted(by_op.items()):
        norm = _normalize_cipher_combo(combos.get(op_char))
        if norm is None:
            continue
        _ol, ok, on, st = norm
        for a_s, b_s, expected, inp_raw, out_raw in group:
            checked += 1
            evaluated = _evaluate_rule(a_s, b_s, op_char, (_ol, ok, on, st))
            got = evaluated[3] if evaluated else "ERR"
            if got == expected:
                compatible += 1
            if len(first_examples) < 3:
                mark = "PASS" if got == expected else "FAIL"
                first_examples.append(
                    f"  {inp_raw}: decoded {a_s}{op_char}{b_s}; {on}/{st} -> {got}; expected {expected}; {mark}"
                )

    support_ops_text = ", ".join(f"{op}:{count}" for op, count in sorted(support_ops.items())) or "none"
    sorted_map = sorted(mapping.items(), key=lambda item: item[1])
    map_str = " ".join(f"{sym}={digit}" for sym, digit in sorted_map)

    lines = [
        "TRANS_ROUTE_V1",
        "",
        "Flow:",
        "  parse support examples and query",
        "  classify surface",
        "  mark cipher digit slots and operator slot",
        "  extract query_op",
        "  count support examples using query_op",
        "  exit to the first matching route",
        "",
        "Surface:",
        "  kind = cipher_digit",
        f"  query = {query_raw}",
        f"  op_pos = {op_pos}",
        f"  digit_slots = {digit_pos}",
        f"  query_op = {q_op}",
        f"  support_ops = {support_ops_text}",
        f"  query_op_support = {support_ops.get(q_op, 0)}",
        "",
        "Route:",
        "  program = Best-Fitting Cipher Rule Fallback",
        "  reason = visible cipher map/rule is not unique enough for a verified card",
        f"  route_check = query_op_support:{support_ops.get(q_op, 0)} -> Best-Fitting Cipher Rule Fallback",
        "",
        "CIPHER_MAP_V1",
        "MapOrder:",
        "  1. from Route.Surface, use op_pos to separate operator symbols from digit symbols",
        "  2. from digit_slots, collect all symbols that must map to digits",
        "  3. enumerate bijective symbol->digit mappings",
        "  4. decode support rows under each candidate mapping",
        "  5. keep the route-consistent best-fit mapping",
        "  6. decode the query and execute the selected pattern",
        "",
        "Mapping:",
        f"  {map_str}",
        "",
        "Decode examples:",
        *decode_lines,
        "",
        f"Query: {query_raw} -> {q_a}{q_op}{q_b}",
        "",
        "Best-Fitting Cipher Rule Fallback",
        "WhenToUse:",
        "  cipher examples do not prove a unique mapping/rule",
        "  use the route's familiar cipher pattern without claiming uniqueness",
        "PatternCue:",
        "  think in four slots: symbol map, operand order, arithmetic operation, output format",
        "  prefer the route-consistent simple decoded pattern",
        "  do not enumerate hidden candidates",
        "RouteFacts:",
        f"  source route: cipher query_op_support={support_ops.get(q_op, 0)}",
        f"  decoded query: {q_a}{q_op}{q_b}",
        f"  selected rule: {ord_label}|{op_name}|{style}",
        "BestFitRule:",
        f"  mapping = {map_str}",
        f"  rule = {ord_label}|{op_name}|{style}",
        f"  compatible replay = {compatible}/{checked}",
        *first_examples,
        "SelfCheck:",
        "  proof status: not unique",
        "  certainty: best-fit fallback",
        "  no support uniqueness claimed",
        "Apply:",
        f"  query = {query_raw} -> {q_a}{q_op}{q_b}",
        f"  ordering = {ord_label} -> L={q_left} R={q_right}",
        f"  operation = {_show_op(op_name, q_left, q_right)}",
    ]
    lines.extend(_show_style_step(q_raw, style, q_op))
    lines.extend([
        f"  formatted digits = {final_digits}",
        f"  encode = {final_digits} -> {encoded}",
        f"Answer: {answer_str}",
    ])

    return "\n".join(lines), answer_str


def build_cipher_missing_symbol_trace(examples_raw: list[tuple[str, str]],
                                      query_raw: str,
                                      answer_str: str,
                                      mapping: dict[str, int],
                                      combos: dict[str, tuple],
                                      op_pos: int = 2) -> tuple[str, str] | None:
    """Build a prior-completion trace for cipher answers with unseen symbols.

    The missing symbol is not visible-proof. The trace gives the model decoded
    state first, then performs one explicitly labeled prior completion step.
    """
    if not examples_raw or not query_raw or answer_str is None or not mapping or not combos:
        return None
    if len(query_raw) <= op_pos:
        return None

    digit_pos = [i for i in range(5) if i != op_pos]
    if len(digit_pos) != 4:
        return None

    visible_symbols: set[str] = set()
    for inp_raw, out_raw in examples_raw:
        visible_symbols.update(inp_raw)
        visible_symbols.update(out_raw)
    visible_symbols.update(query_raw)
    visible_mapping = {sym: digit for sym, digit in mapping.items() if sym in visible_symbols}
    visible_inv = {digit: sym for sym, digit in visible_mapping.items()}

    def decode_sym(text: str) -> str:
        return "".join(str(mapping.get(ch, ch)) for ch in text)

    by_op: dict[str, list[tuple[str, str, str, str, str]]] = {}
    support_ops: dict[str, int] = {}
    decode_lines: list[str] = []
    for inp_raw, out_raw in examples_raw:
        if len(inp_raw) <= op_pos:
            return None
        op_char = inp_raw[op_pos]
        support_ops[op_char] = support_ops.get(op_char, 0) + 1
        try:
            a_s = str(mapping[inp_raw[digit_pos[0]]]) + str(mapping[inp_raw[digit_pos[1]]])
            b_s = str(mapping[inp_raw[digit_pos[2]]]) + str(mapping[inp_raw[digit_pos[3]]])
        except KeyError:
            return None
        out_digits = decode_sym(out_raw)
        by_op.setdefault(op_char, []).append((a_s, b_s, out_digits, inp_raw, out_raw))
        decode_lines.append(f"  {inp_raw} = {out_raw} -> {a_s}{op_char}{b_s} = {out_digits}")

    q_op = query_raw[op_pos]
    try:
        q_a = str(mapping[query_raw[digit_pos[0]]]) + str(mapping[query_raw[digit_pos[1]]])
        q_b = str(mapping[query_raw[digit_pos[2]]]) + str(mapping[query_raw[digit_pos[3]]])
    except KeyError:
        return None

    combo = _normalize_cipher_combo(combos.get(q_op))
    if combo is None:
        return None
    ord_label, order_key, op_name, style = combo
    q_left, q_right = _ordered_ints(q_a, q_b, order_key)
    q_raw = _compute_op(op_name, q_left, q_right)
    if q_raw is None:
        return None
    final_digits = _apply_style(q_raw, style, q_op)
    if len(final_digits) != len(answer_str):
        return None

    encoded_chars: list[str] = []
    prior_bindings: list[tuple[str, int, str, int]] = []
    binding_by_digit: dict[int, str] = {}
    for idx, (digit_ch, ans_ch) in enumerate(zip(final_digits, answer_str), start=1):
        if digit_ch == "-":
            if ans_ch != "-":
                return None
            encoded_chars.append(ans_ch)
            continue
        if not digit_ch.isdigit():
            if digit_ch != ans_ch:
                return None
            encoded_chars.append(ans_ch)
            continue
        digit = int(digit_ch)
        visible_sym = visible_inv.get(digit)
        if visible_sym is not None:
            if ans_ch != visible_sym:
                return None
            encoded_chars.append(ans_ch)
            continue
        if mapping.get(ans_ch) != digit:
            return None
        if ans_ch in visible_symbols:
            return None
        previous = binding_by_digit.get(digit)
        if previous is not None and previous != ans_ch:
            return None
        binding_by_digit[digit] = ans_ch
        prior_bindings.append((digit_ch, digit, ans_ch, idx))
        encoded_chars.append(ans_ch)

    encoded = "".join(encoded_chars)
    if encoded != answer_str or not prior_bindings:
        return None

    compatible = 0
    checked = 0
    first_examples: list[str] = []
    for op_char, group in sorted(by_op.items()):
        norm = _normalize_cipher_combo(combos.get(op_char))
        if norm is None:
            continue
        _ol, ok, on, st = norm
        for a_s, b_s, expected, inp_raw, _out_raw in group:
            checked += 1
            evaluated = _evaluate_rule(a_s, b_s, op_char, (_ol, ok, on, st))
            got = evaluated[3] if evaluated else "ERR"
            if got == expected:
                compatible += 1
            if len(first_examples) < 3:
                mark = "PASS" if got == expected else "FAIL"
                first_examples.append(
                    f"  {inp_raw}: decoded {a_s}{op_char}{b_s}; {on}/{st} -> {got}; expected {expected}; {mark}"
                )

    support_ops_text = ", ".join(f"{op}:{count}" for op, count in sorted(support_ops.items())) or "none"
    visible_map_str = " ".join(
        f"{sym}={digit}" for sym, digit in sorted(visible_mapping.items(), key=lambda item: item[1])
    )
    missing_digit_str = ", ".join(str(d) for d in sorted(binding_by_digit))
    prior_lines = [
        f"  position {pos}: digit {digit} has no visible symbol -> choose {sym}"
        for _digit_ch, digit, sym, pos in prior_bindings
    ]
    encode_lines = []
    for digit_ch, ans_ch in zip(final_digits, answer_str):
        if digit_ch.isdigit():
            digit = int(digit_ch)
            if digit in visible_inv:
                encode_lines.append(f"  {digit_ch} -> {ans_ch} (visible)")
            else:
                encode_lines.append(f"  {digit_ch} -> {ans_ch} (prior completion)")
        else:
            encode_lines.append(f"  {digit_ch} -> {ans_ch}")

    lines = [
        "TRANS_ROUTE_V1",
        "",
        "Flow:",
        "  parse support examples and query",
        "  classify surface",
        "  mark cipher digit slots and operator slot",
        "  extract query_op",
        "  count support examples using query_op",
        "  exit to the first matching route",
        "",
        "Surface:",
        "  kind = cipher_digit",
        f"  query = {query_raw}",
        f"  op_pos = {op_pos}",
        f"  digit_slots = {digit_pos}",
        f"  query_op = {q_op}",
        f"  support_ops = {support_ops_text}",
        f"  query_op_support = {support_ops.get(q_op, 0)}",
        "",
        "Route:",
        "  program = Missing Symbol Cipher Completion",
        "  reason = decoded answer needs a digit whose cipher symbol is not visible",
        f"  route_check = query_op_support:{support_ops.get(q_op, 0)} -> Missing Symbol Cipher Completion",
        "",
        "CIPHER_MAP_V1",
        "MapOrder:",
        "  1. from Route.Surface, use op_pos to separate operator symbols from digit symbols",
        "  2. from digit_slots, collect visible symbols that must map to digits",
        "  3. enumerate bijective symbol->digit mappings",
        "  4. decode support rows under each candidate mapping",
        "  5. keep the route-consistent best-fit mapping",
        "  6. decode query before any missing-symbol completion",
        "",
        "Mapping:",
        f"  known = {visible_map_str}",
        "",
        "Decode examples:",
        *decode_lines,
        "",
        f"Query: {query_raw} -> {q_a}{q_op}{q_b}",
        "",
        "Missing Symbol Cipher Completion",
        "WhenToUse:",
        "  decoded answer contains a digit whose symbol is absent from prompt",
        "  complete only the missing digit-symbol pair",
        "  given the constraints, use a heuristic prior for the missing cipher symbol",
        "PatternCue:",
        "  first build decoded state",
        "  then use cipher-alphabet prior for the unseen symbol",
        "  keep visible mappings fixed",
        "PriorFeatures:",
        f"  known digit count = {len(visible_inv)}",
        f"  missing digits = {missing_digit_str}",
        f"  output digits = {final_digits}",
        f"  selected rule = {ord_label}|{op_name}|{style}",
        f"  compatible replay = {compatible}/{checked}",
        *first_examples,
        "TeleportStep:",
        "  proof status: prior-based, not visible-proof",
        "  certainty: missing-symbol prior",
        *prior_lines,
        "Apply:",
        f"  query = {query_raw} -> {q_a}{q_op}{q_b}",
        f"  ordering = {ord_label} -> L={q_left} R={q_right}",
        f"  operation = {_show_op(op_name, q_left, q_right)}",
    ]
    lines.extend(_show_style_step(q_raw, style, q_op))
    lines.extend([
        f"  formatted digits = {final_digits}",
        "Encode:",
        *encode_lines,
        f"  encoded = {encoded}",
        f"Answer: {answer_str}",
    ])
    return "\n".join(lines), answer_str


def _make_operands(a0, a1, b0, b1, order):
    """Return (L, R) as integers given 4 digits and an ordering."""
    if order in ("BA_DC", "BA,DC"):
        return a1 * 10 + a0, b1 * 10 + b0
    if order in ("AB_DC", "AB,DC"):
        return a0 * 10 + a1, b1 * 10 + b0
    if order in ("BA_CD", "BA,CD"):
        return a1 * 10 + a0, b0 * 10 + b1
    return a0 * 10 + a1, b0 * 10 + b1

def _calc(L, R, op_name):
    """Compute an operation by name, return int. Old calling convention: (L, R, op)."""
    result = _compute_op(op_name, L, R)
    if result is not None:
        return result
    aliases = {"add1": "addp1", "sub1": "addm1", "mul1": "muladd1",
               "cat": "concat", "rcat": "bconcat", "bcat": "bconcat",
               "rsub": "bsub", "diff": "absdiff"}
    alt = aliases.get(op_name)
    if alt:
        return _compute_op(alt, L, R)
    return L + R

def _fmt(value, style, op_char=""):
    """Format a result with a style."""
    if value is None:
        return "0"
    return _apply_style(value, style, op_char)


# ---------------------------------------------------------------------------
# Cipher-digit trace builder
# ---------------------------------------------------------------------------

def build_cipher_trace(source, query_or_answer, answer_or_mapping=None,
                       mapping_or_base=None, combos=None, maj_op=None,
                       op_pos: int = 2) -> tuple[str, str] | None:
    """Build oversolve trace for cipher-digit puzzles.

    Phase 0: Show the symbol→digit mapping
    Phase 1-3: Decode to digits, run numeric oversolve scan
    Phase 4: Re-encode result to symbols

    Args:
        source: either raw puzzle prompt or list[(lhs, rhs)]
        answer_str: known correct answer (in symbols)
        mapping: {symbol: digit} bijection from solver
        base: numeric base (5-10)
        op_states: {op_char: (op_name, rev_in, rev_out)} from solver
    """
    # Support both historical call styles:
    #   build_cipher_trace(prompt, answer, mapping, base, op_states)
    #   build_cipher_trace(examples, query, answer, mapping, combos, maj_op, op_pos)
    if isinstance(source, list):
        examples_raw = source
        query_raw = query_or_answer
        answer_str = answer_or_mapping
        mapping = mapping_or_base
        base = max(mapping.values()) + 1 if mapping else 10
    else:
        prompt = source
        answer_str = query_or_answer
        mapping = answer_or_mapping
        base = mapping_or_base if isinstance(mapping_or_base, int) else (
            max(mapping.values()) + 1 if mapping else 10
        )
        examples_raw = []
        query_raw = None
        for line in prompt.split("\n"):
            line = line.strip()
            m = re.match(r"^(\S{5})\s*=\s*(\S+)$", line)
            if m:
                examples_raw.append((m.group(1), m.group(2)))
            # Parse query: split on "for: " not just ":"
            qm = re.search(r"determine the result for:\s*(\S+)", line, re.IGNORECASE)
            if qm:
                query_raw = qm.group(1)

    if not examples_raw or not query_raw or not answer_str or not mapping:
        return None

    # Build inverse mapping (digit → symbol)
    inv_map = {v: k for k, v in mapping.items()}

    raw_support_ops: dict[str, int] = {}
    if op_pos is not None:
        for inp_raw, _out_raw in examples_raw:
            if len(inp_raw) > op_pos:
                raw_support_ops[inp_raw[op_pos]] = raw_support_ops.get(inp_raw[op_pos], 0) + 1
    q_op_raw = query_raw[op_pos] if len(query_raw) > op_pos else "?"
    q_op_support = raw_support_ops.get(q_op_raw, 0)
    support_ops_text = ", ".join(
        f"{op}:{count}" for op, count in sorted(raw_support_ops.items())
    ) or "none"

    lines = [
        "TRANS_ROUTE_V1",
        "",
        "Flow:",
        "  parse support examples and query",
        "  classify surface",
        "  mark cipher digit slots and operator slot",
        "  extract query_op",
        "  count support examples using query_op",
        "  exit to the first matching route",
        "",
        "Surface:",
        "  kind = cipher_digit",
        f"  query = {query_raw}",
        f"  op_pos = {op_pos}",
        f"  digit_slots = {[i for i in range(5) if i != op_pos]}",
        f"  query_op = {q_op_raw}",
        f"  support_ops = {support_ops_text}",
        f"  query_op_support = {q_op_support}",
        "",
        "Route:",
        "  program = TRANS_MULTI_SUPPORT_V1",
        "  reason = query operator has at least two direct support rows",
        f"  route_check = query_op_support:{q_op_support} -> TRANS_MULTI_SUPPORT_V1",
        "",
        "CIPHER_MAP_V1",
        "MapOrder:",
        "  1. from Route.Surface, use op_pos to separate operator symbols from digit symbols",
        "  2. from digit_slots, collect all symbols that must map to digits",
        "  3. enumerate bijective symbol->digit mappings",
        "  4. decode support rows under each candidate mapping",
        "  5. replay decoded support rows and keep only support-passing mappings",
        "  6. use the kept mapping to decode the query; do not use the stored answer",
        "",
        "Cipher-digit.",
        f"Detect: base={base}, op_pos={op_pos}",
        "",
    ]

    # Phase 0 — show mapping
    lines.append("Mapping:")
    sorted_map = sorted(mapping.items(), key=lambda x: x[1])
    map_str = " ".join(f"{s}={d}" for s, d in sorted_map)
    lines.append(f"  {map_str}")
    lines.append("")

    # Decode all examples
    def decode_sym(s):
        """Decode a symbol string to digit string."""
        return "".join(str(mapping.get(c, c)) for c in s)

    lines.append("Decode examples:")
    decoded_examples = []
    for inp_raw, out_raw in examples_raw:
        inp_dec = decode_sym(inp_raw)
        out_dec = decode_sym(out_raw)
        lines.append(f"  {inp_raw} = {out_raw} → {inp_dec} = {out_dec}")
        decoded_examples.append((inp_dec, out_dec))
    lines.append("")

    # Decode query
    query_dec = decode_sym(query_raw)
    lines.append(f"Query: {query_raw} → {query_dec}")
    lines.append("")

    # Now run numeric scan on decoded examples
    # Group by operator (position 2 in 5-char input)
    by_op = {}
    for inp_dec, out_dec in decoded_examples:
        if len(inp_dec) >= 5:
            a_s = inp_dec[0:2]
            op_char_pos = 2  # operator is always at position 2
            # But decoded operator might be a digit — use RAW operator
            raw_inp = examples_raw[len(by_op.get(inp_dec[2], []) or [])][0] if False else None
        # Actually, operator chars are NOT in the digit mapping — they stay as-is
        # The 5-char input is: sym sym op sym sym
        # After decode: dig dig op dig dig (operator unchanged)
        # Let me re-decode more carefully
        pass

    # Re-parse: operator is at position 2 of the RAW input (not decoded)
    by_op = {}
    raw_to_dec = {}
    digit_pos = [i for i in range(5) if i != op_pos]
    if len(digit_pos) != 4:
        return None

    for (inp_raw, out_raw), (inp_dec, out_dec) in zip(examples_raw, decoded_examples):
        op_char = inp_raw[op_pos]  # operator symbol (stays as-is, not in mapping)
        d0 = mapping.get(inp_raw[digit_pos[0]])
        d1 = mapping.get(inp_raw[digit_pos[1]])
        d3 = mapping.get(inp_raw[digit_pos[2]])
        d4 = mapping.get(inp_raw[digit_pos[3]])
        if d0 is None or d1 is None or d3 is None or d4 is None:
            continue  # skip examples with unmapped symbols
        a_s = str(d0) + str(d1)
        b_s = str(d3) + str(d4)
        # Output: decode all symbols
        out_digits = decode_sym(out_raw)
        by_op.setdefault(op_char, []).append((a_s, b_s, out_digits))

    # Parse query
    if len(query_raw) < 5:
        return None
    q_op = query_raw[op_pos]
    qd0 = mapping.get(query_raw[digit_pos[0]])
    qd1 = mapping.get(query_raw[digit_pos[1]])
    qd3 = mapping.get(query_raw[digit_pos[2]])
    qd4 = mapping.get(query_raw[digit_pos[3]])
    if qd0 is None or qd1 is None or qd3 is None or qd4 is None:
        return None  # can't decode query
    q_a = str(qd0) + str(qd1)
    q_b = str(qd3) + str(qd4)

    ops = sorted(by_op.keys())
    if q_op not in by_op:
        return None
    if len(ops) > 1:
        lines.append("Plan: scan the query operator first; skip support-only operators.")
        lines.append("")
    group = by_op[q_op]
    if len(group) < 2:
        return None
    rule = _first_full_support_rule(group, q_op)
    if rule is None:
        return None
    lines.extend(_render_rule_card(group, q_op, rule))
    lines.append("")

    # Query application
    lines.append("Apply:")
    lines.append(f"  query = {query_raw} -> {q_a}{q_op}{q_b}")
    encoded = None

    if rule:
        ord_label, order_key, op_name, style = rule
        q_a_int, q_b_int = _ordered_ints(q_a, q_b, order_key)

        lines.append(f"  ordering = {ord_label} -> L={q_a_int} R={q_b_int}")
        raw = _compute_op(op_name, q_a_int, q_b_int)
        if raw is None:
            return None
        lines.append(f"  operation = {_show_op(op_name, q_a_int, q_b_int)}")
        final_digits = _apply_style(raw, style, q_op)
        style_lines = _show_style_step(raw, style, q_op)
        lines.extend(style_lines)
        lines.append(f"  formatted = {final_digits}")

        # Phase 4 — re-encode to symbols
        encoded = "".join(inv_map.get(int(c), c) if c.isdigit() else c for c in str(final_digits))
        lines.append(f"  encode = {final_digits} -> {encoded}")

    if encoded != answer_str:
        return None

    lines.append(f"Answer: {answer_str}")

    reasoning = "\n".join(lines)
    return reasoning, answer_str
