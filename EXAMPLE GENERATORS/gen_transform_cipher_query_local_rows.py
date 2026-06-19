#!/usr/bin/env python3
"""Generate cipher rows solved by a query-operator-local decoded rule.

This route is stricter than a global fallback and cheaper than the full cipher
solver. It only trains rows where the selected symbol map and decoded rule
replay the support examples that use the query operator, then predict the
stored query answer.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone

from generators.trace_transform import (
    _apply_style,
    _compute_op,
    _encode_with_inverse_map,
    _evaluate_rule,
    _normalize_cipher_combo,
    _ordered_ints,
    _show_op,
    _show_style_step,
)
from generators.transform_router import (
    extract_examples_and_query,
    is_transformation_prompt,
    route_transform_prompt,
)
from solvers import cipher_digit as cipher
from training.data import BOXED_INSTRUCTION


def _local_query_op_solve(
    prompt: str,
    answer: str,
    timeout: float,
    *,
    min_support: int = 2,
    exact_support: int | None = None,
) -> dict | None:
    """Gold-filtered local solver using same-query-operator support only."""
    examples, query = extract_examples_and_query(prompt)
    op_pos = 2
    if len(query) <= op_pos:
        return None
    query_op = query[op_pos]
    digit_pos = [i for i in range(5) if i != op_pos]
    same = []
    for lhs, rhs in examples:
        if len(lhs) == 5 and lhs[op_pos] == query_op:
            same.append((tuple(lhs[i] for i in digit_pos), rhs))
    if exact_support is not None and len(same) != exact_support:
        return None
    if exact_support is None and len(same) < min_support:
        return None

    query_abcd = tuple(query[i] for i in digit_pos)
    data = same + [(query_abcd, answer)]
    deadline = time.monotonic() + timeout if timeout else None
    cipher._SOLVE_DEADLINE = deadline

    first_abcd, first_rhs = data[0]
    unique_symbols = list(dict.fromkeys(first_abcd))
    for combo in cipher.COMBOS:
        if deadline and time.monotonic() > deadline:
            return None
        order, op_name, fmt = combo
        for perm in itertools.permutations(range(10), len(unique_symbols)):
            if deadline and time.monotonic() > deadline:
                return None
            base_map = dict(zip(unique_symbols, perm))
            digits = tuple(base_map[sym] for sym in first_abcd)
            left, right = cipher.make_op(*digits, order)
            raw = cipher.calc(left, right, op_name)
            formatted = cipher.fmtv(raw, fmt) if raw is not None else None
            if formatted is None or len(formatted) != len(first_rhs):
                continue
            matched = cipher.mrhs(first_rhs, formatted, base_map, set(perm))
            if matched is None:
                continue
            extended = cipher.vext(data[1:], matched[0], matched[1], order, op_name, fmt)
            if extended is None:
                continue
            mapping, _used = extended
            return {
                "answer": answer,
                "mapping": mapping,
                "combos": {query_op: combo},
                "op_pos": op_pos,
                "query_op": query_op,
                "same_query_op_support": len(same),
            }
    return None


def _render_trace(
    prompt: str,
    answer: str,
    result: dict,
    *,
    program_name: str = "Cipher Query-Operator Local Rule",
    route_reason: str = "query-op support rows validate a local decoded rule",
    certainty: str = "query-op local support",
    support_phrase: str = "same query-op support rows replay under the selected local rule",
    min_support: int = 2,
) -> tuple[str, str] | None:
    examples, query = extract_examples_and_query(prompt)
    mapping = result["mapping"]
    combo = _normalize_cipher_combo(result["combos"].get(result["query_op"]))
    op_pos = result.get("op_pos", 2)
    if combo is None or len(query) <= op_pos:
        return None

    digit_pos = [i for i in range(5) if i != op_pos]
    query_op = result["query_op"]
    same_examples = [(lhs, rhs) for lhs, rhs in examples if len(lhs) == 5 and lhs[op_pos] == query_op]
    if len(same_examples) < min_support:
        return None

    inv_map = {v: k for k, v in mapping.items()}

    def decode_sym(text: str) -> str:
        return "".join(str(mapping.get(ch, ch)) for ch in text)

    support_counts: Counter[str] = Counter()
    for lhs, _rhs in examples:
        if len(lhs) > op_pos:
            support_counts[lhs[op_pos]] += 1
    support_ops = ", ".join(f"{op}:{count}" for op, count in sorted(support_counts.items())) or "none"

    decode_lines: list[str] = []
    replay_lines: list[str] = []
    compatible = 0
    checked = 0
    for lhs, rhs in same_examples:
        try:
            left_digits = str(mapping[lhs[digit_pos[0]]]) + str(mapping[lhs[digit_pos[1]]])
            right_digits = str(mapping[lhs[digit_pos[2]]]) + str(mapping[lhs[digit_pos[3]]])
        except KeyError:
            return None
        decoded_rhs = decode_sym(rhs)
        decoded_lhs = f"{left_digits}{query_op}{right_digits}"
        decode_lines.append(f"  {lhs} = {rhs} -> {decoded_lhs} = {decoded_rhs}")
        evaluated = _evaluate_rule(left_digits, right_digits, query_op, combo)
        got = evaluated[3] if evaluated else "ERR"
        checked += 1
        if got == decoded_rhs:
            compatible += 1
        mark = "PASS" if got == decoded_rhs else "FAIL"
        replay_lines.append(
            f"  {lhs}: decoded {decoded_lhs}; {combo[2]}/{combo[3]} -> {got}; expected {decoded_rhs}; {mark}"
        )
    if checked == 0 or compatible != checked:
        return None

    try:
        q_a = str(mapping[query[digit_pos[0]]]) + str(mapping[query[digit_pos[1]]])
        q_b = str(mapping[query[digit_pos[2]]]) + str(mapping[query[digit_pos[3]]])
    except KeyError:
        return None
    q_left, q_right = _ordered_ints(q_a, q_b, combo[1])
    q_raw = _compute_op(combo[2], q_left, q_right)
    if q_raw is None:
        return None
    final_digits = _apply_style(q_raw, combo[3], query_op)
    encoded = _encode_with_inverse_map(final_digits, inv_map)
    if encoded != answer:
        return None

    map_str = " ".join(f"{sym}={digit}" for sym, digit in sorted(mapping.items(), key=lambda item: item[1]))
    lines = [
        "TRANS_ROUTE_V1",
        "",
        "Flow:",
        "  parse support examples and query",
        "  classify surface",
        "  mark cipher digit slots and operator slot",
        "  extract query_op",
        "  count support examples using query_op",
        "  try visible slot rules before local decoded rules",
        "  exit when a local query-op rule replays direct support rows",
        "",
        "Surface:",
        "  kind = cipher_digit",
        f"  query = {query}",
        f"  op_pos = {op_pos}",
        f"  digit_slots = {digit_pos}",
        f"  query_op = {query_op}",
        f"  support_ops = {support_ops}",
        f"  query_op_support = {support_counts.get(query_op, 0)}",
        "",
        "Route:",
        f"  program = {program_name}",
        f"  reason = {route_reason}",
        f"  route_check = query_op_support:{support_counts.get(query_op, 0)} -> {program_name}",
        "",
        "CIPHER_MAP_V1",
        "MapOrder:",
        "  1. from Route.Surface, use op_pos to separate operator symbols from digit symbols",
        "  2. decode only rows whose operator equals query_op",
        "  3. enumerate bijective symbol->digit mappings",
        "  4. keep the local mapping/rule that replays query-op support",
        "  5. decode the query and execute the local query-op rule",
        "",
        "Mapping:",
        f"  {map_str}",
        "",
        "Decode examples:",
        *decode_lines,
        "",
        f"Query: {query} -> {q_a}{query_op}{q_b}",
        "",
        program_name,
        "WhenToUse:",
        f"  {support_phrase}",
        "  use this before global fallback when support-only operators are irrelevant",
        "PatternCue:",
        "  think in four slots: symbol map, operand order, arithmetic operation, output format",
        "  prefer the query-op-local decoded pattern",
        "  do not enumerate hidden candidates",
        "RouteFacts:",
        f"  source route: cipher query_op_support={support_counts.get(query_op, 0)}",
        f"  decoded query: {q_a}{query_op}{q_b}",
        f"  selected rule: {combo[0]}|{combo[2]}|{combo[3]}",
        "BestFitRule:",
        f"  mapping = {map_str}",
        f"  rule = {combo[0]}|{combo[2]}|{combo[3]}",
        f"  compatible replay = {compatible}/{checked}",
        *replay_lines,
        "SelfCheck:",
        f"  proof status: {certainty}",
        f"  certainty: {certainty}",
        f"  local support replay only: op {query_op} has {support_counts.get(query_op, 0)} direct rows",
        "Apply:",
        f"  query = {query} -> {q_a}{query_op}{q_b}",
        f"  ordering = {combo[0]} -> L={q_left} R={q_right}",
        f"  operation = {_show_op(combo[2], q_left, q_right)}",
    ]
    lines.extend(_show_style_step(q_raw, combo[3], query_op))
    lines.extend([
        f"  formatted digits = {final_digits}",
        f"  encode = {final_digits} -> {encoded}",
        f"Answer: {answer}",
    ])
    return "\n".join(lines), answer


def _assistant_content(trace_text: str, answer: str) -> str:
    if "{" in answer or "}" in answer:
        return f"<think>\n{trace_text}\n</think>\nThe final answer is: {answer}"
    return f"<think>\n{trace_text}\n</think>\n\\boxed{{{answer}}}"


def build_rows(train_csv: str, out: str, summary_out: str, timeout: float) -> dict:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    counters: Counter[str] = Counter()
    rows = []

    with open(train_csv) as f:
        for row in csv.DictReader(f):
            prompt = row["prompt"]
            if not is_transformation_prompt(prompt):
                continue
            counters["transform_rows"] += 1
            rid = str(row.get("id", ""))
            answer = row.get("answer", "")
            route = route_transform_prompt(prompt, row_id=rid, stable_trace_ids=set())
            if route["surface"] != "cipher_digit":
                continue
            counters["cipher_rows"] += 1
            counters[f"route:{route['program']}"] += 1
            if route["program"] != "TRANS_MULTI_SUPPORT_V1":
                continue

            result = _local_query_op_solve(prompt, answer, timeout=timeout)
            if result is None:
                counters["local_solve_fail"] += 1
                continue
            rendered = _render_trace(prompt, answer, result)
            if rendered is None:
                counters["trace_builder_reject"] += 1
                continue
            trace_text, pred = rendered
            if pred != answer:
                counters["prediction_mismatch"] += 1
                continue
            if "???" in trace_text or "???" in answer:
                counters["placeholder_reject"] += 1
                continue

            combo = result["combos"][result["query_op"]]
            rows.append({
                "messages": [
                    {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": _assistant_content(trace_text, answer)},
                ],
                "answer": answer,
                "id": rid,
                "puzzle_type": "transformation",
                "mode": "cipher_query_op_local_rule",
                "generator": "gen_transform_cipher_query_local_rows",
                "route_program": route["program"],
                "query_op_local_rule": "|".join(combo),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
            counters["rows"] += 1
            counters[f"rows:{route['program']}"] += 1
            counters[f"combo:{'|'.join(combo)}"] += 1

    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "out": out,
        "timeout": timeout,
        "counters": dict(counters),
        "coverage_by_source_route": {
            key.split(":", 1)[1]: value
            for key, value in counters.items()
            if key.startswith("rows:")
        },
    }
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/competition/train.csv")
    parser.add_argument("--out", default="data/transformation/pool/competition/cipher_query_op_local_rule.jsonl")
    parser.add_argument("--summary", default="data/transformation/pool/competition/cipher_query_op_local_rule.summary.json")
    parser.add_argument("--timeout", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(build_rows(args.train_csv, args.out, args.summary, args.timeout), indent=2))


if __name__ == "__main__":
    main()
