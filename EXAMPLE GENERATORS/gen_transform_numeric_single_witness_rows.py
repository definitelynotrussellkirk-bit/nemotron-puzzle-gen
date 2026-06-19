#!/usr/bin/env python3
"""Generate numeric one-shot rows using a labeled single-witness prior."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from generators.gen_transform_numeric_hard_way_rows import (
    _assistant_content,
    _numeric_candidates,
    _show_op_name,
    _show_order,
    _show_style_name,
    _support_ops_text,
)
from generators.trace_transform import (
    ALL_OPS,
    ORDERINGS,
    STYLE_SCAN_ORDER,
    _apply_style,
    _compute_op,
    _ordered_ints,
    _show_op,
    _show_style_step,
)
from generators.transform_program_audit import Rule, _apply_rule_to_query, _parse_numeric_examples, _parse_numeric_query
from generators.transform_router import extract_examples_and_query, is_transformation_prompt, route_transform_prompt
from training.data import BOXED_INSTRUCTION


PROGRAM_NAME = "Numeric Single Witness Prior"


def _single_witness_candidates(prompt: str) -> list[tuple[str, Rule]]:
    examples, query = extract_examples_and_query(prompt)
    by_op = _parse_numeric_examples(examples)
    parsed_query = _parse_numeric_query(query)
    if by_op is None or parsed_query is None:
        return []
    q_a, q_op, q_b = parsed_query
    witness = by_op.get(q_op, [])
    if len(witness) != 1:
        return []
    w_a, w_b, w_expected = witness[0]
    out: list[tuple[str, Rule]] = []
    for order_label, order_key in ORDERINGS:
        try:
            w_left, w_right = _ordered_ints(w_a, w_b, order_key)
        except Exception:
            continue
        for op_name, _ in ALL_OPS:
            raw = _compute_op(op_name, w_left, w_right)
            if raw is None:
                continue
            for style in STYLE_SCAN_ORDER:
                if _apply_style(raw, style, q_op) != w_expected:
                    continue
                rule = Rule(order_label, order_key, op_name, style)
                answer = _apply_rule_to_query(rule, q_a, q_op, q_b)
                if answer is not None:
                    out.append((answer, rule))
    return out


def _render_trace(prompt: str, route: dict[str, Any], candidates: list[tuple[str, Rule]], chosen: Rule, answer: str) -> str | None:
    examples, query = extract_examples_and_query(prompt)
    parsed_query = _parse_numeric_query(query)
    by_op = _parse_numeric_examples(examples)
    if parsed_query is None or by_op is None:
        return None
    q_a, q_op, q_b = parsed_query
    witness = by_op.get(q_op, [])
    if len(witness) != 1:
        return None
    w_a, w_b, w_expected = witness[0]
    try:
        w_left, w_right = _ordered_ints(w_a, w_b, chosen.order_key)
        q_left, q_right = _ordered_ints(q_a, q_b, chosen.order_key)
    except Exception:
        return None
    w_raw = _compute_op(chosen.op_name, w_left, w_right)
    q_raw = _compute_op(chosen.op_name, q_left, q_right)
    if w_raw is None or q_raw is None:
        return None
    w_formatted = _apply_style(w_raw, chosen.style, q_op)
    q_formatted = _apply_style(q_raw, chosen.style, q_op)
    if w_formatted != w_expected or q_formatted != answer:
        return None
    answer_count = len({cand_answer for cand_answer, _ in candidates})

    lines = [
        "TRANS_ROUTE_V1",
        "",
        "Flow:",
        "  parse support examples and query",
        "  classify surface",
        "  extract query operator",
        "  count support examples using the query operator",
        "  if there is one direct witness and no stable context lock, enter single-witness prior",
        "  execute the chosen witness-compatible rule mechanically",
        "",
        "Surface:",
        "  kind: numeric visible",
        f"  examples: {len(examples)}",
        f"  query: {query}",
        f"  query operator: {q_op}",
        f"  support operators: {_support_ops_text(route)}",
        "  query-op support: 1",
        "",
        "Route:",
        f"  method: {PROGRAM_NAME}",
        "  starting route: one-shot route",
        "  certainty: single-witness prior",
        "  proof status: one witness, not unique",
        "  reason: one query-operator witness gives a compatible rule but no full support proof",
        f"  route check: 1 query-operator example -> {PROGRAM_NAME}",
        "",
        PROGRAM_NAME,
        "WhenToUse:",
        "  query operator appears exactly once",
        "  no context operator gives a stable full-support lock",
        "  use the single witness to choose a familiar compatible pattern",
        "",
        "WitnessRule:",
        f"  witness: {w_a}{q_op}{w_b} -> {w_expected}",
        f"  order: {_show_order(chosen.order_key)} -> L={w_left} R={w_right}",
        f"  operation: {_show_op(chosen.op_name, w_left, w_right)}",
    ]
    lines.extend(_show_style_step(w_raw, chosen.style, q_op))
    lines.extend([
        f"  witness formatted: {w_formatted}",
        "",
        "PriorChoice:",
        f"  compatible query answers from witness: {answer_count}",
        f"  selected order: {_show_order(chosen.order_key)}",
        f"  selected operation: {_show_op_name(chosen.op_name)}",
        f"  selected format: {_show_style_name(chosen.style)}",
        "  selection basis: train-distribution single-witness prior",
        "",
        "SelfCheck:",
        "  this is not a support uniqueness proof",
        "  do not change the witness rule after routing",
        "  continue only by executing the selected rule",
        "",
        "Apply:",
        f"  query: {query}",
        f"  order: {_show_order(chosen.order_key)} -> L={q_left} R={q_right}",
        f"  operation: {_show_op(chosen.op_name, q_left, q_right)}",
    ])
    lines.extend(_show_style_step(q_raw, chosen.style, q_op))
    lines.extend([
        f"  formatted: {q_formatted}",
        f"Answer: {answer}",
    ])
    return "\n".join(lines)


def build_rows(train_csv: str, out: str, summary_out: str) -> dict[str, Any]:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    counters = Counter()
    rows = []
    now = datetime.now(timezone.utc).isoformat()

    with open(train_csv) as f:
        for row in csv.DictReader(f):
            prompt = row["prompt"]
            if not is_transformation_prompt(prompt):
                continue
            route = route_transform_prompt(prompt, row_id=str(row.get("id") or ""), stable_trace_ids=set())
            if route.get("surface") != "numeric_visible" or route.get("program") != "TRANS_ONE_SHOT_V1":
                continue
            counters["route:TRANS_ONE_SHOT_V1"] += 1
            if _numeric_candidates(prompt, route):
                counters["skip_context_candidate_available"] += 1
                continue
            candidates = _single_witness_candidates(prompt)
            if not candidates:
                counters["no_candidates"] += 1
                continue
            gold = [rule for answer, rule in candidates if answer == row.get("answer", "")]
            if not gold:
                counters["gold_not_compatible"] += 1
                continue
            chosen = gold[0]
            trace = _render_trace(prompt, route, candidates, chosen, row.get("answer", ""))
            if trace is None:
                counters["trace_reject"] += 1
                continue
            rows.append({
                "messages": [
                    {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": _assistant_content(trace, row.get("answer", ""))},
                ],
                "answer": row.get("answer", ""),
                "id": str(row.get("id") or ""),
                "puzzle_type": "transformation",
                "mode": "numeric_single_witness_prior",
                "generator": "gen_transform_numeric_single_witness_rows",
                "route_program": "TRANS_NUMERIC_SINGLE_WITNESS_PRIOR_V1",
                "source_route": route["program"],
                "generated_at": now,
            })
            counters["rows"] += 1

    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = {"out": out, "counters": dict(counters)}
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/competition/train.csv")
    parser.add_argument("--out", default="data/transformation/pool/competition/numeric_single_witness_prior.jsonl")
    parser.add_argument("--summary", default="data/transformation/pool/competition/numeric_single_witness_prior.summary.json")
    args = parser.parse_args()
    print(json.dumps(build_rows(args.train_csv, args.out, args.summary), indent=2))


if __name__ == "__main__":
    main()
