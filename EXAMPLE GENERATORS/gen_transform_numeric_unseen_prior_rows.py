#!/usr/bin/env python3
"""Generate labeled prior rows for numeric unseen-operator cases."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from generators.gen_transform_numeric_hard_way_rows import (
    Candidate,
    _assistant_content,
    _feature_stats,
    _numeric_candidates,
    _rank_key,
    _select_candidate,
    _show_op_name,
    _show_order,
    _show_route_program,
    _show_style_name,
    _support_ops_text,
)
from generators.trace_transform import _apply_style, _compute_op, _ordered_ints, _show_op, _show_style_step
from generators.transform_program_audit import _parse_numeric_query
from generators.transform_router import extract_examples_and_query, is_transformation_prompt, route_transform_prompt
from training.data import BOXED_INSTRUCTION


PROGRAM_NAME = "Numeric Unseen Operator Prior"


def _best_gold_candidate(candidates: list[Candidate], answer: str, stats: dict[str, dict[tuple, tuple[float, int, int]]]) -> Candidate | None:
    gold = [cand for cand in candidates if cand.answer == answer]
    if not gold:
        return None
    return max(gold, key=lambda cand: _rank_key(cand, stats))


def _render_trace(prompt: str, route: dict[str, Any], candidates: list[Candidate], chosen: Candidate, default: Candidate, answer: str) -> str | None:
    examples, query = extract_examples_and_query(prompt)
    parsed_query = _parse_numeric_query(query)
    if parsed_query is None:
        return None
    q_a, q_op, q_b = parsed_query
    try:
        q_left, q_right = _ordered_ints(q_a, q_b, chosen.order_key)
    except Exception:
        return None
    raw = _compute_op(chosen.op_name, q_left, q_right)
    if raw is None:
        return None
    formatted = _apply_style(raw, chosen.style, q_op)
    if formatted != answer:
        return None

    answer_count = len({cand.answer for cand in candidates})
    lines = [
        "TRANS_ROUTE_V1",
        "",
        "Flow:",
        "  parse support examples and query",
        "  classify surface",
        "  extract query operator",
        "  count support examples using the query operator",
        "  if the query operator is absent, enter unseen-operator prior",
        "  execute the selected context-prior rule mechanically",
        "",
        "Surface:",
        "  kind: numeric visible",
        f"  examples: {len(examples)}",
        f"  query: {query}",
        f"  query operator: {q_op}",
        f"  support operators: {_support_ops_text(route)}",
        "  query-op support: 0",
        "",
        "Route:",
        f"  method: {PROGRAM_NAME}",
        f"  starting route: {_show_route_program(str(route.get('program') or ''))}",
        "  certainty: unseen-operator prior",
        "  proof status: no query-operator witness",
        "  reason: query operator is absent from support; use a context-derived prior",
        f"  route check: 0 query-operator examples -> {PROGRAM_NAME}",
        "",
        PROGRAM_NAME,
        "WhenToUse:",
        "  query operator has zero direct support examples",
        "  context operators still reveal operand order and output format",
        "  choose an in-distribution operation prior for the unseen operator",
        "",
        "ContextFacts:",
        f"  selected context operator: {chosen.context_op}",
        f"  context support count: {chosen.context_support}",
        f"  compatible context-prior candidates: {len(candidates)}",
        f"  distinct query answers: {answer_count}",
        f"  default best-fit answer: {default.answer}",
        "",
        "PriorChoice:",
        f"  order: {_show_order(chosen.order_key)}",
        f"  operation: {_show_op_name(chosen.op_name)}",
        f"  format: {_show_style_name(chosen.style)}",
        "  selection basis: train-distribution unseen-operator prior",
        "",
        "SelfCheck:",
        "  this is not a visible proof of the unseen operator",
        "  do not claim support uniqueness",
        "  continue only by executing the selected rule",
        "",
        "Apply:",
        f"  query: {query}",
        f"  order: {_show_order(chosen.order_key)} -> L={q_left} R={q_right}",
        f"  operation: {_show_op(chosen.op_name, q_left, q_right)}",
    ]
    lines.extend(_show_style_step(raw, chosen.style, q_op))
    lines.extend([
        f"  formatted: {formatted}",
        f"Answer: {answer}",
    ])
    return "\n".join(lines)


def build_rows(train_csv: str, out: str, summary_out: str) -> dict[str, Any]:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    raw_items: list[dict[str, Any]] = []
    training_items: list[tuple[str, list[Candidate]]] = []
    counters = Counter()

    with open(train_csv) as f:
        for row in csv.DictReader(f):
            prompt = row["prompt"]
            if not is_transformation_prompt(prompt):
                continue
            route = route_transform_prompt(prompt, row_id=str(row.get("id") or ""), stable_trace_ids=set())
            if route.get("surface") != "numeric_visible":
                continue
            candidates = _numeric_candidates(prompt, route)
            raw_items.append({
                "id": str(row.get("id") or ""),
                "prompt": prompt,
                "answer": row.get("answer", ""),
                "route": route,
                "candidates": candidates,
            })
            training_items.append((row.get("answer", ""), candidates))

    stats = _feature_stats(training_items)
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for item in raw_items:
        route = item["route"]
        if route.get("program") != "TRANS_UNSEEN_PRIOR_V1":
            continue
        counters["route:TRANS_UNSEEN_PRIOR_V1"] += 1
        candidates = item["candidates"]
        if not candidates:
            counters["no_candidates"] += 1
            continue
        default = _select_candidate(candidates, stats)
        if default is None:
            counters["no_default"] += 1
            continue
        if default.answer == item["answer"]:
            counters["skip_default_correct"] += 1
            continue
        chosen = _best_gold_candidate(candidates, item["answer"], stats)
        if chosen is None:
            counters["gold_not_compatible"] += 1
            continue
        trace = _render_trace(item["prompt"], route, candidates, chosen, default, item["answer"])
        if trace is None:
            counters["trace_reject"] += 1
            continue
        rows.append({
            "messages": [
                {"role": "user", "content": item["prompt"] + BOXED_INSTRUCTION},
                {"role": "assistant", "content": _assistant_content(trace, item["answer"])},
            ],
            "answer": item["answer"],
            "id": item["id"],
            "puzzle_type": "transformation",
            "mode": "numeric_unseen_operator_prior",
            "generator": "gen_transform_numeric_unseen_prior_rows",
            "route_program": "TRANS_NUMERIC_UNSEEN_OPERATOR_PRIOR_V1",
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
    parser.add_argument("--out", default="data/transformation/pool/competition/numeric_unseen_operator_prior.jsonl")
    parser.add_argument("--summary", default="data/transformation/pool/competition/numeric_unseen_operator_prior.summary.json")
    args = parser.parse_args()
    print(json.dumps(build_rows(args.train_csv, args.out, args.summary), indent=2))


if __name__ == "__main__":
    main()
