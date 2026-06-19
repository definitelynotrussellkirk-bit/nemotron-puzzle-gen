#!/usr/bin/env python3
"""Route transformation rows into stable program families.

This deliberately does not solve the puzzle. It answers the first question a
mechanical trace must answer: what kind of evidence is visible, and which
program should handle it?
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


def is_transformation_prompt(prompt: str) -> bool:
    """Identify equation-transformation rows without catching bit prompts."""
    head = prompt[:240].lower()
    return (
        "secret set of transformation rules" in head
        and "determine the result for:" in prompt.lower()
    )


def extract_examples_and_query(prompt: str) -> tuple[list[tuple[str, str]], str]:
    """Extract support examples and query string from a competition prompt."""
    examples: list[tuple[str, str]] = []
    query = ""
    for raw_line in prompt.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "determine the result for:" in line.lower():
            query = line.split(":", 1)[-1].strip()
            continue
        if "=" in line and not line.startswith("In "):
            lhs, rhs = line.split("=", 1)
            examples.append((lhs.strip(), rhs.strip()))
    return examples, query


def _numeric_operator(expr: str) -> str | None:
    m = re.match(r"^(\d+)(\D)(\d+)$", expr.strip())
    return m.group(2) if m else None


def _cipher_operator(expr: str, op_pos: int = 2) -> str | None:
    expr = expr.strip()
    if len(expr) <= op_pos:
        return None
    return expr[op_pos]


def classify_surface(examples: list[tuple[str, str]], query: str) -> tuple[str, int | None]:
    """Classify the surface syntax and return (surface, op_pos)."""
    lhs_values = [lhs for lhs, _ in examples]
    if query and _numeric_operator(query) and all(_numeric_operator(lhs) for lhs in lhs_values):
        return "numeric_visible", None
    if (
        query
        and len(query) == 5
        and lhs_values
        and all(len(lhs) == 5 for lhs in lhs_values)
        and not any(c.isdigit() for c in lhs_values[0])
    ):
        return "cipher_digit", 2
    return "symbol_structural", None


def _support_operator_counts(
    surface: str,
    examples: list[tuple[str, str]],
    query: str,
    op_pos: int | None,
) -> tuple[str | None, Counter[str]]:
    counts: Counter[str] = Counter()
    if surface == "numeric_visible":
        query_op = _numeric_operator(query)
        for lhs, _ in examples:
            op = _numeric_operator(lhs)
            if op:
                counts[op] += 1
        return query_op, counts
    if surface == "cipher_digit":
        pos = 2 if op_pos is None else op_pos
        query_op = _cipher_operator(query, pos)
        for lhs, _ in examples:
            op = _cipher_operator(lhs, pos)
            if op:
                counts[op] += 1
        return query_op, counts
    return None, counts


def load_stable_trace_ids(path: str | None) -> set[str]:
    """Read ids already solved by the strict rule-card compiler."""
    ids: set[str] = set()
    if not path:
        return ids
    if not os.path.exists(path):
        return ids
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = row.get("id")
            if rid is not None:
                ids.add(str(rid))
    return ids


def route_transform_prompt(
    prompt: str,
    *,
    row_id: str | None = None,
    stable_trace_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Route one transformation prompt into a next-program family."""
    examples, query = extract_examples_and_query(prompt)
    surface, op_pos = classify_surface(examples, query)
    query_op, support_ops = _support_operator_counts(surface, examples, query, op_pos)
    query_op_support = support_ops.get(query_op, 0) if query_op is not None else 0
    stable_direct = bool(row_id is not None and stable_trace_ids and str(row_id) in stable_trace_ids)

    if not examples or not query:
        program = "TRANS_PARSE_UNSUPPORTED_V1"
        training_program = program
        reason = "prompt parse failed"
    elif surface == "symbol_structural":
        program = "TRANS_SYMBOL_STRUCTURAL_V1"
        training_program = program
        reason = "surface is not numeric-visible or 5-character cipher-digit"
    elif query_op_support >= 2:
        # Visible-only route. The strict rule-card vs ranked split is metadata,
        # because strict trace success depends on solver/gold artifacts.
        program = "TRANS_MULTI_SUPPORT_V1"
        training_program = "TRANS_RULE_CARD_V1" if stable_direct else "TRANS_RANKED_V1"
        reason = "query operator has at least two direct support rows"
    elif query_op_support == 1:
        program = "TRANS_ONE_SHOT_V1"
        training_program = program
        reason = "query operator has exactly one direct support row"
    else:
        program = "TRANS_UNSEEN_PRIOR_V1"
        training_program = program
        reason = "query operator is absent from support"

    return {
        "id": row_id,
        "surface": surface,
        "program": program,
        "training_program": training_program,
        "reason": reason,
        "examples": len(examples),
        "query": query,
        "query_op": query_op,
        "query_op_support": query_op_support,
        "support_ops": dict(sorted(support_ops.items())),
        "op_pos": op_pos,
        "stable_direct": stable_direct,
    }


def build_route_trace(route: dict[str, Any]) -> str:
    """Render the route decision as a compact trace block."""
    support_ops = ", ".join(
        f"{op}:{count}" for op, count in route.get("support_ops", {}).items()
    ) or "none"
    return "\n".join([
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
        f"  kind = {route['surface']}",
        f"  examples = {route['examples']}",
        f"  query = {route['query']}",
        f"  query_op = {route['query_op']}",
        f"  support_ops = {support_ops}",
        f"  query_op_support = {route['query_op_support']}",
        "",
        "Route:",
        f"  program = {route['program']}",
        f"  reason = {route['reason']}",
        f"  route_check = query_op_support:{route['query_op_support']} -> {route['program']}",
    ])


def audit_route_invariants(route: dict[str, Any]) -> list[str]:
    """Return invariant violations for a visible route."""
    issues: list[str] = []
    program = route.get("program")
    support = route.get("query_op_support")
    surface = route.get("surface")
    if program == "TRANS_MULTI_SUPPORT_V1" and support < 2:
        issues.append("multi_support_requires_query_op_support_ge2")
    if program == "TRANS_ONE_SHOT_V1" and support != 1:
        issues.append("one_shot_requires_query_op_support_eq1")
    if program == "TRANS_UNSEEN_PRIOR_V1" and support != 0:
        issues.append("unseen_prior_requires_query_op_support_eq0")
    if program == "TRANS_SYMBOL_STRUCTURAL_V1" and surface in {"numeric_visible", "cipher_digit"}:
        issues.append("symbol_structural_requires_non_numeric_non_cipher_surface")
    if program in {"TRANS_MULTI_SUPPORT_V1", "TRANS_ONE_SHOT_V1", "TRANS_UNSEEN_PRIOR_V1"}:
        if surface not in {"numeric_visible", "cipher_digit"}:
            issues.append("numeric_cipher_route_requires_numeric_or_cipher_surface")
    return issues


def route_train_csv(train_csv: str, stable_traces: str | None, out: str, summary_out: str) -> dict[str, Any]:
    stable_ids = load_stable_trace_ids(stable_traces)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_out).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    program_counts: Counter[str] = Counter()
    training_program_counts: Counter[str] = Counter()
    surface_counts: Counter[str] = Counter()
    support_counts: Counter[str] = Counter()
    invariant_counts: Counter[str] = Counter()

    with open(train_csv) as f:
        for row in csv.DictReader(f):
            prompt = row["prompt"]
            if not is_transformation_prompt(prompt):
                continue
            rid = str(row.get("id", ""))
            route = route_transform_prompt(prompt, row_id=rid, stable_trace_ids=stable_ids)
            route["answer"] = row.get("answer", "")
            route["trace"] = build_route_trace(route)
            issues = audit_route_invariants(route)
            route["route_invariant_issues"] = issues
            rows.append(route)
            program_counts[route["program"]] += 1
            training_program_counts[route["training_program"]] += 1
            surface_counts[route["surface"]] += 1
            if route["surface"] in {"numeric_visible", "cipher_digit"}:
                n = route["query_op_support"]
                label = "absent" if n == 0 else "one" if n == 1 else "ge2"
                support_counts[f"{route['surface']}:{label}"] += 1
            for issue in issues:
                invariant_counts[issue] += 1

    with open(out, "w") as f:
        for route in rows:
            f.write(json.dumps(route) + "\n")

    summary = {
        "rows": len(rows),
        "program_counts": dict(program_counts.most_common()),
        "training_program_counts": dict(training_program_counts.most_common()),
        "surface_counts": dict(surface_counts.most_common()),
        "support_counts": dict(support_counts.most_common()),
        "stable_trace_ids": len(stable_ids),
        "route_invariant_issues": dict(invariant_counts.most_common()),
        "output": out,
    }
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/competition/train.csv")
    parser.add_argument("--stable-traces", default="data/transformation/pool/competition/competition_traced.jsonl")
    parser.add_argument("--out", default="data/transformation/routes/competition_routes.jsonl")
    parser.add_argument("--summary", default="data/transformation/routes/competition_routes.summary.json")
    args = parser.parse_args()

    summary = route_train_csv(args.train_csv, args.stable_traces, args.out, args.summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
