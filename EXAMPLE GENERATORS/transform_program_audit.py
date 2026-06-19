#!/usr/bin/env python3
"""Audit simple transformation solve programs inside TRANS_ROUTE_V1 buckets.

This is intentionally an audit, not a data generator. It predicts from visible
prompt evidence first, then compares to the stored label only for reporting.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generators.trace_transform import (
    ALL_OPS,
    ORDERINGS,
    _apply_style,
    _compute_op,
    _detect_styles,
    _ordered_ints,
)
from generators.transform_router import (
    extract_examples_and_query,
    route_transform_prompt,
    load_stable_trace_ids,
    is_transformation_prompt,
)
from solvers.cipher_digit import solve as cipher_solve


@dataclass(frozen=True)
class Rule:
    order_label: str
    order_key: str
    op_name: str
    style: str


def _parse_numeric_examples(examples: list[tuple[str, str]]) -> dict[str, list[tuple[str, str, str]]] | None:
    import re
    by_op: dict[str, list[tuple[str, str, str]]] = {}
    for lhs, rhs in examples:
        m = re.match(r"^(\d+)(\D)(\d+)$", lhs.strip())
        if not m:
            return None
        by_op.setdefault(m.group(2), []).append((m.group(1), m.group(3), rhs.strip()))
    return by_op


def _parse_numeric_query(query: str) -> tuple[str, str, str] | None:
    import re
    m = re.match(r"^(\d+)(\D)(\d+)$", query.strip())
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def _rules_fitting_group(group: list[tuple[str, str, str]], op_char: str) -> list[Rule]:
    """Enumerate rules that replay every row in an operator group."""
    if not group:
        return []
    rules: list[Rule] = []
    for order_label, order_key in ORDERINGS:
        for op_name, _fn in ALL_OPS:
            try:
                first_left, first_right = _ordered_ints(group[0][0], group[0][1], order_key)
            except Exception:
                continue
            first_raw = _compute_op(op_name, first_left, first_right)
            if first_raw is None:
                continue
            styles = _detect_styles(first_raw, group[0][2], op_char)
            if not styles:
                continue
            for style in styles:
                ok = True
                for a_s, b_s, expected in group:
                    try:
                        left, right = _ordered_ints(a_s, b_s, order_key)
                    except Exception:
                        ok = False
                        break
                    raw = _compute_op(op_name, left, right)
                    if raw is None or _apply_style(raw, style, op_char) != expected:
                        ok = False
                        break
                if ok:
                    rules.append(Rule(order_label, order_key, op_name, style))
    return rules


def _apply_rule_to_query(rule: Rule, q_a: str, q_op: str, q_b: str) -> str | None:
    try:
        left, right = _ordered_ints(q_a, q_b, rule.order_key)
    except Exception:
        return None
    raw = _compute_op(rule.op_name, left, right)
    if raw is None:
        return None
    return _apply_style(raw, rule.style, q_op)


def _first_order_style_context(by_op: dict[str, list[tuple[str, str, str]]], query_op: str) -> list[Rule]:
    """Return context rules from non-query operators, most-supported first."""
    contexts: list[tuple[int, str, Rule]] = []
    for op_char, group in by_op.items():
        if op_char == query_op or len(group) < 2:
            continue
        rules = _rules_fitting_group(group, op_char)
        for rule in rules:
            contexts.append((-len(group), op_char, rule))
    contexts.sort(key=lambda item: (item[0], item[1], item[2].order_key, item[2].op_name, item[2].style))
    return [rule for _n, _op, rule in contexts]


def _one_shot_answers(
    by_op: dict[str, list[tuple[str, str, str]]],
    q_a: str,
    q_op: str,
    q_b: str,
) -> tuple[list[str], list[str]]:
    """Candidate answers using context order/style + one qop witness."""
    witness = by_op.get(q_op, [])
    if len(witness) != 1:
        return [], ["bad_witness_count"]
    contexts = _first_order_style_context(by_op, q_op)
    if not contexts:
        return [], ["no_context_operator_ge2"]
    answers: list[str] = []
    issues: list[str] = []
    w_a, w_b, w_expected = witness[0]
    for context in contexts:
        for op_name, _fn in ALL_OPS:
            pseudo = Rule(context.order_label, context.order_key, op_name, context.style)
            try:
                left, right = _ordered_ints(w_a, w_b, pseudo.order_key)
            except Exception:
                continue
            raw = _compute_op(op_name, left, right)
            if raw is None:
                continue
            if _apply_style(raw, pseudo.style, q_op) != w_expected:
                continue
            ans = _apply_rule_to_query(pseudo, q_a, q_op, q_b)
            if ans is not None:
                answers.append(ans)
    if not answers:
        issues.append("no_one_shot_operation")
    return answers, issues


def _unseen_prior_answers(
    by_op: dict[str, list[tuple[str, str, str]]],
    q_a: str,
    q_op: str,
    q_b: str,
) -> tuple[list[str], list[str]]:
    """Candidate answers for unseen qop using context order/style and op priors."""
    contexts = _first_order_style_context(by_op, q_op)
    if not contexts:
        return [], ["no_context_operator_ge2"]
    answers: list[str] = []
    # A deliberately simple prior order. This is not yet a training program.
    prior_ops = ["mul", "add", "sub", "bsub", "absdiff", "concat", "bconcat"]
    for context in contexts:
        for op_name in prior_ops:
            pseudo = Rule(context.order_label, context.order_key, op_name, context.style)
            ans = _apply_rule_to_query(pseudo, q_a, q_op, q_b)
            if ans is not None:
                answers.append(ans)
    return answers, []


def _classify_candidate_answers(answers: list[str], gold: str) -> dict[str, Any]:
    unique_answers = list(dict.fromkeys(answers))
    first = answers[0] if answers else None
    return {
        "candidate_count": len(answers),
        "unique_answer_count": len(unique_answers),
        "first_prediction": first,
        "first_exact": first == gold if first is not None else False,
        "unique_prediction": unique_answers[0] if len(unique_answers) == 1 else None,
        "unique_exact": len(unique_answers) == 1 and unique_answers[0] == gold,
        "contains_gold": gold in unique_answers,
    }


def audit_numeric(prompt: str, answer: str, route: dict[str, Any]) -> dict[str, Any]:
    examples, query = extract_examples_and_query(prompt)
    parsed_query = _parse_numeric_query(query)
    by_op = _parse_numeric_examples(examples)
    if not parsed_query or by_op is None:
        return {"status": "parse_fail"}
    q_a, q_op, q_b = parsed_query

    if route["program"] == "TRANS_MULTI_SUPPORT_V1":
        rules = _rules_fitting_group(by_op.get(q_op, []), q_op)
        answers = [
            ans for rule in rules
            for ans in [_apply_rule_to_query(rule, q_a, q_op, q_b)]
            if ans is not None
        ]
        out = _classify_candidate_answers(answers, answer)
        out["status"] = "multi_support"
        return out
    if route["program"] == "TRANS_ONE_SHOT_V1":
        answers, issues = _one_shot_answers(by_op, q_a, q_op, q_b)
        out = _classify_candidate_answers(answers, answer)
        out["status"] = "one_shot"
        out["issues"] = issues
        return out
    if route["program"] == "TRANS_UNSEEN_PRIOR_V1":
        answers, issues = _unseen_prior_answers(by_op, q_a, q_op, q_b)
        out = _classify_candidate_answers(answers, answer)
        out["status"] = "unseen_prior"
        out["issues"] = issues
        return out
    return {"status": "unsupported_route"}


def audit_cipher(prompt: str, answer: str, route: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Audit current visible cipher solver without gold answer constraints."""
    result = cipher_solve(prompt, answer=None, timeout=timeout)
    pred = result.get("answer") if result else None
    return {
        "status": "cipher_visible_solver",
        "candidate_count": 1 if pred is not None else 0,
        "unique_answer_count": 1 if pred is not None else 0,
        "first_prediction": pred,
        "first_exact": pred == answer if pred is not None else False,
        "unique_prediction": pred,
        "unique_exact": pred == answer if pred is not None else False,
        "contains_gold": pred == answer if pred is not None else False,
    }


def audit_train_csv(
    train_csv: str,
    stable_traces: str,
    out: str,
    summary_out: str,
    cipher_timeout: float,
    surface_filter: str,
    max_rows: int | None,
) -> dict[str, Any]:
    stable_ids = load_stable_trace_ids(stable_traces)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Counter] = {
        "rows": Counter(),
        "first_exact": Counter(),
        "unique_exact": Counter(),
        "contains_gold": Counter(),
        "no_candidates": Counter(),
        "unique_answer": Counter(),
        "multi_answer": Counter(),
        "issues": Counter(),
    }

    with open(train_csv) as f:
        for row in csv.DictReader(f):
            prompt = row["prompt"]
            if not is_transformation_prompt(prompt):
                continue
            rid = str(row.get("id", ""))
            answer = row.get("answer", "")
            route = route_transform_prompt(prompt, row_id=rid, stable_trace_ids=stable_ids)
            if surface_filter != "all" and route["surface"] != surface_filter:
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break
            if route["surface"] == "numeric_visible":
                audit = audit_numeric(prompt, answer, route)
            elif route["surface"] == "cipher_digit":
                audit = audit_cipher(prompt, answer, route, cipher_timeout)
            else:
                audit = {"status": "surface_unsupported"}

            key = f"{route['surface']}|{route['program']}"
            summary["rows"][key] += 1
            if audit.get("first_exact"):
                summary["first_exact"][key] += 1
            if audit.get("unique_exact"):
                summary["unique_exact"][key] += 1
            if audit.get("contains_gold"):
                summary["contains_gold"][key] += 1
            if audit.get("candidate_count", 0) == 0:
                summary["no_candidates"][key] += 1
            if audit.get("unique_answer_count") == 1:
                summary["unique_answer"][key] += 1
            if audit.get("unique_answer_count", 0) > 1:
                summary["multi_answer"][key] += 1
            for issue in audit.get("issues", []) or []:
                summary["issues"][f"{key}|{issue}"] += 1

            rows.append({
                "id": rid,
                "answer": answer,
                "surface": route["surface"],
                "route_program": route["program"],
                "training_program": route["training_program"],
                "query_op_support": route["query_op_support"],
                "audit": audit,
            })

    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    def _counter_to_dict(c: Counter) -> dict[str, int]:
        return dict(c.most_common())

    report = {
        "rows": len(rows),
        "cipher_timeout": cipher_timeout,
        "surface_filter": surface_filter,
        "max_rows": max_rows,
        "summary": {name: _counter_to_dict(counter) for name, counter in summary.items()},
        "out": out,
    }
    with open(summary_out, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/competition/train.csv")
    parser.add_argument("--stable-traces", default="data/transformation/pool/competition/competition_traced.jsonl")
    parser.add_argument("--out", default="data/transformation/routes/program_audit.jsonl")
    parser.add_argument("--summary", default="data/transformation/routes/program_audit.summary.json")
    parser.add_argument("--cipher-timeout", type=float, default=1.0)
    parser.add_argument("--surface", choices=["all", "numeric_visible", "cipher_digit"], default="all")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    report = audit_train_csv(
        args.train_csv,
        args.stable_traces,
        args.out,
        args.summary,
        args.cipher_timeout,
        args.surface,
        args.max_rows,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
