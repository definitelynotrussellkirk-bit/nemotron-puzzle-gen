#!/usr/bin/env python3
"""Generate deterministic best-fit fallback transformation rows.

This is intentionally not a verifier.  It covers rows where routing is
deterministic, but the visible examples do not force a unique mechanical rule.
The builder chooses a route-consistent best-fit pattern learned from the
train.csv distribution and emits a trace only when that deterministic choice
matches the label.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generators.trace_transform import (
    ALL_OPS,
    _apply_style,
    _compute_op,
    _ordered_ints,
    _show_op,
    _show_style_step,
)
from generators.transform_program_audit import (
    Rule,
    _apply_rule_to_query,
    _first_order_style_context,
    _parse_numeric_examples,
    _parse_numeric_query,
    _rules_fitting_group,
)
from generators.transform_router import (
    extract_examples_and_query,
    is_transformation_prompt,
    route_transform_prompt,
)
from training.data import BOXED_INSTRUCTION


BEST_FIT_FALLBACK_NAME = "Best-Fitting Rule Fallback"


@dataclass(frozen=True)
class Candidate:
    answer: str
    route_program: str
    order_label: str
    order_key: str
    op_name: str
    style: str
    source: str
    scan_rank: int
    context_op: str | None = None
    context_support: int = 0

    @property
    def full_feature(self) -> tuple[str, str, str, str, str]:
        return (self.route_program, self.order_key, self.op_name, self.style, self.source)

    @property
    def op_style_feature(self) -> tuple[str, str, str]:
        return (self.route_program, self.op_name, self.style)

    @property
    def op_feature(self) -> tuple[str, str]:
        return (self.route_program, self.op_name)


def _numeric_candidates(prompt: str, route: dict[str, Any]) -> list[Candidate]:
    examples, query = extract_examples_and_query(prompt)
    by_op = _parse_numeric_examples(examples)
    parsed_query = _parse_numeric_query(query)
    if by_op is None or parsed_query is None:
        return []
    q_a, q_op, q_b = parsed_query
    route_program = str(route.get("program") or "")
    out: list[Candidate] = []

    if route_program == "TRANS_MULTI_SUPPORT_V1":
        rules = _rules_fitting_group(by_op.get(q_op, []), q_op)
        for rank, rule in enumerate(rules):
            answer = _apply_rule_to_query(rule, q_a, q_op, q_b)
            if answer is None:
                continue
            out.append(Candidate(
                answer=answer,
                route_program=route_program,
                order_label=rule.order_label,
                order_key=rule.order_key,
                op_name=rule.op_name,
                style=rule.style,
                source="direct_support",
                scan_rank=rank,
                context_op=q_op,
                context_support=int(route.get("query_op_support") or 0),
            ))
        return out

    if route_program == "TRANS_ONE_SHOT_V1":
        witness = by_op.get(q_op, [])
        if len(witness) != 1:
            return []
        w_a, w_b, w_expected = witness[0]
        rank = 0
        # Contexts are already sorted by the deterministic one-shot program:
        # most-supported non-query op, then operator char, then scan order.
        for context in _first_order_style_context(by_op, q_op):
            context_op = next(
                (op for op, group in by_op.items()
                 if op != q_op and len(group) >= 2
                 and any(r.order_key == context.order_key and r.op_name == context.op_name and r.style == context.style
                         for r in _rules_fitting_group(group, op))),
                None,
            )
            context_support = len(by_op.get(context_op, [])) if context_op else 0
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
                answer = _apply_rule_to_query(pseudo, q_a, q_op, q_b)
                if answer is None:
                    continue
                out.append(Candidate(
                    answer=answer,
                    route_program=route_program,
                    order_label=context.order_label,
                    order_key=context.order_key,
                    op_name=op_name,
                    style=context.style,
                    source="one_shot_context",
                    scan_rank=rank,
                    context_op=context_op,
                    context_support=context_support,
                ))
                rank += 1
        return out

    if route_program == "TRANS_UNSEEN_PRIOR_V1":
        rank = 0
        for context in _first_order_style_context(by_op, q_op):
            context_op = next(
                (op for op, group in by_op.items()
                 if op != q_op and len(group) >= 2
                 and any(r.order_key == context.order_key and r.op_name == context.op_name and r.style == context.style
                         for r in _rules_fitting_group(group, op))),
                None,
            )
            context_support = len(by_op.get(context_op, [])) if context_op else 0
            for op_name, _fn in ALL_OPS:
                pseudo = Rule(context.order_label, context.order_key, op_name, context.style)
                answer = _apply_rule_to_query(pseudo, q_a, q_op, q_b)
                if answer is None:
                    continue
                out.append(Candidate(
                    answer=answer,
                    route_program=route_program,
                    order_label=context.order_label,
                    order_key=context.order_key,
                    op_name=op_name,
                    style=context.style,
                    source="unseen_context_prior",
                    scan_rank=rank,
                    context_op=context_op,
                    context_support=context_support,
                ))
                rank += 1
        return out

    return []


def _feature_stats(items: list[tuple[str, list[Candidate]]]) -> dict[str, dict[tuple, tuple[float, int, int]]]:
    """Build deterministic ranking tables from train distribution labels."""
    hits: dict[str, Counter] = {
        "full": Counter(),
        "op_style": Counter(),
        "op": Counter(),
    }
    totals: dict[str, Counter] = {
        "full": Counter(),
        "op_style": Counter(),
        "op": Counter(),
    }
    for answer, candidates in items:
        seen = {"full": set(), "op_style": set(), "op": set()}
        for cand in candidates:
            features = {
                "full": cand.full_feature,
                "op_style": cand.op_style_feature,
                "op": cand.op_feature,
            }
            for name, feature in features.items():
                if feature in seen[name]:
                    continue
                seen[name].add(feature)
                totals[name][feature] += 1
                if any(getattr(other, f"{name}_feature") == feature and other.answer == answer
                       for other in candidates):
                    hits[name][feature] += 1
    stats: dict[str, dict[tuple, tuple[float, int, int]]] = {}
    for name in hits:
        stats[name] = {}
        for feature, total in totals[name].items():
            hit = hits[name][feature]
            stats[name][feature] = (hit / total, hit, total)
    return stats


def _rank_key(cand: Candidate, stats: dict[str, dict[tuple, tuple[float, int, int]]]) -> tuple:
    full = stats["full"].get(cand.full_feature, (0.0, 0, 0))
    op_style = stats["op_style"].get(cand.op_style_feature, (0.0, 0, 0))
    op = stats["op"].get(cand.op_feature, (0.0, 0, 0))
    return (
        full[0], full[1], full[2],
        op_style[0], op_style[1], op_style[2],
        op[0], op[1], op[2],
        cand.context_support,
        -cand.scan_rank,
    )


def _select_candidate(candidates: list[Candidate], stats: dict[str, dict[tuple, tuple[float, int, int]]]) -> Candidate | None:
    if not candidates:
        return None
    return max(candidates, key=lambda cand: _rank_key(cand, stats))


def _reduction_counts(candidates: list[Candidate], chosen: Candidate) -> dict[str, int]:
    """Dense deterministic pool-reduction counters for the trace header."""
    source_count = sum(cand.source == chosen.source for cand in candidates)
    order_count = sum(cand.source == chosen.source and cand.order_key == chosen.order_key for cand in candidates)
    style_count = sum(
        cand.source == chosen.source
        and cand.order_key == chosen.order_key
        and cand.style == chosen.style
        for cand in candidates
    )
    op_count = sum(
        cand.source == chosen.source
        and cand.order_key == chosen.order_key
        and cand.style == chosen.style
        and cand.op_name == chosen.op_name
        for cand in candidates
    )
    answer_count = len({cand.answer for cand in candidates})
    return {
        "source": source_count,
        "order": order_count,
        "style": style_count,
        "op": op_count,
        "answers": answer_count,
    }


def _show_source(source: str) -> str:
    return {
        "direct_support": "direct query-operator support",
        "one_shot_context": "one witness plus context operator",
        "unseen_context_prior": "no query-op witness; context-prior candidates",
    }.get(source, source.replace("_", " "))


def _show_route_program(route_program: str) -> str:
    return {
        "TRANS_MULTI_SUPPORT_V1": "multi-support route",
        "TRANS_ONE_SHOT_V1": "one-shot route",
        "TRANS_UNSEEN_PRIOR_V1": "unseen-operator route",
    }.get(route_program, route_program)


def _show_order(order_key: str) -> str:
    return {
        "AB_CD": "keep both operands as written",
        "BA_DC": "reverse both operands",
        "AB_DC": "keep left, reverse right",
        "BA_CD": "reverse left, keep right",
    }.get(order_key, order_key)


def _show_style_name(style: str) -> str:
    return {
        "plain": "plain digits",
        "rev": "reverse digits",
        "abs": "absolute value",
        "abs_rev": "absolute value, then reverse digits",
        "dsum": "digit sum",
        "opprefix": "prefix query operator",
        "opprefix_rev": "reverse digits, then prefix query operator",
        "opsign": "operator sign only when negative",
        "opsign_always": "operator sign prefix",
        "tailsign": "operator sign suffix only when negative",
        "tailsign_always": "operator sign suffix",
        "rev_opsign": "reverse digits with operator sign prefix if negative",
        "rev_opsign_always": "reverse digits with operator sign prefix",
        "rev_tailsign": "reverse digits with operator sign suffix if negative",
        "rev_tailsign_always": "reverse digits with operator sign suffix",
    }.get(style, style.replace("_", " "))


def _show_op_name(op_name: str) -> str:
    return {
        "add": "add",
        "sub": "left minus right",
        "bsub": "right minus left",
        "mul": "multiply",
        "absdiff": "absolute difference",
        "concat": "concatenate left then right",
        "bconcat": "concatenate right then left",
        "negabsdiff": "negative absolute difference",
        "muladd1": "multiply, then add 1",
        "mulsub1": "multiply, then subtract 1",
        "addp1": "add, then add 1",
        "addm1": "add, then subtract 1",
        "subp1": "left minus right, then add 1",
        "subm1": "left minus right, then subtract 1",
        "floordiv": "left divided by right, floor",
        "bfloordiv": "right divided by left, floor",
        "mod": "left modulo right",
        "bmod": "right modulo left",
        "maxmod": "larger modulo smaller",
    }.get(op_name, op_name.replace("_", " "))


def _show_candidate(cand: Candidate) -> str:
    return (
        f"{_show_order(cand.order_key)}; "
        f"{_show_op_name(cand.op_name)}; "
        f"{_show_style_name(cand.style)}"
    )


def _support_ops_text(route: dict[str, Any]) -> str:
    support_ops = route.get("support_ops") or {}
    return ", ".join(f"{op}:{count}" for op, count in sorted(support_ops.items())) or "none"


def _render_trace(prompt: str, route: dict[str, Any], candidates: list[Candidate], chosen: Candidate, answer: str) -> str | None:
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

    route_program = route["program"]
    q_support = route["query_op_support"]
    if q_support >= 2:
        route_check = f"{q_support} query-operator examples -> best-fit fallback from multi-support route"
    elif q_support == 1:
        route_check = "1 query-operator example -> best-fit fallback from one-shot route"
    else:
        route_check = "0 query-operator examples -> best-fit fallback from unseen-operator route"

    lines = [
        "TRANS_ROUTE_V1",
        "",
        "Flow:",
        "  parse support examples and query",
        "  classify surface",
        "  extract query operator",
        "  count support examples using the query operator",
        "  if no unique rule is proven, enter best-fit fallback",
        "  exit to the first matching route",
        "",
        "Surface:",
        "  kind: numeric visible",
        f"  examples: {len(examples)}",
        f"  query: {query}",
        f"  query operator: {q_op}",
        f"  support operators: {_support_ops_text(route)}",
        f"  query-op support: {q_support}",
        "",
        "Route:",
        f"  method: {BEST_FIT_FALLBACK_NAME}",
        f"  starting route: {_show_route_program(route_program)}",
        "  certainty: best-fit fallback",
        "  proof status: not unique",
        f"  reason: no single verified rule; use the route's familiar fallback pattern",
        f"  route check: {route_check} -> {BEST_FIT_FALLBACK_NAME}",
        "",
        BEST_FIT_FALLBACK_NAME,
        "WhenToUse:",
        "  visible examples do not prove one unique rule",
        "  use a familiar numeric transformation pattern for this route",
        "  then execute the selected pattern mechanically",
        "",
        "PatternCue:",
        "  think in three slots: operand order, arithmetic operation, output format",
        "  prefer the route-consistent simple pattern",
        "  do not enumerate hidden candidates",
        "  do not claim support_pass",
        "",
        "RouteFacts:",
        f"  visible parse: numeric equation, {len(examples)} examples, query operator '{q_op}'",
        f"  query-operator support: {q_support}",
        f"  starting route: {_show_route_program(route_program)}",
        "  stored answer used for choice: no",
        "",
        "BestFitRule:",
        f"  family: {_show_source(chosen.source)}",
        f"  order: {_show_order(chosen.order_key)}",
        f"  operation: {_show_op_name(chosen.op_name)}",
        f"  format: {_show_style_name(chosen.style)}",
        "",
        "SelfCheck:",
        "  this is a best-fit fallback, not a proof of uniqueness",
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


# The renderer needs the active stats to show the same rank scores used by the
# selector without threading a large table through every helper.
_ACTIVE_STATS: dict[str, dict[tuple, tuple[float, int, int]]] = {}


def _assistant_content(trace: str, answer: str) -> str:
    """Use text fallback for answers that cannot safely live inside boxed braces."""
    if "{" in answer or "}" in answer:
        return f"<think>\n{trace}\n</think>\nThe final answer is: {answer}"
    return f"<think>\n{trace}\n</think>\n\\boxed{{{answer}}}"


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
            if route["surface"] != "numeric_visible":
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
            counters["numeric_rows"] += 1
            counters[f"route:{route['program']}"] += 1
            if candidates:
                counters[f"candidate_rows:{route['program']}"] += 1
                if any(cand.answer == row.get("answer", "") for cand in candidates):
                    counters[f"contains_label:{route['program']}"] += 1
            else:
                counters[f"no_candidates:{route['program']}"] += 1

    global _ACTIVE_STATS
    _ACTIVE_STATS = _feature_stats(training_items)
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for item in raw_items:
        chosen = _select_candidate(item["candidates"], _ACTIVE_STATS)
        if chosen is None:
            continue
        if chosen.answer != item["answer"]:
            counters[f"prediction_mismatch:{item['route']['program']}"] += 1
            continue
        trace = _render_trace(item["prompt"], item["route"], item["candidates"], chosen, item["answer"])
        if trace is None:
            counters[f"trace_reject:{item['route']['program']}"] += 1
            continue
        rows.append({
            "messages": [
                {"role": "user", "content": item["prompt"] + BOXED_INSTRUCTION},
                {"role": "assistant", "content": _assistant_content(trace, item["answer"])},
            ],
            "answer": item["answer"],
            "id": item["id"],
            "puzzle_type": "transformation",
            "mode": "numeric_hard_way_prior",
            "generator": "gen_transform_numeric_hard_way_rows",
            "route_program": "TRANS_BEST_FIT_FALLBACK_V1",
            "source_route": item["route"]["program"],
            "generated_at": now,
        })
        counters[f"rows:{item['route']['program']}"] += 1
        counters["rows"] += 1

    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = {
        "out": out,
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
    parser.add_argument("--out", default="data/transformation/pool/competition/numeric_hard_way_prior.jsonl")
    parser.add_argument("--summary", default="data/transformation/pool/competition/numeric_hard_way_prior.summary.json")
    args = parser.parse_args()
    print(json.dumps(build_rows(args.train_csv, args.out, args.summary), indent=2))


if __name__ == "__main__":
    main()
