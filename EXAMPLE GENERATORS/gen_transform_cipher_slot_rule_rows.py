#!/usr/bin/env python3
"""Generate cipher rows solved by visible symbol-slot rules.

This is a stronger route than cipher best-fit fallback: it does not infer a
digit mapping or search arithmetic. It only accepts rows where same-operator
support examples validate one raw slot transform, then applies that exact
transform to the query.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Iterable

from generators.transform_router import (
    extract_examples_and_query,
    is_transformation_prompt,
    route_transform_prompt,
)
from training.data import BOXED_INSTRUCTION

SlotSeq = tuple[str, ...]


NAMED_SLOT_RULES: list[tuple[str, SlotSeq]] = [
    ("ABCD", ("A", "B", "C", "D")),
    ("DCBA", ("D", "C", "B", "A")),
    ("CDAB", ("C", "D", "A", "B")),
    ("BADC", ("B", "A", "D", "C")),
    ("AB", ("A", "B")),
    ("CD", ("C", "D")),
    ("BA", ("B", "A")),
    ("DC", ("D", "C")),
    ("AD", ("A", "D")),
    ("BC", ("B", "C")),
    ("ACBD", ("A", "C", "B", "D")),
    ("BDAC", ("B", "D", "A", "C")),
]


def _slot_templates() -> list[tuple[str, SlotSeq]]:
    """Return deterministic slot-rule candidates, simple rules first."""
    templates: list[tuple[str, SlotSeq]] = []
    seen: set[SlotSeq] = set()
    for name, seq in NAMED_SLOT_RULES:
        templates.append((name, seq))
        seen.add(seq)

    letters = ("A", "B", "C", "D")
    for length in range(1, 5):
        for seq in itertools.permutations(letters, length):
            if seq not in seen:
                templates.append(("".join(seq), seq))
                seen.add(seq)

    # Operator-bearing variants stay visible-only. They handle opprefix-like
    # rows if train.csv exposes them, but still require support replay.
    for length in range(1, 4):
        for seq in itertools.permutations(letters, length):
            for name, seq2 in [
                ("OP+" + "".join(seq), ("OP",) + seq),
                ("".join(seq) + "+OP", seq + ("OP",)),
            ]:
                if seq2 not in seen:
                    templates.append((name, seq2))
                    seen.add(seq2)
    return templates


TEMPLATES = _slot_templates()


def _render_slots(lhs: str, seq: SlotSeq) -> str:
    values = {"A": lhs[0], "B": lhs[1], "OP": lhs[2], "C": lhs[3], "D": lhs[4]}
    return "".join(values[name] for name in seq)


def _same_operator_examples(prompt: str) -> tuple[list[tuple[str, str]], str, str | None]:
    examples, query = extract_examples_and_query(prompt)
    if len(query) != 5:
        return [], query, None
    query_op = query[2]
    same = [(lhs, rhs) for lhs, rhs in examples if len(lhs) == 5 and lhs[2] == query_op]
    return same, query, query_op


def find_visible_slot_rule(prompt: str, answer: str) -> tuple[str, SlotSeq, list[tuple[str, SlotSeq]]] | None:
    """Find a support-validated visible slot rule for one cipher prompt."""
    same, query, _query_op = _same_operator_examples(prompt)
    if len(same) < 2:
        return None

    matches: list[tuple[str, SlotSeq]] = []
    for name, seq in TEMPLATES:
        if not all(_render_slots(lhs, seq) == rhs for lhs, rhs in same):
            continue
        if _render_slots(query, seq) == answer:
            matches.append((name, seq))

    if not matches:
        return None

    matches.sort(key=lambda item: (
        1 if "OP" in item[1] else 0,
        len(item[1]),
        TEMPLATES.index(item),
    ))
    name, seq = matches[0]
    return name, seq, matches


def _support_ops(examples: Iterable[tuple[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for lhs, _rhs in examples:
        if len(lhs) == 5:
            counts[lhs[2]] += 1
    return counts


def _fmt_support_ops(counts: Counter[str]) -> str:
    return ", ".join(f"{op}:{count}" for op, count in sorted(counts.items())) or "none"


def _trace_text(prompt: str, answer: str, rule_name: str, seq: SlotSeq, all_matches: list[tuple[str, SlotSeq]]) -> str:
    examples, query = extract_examples_and_query(prompt)
    same, _query, query_op = _same_operator_examples(prompt)
    support_ops = _support_ops(examples)
    query_op_support = support_ops.get(query_op, 0) if query_op is not None else 0

    lines = [
        "TRANS_ROUTE_V1",
        "",
        "Flow:",
        "  parse support examples and query",
        "  classify surface",
        "  mark cipher digit slots and operator slot",
        "  extract query_op",
        "  count support examples using query_op",
        "  try visible slot rules before hidden-map fallback",
        "  exit when one slot rule passes all query-op support rows",
        "",
        "Surface:",
        "  kind = cipher_digit",
        f"  query = {query}",
        "  op_pos = 2",
        "  digit_slots = [0, 1, 3, 4]",
        f"  query_op = {query_op}",
        f"  support_ops = {_fmt_support_ops(support_ops)}",
        f"  query_op_support = {query_op_support}",
        "",
        "Route:",
        "  program = Cipher Visible Slot Rule",
        "  reason = same query-op support rows validate one raw symbol-slot transform",
        f"  route_check = query_op_support:{query_op_support} -> Cipher Visible Slot Rule",
        "  certainty: visible slot rule",
        "",
        "WhenToUse:",
        "  use this only when same-operator support rows all match the same slot rule",
        "  do not infer hidden digit values",
        "  do not use arithmetic",
        "",
        "SlotRule:",
        "  slots: A=x0 B=x1 OP=x2 C=x3 D=x4",
        f"  rule = {rule_name}",
        f"  output_slots = {' '.join(seq)}",
        f"  matching_slot_rules = {len(all_matches)}",
        "",
        "Verify:",
        f"  support_pass = {len(same)}/{len(same)}",
    ]
    for idx, (lhs, rhs) in enumerate(same, 1):
        got = _render_slots(lhs, seq)
        lines.extend([
            f"  Ex{idx}:",
            f"    input = {lhs}",
            f"    {rule_name} = {got}",
            f"    expected = {rhs}",
            "    match = PASS",
        ])

    query_out = _render_slots(query, seq)
    lines.extend([
        "",
        "Apply:",
        f"  query = {query}",
        f"  {rule_name} = {query_out}",
        f"  output = {query_out}",
        f"Answer: {answer}",
    ])
    return "\n".join(lines)


def _assistant_content(trace_text: str, answer: str) -> str:
    if "{" in answer or "}" in answer:
        return f"<think>\n{trace_text}\n</think>\nThe final answer is: {answer}"
    return f"<think>\n{trace_text}\n</think>\n\\boxed{{{answer}}}"


def build_rows(train_csv: str, out: str, summary_out: str) -> dict:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    counters: Counter[str] = Counter()

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

            found = find_visible_slot_rule(prompt, answer)
            if found is None:
                counters["slot_rule_not_found"] += 1
                continue

            rule_name, seq, all_matches = found
            trace = _trace_text(prompt, answer, rule_name, seq, all_matches)
            if "???" in trace or "???" in answer:
                counters["placeholder_reject"] += 1
                continue

            rows.append({
                "messages": [
                    {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": _assistant_content(trace, answer)},
                ],
                "answer": answer,
                "id": rid,
                "puzzle_type": "transformation",
                "mode": "cipher_visible_slot_rule",
                "generator": "gen_transform_cipher_slot_rule_rows",
                "route_program": route["program"],
                "slot_rule": rule_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
            counters["rows"] += 1
            counters[f"rows:{route['program']}"] += 1
            counters[f"slot_rule:{rule_name}"] += 1
            counters[f"matching_slot_rules:{len(all_matches)}"] += 1

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
    parser.add_argument("--out", default="data/transformation/pool/competition/cipher_visible_slot_rule.jsonl")
    parser.add_argument("--summary", default="data/transformation/pool/competition/cipher_visible_slot_rule.summary.json")
    args = parser.parse_args()
    print(json.dumps(build_rows(args.train_csv, args.out, args.summary), indent=2))


if __name__ == "__main__":
    main()
