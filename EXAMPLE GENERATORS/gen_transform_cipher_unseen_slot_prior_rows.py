#!/usr/bin/env python3
"""Generate cipher unseen-operator rows with reduced answer-space traces.

This route is intentionally a prior, not a proof. It covers train.csv rows
where the query operator is absent from support, but the stored answer is inside
a mechanically computed visible answer space. The trace gives the model a stable
workspace before the final prior choice:

1. QuerySlotSpace from the query's visible A/B/OP/C/D slots.
2. SupportMapArithmeticSpace when the support-only cipher map is solvable.
3. A labeled final prior choice inside the reduced space.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from itertools import islice

from generators.gen_transform_cipher_slot_rule_rows import TEMPLATES, SlotSeq, _render_slots
from generators.transform_router import (
    extract_examples_and_query,
    is_transformation_prompt,
    route_transform_prompt,
)
from solvers.cipher_digit import COMBOS, calc, fmtv, make_op, solve as solve_cipher
from training.data import BOXED_INSTRUCTION


MAX_LISTED_CANDIDATES = 18


def _support_ops(examples: list[tuple[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for lhs, _rhs in examples:
        if len(lhs) == 5:
            counts[lhs[2]] += 1
    return counts


def _fmt_support_ops(counts: Counter[str]) -> str:
    return ", ".join(f"{op}:{count}" for op, count in sorted(counts.items())) or "none"


def _dedupe_stable(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _query_slot_candidates(query: str) -> tuple[list[str], dict[str, str]]:
    """Return all visible query-slot candidates plus the first rule per value."""
    if len(query) != 5:
        return [], {}
    values: list[str] = []
    first_rule: dict[str, str] = {}
    for name, seq in TEMPLATES:
        value = _render_slots(query, seq)
        if value not in first_rule:
            first_rule[value] = name
        values.append(value)
    return _dedupe_stable(values), first_rule


def _encode_with_reverse_map(value: str, reverse_map: dict[int, str]) -> str | None:
    out: list[str] = []
    for char in value:
        if char == "-":
            out.append("-")
            continue
        if not char.isdigit():
            return None
        sym = reverse_map.get(int(char))
        if sym is None:
            return None
        out.append(sym)
    return "".join(out)


def _support_map_candidates(prompt: str, query: str, timeout: float = 0.25) -> tuple[list[str], dict[str, object]]:
    """Enumerate visible encodable answers from a support-only cipher map.

    This does not use the gold answer. If no support-only map can be solved, the
    candidate list is empty and the trace says this mechanical space is closed.
    """
    solved = solve_cipher(prompt, answer=None, timeout=timeout)
    if not solved:
        return [], {"status": "not_solved"}

    op_pos = int(solved.get("op_pos", 2))
    mapping = solved.get("mapping") or {}
    if not isinstance(mapping, dict):
        return [], {"status": "not_solved"}
    if len(query) <= op_pos:
        return [], {"status": "not_solved"}

    q_op = query[op_pos]
    digit_positions = [idx for idx in range(5) if idx != op_pos and idx < len(query)]
    q_abcd = tuple(query[idx] for idx in digit_positions)
    if len(q_abcd) != 4 or any(sym not in mapping for sym in q_abcd):
        return [], {"status": "map_incomplete", "op_pos": op_pos}

    reverse_map = {digit: sym for sym, digit in mapping.items()}
    digits = tuple(mapping[sym] for sym in q_abcd)

    candidates: list[str] = []
    combo_names: list[str] = []
    for order, op, fmt in COMBOS:
        left, right = make_op(*digits, order)
        value = calc(left, right, op)
        formatted = fmtv(value, fmt) if value is not None else None
        if not formatted:
            continue

        direct = _encode_with_reverse_map(formatted, reverse_map)
        if direct:
            candidates.append(direct)
            combo_names.append(f"{order}/{op}/{fmt}")

        if q_op:
            if formatted.startswith("-"):
                encoded_abs = _encode_with_reverse_map(formatted[1:], reverse_map)
                if encoded_abs:
                    candidates.append(q_op + encoded_abs)
                    combo_names.append(f"{order}/{op}/{fmt}/op_abs_prefix")
            encoded_prefix = _encode_with_reverse_map(formatted, reverse_map)
            if encoded_prefix:
                candidates.append(q_op + encoded_prefix)
                combo_names.append(f"{order}/{op}/{fmt}/op_prefix")

    return _dedupe_stable(candidates), {
        "status": "solved",
        "op_pos": op_pos,
        "map_size": len(mapping),
        "combos_tested": len(COMBOS),
        "combo_preview": _dedupe_stable(combo_names)[:5],
    }


def _preview(values: list[str], limit: int = MAX_LISTED_CANDIDATES) -> str:
    if not values:
        return "[]"
    shown = list(islice(values, limit))
    suffix = "" if len(values) <= limit else f", ... +{len(values) - limit}"
    return "[" + ", ".join(shown) + suffix + "]"


def _source_for_answer(
    answer: str,
    query_candidates: list[str],
    support_candidates: list[str],
) -> str | None:
    in_query = answer in query_candidates
    in_support = answer in support_candidates
    if in_query and in_support:
        return "QuerySlotSpace + SupportMapArithmeticSpace"
    if in_query:
        return "QuerySlotSpace"
    if in_support:
        return "SupportMapArithmeticSpace"
    return None


def _trace_text(prompt: str, answer: str) -> tuple[str, dict[str, object]] | None:
    examples, query = extract_examples_and_query(prompt)
    query_op = query[2] if len(query) == 5 else None
    support_ops = _support_ops(examples)
    query_op_support = support_ops.get(query_op, 0) if query_op is not None else 0

    query_candidates, slot_rules = _query_slot_candidates(query)
    support_candidates, support_meta = _support_map_candidates(prompt, query)
    answer_space = _dedupe_stable(query_candidates + support_candidates)
    answer_source = _source_for_answer(answer, query_candidates, support_candidates)
    if answer_source is None:
        return None

    slot_rule = slot_rules.get(answer, "not_query_slot")
    visible_symbols = set("".join(lhs + rhs for lhs, rhs in examples) + query)
    answer_visible_only = all(char in visible_symbols or char == "-" for char in answer)

    lines = [
        "TRANS_ROUTE_V1",
        "",
        "Flow:",
        "  parse support examples and query",
        "  classify surface",
        "  mark cipher digit slots and operator slot",
        "  extract query_op",
        "  count support examples using query_op",
        "  try verified same-operator and local decoded routes first",
        "  if query_op_support is zero, build reduced answer spaces",
        "  make any prior choice only after the reduction",
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
        "  program = Cipher Unseen Answer-Space Prior",
        "  reason = query operator is absent; reduce visible answer spaces before the final prior",
        f"  route_check = query_op_support:{query_op_support} -> Cipher Unseen Answer-Space Prior",
        "  certainty: unseen answer-space prior",
        "",
        "WhenToUse:",
        "  use only after verified same-operator and local decoded routes fail",
        "  use only when query_op_support = 0",
        "  this is an in-distribution prior, not a support proof",
        "  do not claim query-op support replay",
        "",
        "MechanicalReduction:",
        "  step1 = operator slot is known, but query operator has no support rows",
        "  step2 = QuerySlotSpace is computed directly from query slots",
        "  step3 = SupportMapArithmeticSpace is computed only if support rows solve a cipher map",
        "  step4 = final prior must stay inside the reduced visible answer space",
        f"  answer_visible_only = {str(answer_visible_only).lower()}",
        "",
        "QuerySlotSpace:",
        "  slots: A=x0 B=x1 OP=x2 C=x3 D=x4",
        f"  candidate_count = {len(query_candidates)}",
        f"  candidate_preview = {_preview(query_candidates)}",
        "",
        "SupportMapSpace:",
        f"  map_status = {support_meta.get('status')}",
        f"  map_size = {support_meta.get('map_size', 0)}",
        f"  combos_tested = {support_meta.get('combos_tested', 0)}",
        f"  candidate_count = {len(support_candidates)}",
        f"  candidate_preview = {_preview(support_candidates)}",
        "",
        "AnswerSpace:",
        f"  union_count = {len(answer_space)}",
        f"  union_preview = {_preview(answer_space)}",
        f"  selected_space = {answer_source}",
        "",
        "PriorChoice:",
        "  teleport_boundary = final choice only",
        "  basis = train-distribution unseen-op prior inside the reduced answer space",
        f"  slot_rule_if_query_slot = {slot_rule}",
        f"  chosen = {answer}",
        "  certainty: unseen answer-space prior",
        "",
        "SelfCheck:",
        "  proof status: no query-op witness",
        "  support_replay = not available for the query operator",
        "  prior is allowed only after mechanical space reduction",
        "",
        "Apply:",
        f"  query = {query}",
        f"  selected_space = {answer_source}",
        f"  output = {answer}",
        f"Answer: {answer}",
    ]
    meta = {
        "answer_source": answer_source,
        "query_slot_candidates": len(query_candidates),
        "support_map_candidates": len(support_candidates),
        "answer_space_candidates": len(answer_space),
        "answer_visible_only": answer_visible_only,
        "slot_rule": slot_rule,
        "support_map_status": support_meta.get("status"),
    }
    return "\n".join(lines), meta


def _assistant_content(trace_text: str, answer: str) -> str:
    if "{" in answer or "}" in answer:
        return f"<think>\n{trace_text}\n</think>\nThe final answer is: {answer}"
    return f"<think>\n{trace_text}\n</think>\n\\boxed{{{answer}}}"


def build_rows(train_csv: str, out: str, summary_out: str) -> dict:
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
            if route["program"] != "TRANS_UNSEEN_PRIOR_V1":
                continue

            built = _trace_text(prompt, answer)
            if built is None:
                counters["answer_not_in_reduced_space"] += 1
                continue
            trace, meta = built
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
                "mode": "cipher_unseen_answer_space_prior",
                "generator": "gen_transform_cipher_unseen_slot_prior_rows",
                "route_program": route["program"],
                "answer_source": meta["answer_source"],
                "slot_rule": meta["slot_rule"],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
            counters["rows"] += 1
            counters[f"rows:{route['program']}"] += 1
            counters[f"answer_source:{meta['answer_source']}"] += 1
            counters[f"support_map_status:{meta['support_map_status']}"] += 1
            counters[f"answer_visible_only:{meta['answer_visible_only']}"] += 1
            counters[f"slot_rule:{meta['slot_rule']}"] += 1

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
    parser.add_argument("--out", default="data/transformation/pool/competition/cipher_unseen_answer_space_prior.jsonl")
    parser.add_argument("--summary", default="data/transformation/pool/competition/cipher_unseen_answer_space_prior.summary.json")
    args = parser.parse_args()
    print(json.dumps(build_rows(args.train_csv, args.out, args.summary), indent=2))


if __name__ == "__main__":
    main()
