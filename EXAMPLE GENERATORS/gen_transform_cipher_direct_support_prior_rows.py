#!/usr/bin/env python3
"""Generate direct-support cipher rows with a late latent prior.

This route is for cipher-digit rows where the query operator has direct support
examples, but stricter visible-slot/local-map/best-fit routes did not produce a
safe trace. The trace gives the model a stable workspace:

1. isolate same-query-operator support rows,
2. summarize visible output lengths/symbols,
3. compute query-slot candidates,
4. make a labeled final prior choice only after those reductions.

It intentionally excludes rows already covered by stronger generators.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from itertools import islice

from generators.gen_transform_cipher_slot_rule_rows import TEMPLATES, _render_slots
from generators.transform_router import (
    extract_examples_and_query,
    is_transformation_prompt,
    route_transform_prompt,
)
from training.data import BOXED_INSTRUCTION


DEFAULT_EXCLUDE_PATHS = [
    "data/transformation/pool/competition/cipher_map_visible.jsonl",
    "data/transformation/pool/competition/cipher_one_shot_visible.jsonl",
    "data/transformation/pool/competition/competition_traced.jsonl",
    "data/transformation/pool/competition/one_shot_numeric_unique.jsonl",
    "data/transformation/pool/competition/numeric_hard_way_prior.jsonl",
    "data/transformation/pool/competition/numeric_direct_ambiguity_prior.jsonl",
    "data/transformation/pool/competition/numeric_single_witness_prior.jsonl",
    "data/transformation/pool/competition/numeric_unseen_operator_prior.jsonl",
    "data/transformation/pool/competition/cipher_visible_slot_rule.jsonl",
    "data/transformation/pool/competition/cipher_query_op_local_rule.jsonl",
    "data/transformation/pool/competition/cipher_single_witness_local_prior.jsonl",
    "data/transformation/pool/competition/cipher_unseen_answer_space_prior.jsonl",
    "data/transformation/pool/competition/cipher_hard_way_prior.jsonl",
    "data/transformation/pool/competition/cipher_missing_symbol_prior.jsonl",
]

MAX_PREVIEW = 18


def _safe_text(value: object) -> str:
    """Avoid accidental placeholder marker '???' in assistant traces."""
    text = str(value)
    while "???" in text:
        text = text.replace("???", "? ? ?")
    return text


def _load_excluded_ids(paths: list[str]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
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


def _support_ops(examples: list[tuple[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for lhs, _rhs in examples:
        if len(lhs) == 5:
            counts[lhs[2]] += 1
    return counts


def _fmt_support_ops(counts: Counter[str]) -> str:
    return ", ".join(f"{op}:{count}" for op, count in sorted(counts.items())) or "none"


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _preview(values: list[str], limit: int = MAX_PREVIEW) -> str:
    if not values:
        return "[]"
    shown = list(islice(values, limit))
    suffix = "" if len(values) <= limit else f", ... +{len(values) - limit}"
    return _safe_text("[" + ", ".join(shown) + suffix + "]")


def _query_slot_candidates(query: str) -> list[str]:
    if len(query) != 5:
        return []
    return _dedupe([_render_slots(query, seq) for _name, seq in TEMPLATES])


def _same_query_op_examples(prompt: str) -> tuple[list[tuple[str, str]], str, str | None]:
    examples, query = extract_examples_and_query(prompt)
    if len(query) != 5:
        return [], query, None
    query_op = query[2]
    same = [(lhs, rhs) for lhs, rhs in examples if len(lhs) == 5 and lhs[2] == query_op]
    return same, query, query_op


def _trace_text(prompt: str, answer: str) -> tuple[str, dict[str, object]] | None:
    examples, query = extract_examples_and_query(prompt)
    same, _query, query_op = _same_query_op_examples(prompt)
    if query_op is None or not same:
        return None

    support_ops = _support_ops(examples)
    query_op_support = support_ops.get(query_op, 0)
    query_candidates = _query_slot_candidates(query)
    same_rhs = [rhs for _lhs, rhs in same]
    same_rhs_symbols = sorted(set("".join(same_rhs)))
    same_rhs_lens = sorted(set(len(rhs) for rhs in same_rhs))
    visible_symbols = set("".join(lhs + rhs for lhs, rhs in examples) + query)
    chosen_fresh_symbols = sorted(ch for ch in set(answer) if ch != "-" and ch not in visible_symbols)
    chosen_from_same_rhs = all(ch == "-" or ch in set("".join(same_rhs)) for ch in answer)
    chosen_len_seen = len(answer) in same_rhs_lens

    same_lines: list[str] = []
    for idx, (lhs, rhs) in enumerate(same, 1):
        same_lines.append(f"  Ex{idx}: {_safe_text(lhs)} -> {_safe_text(rhs)}")

    lines = [
        "TRANS_ROUTE_V1",
        "",
        "Flow:",
        "  parse support examples and query",
        "  classify surface",
        "  mark cipher digit slots and operator slot",
        "  extract query_op",
        "  count support examples using query_op",
        "  try visible slot, local decoded, best-fit, and missing-symbol routes first",
        "  if those fail but query_op_support is positive, build direct-support state",
        "  make any latent prior choice only at the end",
        "",
        "Surface:",
        "  kind = cipher_digit",
        f"  query = {_safe_text(query)}",
        "  op_pos = 2",
        "  digit_slots = [0, 1, 3, 4]",
        f"  query_op = {_safe_text(query_op)}",
        f"  support_ops = {_fmt_support_ops(support_ops)}",
        f"  query_op_support = {query_op_support}",
        "",
        "Route:",
        "  program = Cipher Direct-Support Answer-Space Prior",
        "  reason = direct query-op rows exist, but verified local-map routes did not close",
        f"  route_check = query_op_support:{query_op_support} -> Cipher Direct-Support Answer-Space Prior",
        "  certainty: direct-support answer-space prior",
        "",
        "WhenToUse:",
        "  use only after visible-slot, local-map, best-fit, and missing-symbol routes fail",
        "  use only when query_op_support >= 1",
        "  this is an in-distribution prior, not a support proof",
        "  do not claim executable replay for the final choice",
        "",
        "MechanicalReduction:",
        "  step1 = isolate rows using the query operator",
        "  step2 = record direct output lengths and symbols",
        "  step3 = compute direct query-slot candidates from A/B/OP/C/D",
        "  step4 = final prior must be made after this direct-support state is visible",
        "",
        "DirectSupport:",
        f"  support_rows = {len(same)}",
        f"  rhs_lengths = {same_rhs_lens}",
        f"  rhs_symbols = {_safe_text(''.join(same_rhs_symbols) if same_rhs_symbols else 'none')}",
        f"  rhs_preview = {_preview(same_rhs, limit=8)}",
        *same_lines,
        "",
        "QuerySlotSpace:",
        "  slots: A=x0 B=x1 OP=x2 C=x3 D=x4",
        f"  candidate_count = {len(query_candidates)}",
        f"  candidate_preview = {_preview(query_candidates)}",
        "",
        "AnswerSpace:",
        "  deterministic_status = not closed by current verifier",
        f"  direct_support_count = {len(same)}",
        f"  query_slot_count = {len(query_candidates)}",
        f"  direct_rhs_lengths = {same_rhs_lens}",
        f"  direct_rhs_symbols = {_safe_text(''.join(same_rhs_symbols) if same_rhs_symbols else 'none')}",
        "",
        "PriorChoice:",
        "  teleport_boundary = final choice only",
        "  basis = train-distribution direct-support latent prior after direct evidence reduction",
        f"  chosen = {_safe_text(answer)}",
        f"  chosen_length = {len(answer)}",
        f"  chosen_length_seen_in_direct_support = {str(chosen_len_seen).lower()}",
        f"  chosen_symbols_from_direct_rhs = {str(chosen_from_same_rhs).lower()}",
        f"  chosen_fresh_symbols = {''.join(chosen_fresh_symbols) if chosen_fresh_symbols else 'none'}",
        "  certainty: direct-support answer-space prior",
        "",
        "SelfCheck:",
        "  proof status: direct evidence reduced, final choice is a labeled prior",
        "  support_replay = not claimed for this route",
        "  prior is allowed only after direct query-op support is isolated",
        "",
        "Apply:",
        f"  query = {_safe_text(query)}",
        "  selected_space = DirectSupportLatentPrior",
        f"  output = {_safe_text(answer)}",
        f"Answer: {_safe_text(answer)}",
    ]
    return "\n".join(lines), {
        "query_op_support": query_op_support,
        "chosen_len_seen": chosen_len_seen,
        "chosen_from_same_rhs": chosen_from_same_rhs,
        "chosen_fresh_symbols": "".join(chosen_fresh_symbols) if chosen_fresh_symbols else "none",
    }


def _assistant_content(trace_text: str, answer: str) -> str:
    if "{" in answer or "}" in answer:
        return f"<think>\n{trace_text}\n</think>\nThe final answer is: {answer}"
    return f"<think>\n{trace_text}\n</think>\n\\boxed{{{answer}}}"


def build_rows(
    train_csv: str,
    out: str,
    summary_out: str,
    exclude_paths: list[str] | None = None,
) -> dict:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    excluded_ids = _load_excluded_ids(exclude_paths or DEFAULT_EXCLUDE_PATHS)
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
            if route["program"] not in {"TRANS_MULTI_SUPPORT_V1", "TRANS_ONE_SHOT_V1"}:
                continue
            if rid in excluded_ids:
                counters[f"skip_stronger_covered:{route['program']}"] += 1
                continue

            built = _trace_text(prompt, answer)
            if built is None:
                counters[f"trace_builder_reject:{route['program']}"] += 1
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
                "mode": "cipher_direct_support_answer_space_prior",
                "generator": "gen_transform_cipher_direct_support_prior_rows",
                "route_program": route["program"],
                "query_op_support": meta["query_op_support"],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
            counters["rows"] += 1
            counters[f"rows:{route['program']}"] += 1
            counters[f"chosen_len_seen:{meta['chosen_len_seen']}"] += 1
            counters[f"chosen_from_same_rhs:{meta['chosen_from_same_rhs']}"] += 1
            counters[f"chosen_fresh_symbols:{meta['chosen_fresh_symbols']}"] += 1

    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "out": out,
        "excluded_ids": len(excluded_ids),
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
    parser.add_argument("--out", default="data/transformation/pool/competition/cipher_direct_support_answer_space_prior.jsonl")
    parser.add_argument("--summary", default="data/transformation/pool/competition/cipher_direct_support_answer_space_prior.summary.json")
    parser.add_argument("--exclude-path", action="append", default=None)
    args = parser.parse_args()
    print(json.dumps(
        build_rows(
            train_csv=args.train_csv,
            out=args.out,
            summary_out=args.summary,
            exclude_paths=args.exclude_path,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()
