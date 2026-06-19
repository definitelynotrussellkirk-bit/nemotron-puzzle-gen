#!/usr/bin/env python3
"""Generate cipher rows requiring prior completion of unseen answer symbols."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone

from generators.trace_transform import build_cipher_missing_symbol_trace
from generators.transform_router import extract_examples_and_query, is_transformation_prompt, route_transform_prompt
from solvers.cipher_digit import find_op_pos, solve as cipher_solve
from training.data import BOXED_INSTRUCTION


def _visible_symbols(prompt: str) -> set[str]:
    symbols: set[str] = set()
    examples, query = extract_examples_and_query(prompt)
    for lhs, rhs in examples:
        symbols.update(lhs)
        symbols.update(rhs)
    symbols.update(query)
    return symbols


def _has_fresh_answer_symbol(prompt: str, answer: str) -> bool:
    symbols = _visible_symbols(prompt)
    return any((ch != "-" and ch not in symbols) for ch in answer)


def _assistant_content(trace_text: str, answer: str) -> str:
    if "{" in answer or "}" in answer:
        return f"<think>\n{trace_text}\n</think>\nThe final answer is: {answer}"
    return f"<think>\n{trace_text}\n</think>\n\\boxed{{{answer}}}"


def build_rows(train_csv: str, out: str, summary_out: str, timeout: float) -> dict:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    counters = Counter()
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

            if not _has_fresh_answer_symbol(prompt, answer):
                counters[f"skip_no_fresh_symbol:{route['program']}"] += 1
                continue

            result = cipher_solve(prompt, answer=answer, timeout=timeout)
            if not result:
                counters[f"gold_solve_fail:{route['program']}"] += 1
                continue
            if result.get("answer") != answer:
                counters[f"prediction_mismatch:{route['program']}"] += 1
                continue

            examples, query = extract_examples_and_query(prompt)
            op_pos = result.get("op_pos") or find_op_pos(examples, query, answer)
            if op_pos is None:
                counters[f"op_pos_fail:{route['program']}"] += 1
                continue
            trace = build_cipher_missing_symbol_trace(
                examples,
                query,
                answer,
                result["mapping"],
                result.get("combos", {}),
                op_pos,
            )
            if trace is None:
                counters[f"trace_builder_reject:{route['program']}"] += 1
                continue
            trace_text, trace_pred = trace
            if trace_pred != answer:
                counters[f"trace_prediction_mismatch:{route['program']}"] += 1
                continue
            if "???" in trace_text or "???" in answer:
                counters[f"placeholder_reject:{route['program']}"] += 1
                continue

            rows.append({
                "messages": [
                    {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": _assistant_content(trace_text, answer)},
                ],
                "answer": answer,
                "id": rid,
                "puzzle_type": "transformation",
                "mode": "cipher_missing_symbol_prior",
                "generator": "gen_transform_cipher_missing_symbol_rows",
                "route_program": route["program"],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
            counters["rows"] += 1
            counters[f"rows:{route['program']}"] += 1

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
    parser.add_argument("--out", default="data/transformation/pool/competition/cipher_missing_symbol_prior.jsonl")
    parser.add_argument("--summary", default="data/transformation/pool/competition/cipher_missing_symbol_prior.summary.json")
    parser.add_argument("--timeout", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(build_rows(args.train_csv, args.out, args.summary, args.timeout), indent=2))


if __name__ == "__main__":
    main()
