#!/usr/bin/env python3
"""Generate audited numeric TRANS_ONE_SHOT_V1 competition rows.

This promotes only rows that are already routed to ONE_SHOT and whose visible
one-shot program predicts the stored label. It writes both:

- deterministic fixed-order accepted rows
- conservative unique-answer accepted rows
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone

from generators.trace_transform import build_numeric_one_shot_trace
from generators.transform_program_audit import (
    _one_shot_answers,
    _parse_numeric_examples,
    _parse_numeric_query,
)
from generators.transform_router import (
    extract_examples_and_query,
    is_transformation_prompt,
    route_transform_prompt,
)
from training.data import BOXED_INSTRUCTION


def _one_shot_answer_stats(prompt: str, answer: str) -> tuple[bool, bool]:
    examples, query = extract_examples_and_query(prompt)
    by_op = _parse_numeric_examples(examples)
    parsed_query = _parse_numeric_query(query)
    if by_op is None or parsed_query is None:
        return False, False
    q_a, q_op, q_b = parsed_query
    answers, _issues = _one_shot_answers(by_op, q_a, q_op, q_b)
    unique_answers = list(dict.fromkeys(answers))
    deterministic_ok = bool(answers and answers[0] == answer)
    unique_ok = bool(len(unique_answers) == 1 and unique_answers[0] == answer)
    return deterministic_ok, unique_ok


def build_rows(train_csv: str, out_first: str, out_unique: str, summary_out: str) -> dict:
    os.makedirs(os.path.dirname(out_first), exist_ok=True)
    os.makedirs(os.path.dirname(out_unique), exist_ok=True)
    counters = Counter()
    first_rows = []
    unique_rows = []

    with open(train_csv) as f:
        for row in csv.DictReader(f):
            prompt = row["prompt"]
            if not is_transformation_prompt(prompt):
                continue
            counters["transform_rows"] += 1
            rid = str(row.get("id", ""))
            answer = row.get("answer", "")
            route = route_transform_prompt(prompt, row_id=rid, stable_trace_ids=set())
            if route["surface"] != "numeric_visible":
                continue
            counters["numeric_rows"] += 1
            if route["program"] != "TRANS_ONE_SHOT_V1":
                continue
            counters["numeric_one_shot_rows"] += 1

            examples, query = extract_examples_and_query(prompt)
            result = build_numeric_one_shot_trace(examples, query, answer)
            if result is None:
                counters["trace_builder_reject"] += 1
                continue
            trace_text, pred = result
            if pred != answer:
                counters["prediction_mismatch"] += 1
                continue

            deterministic_ok, unique_ok = _one_shot_answer_stats(prompt, answer)
            if not deterministic_ok:
                counters["audit_disagrees"] += 1
                continue

            output_row = {
                "messages": [
                    {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": f"<think>\n{trace_text}\n</think>\n\\boxed{{{answer}}}"},
                ],
                "answer": answer,
                "id": rid,
                "puzzle_type": "transformation",
                "mode": "numeric_one_shot",
                "generator": "gen_transform_one_shot_rows",
                "route_program": route["program"],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            first_rows.append(output_row)
            counters["first_order_rows"] += 1
            if unique_ok:
                unique_row = dict(output_row)
                unique_row["mode"] = "numeric_one_shot_unique"
                unique_rows.append(unique_row)
                counters["unique_rows"] += 1
            else:
                counters["first_order_non_unique"] += 1

    with open(out_first, "w") as f:
        for row in first_rows:
            f.write(json.dumps(row) + "\n")
    with open(out_unique, "w") as f:
        for row in unique_rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "out_first": out_first,
        "out_unique": out_unique,
        "counters": dict(counters),
    }
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/competition/train.csv")
    parser.add_argument("--out-first", default="data/transformation/pool/competition/one_shot_numeric_first.jsonl")
    parser.add_argument("--out-unique", default="data/transformation/pool/competition/one_shot_numeric_unique.jsonl")
    parser.add_argument("--summary", default="data/transformation/pool/competition/one_shot_numeric.summary.json")
    args = parser.parse_args()
    print(json.dumps(build_rows(args.train_csv, args.out_first, args.out_unique, args.summary), indent=2))


if __name__ == "__main__":
    main()
