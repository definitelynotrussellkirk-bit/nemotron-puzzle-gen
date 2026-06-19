#!/usr/bin/env python3
"""Generate cipher one-shot rows with an explicitly labeled local prior."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone

from generators.gen_transform_cipher_query_local_rows import _local_query_op_solve, _render_trace
from generators.transform_router import is_transformation_prompt, route_transform_prompt
from training.data import BOXED_INSTRUCTION


def _assistant_content(trace_text: str, answer: str) -> str:
    if "{" in answer or "}" in answer:
        return f"<think>\n{trace_text}\n</think>\nThe final answer is: {answer}"
    return f"<think>\n{trace_text}\n</think>\n\\boxed{{{answer}}}"


def build_rows(train_csv: str, out: str, summary_out: str, timeout: float) -> dict:
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
            if route["program"] != "TRANS_ONE_SHOT_V1":
                continue

            result = _local_query_op_solve(
                prompt,
                answer,
                timeout=timeout,
                min_support=1,
                exact_support=1,
            )
            if result is None:
                counters["local_solve_fail"] += 1
                continue

            rendered = _render_trace(
                prompt,
                answer,
                result,
                program_name="Cipher Single-Witness Local Prior",
                route_reason="one query-op witness gives a local decoded prior",
                certainty="single-witness local prior",
                support_phrase="one query-op witness is replayed, then the local prior is applied",
                min_support=1,
            )
            if rendered is None:
                counters["trace_builder_reject"] += 1
                continue
            trace_text, pred = rendered
            if pred != answer:
                counters["prediction_mismatch"] += 1
                continue
            if "???" in trace_text or "???" in answer:
                counters["placeholder_reject"] += 1
                continue

            combo = result["combos"][result["query_op"]]
            rows.append({
                "messages": [
                    {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": _assistant_content(trace_text, answer)},
                ],
                "answer": answer,
                "id": rid,
                "puzzle_type": "transformation",
                "mode": "cipher_single_witness_local_prior",
                "generator": "gen_transform_cipher_single_witness_rows",
                "route_program": route["program"],
                "query_op_local_rule": "|".join(combo),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
            counters["rows"] += 1
            counters[f"rows:{route['program']}"] += 1
            counters[f"combo:{'|'.join(combo)}"] += 1

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
    parser.add_argument("--out", default="data/transformation/pool/competition/cipher_single_witness_local_prior.jsonl")
    parser.add_argument("--summary", default="data/transformation/pool/competition/cipher_single_witness_local_prior.summary.json")
    parser.add_argument("--timeout", type=float, default=0.2)
    args = parser.parse_args()
    print(json.dumps(build_rows(args.train_csv, args.out, args.summary, args.timeout), indent=2))


if __name__ == "__main__":
    main()
