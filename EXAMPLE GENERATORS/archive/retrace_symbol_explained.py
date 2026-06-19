#!/usr/bin/env python3
"""Offline batch runner: answer-conditioned symbol explainer.

For each symbol transformation row in train.csv, runs solve_answer_conditioned()
with the gold answer as a constraint, renders a unified trace, and writes
training-ready JSONL.

Usage:
    python3 -m generators.retrace_symbol_explained
    python3 -m generators.retrace_symbol_explained --limit 20
"""
from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from solvers.symbol_explainer import parse_prompt, solve_answer_conditioned
from training.data import BOXED_INSTRUCTION
from training.render_symbol_trace import render_symbol_trace

OUTPUT = Path("data/transformation/pool/symbol/symbol_explained.jsonl")
TRAIN_CSV = Path("data/train.csv")
GENERATOR_TAG = "symbol_explained_v1"
TIME_LIMIT = 25.0


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout


def _is_symbol(prompt: str) -> bool:
    """True if this transformation prompt uses symbol (non-digit) operands."""
    for line in prompt.split("\n"):
        if "=" in line and "determine" not in line.lower():
            left = line.split("=")[0].strip()
            if left and not any(c.isdigit() for c in left[:4]):
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Retrace symbol rows with answer-conditioned explainer")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N symbol rows")
    parser.add_argument("--output", type=str, default=str(OUTPUT), help="Output JSONL path")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT, help="Per-row time limit (seconds)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load symbol transformation rows
    with open(TRAIN_CSV, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        rows = [(r[0], r[1], r[2]) for r in reader
                if "transformation rules" in r[1][:120]]

    symbol_rows = [(rid, prompt, gold) for rid, prompt, gold in rows if _is_symbol(prompt)]
    total = len(symbol_rows)
    if args.limit:
        symbol_rows = symbol_rows[:args.limit]
    print(f"Symbol rows: {total} total, processing {len(symbol_rows)}")

    results = []
    solved = 0
    skipped = 0
    timed_out = 0
    t0 = time.time()

    # Set up signal-based timeout (Unix only)
    use_signal = hasattr(signal, "SIGALRM")
    if use_signal:
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)

    for i, (rid, prompt, gold) in enumerate(symbol_rows):
        try:
            if use_signal:
                signal.alarm(int(args.time_limit) + 5)  # hard outer limit

            solution = solve_answer_conditioned(prompt, gold, time_limit=args.time_limit)

            if use_signal:
                signal.alarm(0)

            if solution is None:
                skipped += 1
                continue

            # Render the trace
            trace_text = render_symbol_trace(prompt, gold, solution)

            # Build the full assistant content: <think>TRACE</think>\n\boxed{ANSWER}
            assistant_content = f"<think>\n{trace_text}\n</think>\n\\boxed{{{gold}}}"

            row_out = {
                "messages": [
                    {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": assistant_content},
                ],
                "id": rid,
                "answer": gold,
                "puzzle_type": "transformation",
                "mode": "symbol_explained",
                "generator": GENERATOR_TAG,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "base": solution["base"],
            }
            results.append(json.dumps(row_out, ensure_ascii=False))
            solved += 1

        except _Timeout:
            if use_signal:
                signal.alarm(0)
            timed_out += 1
        except Exception as e:
            if use_signal:
                signal.alarm(0)
            skipped += 1
            if i < 5:
                print(f"  Error on row {rid}: {e}")

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(symbol_rows)}: {solved} solved, {skipped} skipped, "
                  f"{timed_out} timed out ({elapsed:.0f}s)")

    if use_signal:
        signal.signal(signal.SIGALRM, old_handler)

    elapsed = time.time() - t0
    print(f"\nDone: {solved} solved, {skipped} skipped, {timed_out} timed out ({elapsed:.0f}s)")

    # Write output
    with open(output_path, "w") as f:
        for line in results:
            f.write(line + "\n")
    print(f"Written: {output_path} ({len(results)} rows)")

    # Print 2 example traces
    if results:
        print("\n" + "=" * 60)
        print("EXAMPLE TRACES")
        print("=" * 60)
        for idx in range(min(2, len(results))):
            row = json.loads(results[idx])
            print(f"\n--- Row {row['id']} (base {row['base']}) ---")
            print(row["messages"][1]["content"])
            print()


if __name__ == "__main__":
    main()
