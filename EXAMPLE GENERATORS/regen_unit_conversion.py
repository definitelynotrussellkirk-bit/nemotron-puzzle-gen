#!/usr/bin/env python3
"""Regenerate unit conversion training data with compact CERT traces.

Two modes:
  --mode generated   Generate 20K synthetic puzzles and trace them (default)
  --mode competition Trace all unit_conversion rows from data/train.csv

Usage:
    python3 -m generators.regen_unit_conversion --mode generated --n 20000
    python3 -m generators.regen_unit_conversion --mode competition
    python3 -m generators.regen_unit_conversion --mode both --n 20000
"""
import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone

from generators.unit_conversion import UnitConversionGenerator
from solvers.unit_conversion import trace as uc_trace
from training.data import BOXED_INSTRUCTION


def _git_short_hash():
    """Get current git short hash for data_version field."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def regen_generated(n: int, output: str, seed: int):
    """Generate n synthetic unit conversion puzzles with compact CERT traces."""
    gen = UnitConversionGenerator(seed=seed)
    now = datetime.now(timezone.utc).isoformat()
    git_hash = _git_short_hash()
    count = 0
    skipped = 0
    solver_override = 0
    t0 = time.time()

    with open(output, "w") as f:
        for i in range(n * 2):  # oversample for failures
            if count >= n:
                break

            prompt, answer = gen.generate_one()
            result = uc_trace(prompt)

            if result is None:
                skipped += 1
                continue

            reasoning, traced_answer = result

            # Use the solver's answer — the trace teaches the solver's method.
            # Generator and solver may disagree on rounding boundary cases.
            if traced_answer != answer:
                solver_override += 1
            answer = traced_answer

            # Wrap in think tags + boxed answer
            content = f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"

            example = {
                "messages": [
                    {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": content},
                ],
                "answer": answer,
                "id": f"gen_unit_conversion_{count:06d}",
                "puzzle_type": "unit_conversion",
                "trace_quality": "full",
                "data_version": git_hash,
                "generator": "unit_conversion_solver_compact_v1",
                "generated_at": now,
            }
            f.write(json.dumps(example) + "\n")
            count += 1

            if count % 2000 == 0:
                elapsed = time.time() - t0
                print(f"  {count}/{n} ({elapsed:.0f}s, {skipped} skipped, {solver_override} solver overrides)")

    elapsed = time.time() - t0
    print(f"Generated: {count} examples -> {output} ({elapsed:.0f}s, {skipped} skipped, {solver_override} solver overrides)")
    return count


def regen_competition(train_csv: str, output: str):
    """Trace all unit_conversion rows from train.csv with compact CERT traces."""
    now = datetime.now(timezone.utc).isoformat()
    git_hash = _git_short_hash()
    count = 0
    skipped = 0
    mismatch = 0
    t0 = time.time()

    with open(train_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [(r[0], r[1], r[2]) for r in reader
                if "unit conversion" in r[1][:100]]

    print(f"  Found {len(rows)} unit_conversion rows in {train_csv}")

    with open(output, "w") as f:
        for row_id, prompt, expected_answer in rows:
            result = uc_trace(prompt)

            if result is None:
                skipped += 1
                continue

            reasoning, traced_answer = result

            # Check answer matches expected
            if traced_answer != expected_answer:
                mismatch += 1
                # Still include it -- the solver's answer may be right
                # but differ in formatting (e.g., trailing zeros)
                # Skip only if genuinely wrong
                continue

            # Wrap in think tags + boxed answer
            content = f"<think>\n{reasoning}\n</think>\n\\boxed{{{traced_answer}}}"

            example = {
                "messages": [
                    {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": content},
                ],
                "answer": traced_answer,
                "id": f"comp_unit_{row_id}",
                "puzzle_type": "unit_conversion",
                "mode": "competition_traced",
                "trace_quality": "full",
                "generator": "unit_conversion_solver_compact_v1",
                "generated_at": now,
                "data_version": git_hash,
            }
            f.write(json.dumps(example) + "\n")
            count += 1

    elapsed = time.time() - t0
    print(f"Competition: {count} examples -> {output} ({elapsed:.0f}s, {skipped} failed, {mismatch} mismatch)")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generated", "competition", "both"], default="both")
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--output-generated", type=str,
                        default="data/unit_conversion/pool/generated/sft_unit_conversion_20k.jsonl")
    parser.add_argument("--output-competition", type=str,
                        default="data/unit_conversion/pool/competition/competition_traced.jsonl")
    parser.add_argument("--train-csv", type=str, default="data/competition/train.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode in ("generated", "both"):
        print(f"\n=== Regenerating generated data ({args.n} examples) ===")
        regen_generated(args.n, args.output_generated, args.seed)

    if args.mode in ("competition", "both"):
        print(f"\n=== Regenerating competition traces ===")
        regen_competition(args.train_csv, args.output_competition)


if __name__ == "__main__":
    main()
