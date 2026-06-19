#!/usr/bin/env python3
"""Regenerate gravitational training data with compact procedural traces.

Two modes:
  --mode generated   Generate 20K synthetic puzzles and trace them (default)
  --mode competition Trace all gravitational rows from data/train.csv
  --mode both        Both of the above

Usage:
    python3 -m generators.regen_gravitational --mode generated --n 20000
    python3 -m generators.regen_gravitational --mode competition
    python3 -m generators.regen_gravitational --mode both --n 20000
"""
import argparse
import csv
import json
import time
from datetime import datetime, timezone

from generators.gravitational import GravitationalGenerator
from solvers.gravitational import trace as grav_trace
from training.data import BOXED_INSTRUCTION


def _git_short_hash():
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def regen_generated(n: int, output: str, seed: int):
    """Generate n synthetic gravitational puzzles with procedural traces."""
    gen = GravitationalGenerator(seed=seed)
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

            prompt, gen_answer = gen.generate_one()
            result = grav_trace(prompt)

            if result is None:
                skipped += 1
                continue

            reasoning, traced_answer = result

            # Use the solver's answer — the trace teaches the solver's method.
            # Generator and solver may disagree on rounding boundary cases
            # (off-by-one in last decimal). The trace is self-consistent with
            # the solver's answer, so that's what we train on.
            answer = traced_answer
            if traced_answer != gen_answer:
                solver_override += 1

            content = f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"

            example = {
                "messages": [
                    {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": content},
                ],
                "answer": answer,
                "id": f"gen_gravitational_{count:06d}",
                "puzzle_type": "gravitational",
                "trace_quality": "full",
                "data_version": git_hash,
                "generator": "gravitational_solver_compact_v1",
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
    """Trace all gravitational rows from train.csv with procedural traces."""
    now = datetime.now(timezone.utc).isoformat()
    git_hash = _git_short_hash()
    count = 0
    skipped = 0
    mismatch = 0
    t0 = time.time()

    with open(train_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        rows = [(r[0], r[1], r[2]) for r in reader
                if "gravitational constant" in r[1][:100]]

    print(f"  Found {len(rows)} gravitational rows in {train_csv}")

    with open(output, "w") as f:
        for row_id, prompt, expected_answer in rows:
            result = grav_trace(prompt)

            if result is None:
                skipped += 1
                continue

            reasoning, traced_answer = result

            if traced_answer != expected_answer:
                mismatch += 1
                continue  # Competition rows must match exactly

            content = f"<think>\n{reasoning}\n</think>\n\\boxed{{{traced_answer}}}"

            example = {
                "messages": [
                    {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": content},
                ],
                "answer": traced_answer,
                "id": f"comp_grav_{row_id}",
                "puzzle_type": "gravitational",
                "mode": "competition_traced",
                "trace_quality": "full",
                "generator": "gravitational_solver_compact_v1",
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
                        default="data/gravitational/pool/generated/sft_gravitational_20k.jsonl")
    parser.add_argument("--output-competition", type=str,
                        default="data/gravitational/pool/competition/competition_traced.jsonl")
    parser.add_argument("--train-csv", type=str, default="data/competition/train.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode in ("generated", "both"):
        print(f"\n=== Regenerating generated gravitational data ({args.n} examples) ===")
        regen_generated(args.n, args.output_generated, args.seed)

    if args.mode in ("competition", "both"):
        print(f"\n=== Regenerating competition gravitational traces ===")
        regen_competition(args.train_csv, args.output_competition)


if __name__ == "__main__":
    main()
