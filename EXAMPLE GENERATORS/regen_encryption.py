#!/usr/bin/env python3
"""Regenerate encryption training data with improved traces.

Two modes:
  --mode generated   Generate N synthetic puzzles and trace them (default)
  --mode competition Trace all encryption rows from data/competition/train.csv
  --mode both        Both of the above

Usage:
    python3 -m generators.regen_encryption --mode generated --n 20000
    python3 -m generators.regen_encryption --mode competition
    python3 -m generators.regen_encryption --mode both --n 20000
"""
import argparse
import csv
import json
import time
from datetime import datetime, timezone

from generators.encryption import EncryptionGenerator
from solvers.encryption import trace as enc_trace
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
    """Generate n synthetic encryption puzzles with procedural traces."""
    gen = EncryptionGenerator(seed=seed)
    now = datetime.now(timezone.utc).isoformat()
    git_hash = _git_short_hash()
    count = 0
    bare = 0
    t0 = time.time()

    with open(output, "w") as f:
        for i in range(n * 2):  # oversample for failures
            if count >= n:
                break

            gen_result = gen.generate_one()
            if gen_result is None:
                bare += 1
                continue
            prompt, answer = gen_result
            result = enc_trace(prompt)

            if result is None:
                bare += 1
                continue

            reasoning, traced_answer = result

            # Verify answer matches
            if traced_answer.lower() != answer.lower():
                continue

            # Wrap in think tags + boxed answer
            content = f"<think>\n{reasoning}\n</think>\n\\boxed{{{traced_answer}}}"

            # Compute difficulty metadata
            n_words = len(traced_answer.split())
            n_unknown = reasoning.count('=_')
            has_candidates = 'candidates' in reasoning
            n_candidate_lines = reasoning.count('candidates (')

            example = {
                "messages": [
                    {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": content},
                ],
                "answer": traced_answer,
                "id": f"gen_encryption_{count:06d}",
                "puzzle_type": "encryption",
                "mode": "generated",
                "trace_quality": "full",
                "n_query_words": n_words,
                "n_unknown_mappings": n_unknown,
                "generator": "encryption_solver_v11_length_gate",
                "generated_at": now,
                "data_version": git_hash,
            }
            f.write(json.dumps(example) + "\n")
            count += 1

            if count % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  {count}/{n} ({elapsed:.0f}s, {bare} skipped)")

    elapsed = time.time() - t0
    print(f"Generated: {count} examples -> {output} ({elapsed:.0f}s, {bare} skipped)")
    return count


def regen_competition(train_csv: str, output: str):
    """Trace all encryption rows from train.csv with updated traces."""
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
                if "encryption rules" in r[1][:100]]

    print(f"  Found {len(rows)} encryption rows in {train_csv}")

    with open(output, "w") as f:
        for row_id, prompt, expected_answer in rows:
            result = enc_trace(prompt)

            if result is None:
                skipped += 1
                continue

            reasoning, traced_answer = result

            if traced_answer.lower() != expected_answer.lower():
                mismatch += 1
                continue  # Competition rows must match exactly

            content = f"<think>\n{reasoning}\n</think>\n\\boxed{{{traced_answer}}}"

            example = {
                "messages": [
                    {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": content},
                ],
                "answer": traced_answer,
                "id": f"comp_enc_{row_id}",
                "puzzle_type": "encryption",
                "mode": "competition_traced",
                "trace_quality": "full",
                "generator": "encryption_solver_v11_length_gate",
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
                        default="data/encryption/pool/generated/sft_encryption_20k.jsonl")
    parser.add_argument("--output-competition", type=str,
                        default="data/encryption/pool/competition/competition_traced.jsonl")
    parser.add_argument("--train-csv", type=str, default="data/competition/train.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode in ("generated", "both"):
        print(f"\n=== Regenerating generated encryption data ({args.n} examples) ===")
        regen_generated(args.n, args.output_generated, args.seed)

    if args.mode in ("competition", "both"):
        print(f"\n=== Regenerating competition encryption traces ===")
        regen_competition(args.train_csv, args.output_competition)


if __name__ == "__main__":
    main()
