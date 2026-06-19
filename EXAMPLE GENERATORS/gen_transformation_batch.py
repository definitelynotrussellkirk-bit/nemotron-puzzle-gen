#!/usr/bin/env python3
"""Generate transformation training data in small batches.

Usage:
    python -m generators.gen_transformation_batch                    # 200 examples, append
    python -m generators.gen_transformation_batch --n 500            # 500 examples
    python -m generators.gen_transformation_batch --n 200 --hard     # oversample hard cases
    python -m generators.gen_transformation_batch --output data/sft_transformation_extra.jsonl

Each run uses a unique seed derived from current time + file line count,
so repeated runs always produce NEW examples.
"""

import argparse
import json
import time
from pathlib import Path

from generators.transformation import TransformationGenerator
from training.data import BOXED_INSTRUCTION


def generate_batch(n=200, output="data/sft_transformation.jsonl",
                   hard_mode=False, symbol_prob=0.3):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Unique seed: time-based + existing line count so we never repeat
    existing = 0
    if output.exists():
        with open(output) as f:
            existing = sum(1 for _ in f)
    seed = int(time.time()) + existing * 7

    gen = TransformationGenerator(seed=seed, symbol_probability=symbol_prob)

    # Hard mode: increase digit permutation probability, force more operators
    if hard_mode:
        gen.digit_permutation_probability = 0.3  # 30% vs default 7%

    count = 0
    t0 = time.time()

    with open(output, "a") as out:  # append mode
        for i in range(n * 3):  # oversample
            if count >= n:
                break
            result = gen.generate_one_with_trace()
            if result is None:
                continue
            prompt, answer, trace_text = result
            msg = {
                "messages": [
                    {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": trace_text},
                ],
                "id": f"gen_transformation_b{seed}_{count:04d}",
                "puzzle_type": "transformation",
            }
            out.write(json.dumps(msg) + "\n")
            count += 1

            if count % 50 == 0:
                elapsed = time.time() - t0
                rate = count / elapsed
                print(f"  {count}/{n} ({elapsed:.0f}s, {rate:.1f}/s)", flush=True)

    elapsed = time.time() - t0
    total = existing + count
    print(f"Done: +{count} examples in {elapsed:.0f}s. Total in file: {total}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--output", type=str, default="data/sft_transformation.jsonl")
    parser.add_argument("--hard", action="store_true", help="Oversample hard cases")
    parser.add_argument("--symbol-prob", type=float, default=0.3)
    args = parser.parse_args()

    generate_batch(n=args.n, output=args.output,
                   hard_mode=args.hard, symbol_prob=args.symbol_prob)
