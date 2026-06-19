#!/usr/bin/env python3
"""Generate ambiguous bit manipulation training data.

Puzzles with random examples where the solver may not uniquely determine the rule.
Teaches the model to handle genuine uncertainty — identify what IS determined,
acknowledge what ISN'T, and still produce the best answer.

Ambiguity buckets:
  - near_determinate: 1-2 query bits uncertain
  - multi_hypothesis: 2-3 globally plausible rules survive
  - solver_disagree: solver picks different answer than latent circuit

Usage:
    python3 -m generators.gen_bit_ambiguous --n 500 --seed 42
    python3 -m generators.gen_bit_ambiguous --n 500 --output data/bit_manipulation/pool/generated/ambiguous.jsonl
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from generators.bit_manipulation import BitManipulationGenerator, _bits_to_str
from solvers.bit_manipulation import trace as bm_trace
from training.data import BOXED_INSTRUCTION


def generate_ambiguous_puzzle(rng):
    """Generate a bit puzzle with random examples (not solver-optimized)."""
    gen = BitManipulationGenerator.__new__(BitManipulationGenerator)
    gen.rng = rng

    circuit = gen._build_circuit()
    query_input = rng.randrange(256)

    # Random examples — NOT designed to make every bit deterministic
    n_examples = rng.randint(7, 10)
    all_inputs = list(range(256))
    rng.shuffle(all_inputs)
    example_inputs = [x for x in all_inputs if x != query_input][:n_examples]

    examples = []
    for inp in example_inputs:
        out = gen._apply_circuit(circuit, inp)
        examples.append((inp, out))

    rng.shuffle(examples)
    prompt = gen._format_prompt(examples, query_input)
    answer = _bits_to_str(gen._apply_circuit(circuit, query_input))

    return prompt, answer, circuit, examples, query_input


def classify_ambiguity(solver_pred, latent_answer, solver_details):
    """Classify the type of ambiguity in this puzzle."""
    if solver_pred == latent_answer:
        return "solver_agrees"

    # Count differing bits
    if len(solver_pred) == 8 and len(latent_answer) == 8:
        diff_bits = sum(1 for a, b in zip(solver_pred, latent_answer) if a != b)
        if diff_bits <= 2:
            return "near_determinate"
        else:
            return "multi_hypothesis"

    return "solver_disagree"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="data/bit_manipulation/pool/generated/ambiguous.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    bucket_counts = {}

    with open(output, "w") as out:
        for attempt in range(args.n * 3):
            if count >= args.n:
                break

            prompt, answer, circuit, examples, query_input = generate_ambiguous_puzzle(rng)

            # Get solver trace
            result = bm_trace(prompt)
            if result is None:
                continue

            reasoning, solver_pred = result
            solver_pred_raw = solver_pred  # preserve before any rescue
            needed_gold_trace = (solver_pred != answer)
            bucket = classify_ambiguity(solver_pred, answer, None)

            # If solver disagrees, try gold-conditioned trace
            if needed_gold_trace:
                try:
                    from solvers.bit_manipulation import trace_with_gold
                    amb_result = trace_with_gold(prompt, answer)
                    if amb_result:
                        reasoning, solver_pred = amb_result
                    else:
                        continue
                except (ImportError, AttributeError):
                    continue

            # Reject degenerate cases:
            # - trace has no useful content
            if len(reasoning) < 50:
                continue

            msg = {
                "messages": [
                    {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"},
                ],
                "answer": answer,
                "id": f"gen_bit_ambiguous_{count:06d}",
                "puzzle_type": "bit_manipulation",
                "mode": "ambiguous",
                "ambiguity_bucket": bucket,
                "needed_gold_trace": needed_gold_trace,
                "solver_pred_raw": solver_pred_raw,
                "solver_pred_final": solver_pred,
                "generator": "gen_bit_ambiguous",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            out.write(json.dumps(msg) + "\n")
            count += 1
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

            if count % 100 == 0:
                print(f"  {count}/{args.n} ({attempt} attempts)", flush=True)

    print(f"Done: {count} ambiguous bit examples → {output}")
    print(f"Buckets: {bucket_counts}")


if __name__ == "__main__":
    main()
