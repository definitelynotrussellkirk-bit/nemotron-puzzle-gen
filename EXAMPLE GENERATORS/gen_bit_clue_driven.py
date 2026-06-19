#!/usr/bin/env python3
"""Generate clue-driven bit manipulation training data.

Creates puzzles where most bits are determined by examples but 1-3 bits
remain genuinely ambiguous. Examples are chosen to maximize information
gain (kill wrong candidates) while preserving deliberate ambiguity on
target bits. Teaches the model to reason about what IS determined vs
what requires a prior.

Usage:
    python3 -m generators.gen_bit_clue_driven --n 500
"""

import argparse
import json
import random
import time
from pathlib import Path

from generators.bit_manipulation import BitManipulationGenerator, _bits_to_str, _get_bit
from solvers.bit_manipulation import (
    _enumerate_candidates, trace as bm_trace,
)
from training.data import BOXED_INSTRUCTION


def _count_ambiguous_bits(examples, query, circuit, gen):
    """Count how many bits have multiple surviving candidates at query."""
    query_output = gen._apply_circuit(circuit, query)
    input_bit_matrix = [[_get_bit(inp, pos) for inp, _ in examples] for pos in range(8)]
    query_bits = [_get_bit(query, pos) for pos in range(8)]
    ambiguous = 0
    for bp in range(8):
        target_bits = [_get_bit(out, bp) for _, out in examples]
        cands = _enumerate_candidates(input_bit_matrix, target_bits, query_bits, bp)
        query_vals = set(qbit for _, _, qbit, _ in cands)
        if len(query_vals) > 1:
            ambiguous += 1
    return ambiguous


def _bit_determined(examples, query, circuit, gen, bp):
    """Check if a specific bit position is uniquely determined."""
    input_bit_matrix = [[_get_bit(inp, pos) for inp, _ in examples] for pos in range(8)]
    query_bits = [_get_bit(query, pos) for pos in range(8)]
    target_bits = [_get_bit(out, bp) for _, out in examples]
    cands = _enumerate_candidates(input_bit_matrix, target_bits, query_bits, bp)
    query_vals = set(qbit for _, _, qbit, _ in cands)
    return len(query_vals) <= 1


def generate_clue_driven_puzzle(rng):
    """Generate a puzzle with targeted ambiguity on 1-3 bits.

    Strategy:
    1. Build a random circuit
    2. Start with minimal examples (3-4)
    3. Greedily add examples that determine MORE bits
    4. Stop when exactly 1-3 bits remain ambiguous
    5. Use solver trace (which honestly reports ambiguity)
    """
    gen = BitManipulationGenerator.__new__(BitManipulationGenerator)
    gen.rng = rng

    for circuit_attempt in range(30):
        circuit = gen._build_circuit()
        query = rng.randrange(256)
        answer = _bits_to_str(gen._apply_circuit(circuit, query))

        all_inputs = [x for x in range(256) if x != query]
        rng.shuffle(all_inputs)

        # Start with 3 random examples
        examples = [(x, gen._apply_circuit(circuit, x)) for x in all_inputs[:3]]

        # Greedily add examples to reduce ambiguity, but stop at 1-3 ambiguous bits
        for _ in range(10):
            n_amb = _count_ambiguous_bits(examples, query, circuit, gen)
            if 1 <= n_amb <= 3:
                break  # sweet spot
            if n_amb == 0:
                break  # over-determined, restart

            # Find example that reduces ambiguity most (but not to 0)
            best_input = None
            best_amb = n_amb

            for x in rng.sample(all_inputs, min(40, len(all_inputs))):
                if any(x == inp for inp, _ in examples):
                    continue
                trial = examples + [(x, gen._apply_circuit(circuit, x))]
                trial_amb = _count_ambiguous_bits(trial, query, circuit, gen)
                # Prefer: reduces ambiguity but keeps 1-3 bits ambiguous
                if 1 <= trial_amb <= 3 and trial_amb < best_amb:
                    best_amb = trial_amb
                    best_input = x
                elif trial_amb < best_amb and trial_amb >= 1:
                    best_amb = trial_amb
                    best_input = x

            if best_input is not None:
                examples.append((best_input, gen._apply_circuit(circuit, best_input)))
            else:
                break

        n_amb = _count_ambiguous_bits(examples, query, circuit, gen)
        if not (1 <= n_amb <= 3):
            continue  # wrong range, retry

        # Get trace from solver (will show ambiguity honestly)
        rng.shuffle(examples)
        prompt = gen._format_prompt(examples, query)
        result = bm_trace(prompt)
        if result is None:
            continue

        reasoning, pred = result

        # If solver disagrees with latent circuit, use ambiguity-aware trace
        if pred != answer:
            from solvers.bit_manipulation import trace_with_gold
            amb_result = trace_with_gold(prompt, answer)
            if amb_result:
                reasoning, pred = amb_result
            else:
                continue

        from datetime import datetime, timezone
        return {
            "messages": [
                {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"},
            ],
            "answer": answer,
            "id": f"gen_bit_clue_{rng.randint(0,999999):06d}",
            "puzzle_type": "bit_manipulation",
            "mode": f"clue_driven_amb{n_amb}",
            "n_ambiguous_bits": n_amb,
            "n_examples": len(examples),
            "generator": "gen_bit_clue_driven",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--output", type=str,
                        default="data/bit_manipulation/pool/generated/clue_driven.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    t0 = time.time()
    with open(output, "w") as out:
        for i in range(args.n * 3):
            if count >= args.n:
                break
            result = generate_clue_driven_puzzle(rng)
            if result:
                out.write(json.dumps(result) + "\n")
                count += 1
                if count % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  {count}/{args.n} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"Done: {count} clue-driven examples in {elapsed:.0f}s → {output}")


if __name__ == "__main__":
    main()
