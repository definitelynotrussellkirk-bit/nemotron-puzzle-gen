#!/usr/bin/env python3
"""Generate CONST-trap bit manipulation training data.

Puzzles where a non-constant function looks constant on most examples,
then one "killer" example reveals the true function. Teaches the model
not to trust apparent constancy.

Usage:
    python3 -m generators.gen_bit_const_trap --n 300
"""

from datetime import datetime, timezone
import argparse
import json
import random
import time
from pathlib import Path

from generators.bit_manipulation import BitManipulationGenerator, _bits_to_str, _apply_bit_spec, _get_bit
from solvers.bit_manipulation import trace as bm_trace
from training.data import BOXED_INSTRUCTION


def generate_const_trap(rng):
    """Generate a puzzle where at least one bit looks constant but isn't."""
    gen = BitManipulationGenerator.__new__(BitManipulationGenerator)
    gen.rng = rng

    for attempt in range(50):
        circuit = gen._build_circuit()
        query_input = rng.randrange(256)
        query_output = gen._apply_circuit(circuit, query_input)

        # Find bits where the circuit is NOT constant but could look constant
        # on a carefully chosen example set
        trap_bits = []
        for bp in range(8):
            spec = circuit[bp]
            if spec['family'] in ('CONST_0', 'CONST_1'):
                continue  # skip actually constant bits

            # Check: what does this bit output on the query?
            query_bit = _get_bit(query_output, bp)

            # Find inputs that all produce the SAME output for this bit
            # (making it look constant), but query produces the OPPOSITE
            same_inputs = []
            diff_inputs = []
            for x in range(256):
                if x == query_input:
                    continue
                out_bit = _get_bit(gen._apply_circuit(circuit, x), bp)
                if out_bit == query_bit:
                    same_inputs.append(x)
                else:
                    diff_inputs.append(x)

            # We want: many same-as-query inputs AND at least one different
            if len(same_inputs) >= 6 and len(diff_inputs) >= 2:
                trap_bits.append((bp, same_inputs, diff_inputs))

        if not trap_bits:
            continue

        # Pick examples: mostly from same_inputs (looks constant),
        # plus 1-2 from diff_inputs (killer)
        bp, same, diff = rng.choice(trap_bits)

        n_same = rng.randint(5, 7)
        n_diff = rng.randint(1, 2)

        chosen_same = rng.sample(same, min(n_same, len(same)))
        chosen_diff = rng.sample(diff, min(n_diff, len(diff)))

        all_inputs = chosen_same + chosen_diff
        rng.shuffle(all_inputs)

        examples = [(x, gen._apply_circuit(circuit, x)) for x in all_inputs]
        prompt = gen._format_prompt(examples, query_input)
        answer = _bits_to_str(query_output)

        # Get trace from solver — only keep if solver agrees (compact format)
        result = bm_trace(prompt)
        if result is None:
            continue

        reasoning, pred = result
        if pred != answer:
            continue

        return {
            "messages": [
                {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"},
            ],
            "answer": answer,
            "id": f"gen_const_trap_{rng.randint(0,999999):06d}",
            "puzzle_type": "bit_manipulation",
            "mode": "const_trap",
            "trap_bit": bp,
            "generator": "gen_bit_const_trap",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--output", type=str,
                        default="data/bit_manipulation/pool/generated/const_trap.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output, "w") as out:
        for i in range(args.n * 3):
            if count >= args.n:
                break
            result = generate_const_trap(rng)
            if result:
                out.write(json.dumps(result) + "\n")
                count += 1
                if count % 50 == 0:
                    print(f"  {count}/{args.n}", flush=True)

    print(f"Done: {count} CONST-trap examples → {output}")


if __name__ == "__main__":
    main()
