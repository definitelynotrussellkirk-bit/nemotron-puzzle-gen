#!/usr/bin/env python3
"""Generate anti-prior bit manipulation training data.

Puzzles where high-prior families (COPY, CONST, NOT) look correct at first
but are killed by later evidence. Teaches the model not to trust shortcuts.

Usage:
    python3 -m generators.gen_bit_anti_prior --n 200
"""

from datetime import datetime, timezone
import argparse
import json
import random
import time
from pathlib import Path

from generators.bit_manipulation import BitManipulationGenerator, _bits_to_str, _get_bit, _apply_bit_spec
from solvers.bit_manipulation import trace as bm_trace
from training.data import BOXED_INSTRUCTION


# Target families that should be the TRUE function (not the shortcut)
ANTI_TARGETS = [
    # True function families that get overshadowed by COPY/CONST/NOT
    "XNOR", "NAND", "NOR", "a_OR_NOTb", "NOTa_AND_b", "NOTa_OR_b", "a_AND_NOTb",
]

TT2_MAP = {
    "AND": 0b0001, "XOR": 0b0110, "XNOR": 0b1001,
    "OR": 0b0111, "NAND": 0b1110, "NOR": 0b1000,
    "a_OR_NOTb": 0b1101, "NOTa_AND_b": 0b0100,
    "NOTa_OR_b": 0b1011, "a_AND_NOTb": 0b0010,
}


def generate_anti_prior_puzzle(rng):
    """Generate puzzle where at least one bit uses an asymmetric gate
    that could be confused with COPY/CONST/NOT on most examples."""
    gen = BitManipulationGenerator.__new__(BitManipulationGenerator)
    gen.rng = rng

    for attempt in range(30):
        # Build circuit: most bits use simple families, but 2-3 use asymmetric gates
        circuit = []
        anti_bits = rng.sample(range(8), rng.randint(2, 3))

        for bp in range(8):
            if bp in anti_bits:
                # Use an asymmetric gate
                family = rng.choice(ANTI_TARGETS)
                tt = TT2_MAP.get(family, 0b0110)
                left, right = rng.sample(range(8), 2)
                circuit.append({"family": family, "inputs": (left, right), "tt": tt})
            else:
                # Use a simple family
                simple = rng.choice(["COPY", "NOT", "CONST_0", "CONST_1"])
                if simple == "CONST_0":
                    circuit.append({"family": "CONST_0", "inputs": (), "tt": 0})
                elif simple == "CONST_1":
                    circuit.append({"family": "CONST_1", "inputs": (), "tt": 1})
                elif simple == "COPY":
                    pos = rng.randint(0, 7)
                    circuit.append({"family": "COPY", "inputs": (pos,)})
                else:
                    pos = rng.randint(0, 7)
                    circuit.append({"family": "NOT", "inputs": (pos,)})

        query = rng.randrange(256)

        def apply_circuit(inp):
            out = 0
            for pos, spec in enumerate(circuit):
                out |= (_apply_bit_spec(spec, inp) << (7 - pos))
            return out & 0xFF

        # Generate examples
        n_examples = rng.randint(7, 10)
        inputs = rng.sample([x for x in range(256) if x != query], n_examples)
        examples = [(x, apply_circuit(x)) for x in inputs]
        rng.shuffle(examples)

        prompt = gen._format_prompt(examples, query)
        answer = _bits_to_str(apply_circuit(query))

        result = bm_trace(prompt)
        if result is None:
            continue

        reasoning, pred = result
        # Only keep puzzles the solver gets right — ensures compact format trace
        if pred != answer:
            continue

        return {
            "messages": [
                {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"},
            ],
            "answer": answer,
            "id": f"gen_bit_anti_{rng.randint(0,999999):06d}",
            "puzzle_type": "bit_manipulation",
            "mode": "anti_prior",
            "anti_bits": anti_bits,
            "generator": "gen_bit_anti_prior",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--output", type=str,
                        default="data/bit_manipulation/pool/generated/anti_prior.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output, "w") as out:
        for i in range(args.n * 2):
            if count >= args.n:
                break
            result = generate_anti_prior_puzzle(rng)
            if result:
                out.write(json.dumps(result) + "\n")
                count += 1
                if count % 50 == 0:
                    print(f"  {count}/{args.n}", flush=True)

    print(f"Done: {count} anti-prior examples → {output}")


if __name__ == "__main__":
    main()
