#!/usr/bin/env python3
"""Generate row-template bit manipulation training data.

Circuits where multiple output bits share the same family + relative offsets,
making cross-bit motifs obvious. This teaches the model to look for
row-level reuse patterns.

Usage:
    python3 -m generators.gen_bit_template --n 300
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


# Template families with relative offsets
TEMPLATES = [
    # (family, n_inputs, relative_offsets)
    ("XNOR", 2, (1, 3)),
    ("XNOR", 2, (2, 5)),
    ("XNOR", 2, (6, 1)),
    ("XOR", 2, (1, 4)),
    ("XOR", 2, (3, 7)),
    ("AND", 2, (2, 3)),
    ("OR", 2, (2, 5)),
    ("OR", 2, (1, 3)),
    ("NAND", 2, (1, 6)),
    ("NOR", 2, (3, 5)),
    ("NOT", 1, (2,)),
    ("NOT", 1, (6,)),
    ("COPY", 1, (3,)),
    ("COPY", 1, (5,)),
]

TT2_MAP = {
    "AND": 0b0001, "XOR": 0b0110, "XNOR": 0b1001,
    "OR": 0b0111, "NAND": 0b1110, "NOR": 0b1000,
}


def _make_spec_from_template(out_pos, family, offsets):
    """Create a bit spec from a template applied at out_pos."""
    inputs = tuple((out_pos + off) % 8 for off in offsets)

    if family == "CONST_0":
        return {"family": "CONST_0", "inputs": (), "tt": 0}
    elif family == "CONST_1":
        return {"family": "CONST_1", "inputs": (), "tt": 1}
    elif family == "COPY":
        return {"family": "COPY", "inputs": (inputs[0],)}
    elif family == "NOT":
        return {"family": "NOT", "inputs": (inputs[0],)}
    elif family in TT2_MAP:
        return {"family": family, "inputs": inputs, "tt": TT2_MAP[family]}
    return None


def generate_template_puzzle(rng):
    """Generate a circuit with shared row templates."""
    gen = BitManipulationGenerator.__new__(BitManipulationGenerator)
    gen.rng = rng

    # Pick 1-2 templates
    t1 = rng.choice(TEMPLATES)
    t2 = rng.choice(TEMPLATES)

    # Assign templates to output bits
    # t1 gets 4-6 bits, t2 or random gets the rest
    n_t1 = rng.randint(4, 6)
    t1_bits = rng.sample(range(8), n_t1)
    t2_bits = [b for b in range(8) if b not in t1_bits]

    circuit = [None] * 8
    for bp in t1_bits:
        spec = _make_spec_from_template(bp, t1[0], t1[2])
        if spec:
            circuit[bp] = spec

    for bp in t2_bits:
        spec = _make_spec_from_template(bp, t2[0], t2[2])
        if spec:
            circuit[bp] = spec

    # Fill any remaining with random
    for bp in range(8):
        if circuit[bp] is None:
            circuit[bp] = gen._make_spec(rng.choice(["COPY", "NOT", "CONST_0", "CONST_1"]))

    # Generate examples
    query = rng.randrange(256)
    n_examples = rng.randint(7, 10)
    inputs = rng.sample([x for x in range(256) if x != query], n_examples)

    def apply_circuit(inp):
        out = 0
        for pos, spec in enumerate(circuit):
            out |= (_apply_bit_spec(spec, inp) << (7 - pos))
        return out & 0xFF

    examples = [(x, apply_circuit(x)) for x in inputs]
    rng.shuffle(examples)

    prompt = gen._format_prompt(examples, query)
    answer = _bits_to_str(apply_circuit(query))

    # Get trace — only keep if solver agrees (compact format)
    result = bm_trace(prompt)
    if result is None:
        return None

    reasoning, pred = result
    if pred != answer:
        return None

    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"},
        ],
        "answer": answer,
        "id": f"gen_bit_template_{rng.randint(0,999999):06d}",
        "puzzle_type": "bit_manipulation",
        "mode": "row_template",
        "template_1": f"{t1[0]}({t1[2]})",
        "template_1_bits": t1_bits,
        "generator": "gen_bit_template",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--output", type=str,
                        default="data/bit_manipulation/pool/generated/template.jsonl")
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
            result = generate_template_puzzle(rng)
            if result:
                out.write(json.dumps(result) + "\n")
                count += 1
                if count % 50 == 0:
                    print(f"  {count}/{args.n}", flush=True)

    print(f"Done: {count} template examples → {output}")


if __name__ == "__main__":
    main()
