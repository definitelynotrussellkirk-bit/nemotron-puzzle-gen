#!/usr/bin/env python3
"""
UNIT CONVERSION puzzle generator  —  out = in * factor
======================================================

WHAT THE PUZZLE IS
------------------
A linear conversion with an UNKNOWN factor. A few (input, output) pairs, then
convert a query input.

HOW WE CREATE ONE (the generative model)
----------------------------------------
    1. pick an unknown positive factor.
    2. for several inputs, emit  out = in * factor  (rounded for display).
    3. query a fresh input.

The constant rate across examples is f = out / in. Displayed outputs are rounded,
so instances are graded with a small relative tolerance.
"""

import argparse
import json
import random


def sample(rng, n_examples=3, round_dp=2):
    """
    Create ONE unit-conversion puzzle.

    Returns:
        examples : list of {"in","out"}  (out rounded to round_dp)
        query    : {"in"}
        answer   : converted output (rounded)
        rule     : {"factor": value}  (how it was built)
        tolerance: suggested relative tolerance for grading
    """
    factor = round(rng.uniform(0.05, 40.0), 3)
    ins = rng.sample([round(v, 1) for v in (n / 2 for n in range(2, 60))], n_examples + 1)

    def out(v):
        return round(v * factor, round_dp)

    return {
        "type": "unit_conversion",
        "examples": [{"in": v, "out": out(v)} for v in ins[:-1]],
        "query": {"in": ins[-1]},
        "answer": out(ins[-1]),
        "rule": {"factor": factor},
        "tolerance": 0.01,
    }


def main():
    ap = argparse.ArgumentParser(description="Create UNIT CONVERSION puzzles (out=in*factor).")
    ap.add_argument("-n", "--num", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--jsonl", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    for _ in range(args.num):
        p = sample(rng, args.examples)
        if args.jsonl:
            print(json.dumps(p))
            continue
        print(f"\nRULE  factor={p['rule']['factor']}")
        for e in p["examples"]:
            print(f"  in={e['in']:<6} out={e['out']}")
        print(f"  QUERY in={p['query']['in']} -> out={p['answer']}")


if __name__ == "__main__":
    main()
