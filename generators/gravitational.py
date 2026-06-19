#!/usr/bin/env python3
"""
GRAVITATIONAL puzzle generator  —  d = 1/2 * g * t^2
====================================================

WHAT THE PUZZLE IS
------------------
A falling-distance relation with an UNKNOWN gravitational constant g. A few
(t, d) examples, then predict d for a query t.

HOW WE CREATE ONE (the generative model)
----------------------------------------
    1. pick an unknown g.
    2. for several times t, emit d = 0.5 * g * t^2  (rounded for display).
    3. query a fresh t.

The constant rate across examples is r = d / t^2 = g/2. Displayed distances are
rounded, so instances are graded with a small relative tolerance.
"""

import argparse
import json
import random


def sample(rng, n_examples=3, round_dp=2):
    """
    Create ONE gravitational puzzle.

    Returns:
        examples : list of {"t","d"}  (d rounded to round_dp)
        query    : {"t"}
        answer   : d for the query (rounded)
        rule     : {"g": value, "rate": g/2}  (how it was built)
        tolerance: suggested relative tolerance for grading
    """
    g = round(rng.uniform(1.5, 25.0), 2)
    rate = g / 2.0
    ts = rng.sample([round(t, 1) for t in (v / 2 for v in range(2, 25))], n_examples + 1)

    def d(t):
        return round(rate * t * t, round_dp)

    return {
        "type": "gravitational",
        "examples": [{"t": t, "d": d(t)} for t in ts[:-1]],
        "query": {"t": ts[-1]},
        "answer": d(ts[-1]),
        "rule": {"g": g, "rate": rate},
        "tolerance": 0.01,
    }


def main():
    ap = argparse.ArgumentParser(description="Create GRAVITATIONAL puzzles (d=1/2 g t^2).")
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
        print(f"\nRULE  g={p['rule']['g']}  (rate=d/t^2={p['rule']['rate']})")
        for e in p["examples"]:
            print(f"  t={e['t']:<5} d={e['d']}")
        print(f"  QUERY t={p['query']['t']} -> d={p['answer']}")


if __name__ == "__main__":
    main()
