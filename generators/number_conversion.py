#!/usr/bin/env python3
"""
NUMBER CONVERSION puzzle generator  —  integer <-> Roman numeral
================================================================

WHAT THE PUZZLE IS
------------------
Convert between Arabic integers and Roman numerals. The examples establish the
direction (int->roman or roman->int); the query asks for one more conversion.

HOW WE CREATE ONE (the generative model)
----------------------------------------
    1. pick a direction.
    2. sample integers in [1, 3999], render each side with the fixed value table.
    3. query a fresh integer.

Roman numerals are a fixed, table-driven additive/subtractive notation (not a
positional system), so rendering is a deterministic greedy walk of the value
table -- no unknown parameter at all.
"""

import argparse
import json
import random

_VALUES = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def to_roman(n):
    """Greedy value-table decode: append each numeral while it fits."""
    out = []
    for v, sym in _VALUES:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def sample(rng, n_examples=3, direction=None):
    """
    Create ONE number-conversion puzzle.

    Returns:
        direction : "int2roman" or "roman2int"
        examples  : list of {"input","output"}
        query     : {"input"}
        answer    : converted value
        rule      : {"direction": ...}  (how it was built)
    """
    direction = direction or rng.choice(["int2roman", "roman2int"])
    nums = rng.sample(range(1, 4000), n_examples + 1)

    def render(n):
        if direction == "int2roman":
            return {"input": str(n), "output": to_roman(n)}
        return {"input": to_roman(n), "output": str(n)}

    examples = [render(n) for n in nums[:-1]]
    q = render(nums[-1])
    return {
        "type": "number_conversion",
        "direction": direction,
        "examples": examples,
        "query": {"input": q["input"]},
        "answer": q["output"],
        "rule": {"direction": direction},
    }


def main():
    ap = argparse.ArgumentParser(description="Create NUMBER CONVERSION puzzles (int<->roman).")
    ap.add_argument("-n", "--num", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--direction", choices=["int2roman", "roman2int"], default=None)
    ap.add_argument("--jsonl", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    for _ in range(args.num):
        p = sample(rng, args.examples, args.direction)
        if args.jsonl:
            print(json.dumps(p))
            continue
        print(f"\nRULE  direction={p['direction']}")
        for e in p["examples"]:
            print(f"  {e['input']:<6} -> {e['output']}")
        print(f"  QUERY {p['query']['input']} -> {p['answer']}")


if __name__ == "__main__":
    main()
