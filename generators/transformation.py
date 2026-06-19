#!/usr/bin/env python3
"""
TRANSFORMATION puzzle generator  —  two numbers -> hidden arithmetic recipe
===========================================================================

WHAT THE PUZZLE IS
------------------
Each example is "A op B = RESULT" where RESULT is produced by a hidden recipe
with THREE independent knobs:
    1. OPERATION   : add, sub, mul, absdiff, mod, floordiv, concat, ...
    2. ORDERING    : operands taken as (A,B) or swapped (B,A), optionally with
                     each operand's digits reversed first.
    3. STYLE       : how the result is written -- plain, digit-reversed,
                     sign-prefixed ("-12"), or sign-suffixed ("12-").

HOW WE CREATE ONE (the generative model)
----------------------------------------
Pick (operation, ordering, style), sample operand pairs, render each example
through the recipe, hold one out as the query. The three knobs are independent,
so the result string carries an encoding (ordering + style) on top of the raw
arithmetic -- that layering is the whole point of the type.

CIPHER VARIANT
--------------
`sample(..., cipher=True)` disguises digits 0-9 as symbols via a random bijection;
the symbol map is implied by the examples.
"""

import argparse
import json
import random

OPS = {
    "add":      lambda a, b: a + b,
    "sub":      lambda a, b: a - b,
    "mul":      lambda a, b: a * b,
    "absdiff":  lambda a, b: abs(a - b),
    "mod":      lambda a, b: a % b if b else 0,
    "floordiv": lambda a, b: a // b if b else 0,
    "concat":   lambda a, b: int(f"{a}{b}"),
}
OP_WEIGHTS = {"add": 5, "sub": 4, "mul": 4, "absdiff": 3, "mod": 2, "floordiv": 2, "concat": 2}

ORDERINGS = ["AB", "BA", "AB_rev", "BA_rev"]   # _rev = reverse each operand's digits first
STYLES = ["plain", "rev", "opsign", "tailsign"]


def _rev(n):
    return int(str(n)[::-1] or "0")


def _operands(a, b, ordering):
    if ordering == "AB":
        return a, b
    if ordering == "BA":
        return b, a
    if ordering == "AB_rev":
        return _rev(a), _rev(b)
    if ordering == "BA_rev":
        return _rev(b), _rev(a)
    raise ValueError(ordering)


def _style(value, style):
    """Render a numeric result string under the chosen style."""
    neg = value < 0
    digits = str(abs(value))
    if style == "plain":
        return ("-" if neg else "") + digits
    if style == "rev":
        return ("-" if neg else "") + digits[::-1]
    if style == "opsign":          # always prefix a marker
        return "-" + digits
    if style == "tailsign":        # marker as suffix
        return digits + "-"
    raise ValueError(style)


def render(a, b, op, ordering, style):
    x, y = _operands(a, b, ordering)
    return _style(OPS[op](x, y), style)


def _digit_map(rng):
    syms = list("@#$%&*+=?!")
    rng.shuffle(syms)
    return {str(d): syms[d] for d in range(10)}


def _enc(s, dmap):
    """Encode a numeric string (may carry a sign char) symbol-by-symbol."""
    return "".join(dmap.get(c, c) for c in str(s))


def sample(rng, n_examples=4, cipher=False):
    """
    Create ONE transformation puzzle.

    Returns:
        examples : list of {"a","b","result"} (symbol-encoded if cipher)
        query    : {"a","b"}
        answer   : result string
        rule     : {"op","ordering","style"[,"digit_map"]}  (how it was built)
    """
    op = rng.choices(list(OP_WEIGHTS), weights=list(OP_WEIGHTS.values()))[0]
    ordering = rng.choice(ORDERINGS)
    style = rng.choice(STYLES)
    dmap = _digit_map(rng) if cipher else None

    def enc(s):
        return _enc(s, dmap) if cipher else str(s)

    pairs = [(rng.randint(10, 99), rng.randint(10, 99)) for _ in range(n_examples + 1)]
    examples = [{"a": enc(a), "b": enc(b), "result": enc(render(a, b, op, ordering, style))}
                for a, b in pairs[:-1]]
    qa, qb = pairs[-1]
    rule = {"op": op, "ordering": ordering, "style": style}
    if cipher:
        rule["digit_map"] = dmap
    return {
        "type": "transformation_cipher" if cipher else "transformation",
        "examples": examples,
        "query": {"a": enc(qa), "b": enc(qb)},
        "answer": enc(render(qa, qb, op, ordering, style)),
        "rule": rule,
    }


def main():
    ap = argparse.ArgumentParser(description="Create TRANSFORMATION puzzles (numeric / cipher).")
    ap.add_argument("-n", "--num", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--cipher", action="store_true", help="symbol-disguised digits")
    ap.add_argument("--jsonl", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    for _ in range(args.num):
        p = sample(rng, args.examples, cipher=args.cipher)
        if args.jsonl:
            print(json.dumps(p))
            continue
        r = p["rule"]
        print(f"\nRULE  op={r['op']} ordering={r['ordering']} style={r['style']}")
        for e in p["examples"]:
            print(f"  {e['a']} ? {e['b']} = {e['result']}")
        print(f"  QUERY {p['query']['a']} ? {p['query']['b']} = {p['answer']}")


if __name__ == "__main__":
    main()
