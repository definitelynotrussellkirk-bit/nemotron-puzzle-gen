#!/usr/bin/env python3
"""
BIT puzzle generator  —  output = GATE(f(x), g(x), h(x))
=========================================================

WHAT THE PUZZLE IS
------------------
Several (input_byte -> output_byte) examples of an unknown 8-bit boolean
transform, plus one query input.

HOW WE CREATE ONE (the generative model)
----------------------------------------
    1. pick a boolean GATE family       (e.g. OR_XNOR)
    2. pick a STREAM TRIPLE (f, g, h)    — three whole-byte views of the input
    3. for N random input bytes x, emit  x -> GATE(f(x), g(x), h(x))
    4. hold one pair out as the QUERY.

A "stream" is the input run through one cheap whole-byte op (shift / rotate /
complement / identity). The output byte is one gate applied across three streams.

ALIAS NOTE (matters when picking streams)
-----------------------------------------
For an 8-bit byte, rol_k == ror_(8-k). So the 58 *named* streams below are only
44 *distinct functions* — two names can denote the same op.
"""

import argparse
import json
import random

MASK = 0xFF


# ----------------------------------------------------------------------
# Whole-byte stream operations
# ----------------------------------------------------------------------
def rol(x, k):
    """Rotate-left by k bits (wrap-around)."""
    return ((x << k) | (x >> (8 - k))) & MASK


def ror(x, k):
    """Rotate-right by k bits (wrap-around)."""
    return ((x >> k) | (x << (8 - k))) & MASK


def stream_catalog():
    """
    {name: fn(x) -> byte} for every stream.

        x, ~x                       identity + complement      (2)
        shl1..7, shr1..7            logical shift, zero-fill   (14)
        rol1..7, ror1..7            rotate, wrap-around        (14)
        ~shl/~shr/~rol/~ror         complement of each above   (28)
    """
    s = {"x": lambda x: x & MASK, "~x": lambda x: (~x) & MASK}
    for k in range(1, 8):
        s[f"shl{k}"] = lambda x, k=k: (x << k) & MASK
        s[f"shr{k}"] = lambda x, k=k: (x >> k) & MASK
        s[f"rol{k}"] = lambda x, k=k: rol(x, k)
        s[f"ror{k}"] = lambda x, k=k: ror(x, k)
        s[f"~shl{k}"] = lambda x, k=k: (~(x << k)) & MASK
        s[f"~shr{k}"] = lambda x, k=k: (~(x >> k)) & MASK
        s[f"~rol{k}"] = lambda x, k=k: (~rol(x, k)) & MASK
        s[f"~ror{k}"] = lambda x, k=k: (~ror(x, k)) & MASK
    return s


STREAMS = stream_catalog()


# ----------------------------------------------------------------------
# Gate families  (h is the OR / selector term where relevant; ignored for
# 2-input families, kept in a uniform 3-arg signature for simplicity)
# ----------------------------------------------------------------------
GATES = {
    "OR_XNOR":          lambda f, g, h: (h | (f & g) | ((~f) & (~g))) & MASK,
    "GATED_XNOR_NAND":  lambda f, g, h: (((~h) & ((f & g) | ((~f) & (~g)))) | (h & (~(f & g)))) & MASK,
    "CH":               lambda f, g, h: ((f & g) | ((~f) & h)) & MASK,
    "MAJ3":             lambda f, g, h: ((f & g) | (f & h) | (g & h)) & MASK,
    "PAR3":             lambda f, g, h: (f ^ g ^ h) & MASK,
    "AO":               lambda f, g, h: ((f & g) | h) & MASK,
    "AX":               lambda f, g, h: ((f & g) ^ h) & MASK,
    "AND":              lambda f, g, h: (f & g) & MASK,          # h ignored
    "OR":               lambda f, g, h: (f | g) & MASK,          # h ignored
    "XOR":              lambda f, g, h: (f ^ g) & MASK,          # h ignored
    "AND_NOT":          lambda f, g, h: (g & (~f)) & MASK,       # h ignored
}
_TWO_INPUT = {"AND", "OR", "XOR", "AND_NOT"}

# Empirical family frequency on the 1,602 competition rows. Sampling with these
# weights yields a realistic family mix (uniform-over-families would massively
# over-represent the exotic 3-input gates, which are ~1% of real bits).
GATE_WEIGHTS = {
    "OR_XNOR": 72, "GATED_XNOR_NAND": 40, "OR": 30, "AND": 30, "XOR": 25,
    "CH": 5, "AND_NOT": 3, "MAJ3": 2, "AO": 2, "AX": 2, "PAR3": 1,
}


def _fmt(b):
    return format(b & MASK, "08b")


def sample(rng, n_examples=8):
    """
    Create ONE bit puzzle.

    Returns a dict:
        examples : list of {"input","output"} (8-char binary strings)
        query    : 8-char binary string
        answer   : 8-char binary string  (the gate applied to the query)
        rule     : {"gate","f","g","h"}  (exactly how it was built)
    """
    gate = rng.choices(list(GATE_WEIGHTS), weights=list(GATE_WEIGHTS.values()))[0]
    fn = GATES[gate]
    two = gate in _TWO_INPUT

    names = list(STREAMS)
    f = rng.choice(names)
    g = rng.choice(names)
    h = rng.choice(names) if not two else f

    def apply(x):
        return fn(STREAMS[f](x), STREAMS[g](x), STREAMS[h](x))

    xs = rng.sample(range(256), n_examples + 1)   # distinct inputs + query
    examples = [{"input": _fmt(x), "output": _fmt(apply(x))} for x in xs[:-1]]
    q = xs[-1]
    return {
        "type": "bit",
        "examples": examples,
        "query": _fmt(q),
        "answer": _fmt(apply(q)),
        "rule": {"gate": gate, "f": f, "g": g, "h": (None if two else h)},
    }


def main():
    ap = argparse.ArgumentParser(description="Create BIT puzzles: GATE(f,g,h).")
    ap.add_argument("-n", "--num", type=int, default=3, help="how many to create")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples", type=int, default=8, help="examples per puzzle")
    ap.add_argument("--jsonl", action="store_true", help="emit JSONL instead of pretty")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    for _ in range(args.num):
        p = sample(rng, args.examples)
        if args.jsonl:
            print(json.dumps(p))
            continue
        r = p["rule"]
        print(f"\nRULE  {r['gate']}(f={r['f']}, g={r['g']}, h={r['h']})")
        for e in p["examples"]:
            print(f"  {e['input']} -> {e['output']}")
        print(f"  QUERY {p['query']} -> {p['answer']}")


if __name__ == "__main__":
    main()
