#!/usr/bin/env python3
"""
ENCRYPTION puzzle generator  —  1:1 letter substitution over a fixed vocabulary
===============================================================================

WHAT THE PUZZLE IS
------------------
A monoalphabetic substitution cipher: every plaintext letter maps to a unique
cipher letter (a bijection on a..z), spaces preserved. All plaintext words come
from a small FIXED VOCABULARY. You see a few (cipher -> plain) sentences, then
decode a query.

HOW WE CREATE ONE (the generative model)
----------------------------------------
    1. fix a vocabulary V (the competition used a 77-word list; here a compact
       illustrative set).
    2. sample a random bijection  key : a..z -> a..z   (the cipher).
    3. build example sentences and a query from V, encode them with `key`.

The closed vocabulary is what keeps instances tractable: every plaintext word is
guaranteed to be in V, so unknown cipher-words can be resolved by testing every
same-length vocabulary word. (Decoding is usually but not always unique — a short
query can occasionally admit more than one vocabulary-consistent reading.)
"""

import argparse
import json
import random
import string

# Compact illustrative vocabulary (closed world).
VOCAB = [
    "cat", "key", "map", "the", "and", "you",
    "bird", "book", "cave", "dark", "door", "king", "near", "wise",
    "queen", "reads", "river", "sees",
    "dragon", "castle", "golden", "secret",
    "princess", "treasure", "mountain", "explores",
]


def _random_key(rng):
    """A random bijection a..z -> a..z (the substitution cipher)."""
    letters = list(string.ascii_lowercase)
    shuffled = letters[:]
    rng.shuffle(shuffled)
    return dict(zip(letters, shuffled))


def _encode(word, key):
    return "".join(key[c] for c in word)


def sample(rng, n_examples=4):
    """
    Create ONE encryption puzzle.

    Returns:
        examples : list of {"cipher","plain"} sentences (space-separated)
        query    : cipher sentence to decode
        answer   : plaintext sentence
        rule     : {"key": {plain_letter: cipher_letter, ...}}  (the cipher used)
    """
    key = _random_key(rng)
    examples = []
    for _ in range(n_examples):
        words = [rng.choice(VOCAB) for _ in range(rng.randint(3, 5))]
        examples.append({
            "plain": " ".join(words),
            "cipher": " ".join(_encode(w, key) for w in words),
        })
    qwords = [rng.choice(VOCAB) for _ in range(rng.randint(2, 4))]
    return {
        "type": "encryption",
        "examples": examples,
        "query": " ".join(_encode(w, key) for w in qwords),
        "answer": " ".join(qwords),
        "rule": {"key": key},
    }


def main():
    ap = argparse.ArgumentParser(description="Create ENCRYPTION (substitution) puzzles.")
    ap.add_argument("-n", "--num", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--jsonl", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    for _ in range(args.num):
        p = sample(rng, args.examples)
        if args.jsonl:
            print(json.dumps(p))
            continue
        print("\nEXAMPLES")
        for e in p["examples"]:
            print(f"  {e['cipher']:<28} -> {e['plain']}")
        print(f"  QUERY {p['query']:<22} -> {p['answer']}")


if __name__ == "__main__":
    main()
