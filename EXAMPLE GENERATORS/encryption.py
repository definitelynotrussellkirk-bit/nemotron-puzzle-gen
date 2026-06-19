"""Generator for text encryption puzzles — 77-word competition vocabulary ONLY.

=== STRATEGY ===

Generate puzzles using EXACTLY the 77-word competition vocabulary.
Previous version used ~400 words — this directly caused 53% of enc failures
(model hallucinated non-vocab words because it was TRAINED on them).

1. Pick a random substitution cipher (bijection on 26 letters)
2. Pick random words from the 77-word vocab for 2-5 word phrases
3. Generate 3-5 example pairs
4. Run the solver on the query
5. KEEP only if deterministically solvable

CRITICAL: Every answer word MUST be in the 77-word competition vocabulary.
"""

import random
import string
import re
from solvers.dictionary import words_of_length, all_words
from wordfreq import zipf_frequency
from .base import BaseGenerator


def _make_cipher(rng: random.Random) -> dict[str, str]:
    """Create a random monoalphabetic substitution cipher."""
    letters = list(string.ascii_lowercase)
    shuffled = letters[:]
    rng.shuffle(shuffled)
    return dict(zip(letters, shuffled))


def _encrypt(text: str, cipher: dict[str, str]) -> str:
    """Encrypt plaintext using the substitution cipher."""
    return "".join(cipher.get(c, c) for c in text)


def _random_phrase(rng: random.Random, grammar: dict[str, list[str]],
                   min_words: int = 3, max_words: int = 5) -> str:
    """Generate a grammatical phrase matching competition sentence structure.

    Real data patterns:
      3-word: [subject] [verb] [object]
      4-word: [subject] [verb] [prep] [place]
           or [the] [adj] [subject] [verb]
      5-word: [subject] [verb] the [adj] [object]
    """
    # Weighted toward 4-5 words (competition distribution: 3=24%, 4=52%, 5=25%)
    n = rng.choices([3, 4, 5], weights=[24, 52, 25])[0]
    n = max(min_words, min(n, max_words))

    if n == 3:
        templates = [
            # subject verb object
            lambda: f"{rng.choice(grammar['subjects'])} {rng.choice(grammar['verbs'])} {rng.choice(grammar['objects'])}",
            # subject verb prep
            lambda: f"{rng.choice(grammar['subjects'])} {rng.choice(grammar['verbs'])} {rng.choice(grammar['preps'])}",
            # the subject verb (only if 'the' is in vocab)
            lambda: f"the {rng.choice(grammar['subjects'])} {rng.choice(grammar['verbs'])}",
        ]
        return rng.choice(templates)()
    elif n == 4:
        templates = [
            # subject verb prep object
            lambda: f"{rng.choice(grammar['subjects'])} {rng.choice(grammar['verbs'])} {rng.choice(grammar['preps'])} {rng.choice(grammar['objects'])}",
            # the adj subject verb
            lambda: f"the {rng.choice(grammar['adjs'])} {rng.choice(grammar['subjects'])} {rng.choice(grammar['verbs'])}",
            # subject verb subject object
            lambda: f"{rng.choice(grammar['subjects'])} {rng.choice(grammar['verbs'])} {rng.choice(grammar['subjects'])} {rng.choice(grammar['objects'])}",
            # subject verb the object
            lambda: f"{rng.choice(grammar['subjects'])} {rng.choice(grammar['verbs'])} the {rng.choice(grammar['objects'])}",
        ]
        return rng.choice(templates)()
    else:
        templates = [
            # subject verb the adj object
            lambda: f"{rng.choice(grammar['subjects'])} {rng.choice(grammar['verbs'])} the {rng.choice(grammar['adjs'])} {rng.choice(grammar['objects'])}",
            # the adj subject verb object
            lambda: f"the {rng.choice(grammar['adjs'])} {rng.choice(grammar['subjects'])} {rng.choice(grammar['verbs'])} {rng.choice(grammar['objects'])}",
            # subject verb prep adj object
            lambda: f"{rng.choice(grammar['subjects'])} {rng.choice(grammar['verbs'])} {rng.choice(grammar['preps'])} {rng.choice(grammar['adjs'])} {rng.choice(grammar['objects'])}",
            # subject verb adj object prep
            lambda: f"{rng.choice(grammar['subjects'])} {rng.choice(grammar['verbs'])} {rng.choice(grammar['adjs'])} {rng.choice(grammar['objects'])} {rng.choice(grammar['preps'])}",  # NEW
            # adj subject verb prep object
            lambda: f"{rng.choice(grammar['adjs'])} {rng.choice(grammar['subjects'])} {rng.choice(grammar['verbs'])} {rng.choice(grammar['preps'])} {rng.choice(grammar['objects'])}",  # NEW
        ]
        return rng.choice(templates)()


def _build_mapping_from_pairs(pairs: list[tuple[str, str]]) -> tuple[dict, dict]:
    """Build cipher→plain and plain→cipher from (ciphertext, plaintext) pairs."""
    c2p, p2c = {}, {}
    for ct, pt in pairs:
        for c, p in zip(ct, pt):
            if c != " " and p != " ":
                if c not in c2p:
                    c2p[c] = p
                    p2c[p] = c
    return c2p, p2c


def _find_matches(cipher_word: str, c2p: dict, p2c: dict) -> list[str]:
    """Find English words matching a cipher word under bijection."""
    n = len(cipher_word)
    used_plains = set(p2c.keys())
    candidates = []

    for word in words_of_length(n):
        ok = True
        new_mappings = {}

        for cc, pp in zip(cipher_word, word):
            if cc in c2p:
                if c2p[cc] != pp:
                    ok = False; break
            elif cc in new_mappings:
                if new_mappings[cc] != pp:
                    ok = False; break
            else:
                if pp in used_plains or pp in new_mappings.values():
                    ok = False; break
                new_mappings[cc] = pp

        if ok:
            candidates.append(word)

    candidates.sort(key=lambda w: -zipf_frequency(w, "en"))
    return candidates


def _is_deterministic(cipher_words: list[str], plain_words: list[str],
                      c2p: dict, p2c: dict) -> tuple[bool, bool]:
    """Check if the puzzle is deterministically solvable.

    Returns (solvable, needed_crossword):
    - solvable: True if solver gets the right answer
    - needed_crossword: True if cross-word constraints were needed
    """
    c2p = dict(c2p)
    p2c = dict(p2c)
    result = [None] * len(cipher_words)
    needed_crossword = False

    # Phase 1: solve fully known words
    for i, cw in enumerate(cipher_words):
        if all(c in c2p for c in cw):
            result[i] = "".join(c2p[c] for c in cw)

    # Phase 2: iterative constraint propagation
    changed = True
    while changed:
        changed = False
        for i, cw in enumerate(cipher_words):
            if result[i] is not None:
                continue
            matches = _find_matches(cw, c2p, p2c)
            if len(matches) == 0:
                return False, False  # contradiction
            if len(matches) == 1:
                result[i] = matches[0]
                for cc, pp in zip(cw, matches[0]):
                    if cc not in c2p:
                        c2p[cc] = pp
                        p2c[pp] = cc
                changed = True

    # Phase 3: check if unsolved words need cross-word
    unsolved = [i for i in range(len(result)) if result[i] is None]
    if not unsolved:
        return all(r == p for r, p in zip(result, plain_words)), needed_crossword

    # Try backtracking with cross-word constraints
    needed_crossword = True

    def backtrack(c2p, p2c, result):
        unsolved = [i for i in range(len(result)) if result[i] is None]
        if not unsolved:
            return all(r == p for r, p in zip(result, plain_words))

        # Pick most constrained
        best_i = min(unsolved, key=lambda i: len(_find_matches(cipher_words[i], c2p, p2c)))
        matches = _find_matches(cipher_words[best_i], c2p, p2c)

        if len(matches) == 0:
            return False

        for candidate in matches:
            new_c2p = dict(c2p)
            new_p2c = dict(p2c)
            conflict = False
            for cc, pp in zip(cipher_words[best_i], candidate):
                if cc in new_c2p:
                    if new_c2p[cc] != pp:
                        conflict = True; break
                elif pp in new_p2c:
                    if new_p2c[pp] != cc:
                        conflict = True; break
                else:
                    new_c2p[cc] = pp
                    new_p2c[pp] = cc
            if conflict:
                continue

            new_result = list(result)
            new_result[best_i] = candidate

            # Propagate
            changed = True
            while changed:
                changed = False
                for i in range(len(cipher_words)):
                    if new_result[i] is not None:
                        continue
                    m = _find_matches(cipher_words[i], new_c2p, new_p2c)
                    if len(m) == 1:
                        new_result[i] = m[0]
                        for cc, pp in zip(cipher_words[i], m[0]):
                            if cc not in new_c2p:
                                new_c2p[cc] = pp
                                new_p2c[pp] = cc
                        changed = True
                    elif len(m) == 0:
                        break

            if backtrack(new_c2p, new_p2c, new_result):
                return True

        return False

    return backtrack(c2p, p2c, result), needed_crossword


# Word pools by grammatical role — common English words
# Broad enough to be diverse, common enough to be solvable
# EXACT 77-word competition vocabulary — NOTHING ELSE
# The generator was using ~400 words. The model was being trained on
# answers it would NEVER see in competition. This directly caused
# 53% of enc failures (hallucinated non-vocab words).
_GRAMMAR = {
    "subjects": [
        "alice", "bird", "cat", "dragon", "hatter", "king", "knight",
        "mouse", "princess", "queen", "rabbit", "student", "teacher",
        "turtle", "wizard",
    ],
    "verbs": [
        "chases", "creates", "discovers", "draws", "dreams", "explores",
        "follows", "found", "imagines", "reads", "sees",
        "studies", "watches", "writes",
    ],
    "preps": [
        "above", "around", "beyond", "in", "inside", "near",
        "through", "under",
    ],
    "adjs": [
        "ancient", "bright", "clever", "colorful", "curious", "dark",
        "golden", "hidden", "magical", "mysterious", "secret", "silver",
        "strange", "wise",
    ],
    "objects": [
        "book", "castle", "cave", "crystal", "door", "forest", "garden",
        "island", "key", "library", "map", "message", "mirror",
        "mountain", "ocean", "palace", "potion", "puzzle", "school",
        "secret", "story", "tower", "treasure", "valley", "village",
        "wonderland",
    ],
}


class EncryptionGenerator(BaseGenerator):
    """Generates encryption puzzles over a fantasy-domain grammar, deterministic only.

    Uses EXACTLY the 77-word competition vocabulary. No other words allowed.
    Filters for puzzles where the solver achieves 100% correct decryption.
    Optionally prefers puzzles requiring cross-word resolution.
    """

    name = "encryption"

    def __init__(self, seed: int | None = None,
                 require_crossword: float = 0.3,
                 max_attempts: int = 50):
        """
        Args:
            seed: Random seed.
            require_crossword: Probability of requiring cross-word resolution.
                              Set higher to get more hard puzzles.
            max_attempts: Max tries before accepting a non-crossword puzzle.
        """
        super().__init__(seed)
        self.require_crossword = require_crossword
        self.max_attempts = max_attempts
    def generate_one(self) -> tuple[str, str]:
        """Generate one deterministic encryption puzzle."""
        want_crossword = self.rng.random() < self.require_crossword

        for attempt in range(self.max_attempts):
            # Step 1: random cipher
            cipher = _make_cipher(self.rng)
            reverse_cipher = {v: k for k, v in cipher.items()}

            # Step 2: generate example phrases
            n_examples = self.rng.choices([3, 4, 5, 6, 7], weights=[15, 30, 30, 15, 10])[0]
            example_plains = [_random_phrase(self.rng, _GRAMMAR) for _ in range(n_examples)]
            example_ciphers = [_encrypt(p, cipher) for p in example_plains]
            pairs = list(zip(example_ciphers, example_plains))

            # Step 3: generate query phrase
            query_plain = _random_phrase(self.rng, _GRAMMAR, min_words=3, max_words=5)
            query_cipher = _encrypt(query_plain, cipher)

            # Step 4: build mapping from examples
            c2p, p2c = _build_mapping_from_pairs(pairs)

            # Step 5: check if deterministic by running the ACTUAL solver
            # Import here to avoid circular dependency
            from solvers.encryption import solve as _solve

            prompt_candidate = (
                "In Alice's Wonderland, secret encryption rules are used on text. "
                "Here are some examples:\n"
                + "\n".join(f"{ct} -> {pt}" for ct, pt in zip(example_ciphers, example_plains))
                + f"\nNow, decrypt the following text: {query_cipher}"
            )

            predicted = _solve(prompt_candidate)
            if predicted != query_plain:
                continue  # solver can't crack it — discard

            # Compute coverage: what fraction of query letters are known from support?
            query_chars = [c for c in query_cipher.replace(" ", "")]
            known_chars = sum(1 for c in query_chars if c in c2p)
            coverage_ratio = known_chars / len(query_chars) if query_chars else 0

            # Difficulty buckets: shaped distribution toward medium/hard
            # easy (>85% known): 30% of output
            # medium (60-85% known): 45% of output
            # hard (<60% known): 25% of output
            if coverage_ratio > 0.85 and self.rng.random() < 0.50:
                continue  # reject half of easy puzzles → pushes toward medium/hard
            if coverage_ratio > 0.95 and self.rng.random() < 0.70:
                continue  # reject most trivial puzzles

            # Check if cross-word was needed (any gap word had >1 match initially)
            cipher_words = query_cipher.split()
            needed_crossword = False
            unknown_cipher_chars = set()
            for cw in cipher_words:
                partial = "".join(c2p.get(c, "_") for c in cw)
                if "_" in partial:
                    for c in cw:
                        if c not in c2p:
                            unknown_cipher_chars.add(c)
                    matches = _find_matches(cw, c2p, p2c)
                    if len(matches) > 1:
                        needed_crossword = True
            # Check for shared unknowns across words
            shared_unknowns = 0
            for uc in unknown_cipher_chars:
                words_with_uc = sum(1 for cw in cipher_words if uc in cw)
                if words_with_uc >= 2:
                    shared_unknowns += 1

            if want_crossword and not needed_crossword and attempt < self.max_attempts - 10:
                continue

            # Step 6: format the prompt (shuffle example order for diversity)
            paired = list(zip(example_ciphers, example_plains))
            self.rng.shuffle(paired)
            example_lines = [f"{ct} -> {pt}" for ct, pt in paired]
            prompt = (
                "In Alice's Wonderland, secret encryption rules are used on text. "
                "Here are some examples:\n"
                + "\n".join(example_lines)
                + f"\nNow, decrypt the following text: {query_cipher}"
            )

            return prompt, query_plain

        # All attempts failed — return None instead of unverified row
        return None
