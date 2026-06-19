#!/usr/bin/env python3
"""Micro-skill implementations.

Each skill is a class registered with @register. Add new skills here.
Run `python3 -m generators.microskill_framework --list` to see all.

Weight guide:
  5.0 = critical inference skill (rule discrimination, verification)
  3.0 = foundational operation (shifts, gates, mappings)
  2.0 = important supporting skill (edge cases, properties)
  1.0 = standard skill
  0.5 = maintenance/niche skill
"""
from generators.microskill_framework import (
    MicroSkill, register, BYTE,
    SHIFT_OPS, GATE_OPS, shift_str, gate_position_by_position, load_vocab,
)
import random


# ============================================================
# BIT MANIPULATION SKILLS
# ============================================================

@register
class BitShift(MicroSkill):
    name = "bit_shift"
    puzzle_type = "bit_manipulation"
    description = "Execute 3 shift/rotate operations on same input (batched for gradient density)"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_shift.jsonl"
    weight = 3.0       # foundational — needed for everything
    max_pool = 10000   # 28 ops x 256 inputs, batched 3 = huge space

    # CATALOGUE:
    # Sample space: 28 ops x 256 inputs, batched 3 per example = ~2400 unique combos per 2K
    # Difficulty:
    #   easy: shl1/shr1 (just move by 1)
    #   medium: any shift 1-4
    #   hard: shift 5-7, edge inputs, shift vs rotate on same input

    EASY_OPS = ["shl1", "shr1", "rol1", "ror1"]
    MED_OPS = [f"{op}{k}" for op in ["shl","shr","rol","ror"] for k in range(1,5)]
    HARD_OPS = [f"{op}{k}" for op in ["shl","shr","rol","ror"] for k in range(5,8)]
    ALL_OPS = [f"{op}{k}" for op in ["shl","shr","rol","ror"] for k in range(1,8)]

    def generate_one(self, rng, difficulty="medium"):
        x = rng.randint(0, 255)
        if difficulty == "hard":
            x = rng.choice([0, 1, 127, 128, 255, rng.randint(0, 255)])
            pool = self.HARD_OPS + self.MED_OPS
        elif difficulty == "easy":
            pool = self.EASY_OPS
        else:
            pool = self.ALL_OPS

        # Pick 3 different ops for same input
        ops = rng.sample(pool, min(3, len(pool)))
        bits = format(x, "08b")

        think_lines = []
        results = []
        for op in ops:
            result = format(SHIFT_OPS[op](x), "08b")
            think_lines.append(shift_str(op, bits))
            results.append(result)

        return {
            "user": f"Compute these on x={bits}: {', '.join(ops)}",
            "think": "\n".join(think_lines),
            "answer": ", ".join(results),
        }


@register
class BitGate(MicroSkill):
    name = "bit_gate"
    puzzle_type = "bit_manipulation"
    description = "Execute 2 bitwise gates on same inputs (batched for gradient density)"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_gate.jsonl"

    # CATALOGUE:
    # Sample space: 5 gates x 256^2 input pairs, 2 gates per example
    # Difficulty:
    #   easy: AND + OR (intuitive pair)
    #   medium: any 2 gates
    #   hard: edge inputs + near-miss gates (XOR vs XNOR)

    def generate_one(self, rng, difficulty="medium"):
        a = rng.randint(0, 255); b = rng.randint(0, 255)
        if difficulty == "hard":
            a = rng.choice([0, 255, rng.randint(0, 255)])
            b = rng.choice([0, 255, rng.randint(0, 255)])

        gates = rng.sample(list(GATE_OPS.keys()), 2)
        a_bits = format(a, "08b")
        b_bits = format(b, "08b")

        think_parts = [f"A={a_bits}, B={b_bits}\n"]
        results = []
        for gate in gates:
            result, lines = gate_position_by_position(a_bits, b_bits, gate)
            think_parts.append(f"{gate}:\n" + "\n".join(lines))
            results.append(result)

        return {
            "user": f"Compute {gates[0]} and {gates[1]} for A={a_bits}, B={b_bits}",
            "think": "\n\n".join(think_parts),
            "answer": ", ".join(results),
        }


@register
class BitRuleCheck(MicroSkill):
    name = "bit_rule_check"
    puzzle_type = "bit_manipulation"
    description = "Does this rule fit ALL given examples? Verify step by step"
    output_dir = "data/bit_manipulation/pool/generated/ms_rule_check.jsonl"

    # CATALOGUE:
    # Sample space: ~28^2 source pairs x 5 gates x correct/wrong = huge
    # Shows 4 examples, model checks each
    # Difficulty:
    #   easy: wrong rule fails on example 1 (obvious)
    #   medium: wrong rule fails on example 2-3
    #   hard: wrong rule passes 3 examples, fails on 4th (near-miss)

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        gate_names = list(GATE_OPS.keys())

        sa = rng.choice(src_names)
        sb = rng.choice([s for s in src_names if s != sa])
        gate = rng.choice(gate_names)

        def compute(x):
            return GATE_OPS[gate](SHIFT_OPS[sa](x), SHIFT_OPS[sb](x))

        inputs = rng.sample(range(256), 4)
        examples = [(format(x, "08b"), format(compute(x), "08b")) for x in inputs]

        is_correct = rng.random() < 0.5

        if is_correct:
            pa, pb, pg = sa, sb, gate
        else:
            change = rng.choice(["gate", "src"])
            if change == "gate":
                pg = rng.choice([g for g in gate_names if g != gate])
                pa, pb = sa, sb
            else:
                pa = rng.choice([s for s in src_names if s != sa])
                pb, pg = sb, gate

        pa_fn, pb_fn, pg_fn = SHIFT_OPS[pa], SHIFT_OPS[pb], GATE_OPS[pg]

        ex_str = "\n".join(f"  {inp} -> {out}" for inp, out in examples)
        rule_str = f"A={pa}, B={pb}, output={pg}"

        think_lines = []
        all_pass = True
        for j, (inp, expected) in enumerate(examples):
            x = int(inp, 2)
            computed = format(pg_fn(pa_fn(x), pb_fn(x)), "08b")
            mark = "→ MATCH" if computed == expected else "→ MISMATCH"
            think_lines.append(f"Ex {j+1}: {pg}={computed} vs {expected} {mark}")
            if computed != expected:
                all_pass = False
                think_lines.append(f"FAIL at example {j+1}")
                break

        if all_pass:
            think_lines.append("All match -> rule fits")

        return {
            "user": f"Does this rule fit all examples?\nRule: {rule_str}\n{ex_str}",
            "think": "\n".join(think_lines),
            "answer": "Yes" if all_pass else "No",
        }


@register
class BitDistinguish(MicroSkill):
    name = "bit_distinguish"
    puzzle_type = "bit_manipulation"
    description = "Which example distinguishes two candidate rules?"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_distinguish.jsonl"

    # CATALOGUE:
    # Sample space: pairs of rules that share sources but differ in gate (or vice versa)
    # Difficulty:
    #   easy: rules disagree on most examples
    #   medium: rules agree on 2-3, disagree on 1-2
    #   hard: rules agree on all but 1 example (the critical discriminator)

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        gate_names = list(GATE_OPS.keys())

        sa = rng.choice(src_names)
        sb = rng.choice([s for s in src_names if s != sa])
        g1, g2 = rng.sample(gate_names, 2)

        inputs = rng.sample(range(256), 5)

        def compute(gate, x):
            return GATE_OPS[gate](SHIFT_OPS[sa](x), SHIFT_OPS[sb](x))

        distinguishing = []
        for j, x in enumerate(inputs):
            if compute(g1, x) != compute(g2, x):
                distinguishing.append(j)

        if not distinguishing:
            return None

        examples = [(format(x, "08b"), format(compute(g1, x), "08b")) for x in inputs]
        ex_str = "\n".join(f"  Ex {j+1}: {inp} -> {out}" for j, (inp, out) in enumerate(examples))

        think_lines = []
        for j, x in enumerate(inputs):
            o1 = format(compute(g1, x), "08b")
            o2 = format(compute(g2, x), "08b")
            if o1 == o2:
                think_lines.append(f"Ex {j+1}: both give {o1} -- same")
            else:
                think_lines.append(f"Ex {j+1}: Rule1={o1}, Rule2={o2} -- DIFFERENT")

        return {
            "user": f"Two rules:\n  Rule 1: A={sa}, B={sb}, {g1}\n  Rule 2: A={sa}, B={sb}, {g2}\nExamples:\n{ex_str}\nWhich example distinguishes them?",
            "think": "\n".join(think_lines),
            "answer": f"Example {distinguishing[0] + 1}",
        }


@register
class BitSimilarity(MicroSkill):
    name = "bit_similarity"
    puzzle_type = "bit_manipulation"
    description = "How many bits match between shift(input) and output?"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_similarity.jsonl"

    # CATALOGUE:
    # Sample space: 28 shifts x 256 inputs x 256 outputs
    # Teaches the model to SCORE candidate sources — the key step in source identification
    # Difficulty:
    #   easy: high match (7-8 bits) or low match (0-1)
    #   medium: moderate match (3-5)
    #   hard: near-miss (6 bits match but 2 critical ones don't)

    def generate_one(self, rng, difficulty="medium"):
        op_name = rng.choice(list(SHIFT_OPS.keys()))
        x = rng.randint(0, 255)
        shifted = SHIFT_OPS[op_name](x)

        # The "output" is from the actual puzzle rule (unknown to model)
        # For training, we just generate a random output
        if difficulty == "easy":
            output = shifted if rng.random() < 0.5 else rng.randint(0, 255)
        elif difficulty == "hard":
            # Near-miss: flip 1-2 bits from shifted
            output = shifted
            for _ in range(rng.randint(1, 2)):
                output ^= (1 << rng.randint(0, 7))
        else:
            output = rng.randint(0, 255)

        x_bits = format(x, "08b")
        s_bits = format(shifted, "08b")
        o_bits = format(output, "08b")

        matches = sum(1 for i in range(8) if s_bits[i] == o_bits[i])

        think = f"{op_name}({x_bits}) = {s_bits}\nCompare to output {o_bits}:\n"
        think += f"  {' '.join(s_bits)}\n  {' '.join(o_bits)}\n"
        think += f"  {''.join('→ MATCH' if s_bits[i] == o_bits[i] else '→ MISMATCH' for i in range(8))}\n"
        think += f"  {matches}/8 bits match"

        return {
            "user": f"How many bits of {op_name}({x_bits}) match output {o_bits}?",
            "think": think,
            "answer": f"{matches}/8",
        }


# ============================================================
# ENCRYPTION SKILLS
# ============================================================

@register
class EncExtractMapping(MicroSkill):
    name = "enc_extract_mapping"
    puzzle_type = "encryption"
    description = "Extract cipher->plain mappings from one sentence pair"
    output_dir = "data/encryption/pool/generated/ms_enc_extract.jsonl"

    # CATALOGUE:
    # Sample space: all Wonderland sentence pairs
    # The FIRST step of encryption solving — model must learn this perfectly
    # Difficulty:
    #   easy: short words (3-4 letters), no repeated letters
    #   medium: standard sentence (4-5 words)
    #   hard: words with repeated letters, long sentences

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab:
            return None

        if difficulty == "easy":
            n_words = rng.randint(2, 3)
            pool = [w for w in vocab if len(w) <= 4 and len(set(w)) == len(w)]
        elif difficulty == "hard":
            n_words = rng.randint(4, 6)
            pool = vocab
        else:
            n_words = rng.randint(3, 4)
            pool = [w for w in vocab if len(w) >= 3]

        if len(pool) < n_words:
            pool = vocab

        words = rng.sample(pool, min(n_words, len(pool)))
        plain = " ".join(words)

        # Build bijection
        plain_chars = sorted(set(plain.replace(" ", "")))
        available = list("abcdefghijklmnopqrstuvwxyz")
        rng.shuffle(available)
        p2c = {}
        for ch in plain_chars:
            for cc in available:
                if cc not in p2c.values():
                    p2c[ch] = cc
                    break

        cipher = "".join(p2c.get(ch, " ") if ch != " " else " " for ch in plain)

        # Build trace
        think_lines = [f'"{cipher}" = "{plain}"']
        mappings = []
        seen = set()
        for cc, pp in zip(cipher.replace(" ", ""), plain.replace(" ", "")):
            if cc not in seen:
                mappings.append(f"{cc}->{pp}")
                seen.add(cc)
        think_lines.append(f"Mappings: {', '.join(mappings)}")

        return {
            "user": f'Extract all cipher->plain mappings from:\n  "{cipher}" = "{plain}"',
            "think": "\n".join(think_lines),
            "answer": ", ".join(mappings),
        }


@register
class EncApplyMapping(MicroSkill):
    name = "enc_apply_mapping"
    puzzle_type = "encryption"
    description = "Apply a known mapping to partially decrypt a word"
    output_dir = "data/encryption/pool/generated/ms_enc_apply.jsonl"

    # CATALOGUE:
    # Sample space: any word with partial mapping
    # Difficulty:
    #   easy: all letters known -> full decrypt
    #   medium: 1-2 unknown letters -> partial with underscores
    #   hard: 3+ unknown + repeated letters

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab:
            return None

        word = rng.choice([w for w in vocab if len(w) >= 4])
        letters = sorted(set(word))

        # Build partial mapping
        if difficulty == "easy":
            known = letters  # all known
        elif difficulty == "hard":
            n_known = max(1, len(letters) - 3)
            known = rng.sample(letters, n_known)
        else:
            n_known = max(1, len(letters) - rng.randint(1, 2))
            known = rng.sample(letters, n_known)

        available = list("abcdefghijklmnopqrstuvwxyz")
        rng.shuffle(available)
        p2c = {}
        for ch in known:
            for cc in available:
                if cc not in p2c.values():
                    p2c[ch] = cc
                    break

        cipher_word = "".join(p2c.get(ch, "?") for ch in word)
        c2p = {v: k for k, v in p2c.items()}
        partial = "".join(c2p.get(cc, "_") for cc in cipher_word)

        mapping_str = ", ".join(f"{cc}->{pp}" for cc, pp in sorted(c2p.items()))

        think = f"Apply mapping to '{cipher_word}':\n"
        for cc in cipher_word:
            if cc in c2p:
                think += f"  {cc} -> {c2p[cc]}\n"
            else:
                think += f"  {cc} -> _\n"
        think += f"Result: {partial}"

        return {
            "user": f"Known mapping: {mapping_str}\nDecrypt: {cipher_word}",
            "think": think,
            "answer": partial,
        }


@register
class EncStringCount(MicroSkill):
    name = "str_count"
    puzzle_type = "encryption"
    description = "Count characters in 3 words + compare lengths (batched)"
    output_dir = "data/encryption/pool/generated/ms_str_count.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None
        words = rng.sample(vocab, 3)
        think_lines = []
        for w in words:
            think_lines.append(f'  "{w}": {len(w)} letters')
        longest = max(words, key=len)
        shortest = min(words, key=len)
        think_lines.append(f'Longest: "{longest}" ({len(longest)}), Shortest: "{shortest}" ({len(shortest)})')
        return {
            "user": f'Count letters in: "{words[0]}", "{words[1]}", "{words[2]}". Which is longest?',
            "think": "\n".join(think_lines),
            "answer": f'{longest} ({len(longest)} letters)',
        }


@register
class EncStringCompare(MicroSkill):
    name = "str_compare"
    puzzle_type = "encryption"
    description = "Compare two strings character by character, find differences"
    output_dir = "data/encryption/pool/generated/ms_str_compare.jsonl"

    # CATALOGUE:
    # Sample space: pairs of similar strings (near-misses)
    # Difficulty:
    #   easy: completely different strings
    #   medium: 1-2 chars different
    #   hard: differ only in one position, or transposition

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab:
            return None

        word = rng.choice([w for w in vocab if len(w) >= 4])

        if difficulty == "easy":
            other = rng.choice([w for w in vocab if len(w) == len(word) and w != word])
        elif difficulty == "hard":
            # Near-miss: change just 1 char, or swap 2 adjacent
            other = list(word)
            if rng.random() < 0.5 and len(other) > 1:
                # swap adjacent
                pos = rng.randint(0, len(other) - 2)
                other[pos], other[pos + 1] = other[pos + 1], other[pos]
            else:
                pos = rng.randint(0, len(other) - 1)
                other[pos] = rng.choice("abcdefghijklmnopqrstuvwxyz")
            other = "".join(other)
        else:
            other = list(word)
            n_changes = rng.randint(1, 2)
            for _ in range(n_changes):
                pos = rng.randint(0, len(other) - 1)
                other[pos] = rng.choice("abcdefghijklmnopqrstuvwxyz")
            other = "".join(other)

        same = word == other
        diffs = []
        if len(word) == len(other):
            for j in range(len(word)):
                if word[j] != other[j]:
                    diffs.append(f"pos {j}: '{word[j]}' vs '{other[j]}'")
            match_count = len(word) - len(diffs)
        else:
            diffs.append(f"different lengths: {len(word)} vs {len(other)}")
            match_count = 0

        think = f'"{word}" vs "{other}":\n'
        if len(word) == len(other):
            think += f"  {' '.join(word)}\n  {' '.join(other)}\n"
            think += f"  {''.join('→ MATCH' if word[j] == other[j] else '→ MISMATCH' for j in range(len(word)))}\n"
        think += f"  {match_count}/{max(len(word),len(other))} match"
        if diffs:
            think += f"\n  Differences: {'; '.join(diffs)}"

        answer = "Same" if same else f"Different ({len(diffs)} position{'s' if len(diffs)>1 else ''})"

        return {
            "user": f'Are these the same? "{word}" vs "{other}"',
            "think": think,
            "answer": answer,
        }


# ============================================================
# TRANSFORMATION SKILLS
# ============================================================

@register
class TransParse(MicroSkill):
    name = "trans_parse"
    puzzle_type = "transformation"
    description = "Parse an equation into left operand, operator, right operand, result"
    output_dir = "data/transformation/pool/generated/ms_trans_parse.jsonl"

    # CATALOGUE:
    # Sample space: numeric equations with various operators
    # Difficulty:
    #   easy: single-digit operands, standard operators (+, -, *)
    #   medium: two-digit operands
    #   hard: negative results, sign-encoded results

    def generate_one(self, rng, difficulty="medium"):
        if difficulty == "easy":
            a = rng.randint(1, 9)
            b = rng.randint(1, 9)
            op = rng.choice(["+", "-", "*"])
        elif difficulty == "hard":
            a = rng.randint(10, 99)
            b = rng.randint(10, 99)
            op = rng.choice(["+", "-", "*", "^", "#"])
        else:
            a = rng.randint(10, 99)
            b = rng.randint(1, 50)
            op = rng.choice(["+", "-", "*"])

        if op in ["+", "^", "#"]:
            result = a + b
        elif op == "-":
            result = a - b
        else:
            result = a * b

        eq = f"{a}{op}{b}={result}"
        think = f"Parse: {eq}\n  Left: {a}\n  Operator: {op}\n  Right: {b}\n  Result: {result}"

        return {
            "user": f'Parse this equation: {eq}',
            "think": think,
            "answer": f"Left={a}, Op={op}, Right={b}, Result={result}",
        }


@register
class TransOpFromExamples(MicroSkill):
    name = "trans_op_from_examples"
    puzzle_type = "transformation"
    description = "Given 3 equations with hidden operator, identify the operation"
    output_dir = "data/transformation/pool/generated/ms_trans_op_examples.jsonl"

    # CATALOGUE:
    # Sample space: all arithmetic operations used in competition
    # Difficulty:
    #   easy: addition/subtraction (obvious)
    #   medium: multiplication, abs difference
    #   hard: near-miss operations (a+b vs a+b+1)

    OPS = {
        "a+b": lambda a, b: a + b,
        "a-b": lambda a, b: a - b,
        "a*b": lambda a, b: a * b,
        "a+b+1": lambda a, b: a + b + 1,
        "a*b-1": lambda a, b: a * b - 1,
        "abs(a-b)": lambda a, b: abs(a - b),
    }

    def generate_one(self, rng, difficulty="medium"):
        if difficulty == "easy":
            ops = ["a+b", "a-b"]
        elif difficulty == "hard":
            ops = list(self.OPS.keys())
        else:
            ops = ["a+b", "a-b", "a*b"]

        correct_name = rng.choice(ops)
        fn = self.OPS[correct_name]

        examples = []
        for _ in range(3):
            a = rng.randint(5, 50)
            b = rng.randint(5, 50)
            examples.append((a, b, fn(a, b)))

        ex_str = "\n".join(f"  {a} ? {b} = {r}" for a, b, r in examples)

        # Show checking each candidate
        think_lines = []
        for name, test_fn in self.OPS.items():
            fits = all(test_fn(a, b) == r for a, b, r in examples)
            mark = "→ MATCH fits all" if fits else "→ MISMATCH"
            think_lines.append(f"  {name}: {mark}")
            if fits and name == correct_name:
                pass  # keep going to show others don't fit

        return {
            "user": f"What operation fits all examples?\n{ex_str}",
            "think": "\n".join(think_lines),
            "answer": correct_name,
        }


# ============================================================
# BIT: COMPOSITION (stage 2+3)
# ============================================================

@register
class BitCompose2(MicroSkill):
    name = "bit_compose2"
    puzzle_type = "bit_manipulation"
    description = "Two-step: shift then gate, both shown step by step"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_compose2.jsonl"
    # Sample space: 28 shifts x 28 shifts x 5 gates x 256 inputs

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        gate_names = list(GATE_OPS.keys())
        sa, sb = rng.sample(src_names, 2)
        gate = rng.choice(gate_names)
        x = rng.randint(0, 255)
        bits = format(x, "08b")
        a = SHIFT_OPS[sa](x)
        b = SHIFT_OPS[sb](x)
        a_bits = format(a, "08b")
        b_bits = format(b, "08b")
        result, g_lines = gate_position_by_position(a_bits, b_bits, gate)

        think = f"A = {shift_str(sa, bits)}\nB = {shift_str(sb, bits)}\n{gate}:\n" + "\n".join(g_lines)
        return {
            "user": f"Compute {gate} where A={sa}(x), B={sb}(x), x={bits}",
            "think": think,
            "answer": result,
        }


@register
class BitCompose3(MicroSkill):
    name = "bit_compose3"
    puzzle_type = "bit_manipulation"
    description = "Three-step: 3 shifts + family formula with helper bytes"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_compose3.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        sa, sb, sc = rng.sample(src_names, 3)
        x = rng.randint(0, 255)
        bits = format(x, "08b")
        a = SHIFT_OPS[sa](x); b = SHIFT_OPS[sb](x); c = SHIFT_OPS[sc](x)
        ab, al = gate_position_by_position(format(a,"08b"), format(b,"08b"), "A ^ B")
        p = int(ab, 2)
        cb, cl = gate_position_by_position(format(c,"08b"), ab, "A | B")

        think = f"A = {shift_str(sa, bits)}\nB = {shift_str(sb, bits)}\nC = {shift_str(sc, bits)}\n"
        think += f"P = A ^ B:\n" + "\n".join(al) + f"\noutput = C | P:\n" + "\n".join(cl)
        return {
            "user": f"Compute C | (A ^ B) where A={sa}, B={sb}, C={sc}, x={bits}",
            "think": think,
            "answer": cb,
        }


# ============================================================
# BIT: EDGE CASES
# ============================================================

@register
class BitEdgeShiftRotate(MicroSkill):
    name = "bit_edge_shift_vs_rotate"
    puzzle_type = "bit_manipulation"
    description = "Side-by-side shift vs rotate on same input — teaches the distinction"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_edge_sr.jsonl"
    # The model confuses shifts and rotations. This drills the difference.

    def generate_one(self, rng, difficulty="medium"):
        k = rng.randint(1, 4) if difficulty != "hard" else rng.randint(5, 7)
        x = rng.randint(1, 254) if difficulty != "hard" else rng.choice([1, 128, 255, rng.randint(0,255)])
        bits = format(x, "08b")

        shl_r = format(SHIFT_OPS[f"shl{k}"](x), "08b")
        rol_r = format(SHIFT_OPS[f"rol{k}"](x), "08b")
        same = shl_r == rol_r

        lost = bits[:k]
        think = f"shl{k}({bits}): drop first {k}, append zeros -> {shl_r}\n"
        think += f"rol{k}({bits}): move first {k} to end -> {rol_r}\n"
        think += f"Lost bits in shift: {lost}\n"
        think += f"Same result: {'yes (lost bits were all 0)' if same else 'no (lost bits had 1s that wrapped in rotate)'}"

        return {
            "user": f"Compare shl{k} vs rol{k} on {bits}. Same result?",
            "think": think,
            "answer": "Yes" if same else "No",
        }


@register
class BitEdgeZeros(MicroSkill):
    name = "bit_edge_zeros"
    puzzle_type = "bit_manipulation"
    description = "Apply 3 different ops to an extreme input — compare behaviors"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_edge_zeros.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        x = rng.choice([0, 255, 1, 128, 64, 32, 2, 4, 8, 16, 127, 254, 3, 192])
        ops = rng.sample(list(SHIFT_OPS.keys()), 3)
        bits = format(x, "08b")

        if x == 0: note = "All zeros input"
        elif x == 255: note = "All ones input"
        elif bin(x).count('1') == 1: note = f"Single bit at position {7 - format(x,'08b').index('1')}"
        else: note = f"Edge value {x}"

        think_lines = [f"{note}: {bits}\n"]
        results = []
        for o in ops:
            r = format(SHIFT_OPS[o](x), "08b")
            results.append((o, r))
            desc = shift_str(o, bits)
            think_lines.append(f"  {desc}")

        think = "\n".join(think_lines)
        answers = ", ".join(f"{o}={r}" for o, r in results)
        prompt = "Compute each:\n" + "\n".join(f"  {o}({bits})" for o, _, in results)
        return {
            "user": prompt,
            "think": think,
            "answer": answers,
        }


@register
class BitEdgeGate(MicroSkill):
    name = "bit_edge_gate"
    puzzle_type = "bit_manipulation"
    description = "Gate operations with extreme inputs (all 0, all 1, identical)"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_edge_gate.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        gate = rng.choice(list(GATE_OPS.keys()))
        case = rng.choice(["identical", "complement", "zero_a", "one_b"])

        x = rng.randint(0, 255)
        if case == "identical":
            a, b = x, x
            note = "Identical inputs"
        elif case == "complement":
            a, b = x, (~x) & BYTE
            note = "Complementary inputs"
        elif case == "zero_a":
            a, b = 0, x
            note = "A is all zeros"
        else:
            a, b = x, 255
            note = "B is all ones"

        ab, al = gate_position_by_position(format(a,"08b"), format(b,"08b"), gate)
        think = f"{note}\n{gate}:\n" + "\n".join(al)
        return {
            "user": f"Compute {gate}: A={format(a,'08b')}, B={format(b,'08b')}",
            "think": think,
            "answer": ab,
        }


@register
class BitErrorDetect(MicroSkill):
    name = "bit_error_detect"
    puzzle_type = "bit_manipulation"
    description = "Find the error in a computation"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_error.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        op = rng.choice(list(SHIFT_OPS.keys()))
        x = rng.randint(0, 255)
        bits = format(x, "08b")
        correct = format(SHIFT_OPS[op](x), "08b")
        wrong = list(correct)
        pos = rng.randint(0, 7)
        wrong[pos] = "1" if wrong[pos] == "0" else "0"
        wrong = "".join(wrong)

        think = f"Claimed: {op}({bits}) = {wrong}\nActual: {shift_str(op, bits)}\nBit {pos} wrong: got {wrong[pos]}, should be {correct[pos]}"
        return {
            "user": f"Find the error: {op}({bits}) = {wrong}",
            "think": think,
            "answer": f"Bit {pos} wrong. Correct: {correct}",
        }


@register
class BitFullVerify(MicroSkill):
    name = "bit_full_verify"
    puzzle_type = "bit_manipulation"
    description = "Verify a rule against ALL 5 examples step by step"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_full_verify.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        sa, sb = rng.sample(src_names, 2)
        gate = rng.choice(list(GATE_OPS.keys()))

        def compute(x):
            return GATE_OPS[gate](SHIFT_OPS[sa](x), SHIFT_OPS[sb](x))

        inputs = rng.sample(range(256), 5)
        examples = [(format(x,"08b"), format(compute(x),"08b")) for x in inputs]
        ex_str = "\n".join(f"  {inp} -> {out}" for inp, out in examples)

        think_lines = []
        for j, (inp, expected) in enumerate(examples):
            x = int(inp, 2)
            out = format(compute(x), "08b")
            think_lines.append(f"Ex {j+1}: {gate}({sa},{sb})={out} vs {expected} → MATCH")
        think_lines.append("All 5 match")

        return {
            "user": f"Verify rule A={sa}, B={sb}, {gate} against all:\n{ex_str}",
            "think": "\n".join(think_lines),
            "answer": "All 5 match",
        }


# ============================================================
# ENCRYPTION: REMAINING SKILLS
# ============================================================

@register
class EncBijection(MicroSkill):
    name = "enc_bijection"
    puzzle_type = "encryption"
    description = "Is this mapping bijective? Check for duplicates"
    output_dir = "data/encryption/pool/generated/ms_enc_bijection.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        letters = list("abcdefghijklmnopqrstuvwxyz")
        n = rng.randint(4, 8)
        cipher = rng.sample(letters, n)
        plain = rng.sample(letters, n)
        valid = rng.random() < 0.5
        if not valid:
            plain[rng.randint(1, n-1)] = plain[0]
        mapping = ", ".join(f"{c}->{p}" for c, p in zip(cipher, plain))
        dup = plain[0] if not valid else None
        think = f"Check: {mapping}\n"
        if valid:
            think += "All plain letters unique -> bijective"
        else:
            think += f"'{dup}' appears twice -> NOT bijective"
        return {
            "user": f"Is this mapping bijective? {mapping}",
            "think": think,
            "answer": "Yes" if valid else "No",
        }


@register
class EncForcedMapping(MicroSkill):
    name = "enc_forced_mapping"
    puzzle_type = "encryption"
    description = "Given cipher=plain word, what mappings are forced?"
    output_dir = "data/encryption/pool/generated/ms_enc_forced.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None
        word = rng.choice([w for w in vocab if len(w) >= 4])
        letters = list("abcdefghijklmnopqrstuvwxyz"); rng.shuffle(letters)
        c2p = {}; cipher = []
        for ch in word:
            existing = [k for k,v in c2p.items() if v == ch]
            if existing:
                cipher.append(existing[0])
            else:
                cc = next(c for c in letters if c not in c2p)
                c2p[cc] = ch; cipher.append(cc)
        cipher_str = "".join(cipher)
        pairs = [f"{cc}->{pp}" for cc, pp in sorted(set(zip(cipher, word)))]
        think = f'"{cipher_str}" = "{word}"\n' + "\n".join(f"  {cc} -> {pp}" for cc, pp in zip(cipher, word))
        think += f"\nForced: {', '.join(pairs)}"
        return {
            "user": f'If cipher "{cipher_str}" = plain "{word}", what mappings are forced?',
            "think": think,
            "answer": ", ".join(pairs),
        }


@register
class EncPropagation(MicroSkill):
    name = "enc_propagation"
    puzzle_type = "encryption"
    description = "If we learn a new mapping, what's blocked/unused?"
    output_dir = "data/encryption/pool/generated/ms_enc_propagation.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        letters = list("abcdefghijklmnopqrstuvwxyz")
        n = rng.randint(5, 12)
        c_set = rng.sample(letters, n)
        p_set = rng.sample(letters, n)
        c2p = dict(zip(c_set, p_set))
        p2c = dict(zip(p_set, c_set))
        new_c = rng.choice([c for c in letters if c not in c2p])
        new_p = rng.choice([p for p in letters if p not in p2c])
        new_unused = sorted(p for p in letters if p not in p2c and p != new_p)
        existing = ", ".join(f"{c}->{p}" for c, p in sorted(c2p.items()))
        think = f"Current: {existing}\nNew: {new_c}->{new_p}\nNow unused: {', '.join(new_unused)}"
        return {
            "user": f"Mapping: {existing}\nWe learn {new_c}->{new_p}. What plain letters still unused?",
            "think": think,
            "answer": ", ".join(new_unused),
        }


@register
class EncMostConstrained(MicroSkill):
    name = "enc_most_constrained"
    puzzle_type = "encryption"
    description = "Which partial word has fewest valid completions?"
    output_dir = "data/encryption/pool/generated/ms_most_constrained.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None
        from collections import defaultdict
        by_len = defaultdict(list)
        for w in vocab: by_len[len(w)].append(w)

        data = []
        for _ in range(3):
            length = rng.choice([4,5,6,7])
            pool = by_len.get(length, [])
            if not pool: continue
            target = rng.choice(pool)
            hide = rng.randint(1, min(2, length-1))
            hide_pos = set(rng.sample(range(length), hide))
            pattern = "".join("_" if j in hide_pos else target[j] for j in range(length))
            matches = sum(1 for w in pool if all(pattern[j]=="_" or pattern[j]==w[j] for j in range(length)))
            data.append((pattern, matches))

        if len(data) < 3: return None
        labels = ["A","B","C"]
        best = min(range(3), key=lambda j: data[j][1])
        options = "\n".join(f"{labels[j]}) {p} ({n} matches)" for j,(p,n) in enumerate(data))
        think = "\n".join(f"{labels[j]}) {p}: {n} matches{'  <- fewest' if j==best else ''}" for j,(p,n) in enumerate(data))
        return {
            "user": f"Which has fewest completions?\n{options}",
            "think": think,
            "answer": labels[best],
        }


@register
class EncVocab(MicroSkill):
    name = "enc_vocab"
    puzzle_type = "encryption"
    description = "List words matching a random pattern from Alice's vocabulary"
    output_dir = "data/encryption/pool/generated/ms_enc_vocab.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None
        from collections import defaultdict
        by_len = defaultdict(list)
        for w in vocab: by_len[len(w)].append(w)
        length = rng.choice(list(by_len.keys()))
        pool = by_len[length]
        # Pick a random word and create a pattern with varying reveal strategies
        word = rng.choice(pool)
        strategy = rng.choice(["random_reveal", "random_reveal", "random_reveal", "first_last"])
        if strategy == "first_last":
            reveal_pos = {0, length - 1}
        elif strategy == "consonants":
            reveal_pos = {i for i in range(length) if word[i] not in "aeiou"}
            if len(reveal_pos) >= length: reveal_pos = {0}  # fallback
        elif strategy == "vowels":
            reveal_pos = {i for i in range(length) if word[i] in "aeiou"}
            if not reveal_pos: reveal_pos = {0}  # fallback
        else:
            n_reveal = rng.randint(1, max(1, length - 2))
            reveal_pos = set(rng.sample(range(length), n_reveal))
        pattern = "".join(word[i] if i in reveal_pos else "_" for i in range(length))
        matches = [w for w in pool if all(pattern[i] == "_" or pattern[i] == w[i] for i in range(length))]
        rng.shuffle(matches)
        think = f"Pattern: {pattern} ({length} letters)\nMatches: {', '.join(matches)}\n({len(matches)} words)"
        return {
            "user": f"What Alice words match the pattern \"{pattern}\"?",
            "think": think,
            "answer": ", ".join(matches),
        }


@register
class EncNotForced(MicroSkill):
    name = "enc_not_forced"
    puzzle_type = "encryption"
    description = "Which mapping is NOT forced by this cipher=plain pair?"
    output_dir = "data/encryption/pool/generated/ms_enc_not_forced.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None
        candidates = [w for w in vocab if len(w) >= 5 and len(set(w)) == len(w)]
        if not candidates: return None
        word = rng.choice(candidates)
        letters = list("abcdefghijklmnopqrstuvwxyz"); rng.shuffle(letters)
        cipher = letters[:len(word)]
        pairs = list(zip(cipher, word))
        extra_c = rng.choice([c for c in "abcdefghijklmnopqrstuvwxyz" if c not in cipher])
        extra_p = rng.choice([c for c in "abcdefghijklmnopqrstuvwxyz" if c not in word])
        all_claims = pairs + [(extra_c, extra_p)]; rng.shuffle(all_claims)
        labels = [chr(65+j) for j in range(len(all_claims))]
        correct = labels[all_claims.index((extra_c, extra_p))]
        options = "\n".join(f"{labels[j]}) {c}->{p}" for j,(c,p) in enumerate(all_claims))
        think = "\n".join(f"{labels[j]}) {c}->{p} {'-- NOT in cipher' if (c,p)==(extra_c,extra_p) else '-- forced'}" for j,(c,p) in enumerate(all_claims))
        return {
            "user": f'Cipher "{"".join(cipher)}" = "{word}". Which NOT forced?\n{options}',
            "think": think,
            "answer": correct,
        }


@register
class EncReversed(MicroSkill):
    name = "enc_reverse_decrypt"
    puzzle_type = "encryption"
    description = "Given plaintext + mapping, generate the cipher"
    output_dir = "data/encryption/pool/generated/ms_enc_reverse.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None
        word = rng.choice([w for w in vocab if len(w) >= 3])
        letters = list("abcdefghijklmnopqrstuvwxyz"); rng.shuffle(letters)
        p2c = {}
        for ch in set(word):
            cc = next(c for c in letters if c not in p2c.values())
            p2c[ch] = cc
        cipher = "".join(p2c[ch] for ch in word)
        mapping = ", ".join(f"{p}->{c}" for p,c in sorted(p2c.items()))
        think = "\n".join(f"  {ch} -> {p2c[ch]}" for ch in word) + f'\nCipher: "{cipher}"'
        return {
            "user": f'Encrypt "{word}" using: {mapping}',
            "think": think,
            "answer": cipher,
        }


@register
class EncRepeatedLetters(MicroSkill):
    name = "enc_repeated_letters"
    puzzle_type = "encryption"
    description = "Given cipher word, does the repeat pattern match a candidate plain word?"
    output_dir = "data/encryption/pool/generated/ms_enc_repeated.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        from collections import Counter
        repeat_words = [w for w in vocab if len(set(w)) < len(w) and len(w) >= 4]
        if not repeat_words: return None
        word = rng.choice(repeat_words)
        # Build letter-pattern (aabca etc)
        seen = {}; nl = 'a'; pattern = []
        for ch in word:
            if ch not in seen: seen[ch] = nl; nl = chr(ord(nl)+1)
            pattern.append(seen[ch])
        pat_str = "".join(pattern)
        # Generate a random cipher with same pattern
        letters = list("abcdefghijklmnopqrstuvwxyz"); rng.shuffle(letters)
        c_map = {}; cipher = []
        for p in pattern:
            if p not in c_map: c_map[p] = letters.pop()
            cipher.append(c_map[p])
        cipher_str = "".join(cipher)
        # Pick a wrong candidate with DIFFERENT pattern
        wrong_candidates = [w for w in vocab if len(w) == len(word) and w != word]
        if not wrong_candidates:
            return None
        wrong = rng.choice(wrong_candidates)
        ws = {}; wnl = 'a'; wp = []
        for ch in wrong:
            if ch not in ws: ws[ch] = wnl; wnl = chr(ord(wnl)+1)
            wp.append(ws[ch])
        wrong_pat = "".join(wp)

        think = f'Cipher: {cipher_str}\n'
        think += f'Repeat pattern: {pat_str}\n'
        think += f'Candidate "{word}": pattern {pat_str} — MATCHES → MATCH\n'
        think += f'Candidate "{wrong}": pattern {wrong_pat} — {"MATCHES" if wrong_pat == pat_str else "NO MATCH → MISMATCH"}'

        fits = "Yes" if wrong_pat == pat_str else "No"
        return {
            "user": f'Cipher "{cipher_str}" has repeat pattern {pat_str}. Does "{wrong}" fit this pattern?',
            "think": think,
            "answer": fits,
        }


# ============================================================
# TRANSFORMATION: REMAINING SKILLS
# ============================================================

@register
class TransBase(MicroSkill):
    name = "trans_base"
    puzzle_type = "transformation"
    description = "Base identification, encode/decode in base N"
    output_dir = "data/transformation/pool/generated/ms_base.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        base = rng.randint(6, 12)
        qtype = rng.choice(["encode", "decode", "encode_large", "compare_bases", "encode", "decode"])
        if qtype == "encode":
            num = rng.randint(0, base**3 - 1)  # larger range
            digits = []
            n = num
            if n == 0: digits = [0]
            else:
                while n > 0: digits.append(n % base); n //= base
                digits.reverse()
            answer = "".join(str(d) for d in digits)
            steps = []
            n = num
            while n > 0:
                steps.append(f"  {n} ÷ {base} = {n//base} remainder {n%base}")
                n //= base
            think = f"{num} in base {base}:\n" + "\n".join(steps) + f"\nDigits: {answer}"
            return {"user": f"Encode {num} in base {base}", "think": think, "answer": answer}
        elif qtype == "encode_large":
            num = rng.randint(base**2, base**4 - 1)
            digits = []
            n = num
            while n > 0: digits.append(n % base); n //= base
            digits.reverse()
            answer = "".join(str(d) for d in digits)
            think = f"{num} in base {base}: {answer} ({len(digits)} digits)"
            return {"user": f"Encode {num} in base {base}", "think": think, "answer": answer}
        elif qtype == "compare_bases":
            num = rng.randint(10, 200)
            b1 = rng.randint(6, 10)
            b2 = rng.randint(8, 12)
            while b1 == b2: b2 = rng.randint(8, 12)
            def to_base(v, b):
                if v == 0: return "0"
                d = []
                while v > 0: d.append(str(v % b)); v //= b
                return "".join(reversed(d))
            r1 = to_base(num, b1); r2 = to_base(num, b2)
            think = f"{num} in base {b1}: {r1}\n{num} in base {b2}: {r2}\nMore digits in base {b1 if len(r1) > len(r2) else b2}"
            return {"user": f"Encode {num} in both base {b1} and base {b2}. Which uses more digits?", "think": think, "answer": f"base {b1}: {r1}, base {b2}: {r2}"}
        elif qtype == "decode":
            digits = [rng.randint(0, base-1) for _ in range(rng.randint(1,3))]
            if digits[0] == 0 and len(digits) > 1: digits[0] = rng.randint(1, base-1)
            value = sum(d * base**p for p, d in enumerate(reversed(digits)))
            ds = "".join(str(d) for d in digits)
            think = f'"{ds}" base {base} = {" + ".join(f"{d}*{base}^{len(digits)-1-p}" for p,d in enumerate(digits))} = {value}'
            return {"user": f'Decode "{ds}" from base {base}', "think": think, "answer": str(value)}
        else:  # fallback encode
            num = rng.randint(base, base**3)
            digits = []; n = num
            while n > 0: digits.append(n % base); n //= base
            digits.reverse()
            answer = "".join(str(d) for d in digits)
            think = f"{num} in base {base} = {answer}"
            return {"user": f"Encode {num} in base {base}", "think": think, "answer": answer}


@register
class TransSign(MicroSkill):
    name = "trans_sign"
    puzzle_type = "transformation"
    description = "Opsign/tailsign/plain sign encoding conventions"
    output_dir = "data/transformation/pool/generated/ms_trans_sign.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        value = rng.randint(-99, 99)
        op = rng.choice(["+","-","*","^","#"])
        mode = rng.choice(["opsign", "tailsign", "plain"])
        if mode == "opsign":
            encoded = f"{op}{abs(value)}" if value < 0 else str(abs(value))
        elif mode == "tailsign":
            encoded = f"{abs(value)}{op}" if value < 0 else str(abs(value))
        else:
            encoded = str(value)
        think = f"{mode}: {value} -> {encoded}"
        return {"user": f"Encode {value} in {mode} (op='{op}')", "think": think, "answer": encoded}


@register
class TransChain(MicroSkill):
    name = "trans_chain"
    puzzle_type = "transformation"
    description = "Chained arithmetic: do X then Y"
    output_dir = "data/transformation/pool/generated/ms_trans_chain.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        a = rng.randint(10, 99); b = rng.randint(10, 99)
        chain = rng.choice(["reverse_add", "add_reverse", "mul_sub1", "mul_add1"])
        if chain == "reverse_add":
            rev_a = int(str(a)[::-1])
            result = rev_a + b
            think = f"Step 1: reverse {a} -> {rev_a}\nStep 2: {rev_a} + {b} = {result}"
        elif chain == "add_reverse":
            s = a + b; result = int(str(s)[::-1])
            think = f"Step 1: {a} + {b} = {s}\nStep 2: reverse {s} -> {result}"
        elif chain == "mul_sub1":
            p = a * b; result = p - 1
            think = f"Step 1: {a} * {b} = {p}\nStep 2: {p} - 1 = {result}"
        else:
            p = a * b; result = p + 1
            think = f"Step 1: {a} * {b} = {p}\nStep 2: {p} + 1 = {result}"
        return {"user": f"Compute: {chain.replace('_',' ')} with a={a}, b={b}", "think": think, "answer": str(result)}


@register
class TransReverse(MicroSkill):
    name = "trans_reverse"
    puzzle_type = "transformation"
    description = "Is string X the reverse of string Y?"
    output_dir = "data/transformation/pool/generated/ms_trans_reverse.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        word = rng.choice(vocab) if vocab else "castle"
        rev = word[::-1]
        is_rev = rng.random() < 0.5
        if is_rev:
            test = rev
        else:
            test = list(rev)
            if len(test) > 2:
                pos = rng.randint(0, len(test)-1)
                test[pos] = rng.choice("abcdefghijklmnopqrstuvwxyz")
            test = "".join(test)
        correct = test == rev
        think = f'Reverse "{word}": {rev}\n"{test}" {"matches" if correct else "does not match"}'
        return {
            "user": f'Is "{test}" the reverse of "{word}"?',
            "think": think,
            "answer": "Yes" if correct else "No",
        }


# ============================================================
# GRAV/UNIT SKILLS
# ============================================================

@register
class ArithRound(MicroSkill):
    name = "arith_round"
    puzzle_type = "gravitational"
    description = "Divide and round using 2r rule"
    output_dir = "data/gravitational/pool/generated/ms_arith_round.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        num = rng.randint(1000, 99999)
        div = rng.randint(100, 9999)
        q, r = divmod(num, div)
        up = 2 * r >= div
        result = q + 1 if up else q
        cmp = ">=" if up else "<"
        act = f"round up to {q+1}" if up else f"keep {q}"
        think = f"{num} / {div} = {q} remainder {r}\n2*{r} = {2*r} {cmp} {div} -> {act}"
        return {"user": f"Divide {num} by {div}, round with 2r rule", "think": think, "answer": str(result)}


@register
class ArithHundredths(MicroSkill):
    name = "arith_hundredths"
    puzzle_type = "unit_conversion"
    description = "Convert 3 values between hundredths and decimal (batched)"
    output_dir = "data/unit_conversion/pool/generated/ms_arith_hundredths.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        think_lines = []
        answers = []
        for _ in range(3):
            h = rng.randint(100, 9999)
            d = f"{h/100:.2f}"
            if rng.random() < 0.5:
                think_lines.append(f"  {h} hundredths = {h}/100 = {d}")
                answers.append(d)
            else:
                think_lines.append(f"  {d} = {d} × 100 = {h} hundredths")
                answers.append(str(h))
        return {
            "user": "Convert these between hundredths and decimal",
            "think": "\n".join(think_lines),
            "answer": ", ".join(answers),
        }


@register
class ArithLongMultiply(MicroSkill):
    name = "arith_long_multiply"
    puzzle_type = "gravitational"
    description = "Multiply two numbers step by step"
    output_dir = "data/gravitational/pool/generated/ms_arith_multiply.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        if difficulty == "easy":
            a, b = rng.randint(10, 99), rng.randint(2, 9)
        elif difficulty == "hard":
            a, b = rng.randint(1000, 9999), rng.randint(100, 999)
        else:
            a, b = rng.randint(100, 999), rng.randint(10, 99)
        result = a * b
        think = f"{a} * {b} = {result}"
        return {"user": f"Compute {a} * {b}", "think": think, "answer": str(result)}


@register
class ArithLongDivide(MicroSkill):
    name = "arith_long_divide"
    puzzle_type = "gravitational"
    description = "Divide with quotient and remainder"
    output_dir = "data/gravitational/pool/generated/ms_arith_divide.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        if difficulty == "easy":
            div = rng.randint(10, 99); q = rng.randint(10, 99)
        elif difficulty == "hard":
            div = rng.randint(1000, 9999); q = rng.randint(100, 999)
        else:
            div = rng.randint(100, 999); q = rng.randint(10, 99)
        r = rng.randint(0, div - 1)
        num = q * div + r
        think = f"{num} / {div} = {q} remainder {r}\nCheck: {q} * {div} + {r} = {q*div} + {r} = {num} → MATCH"
        return {"user": f"Divide {num} by {div}", "think": think, "answer": f"{q} remainder {r}"}


@register
class NumconvBase(MicroSkill):
    name = "numconv_base"
    puzzle_type = "number_conversion"
    description = "Convert between number bases"
    output_dir = "data/number_conversion/pool/generated/ms_numconv_base.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        from_base = rng.choice([8, 10, 16])
        to_base = rng.choice([b for b in [8, 10, 16] if b != from_base])
        value = rng.randint(10, 255)

        def to_str(v, base):
            if base == 10: return str(v)
            if base == 16: return hex(v)[2:].upper()
            if base == 8: return oct(v)[2:]
            digits = []
            n = v
            while n > 0: digits.append(str(n % base)); n //= base
            return "".join(reversed(digits)) or "0"

        from_str = to_str(value, from_base)
        to_str_r = to_str(value, to_base)
        think = f"{from_str} (base {from_base}) = {value} (base 10) = {to_str_r} (base {to_base})"
        return {
            "user": f"Convert {from_str} from base {from_base} to base {to_base}",
            "think": think,
            "answer": to_str_r,
        }


@register
class GeneralStringCompare(MicroSkill):
    name = "general_string_diff"
    puzzle_type = "bit_manipulation"
    description = "Compare two 8-bit strings, count matching positions"
    output_dir = "data/bit_manipulation/pool/generated/ms_general_diff.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        a = format(rng.randint(0, 255), "08b")
        b = format(rng.randint(0, 255), "08b")
        matches = sum(1 for i in range(8) if a[i] == b[i])
        think = f"  {' '.join(a)}\n  {' '.join(b)}\n  {''.join('→ MATCH' if a[i]==b[i] else '→ MISMATCH' for i in range(8))}\n  {matches}/8 match"
        return {
            "user": f"Compare {a} and {b} position by position. How many match?",
            "think": think,
            "answer": f"{matches}/8",
        }


# ============================================================
# REMAINING BIT SKILLS
# ============================================================

@register
class BitWhichOp(MicroSkill):
    name = "bit_which_op"
    puzzle_type = "bit_manipulation"
    description = "Given input->output, identify which single shift/rotate was applied"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_which_op.jsonl"
    # Sample space: 28 ops x 256 inputs, 3 options shown

    def generate_one(self, rng, difficulty="medium"):
        ops = list(SHIFT_OPS.keys())
        correct = rng.choice(ops)
        x = rng.randint(0, 255)
        bits = format(x, "08b")
        result = format(SHIFT_OPS[correct](x), "08b")
        # Pick 2 wrong alternatives
        wrong_candidates = [o for o in ops if SHIFT_OPS[o](x) != SHIFT_OPS[correct](x)]
        if len(wrong_candidates) < 2:
            return None
        wrongs = rng.sample(wrong_candidates[:5], min(2, len(wrong_candidates[:5])))
        options = [(correct, result)] + [(w, format(SHIFT_OPS[w](x), "08b")) for w in wrongs]
        rng.shuffle(options)
        labels = ["A", "B", "C"]
        opt_str = "\n".join(f"{labels[j]}) {name}" for j, (name, _) in enumerate(options))
        think_lines = []
        correct_label = "?"
        for j, (name, out) in enumerate(options):
            r = format(SHIFT_OPS[name](x), "08b")
            mark = "→ MATCH" if r == result else "→ MISMATCH"
            think_lines.append(f"{labels[j]}) {name}({bits}) = {r} {'= ' + result + ' → MATCH' if mark == '→ MATCH' else '≠ ' + result + ' → MISMATCH'}")
            if mark == "→ MATCH":
                correct_label = labels[j]
        return {
            "user": f"x={bits}, output={result}. Which operation?\n{opt_str}",
            "think": "\n".join(think_lines),
            "answer": correct_label,
        }


@register
class BitCounterfactual(MicroSkill):
    name = "bit_counterfactual"
    puzzle_type = "bit_manipulation"
    description = "If op(x) = Y, which input CANNOT be x?"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_counterfactual.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        op = rng.choice(list(SHIFT_OPS.keys()))
        x = rng.randint(0, 255)
        result = SHIFT_OPS[op](x)
        result_bits = format(result, "08b")
        # Find 1 valid input and 1 invalid
        valid = format(x, "08b")
        invalid_x = rng.randint(0, 255)
        while SHIFT_OPS[op](invalid_x) == result:
            invalid_x = rng.randint(0, 255)
        invalid = format(invalid_x, "08b")
        options = [(valid, True), (invalid, False)]
        rng.shuffle(options)
        labels = ["A", "B"]
        opt_str = "\n".join(f"{labels[j]}) {v}" for j, (v, _) in enumerate(options))
        think_lines = []
        answer = "?"
        for j, (v, ok) in enumerate(options):
            out = format(SHIFT_OPS[op](int(v, 2)), "08b")
            if ok:
                think_lines.append(f"{labels[j]}) {op}({v}) = {out} = {result_bits} → MATCH")
            else:
                think_lines.append(f"{labels[j]}) {op}({v}) = {out} ≠ {result_bits} → MISMATCH CANNOT")
                answer = labels[j]
        return {
            "user": f"If {op}(x) = {result_bits}, which CANNOT be x?\n{opt_str}",
            "think": "\n".join(think_lines),
            "answer": answer,
        }


@register
class BitProperties(MicroSkill):
    name = "bit_properties"
    puzzle_type = "bit_manipulation"
    description = "True/false about operations — verified with a random concrete example"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_properties.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        from generators.microskill_framework import rol
        x = rng.randint(0, 255)
        x_bits = format(x, "08b")
        k = rng.randint(1, 6)

        # Generate a concrete property claim with a specific input
        props = [
            (f"shr{k}({x_bits}) has first {k} bits all 0",
             all(c == '0' for c in format((x >> k) & 0xFF, '08b')[:k]),
             f"shr{k}({x_bits}) = {format((x >> k) & 0xFF, '08b')} — first {k} bits: {format((x >> k) & 0xFF, '08b')[:k]}"),
            (f"shl{k}({x_bits}) has last {k} bits all 0",
             all(c == '0' for c in format((x << k) & 0xFF, '08b')[-k:]),
             f"shl{k}({x_bits}) = {format((x << k) & 0xFF, '08b')} — last {k} bits: {format((x << k) & 0xFF, '08b')[-k:]}"),
            (f"rol{k}({x_bits}) has same number of 1s as {x_bits}",
             bin(rol(x, k)).count('1') == bin(x).count('1'),
             f"rol{k}({x_bits}) = {format(rol(x, k), '08b')} — {bin(x).count('1')} ones vs {bin(rol(x, k)).count('1')} ones"),
            (f"shr{k}({x_bits}) has same number of 1s as {x_bits}",
             bin((x >> k) & 0xFF).count('1') == bin(x).count('1'),
             f"shr{k}({x_bits}) = {format((x >> k) & 0xFF, '08b')} — {bin(x).count('1')} ones vs {bin((x >> k) & 0xFF).count('1')} ones"),
            (f"{x_bits} XOR {x_bits} = 00000000",
             True,
             f"{x_bits} XOR {x_bits} = {format(x ^ x, '08b')} — identical inputs always give 0"),
            (f"{x_bits} AND {format(0, '08b')} = 00000000",
             True,
             f"{x_bits} AND 00000000 = {format(x & 0, '08b')} — AND with zero is always zero"),
            (f"NOT({x_bits}) = {format((~x) & 0xFF, '08b')}",
             True,
             f"NOT({x_bits}) = {format((~x) & 0xFF, '08b')} — flip every bit"),
        ]

        prop_text, correct, explanation = rng.choice(props)

        # Sometimes make a FALSE version
        if rng.random() < 0.3 and correct:
            # Corrupt the claim
            prop_text = prop_text.replace("same number", "different number") if "same number" in prop_text else prop_text.replace("all 0", "all 1")
            correct = False
            explanation += " — CLAIM IS FALSE"

        think = f'{prop_text}\nCheck: {explanation}\n{"True" if correct else "False"}'
        return {
            "user": f"True or false: {prop_text}",
            "think": think,
            "answer": "True" if correct else "False",
        }


@register
class BitStepByStep(MicroSkill):
    name = "bit_step_by_step"
    puzzle_type = "bit_manipulation"
    description = "Detailed step-by-step single operation with explanation"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_stepbystep.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        op = rng.choice(list(SHIFT_OPS.keys()))
        x = rng.randint(0, 255)
        bits = format(x, "08b")
        result = format(SHIFT_OPS[op](x), "08b")
        desc = shift_str(op, bits)
        import re
        m = re.match(r'(shl|shr|rol|ror)(\d+)', op)
        kind, k = m.group(1), int(m.group(2))
        if kind == "shr":
            detail = f"Input: {bits}\nPrepend {k} zeros: {'0'*k}{bits}\nDrop last {k}: {result}\nResult: {result}"
        elif kind == "shl":
            detail = f"Input: {bits}\nDrop first {k}: {bits[k:]}\nAppend {k} zeros: {bits[k:]}{'0'*k}\nResult: {result}"
        elif kind == "rol":
            detail = f"Input: {bits}\nFirst {k} bits: {bits[:k]}\nRemaining: {bits[k:]}\nJoin remaining + first: {bits[k:]}{bits[:k]}\nResult: {result}"
        else:
            detail = f"Input: {bits}\nLast {k} bits: {bits[-k:]}\nRemaining: {bits[:-k]}\nJoin last + remaining: {bits[-k:]}{bits[:-k]}\nResult: {result}"
        return {
            "user": f"Show step by step: {op}({bits})",
            "think": detail,
            "answer": result,
        }


@register
class BitReverseFind(MicroSkill):
    name = "bit_reverse_find"
    puzzle_type = "bit_manipulation"
    description = "Given operation + output, find a valid input"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_reverse.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        import re
        op = rng.choice(list(SHIFT_OPS.keys()))
        x = rng.randint(0, 255)
        result = SHIFT_OPS[op](x)
        result_bits = format(result, "08b")
        x_bits = format(x, "08b")
        m = re.match(r'(shl|shr|rol|ror)(\d+)', op)
        kind, k = m.group(1), int(m.group(2))
        if kind in ("rol", "ror"):
            inv_op = f"ror{k}" if kind == "rol" else f"rol{k}"
            think = f"Inverse of {op} is {inv_op}\n{inv_op}({result_bits}) = {x_bits}\nCheck: {op}({x_bits}) = {result_bits} → MATCH"
        else:
            think = f"{op}(x) = {result_bits}\nOne valid x: {x_bits}\nCheck: {op}({x_bits}) = {result_bits} → MATCH"
        return {
            "user": f"If {op}(x) = {result_bits}, find x",
            "think": think,
            "answer": x_bits,
        }


@register
class BitTwoStepId(MicroSkill):
    name = "bit_two_step_id"
    puzzle_type = "bit_manipulation"
    description = "Identify which shift was used in gate(x, shift(x)) = result"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_twostep.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        gate = rng.choice(["A ^ B", "A & B", "A | B"])
        gate_fn = GATE_OPS[gate]
        ops = list(SHIFT_OPS.keys())
        correct = rng.choice(ops)
        x = rng.randint(1, 254)
        x_bits = format(x, "08b")
        shifted = SHIFT_OPS[correct](x)
        result = gate_fn(x, shifted)
        result_bits = format(result, "08b")
        wrong = rng.choice([o for o in ops if o != correct])
        options = [correct, wrong]; rng.shuffle(options)
        labels = ["A", "B"]
        think_lines = []
        answer = "?"
        for j, op in enumerate(options):
            s = SHIFT_OPS[op](x)
            r = format(gate_fn(x, s), "08b")
            mark = "→ MATCH" if r == result_bits else "→ MISMATCH"
            think_lines.append(f"{labels[j]}) {gate} with {op}: {r} {'= '+result_bits+' → MATCH' if mark == '→ MATCH' else '≠ '+result_bits+' → MISMATCH'}")
            if mark == "→ MATCH": answer = labels[j]
        return {
            "user": f"x={x_bits}, {gate}(x, shift(x)) = {result_bits}. Which shift?\nA) {options[0]}  B) {options[1]}",
            "think": "\n".join(think_lines),
            "answer": answer,
        }


@register
class BitPopcount(MicroSkill):
    name = "bit_popcount"
    puzzle_type = "bit_manipulation"
    description = "Count 1-bits and check popcount preservation after operations"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_popcount.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        x = rng.randint(0, 255)
        bits = format(x, "08b")
        pc = bin(x).count('1')
        qtype = rng.choice(["count", "compare"])
        if qtype == "count":
            think = f"{bits}: {' '.join(bits)}\nCount 1s: {pc}"
            return {"user": f"How many 1-bits in {bits}?", "think": think, "answer": str(pc)}
        else:
            op = rng.choice(list(SHIFT_OPS.keys()))
            result = SHIFT_OPS[op](x)
            r_bits = format(result, "08b")
            r_pc = bin(result).count('1')
            preserved = pc == r_pc
            import re
            kind = re.match(r'(shl|shr|rol|ror)', op).group(1)
            note = "Rotations always preserve the number of ones" if kind in ("rol","ror") else "Shifts can lose 1-bits"
            think = f"{bits} has {pc} ones\n{op}({bits}) = {r_bits} has {r_pc} ones\n{'Preserved' if preserved else f'Changed: {pc} -> {r_pc}'}\n{note}"
            return {"user": f"{bits} has {pc} ones. After {op}, how many?", "think": think, "answer": str(r_pc)}


@register
class BitNojump(MicroSkill):
    name = "bit_nojump"
    puzzle_type = "bit_manipulation"
    description = "Full mini-puzzle: find rule and apply with step-by-step execution"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_nojump.jsonl"
    # This is a BRIDGE skill — Wonderland prompt format + step-by-step execution

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        gate_names = list(GATE_OPS.keys())
        sa, sb = rng.sample(src_names, 2)
        gate = rng.choice(gate_names)

        def compute(x):
            return GATE_OPS[gate](SHIFT_OPS[sa](x), SHIFT_OPS[sb](x))

        n_ex = rng.randint(4, 6)
        inputs = rng.sample(range(256), n_ex + 1)
        query = inputs[-1]
        examples = [(format(x, "08b"), format(compute(x), "08b")) for x in inputs[:-1]]
        q_bits = format(query, "08b")
        answer = format(compute(query), "08b")

        ex_str = "\n".join(f"  {inp} -> {out}" for inp, out in examples)

        # No-jump trace: show rule + step-by-step check + query
        a = SHIFT_OPS[sa](int(examples[0][0], 2))
        b = SHIFT_OPS[sb](int(examples[0][0], 2))
        check_result, check_lines = gate_position_by_position(format(a,"08b"), format(b,"08b"), gate)

        qa = SHIFT_OPS[sa](query)
        qb = SHIFT_OPS[sb](query)
        q_result, q_lines = gate_position_by_position(format(qa,"08b"), format(qb,"08b"), gate)

        think = f"Rule: A={sa}, B={sb}, {gate}\n\n"
        think += f"Check: x={examples[0][0]}\n"
        think += f"  A = {shift_str(sa, examples[0][0])}\n"
        think += f"  B = {shift_str(sb, examples[0][0])}\n"
        think += f"  {gate}:\n" + "\n".join("  " + l for l in check_lines) + "\n"
        think += f"  = {check_result} vs {examples[0][1]} {'→ MATCH' if check_result == examples[0][1] else '→ MISMATCH'}\n\n"
        think += f"Query: x={q_bits}\n"
        think += f"  A = {shift_str(sa, q_bits)}\n"
        think += f"  B = {shift_str(sb, q_bits)}\n"
        think += f"  {gate}:\n" + "\n".join("  " + l for l in q_lines) + "\n"
        think += f"  output = {answer}"

        user = f"Bit manipulation puzzle:\n{ex_str}\nQuery: {q_bits}"

        return {"user": user, "think": think, "answer": answer}


# ============================================================
# REMAINING ENCRYPTION SKILLS
# ============================================================

@register
class EncPatternFill(MicroSkill):
    name = "enc_pattern_fill"
    puzzle_type = "encryption"
    description = "Fill pattern with bijection constraints, list valid words"
    output_dir = "data/encryption/pool/generated/ms_enc_pattern.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None
        from collections import defaultdict
        by_len = defaultdict(list)
        for w in vocab: by_len[len(w)].append(w)

        length = rng.choice([4, 5, 6, 7])
        pool = by_len.get(length, [])
        if not pool: return None
        target = rng.choice(pool)
        hide = rng.randint(1, min(2, length - 1))
        hide_pos = set(rng.sample(range(length), hide))
        pattern = "".join("_" if j in hide_pos else target[j] for j in range(length))
        matches = [w for w in pool if all(pattern[j] == "_" or pattern[j] == w[j] for j in range(length))]
        if not matches: return None
        rng.shuffle(matches)
        think = f"Pattern: {pattern}\nMatches: {', '.join(matches[:8])}\n({len(matches)} total)"
        return {
            "user": f"What words fit the pattern \"{pattern}\"?",
            "think": think,
            "answer": ", ".join(matches[:8]),
        }


@register
class EncCanFit(MicroSkill):
    name = "enc_can_fit"
    puzzle_type = "encryption"
    description = "Can this word fit given blocked letters? Yes/No with check"
    output_dir = "data/encryption/pool/generated/ms_enc_canfit.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None
        word = rng.choice([w for w in vocab if len(w) >= 4])
        n = len(word)
        hide = rng.randint(1, min(2, n - 1))
        hide_pos = set(rng.sample(range(n), hide))
        pattern = "".join("_" if j in hide_pos else word[j] for j in range(n))
        used = set(word[j] for j in range(n) if j not in hide_pos)
        needed = set(word[j] for j in hide_pos)

        can_fit = rng.random() < 0.5
        if can_fit:
            extra = set(rng.sample([c for c in "abcdefghijklmnopqrstuvwxyz" if c not in used and c not in needed], min(2, 26 - len(used) - len(needed))))
            blocked = sorted(used | extra)
            think = f'Pattern: {pattern}, blocked: {", ".join(blocked)}\n"{word}" needs: {", ".join(sorted(needed))}\nNone blocked -> fits'
        else:
            if not needed: return None
            block_letter = rng.choice(list(needed))
            blocked = sorted(used | {block_letter})
            think = f'Pattern: {pattern}, blocked: {", ".join(blocked)}\n"{word}" needs "{block_letter}" but blocked -> no'

        return {
            "user": f'Can "{word}" fit "{pattern}" if blocked: {", ".join(blocked)}?',
            "think": think,
            "answer": "Yes" if can_fit else "No",
        }


@register
class EncWhyWrong(MicroSkill):
    name = "enc_why_wrong"
    puzzle_type = "encryption"
    description = "What goes wrong if we pick word X instead of Y?"
    output_dir = "data/encryption/pool/generated/ms_enc_whywrong.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None
        word = rng.choice([w for w in vocab if len(w) >= 4])
        same_len = [w for w in vocab if len(w) == len(word) and w != word]
        if not same_len: return None
        wrong = rng.choice(same_len)
        used = set(rng.sample(list("abcdefghijklmnopqrstuvwxyz"), 10))
        conflicts = [c for c in wrong if c in used and c not in word]
        if not conflicts: return None
        conflict = conflicts[0]
        pos = wrong.index(conflict)
        think = f'"{wrong}" position {pos} needs "{conflict}"\n"{conflict}" already used\nBijection violated\n"{word}" avoids this'
        return {
            "user": f'Used letters: {", ".join(sorted(used))}. Why not "{wrong}" instead of "{word}"?',
            "think": think,
            "answer": f'"{conflict}" already used',
        }


# ============================================================
# REMAINING TRANSFORMATION SKILLS
# ============================================================

@register
class TransSymbolEdit(MicroSkill):
    name = "trans_symbol_edit"
    puzzle_type = "transformation"
    description = "Identify symbol edit operations: delete center, swap halves"
    output_dir = "data/transformation/pool/generated/ms_trans_edit.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        import string
        symbols = rng.sample(list("!@#$%^&*()[]{}|\\/<>?~`"), 5)
        inp = "".join(symbols)
        edit = rng.choice(["delete_center", "swap_halves", "none"])
        if edit == "delete_center":
            output = inp[:2] + inp[3:]
            think = f'"{inp}" -> "{output}"\nCenter char "{inp[2]}" removed\nEdit: delete_center'
            answer = "delete_center"
        elif edit == "swap_halves":
            output = inp[3:] + inp[:2]
            think = f'"{inp}" -> "{output}"\nLeft "{inp[:2]}" swapped with right "{inp[3:]}"\nEdit: swap_halves'
            answer = "swap_halves"
        else:
            output = "".join(rng.sample(list("!@#$%^&*"), 3))
            think = f'"{inp}" -> "{output}"\nOutput has different symbols/length\nNot a simple edit -> arithmetic'
            answer = "arithmetic"
        return {
            "user": f'What type of transformation? "{inp}" -> "{output}"',
            "think": think,
            "answer": answer,
        }


@register
class TransRevChain(MicroSkill):
    name = "trans_rev_chain"
    puzzle_type = "transformation"
    description = "Apply rev_input modifier then operate"
    output_dir = "data/transformation/pool/generated/ms_trans_revchain.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        op = rng.choice(["+", "-", "*"])
        modifier = rng.choice(["rev_input", "rev_output", "both"])

        if modifier == "rev_input":
            ra = int(str(a)[::-1]); rb = int(str(b)[::-1])
            if op == "+": result = ra + rb
            elif op == "-": result = ra - rb
            else: result = ra * rb
            think = f"rev_input: reverse operands first\n  {a} -> {ra}, {b} -> {rb}\n  {ra} {op} {rb} = {result}"
        elif modifier == "rev_output":
            if op == "+": raw = a + b
            elif op == "-": raw = a - b
            else: raw = a * b
            result = int(str(abs(raw))[::-1]) * (1 if raw >= 0 else -1)
            think = f"rev_output: compute then reverse\n  {a} {op} {b} = {raw}\n  reverse {raw} -> {result}"
        else:
            ra = int(str(a)[::-1]); rb = int(str(b)[::-1])
            if op == "+": raw = ra + rb
            elif op == "-": raw = ra - rb
            else: raw = ra * rb
            result = int(str(abs(raw))[::-1]) * (1 if raw >= 0 else -1)
            think = f"rev_input + rev_output:\n  reverse inputs: {a}->{ra}, {b}->{rb}\n  {ra} {op} {rb} = {raw}\n  reverse output: {raw} -> {result}"

        return {
            "user": f"Apply {modifier}: {a} {op} {b}",
            "think": think,
            "answer": str(result),
        }


# ============================================================
# HIGH-GRADIENT: IMPOSSIBILITY / ERROR SPOTTING
# ============================================================

@register
class BitImpossible(MicroSkill):
    name = "bit_impossible"
    puzzle_type = "bit_manipulation"
    description = "Can this output be produced by this operation? Spot impossible claims"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_impossible.jsonl"
    # HIGH GRADIENT: answer is ONE fact (which constraint is violated)

    def generate_one(self, rng, difficulty="medium"):
        import re
        op = rng.choice(list(SHIFT_OPS.keys()))
        m = re.match(r'(shl|shr|rol|ror)(\d+)', op)
        kind, k = m.group(1), int(m.group(2))

        possible = rng.random() < 0.5

        if possible:
            x = rng.randint(0, 255)
            result = format(SHIFT_OPS[op](x), "08b")
            think = f"{op} can produce {result}\nExample: {op}({format(x,'08b')}) = {result}"
            answer = "Yes"
        else:
            # Generate impossible result
            result = format(rng.randint(0, 255), "08b")
            if kind == "shr":
                # shr must have first k bits as 0
                if all(c == '0' for c in result[:k]):
                    # Make it impossible by setting a top bit
                    result = list(result)
                    result[rng.randint(0, k-1)] = '1'
                    result = "".join(result)
                reason = f"shr{k} output must have first {k} bits = 0, but got '{result[:k]}'"
            elif kind == "shl":
                if all(c == '0' for c in result[-k:]):
                    result = list(result)
                    result[8 - rng.randint(1, k)] = '1'
                    result = "".join(result)
                reason = f"shl{k} output must have last {k} bits = 0, but got '{result[-k:]}'"
            elif kind in ("rol", "ror"):
                # Rotations preserve popcount
                x = rng.randint(0, 255)
                real = SHIFT_OPS[op](x)
                pc_real = bin(real).count('1')
                # Pick result with different popcount
                target_pc = pc_real + rng.choice([-2, -1, 1, 2])
                target_pc = max(0, min(8, target_pc))
                if target_pc == pc_real:
                    target_pc = (pc_real + 1) % 9
                # Build result with that popcount
                bits = ['0'] * 8
                for pos in rng.sample(range(8), target_pc):
                    bits[pos] = '1'
                result = "".join(bits)
                reason = f"{op} preserves ones count. Input has {bin(x).count('1')} ones but result has {target_pc} ones"
            else:
                return None

            think = f"Can {op}(x) = {result}?\n{reason}\nImpossible."
            answer = "No"

        return {
            "user": f"Can {op}(x) produce {result} for some input x?",
            "think": think,
            "answer": answer,
        }


@register
class BitSpotError(MicroSkill):
    name = "bit_spot_error"
    puzzle_type = "bit_manipulation"
    description = "Find the EXACT position of an error in a computation"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_spot_error.jsonl"
    # HIGH GRADIENT: answer is one position number

    def generate_one(self, rng, difficulty="medium"):
        qtype = rng.choice(["shift", "gate"])

        if qtype == "shift":
            op = rng.choice(list(SHIFT_OPS.keys()))
            x = rng.randint(0, 255)
            bits = format(x, "08b")
            correct = format(SHIFT_OPS[op](x), "08b")
            pos = rng.randint(0, 7)
            wrong = list(correct)
            wrong[pos] = '1' if wrong[pos] == '0' else '0'
            wrong = "".join(wrong)
            think = f"Claimed: {op}({bits}) = {wrong}\nCorrect: {correct}\nPosition {pos}: claimed '{wrong[pos]}' but should be '{correct[pos]}'"
            return {
                "user": f"Which bit position is wrong? {op}({bits}) = {wrong}",
                "think": think,
                "answer": f"Position {pos}",
            }
        else:
            gate = rng.choice(list(GATE_OPS.keys()))
            a = rng.randint(0, 255); b = rng.randint(0, 255)
            ab = format(a, "08b"); bb = format(b, "08b")
            correct = format(GATE_OPS[gate](a, b), "08b")
            pos = rng.randint(0, 7)
            wrong = list(correct)
            wrong[pos] = '1' if wrong[pos] == '0' else '0'
            wrong = "".join(wrong)
            think = f"Claimed: {gate}({ab}, {bb}) = {wrong}\nPosition {pos}: A[{pos}]={ab[pos]}, B[{pos}]={bb[pos]}"
            if "^" in gate:
                think += f" -> {'diff' if ab[pos]!=bb[pos] else 'same'} -> should be {correct[pos]}, got {wrong[pos]}"
            elif "&" in gate and "~" not in gate:
                think += f" -> both {'1' if ab[pos]=='1' and bb[pos]=='1' else 'not 1'} -> should be {correct[pos]}"
            elif "|" in gate:
                think += f" -> either {'1' if ab[pos]=='1' or bb[pos]=='1' else '0'} -> should be {correct[pos]}"
            return {
                "user": f"Which bit is wrong? {gate}({ab}, {bb}) = {wrong}",
                "think": think,
                "answer": f"Position {pos}",
            }


@register
class EncImpossible(MicroSkill):
    name = "enc_impossible"
    puzzle_type = "encryption"
    description = "Is this mapping/word possible given constraints? Spot violations"
    output_dir = "data/encryption/pool/generated/ms_enc_impossible.jsonl"
    # HIGH GRADIENT: answer is which specific constraint is violated

    def generate_one(self, rng, difficulty="medium"):
        qtype = rng.choice(["dup_plain", "dup_cipher", "length_mismatch", "valid"])

        if qtype == "dup_plain":
            # Two cipher letters map to same plain
            c1, c2 = rng.sample("abcdefghijklmnopqrstuvwxyz", 2)
            p = rng.choice("abcdefghijklmnopqrstuvwxyz")
            think = f"{c1}->{p} and {c2}->{p}\nTwo cipher letters map to same plain '{p}'\nViolates bijection: each plain letter must map from exactly one cipher letter"
            return {
                "user": f"Is this valid? {c1}->{p}, {c2}->{p}",
                "think": think,
                "answer": f"No: '{p}' mapped from both '{c1}' and '{c2}'",
            }
        elif qtype == "dup_cipher":
            c = rng.choice("abcdefghijklmnopqrstuvwxyz")
            p1, p2 = rng.sample("abcdefghijklmnopqrstuvwxyz", 2)
            think = f"{c}->{p1} and {c}->{p2}\nSame cipher '{c}' maps to two different plain letters\nViolates: each cipher letter must map to exactly one plain letter"
            return {
                "user": f"Is this valid? {c}->{p1}, {c}->{p2}",
                "think": think,
                "answer": f"No: '{c}' maps to both '{p1}' and '{p2}'",
            }
        elif qtype == "length_mismatch":
            vocab = load_vocab()
            if not vocab: return None
            word = rng.choice([w for w in vocab if len(w) >= 4])
            wrong_len = len(word) + rng.choice([-1, 1])
            cipher = "".join(rng.sample("abcdefghijklmnopqrstuvwxyz", wrong_len))
            think = f'Cipher "{cipher}" has {wrong_len} chars, plain "{word}" has {len(word)} chars\nIn substitution cipher, lengths must match'
            return {
                "user": f'Can cipher "{cipher}" ({wrong_len} chars) decrypt to "{word}" ({len(word)} chars)?',
                "think": think,
                "answer": f"No: length mismatch ({wrong_len} vs {len(word)})",
            }
        else:
            # Valid mapping
            letters = list("abcdefghijklmnopqrstuvwxyz")
            n = rng.randint(3, 6)
            c = rng.sample(letters, n)
            p = rng.sample(letters, n)
            mapping = ", ".join(f"{ci}->{pi}" for ci, pi in zip(c, p))
            think = f"{mapping}\nAll cipher letters unique, all plain letters unique\nValid bijection"
            return {
                "user": f"Is this mapping valid? {mapping}",
                "think": think,
                "answer": "Yes: valid bijection",
            }


@register
class TransImpossible(MicroSkill):
    name = "trans_impossible"
    puzzle_type = "transformation"
    description = "Can this operation produce this result? Spot impossible claims"
    output_dir = "data/transformation/pool/generated/ms_trans_impossible.jsonl"

    def generate_one(self, rng, difficulty="medium"):
        qtype = rng.choice(["add_too_small", "mul_wrong", "sub_sign", "valid"])

        a = rng.randint(10, 50); b = rng.randint(10, 50)

        if qtype == "add_too_small":
            wrong = a + b - rng.randint(2, 10)
            think = f"{a} + {b} = {a+b}, not {wrong}\nOff by {a+b - wrong}"
            return {
                "user": f"Is {a} + {b} = {wrong} correct?",
                "think": think,
                "answer": f"No: should be {a+b}",
            }
        elif qtype == "mul_wrong":
            wrong = a * b + rng.choice([-1, 1, -a, b])
            if wrong == a * b: wrong += 1
            think = f"{a} × {b} = {a*b}, not {wrong}\nOff by {a*b - wrong}"
            return {
                "user": f"Is {a} × {b} = {wrong} correct?",
                "think": think,
                "answer": f"No: should be {a*b}",
            }
        elif qtype == "sub_sign":
            if a < b:
                wrong = a - b  # negative, but claim positive
                claim = abs(wrong)
                think = f"{a} - {b} = {a-b} (negative), not {claim}\nSign error"
                return {
                    "user": f"Is {a} - {b} = {claim} correct?",
                    "think": think,
                    "answer": f"No: should be {a-b} (negative)",
                }
            else:
                result = a - b
                think = f"{a} - {b} = {result}\nCorrect"
                return {
                    "user": f"Is {a} - {b} = {result} correct?",
                    "think": think,
                    "answer": "Yes",
                }
        else:
            op = rng.choice(["+", "-", "*"])
            if op == "+": result = a + b
            elif op == "-": result = a - b
            else: result = a * b
            think = f"{a} {op} {b} = {result}\nCorrect"
            return {
                "user": f"Is {a} {op} {b} = {result} correct?",
                "think": think,
                "answer": "Yes",
            }


# ============================================================
# HIGH-GRADIENT: RULE DISCRIMINATION (THE CORE INFERENCE TASK)
# ============================================================

@register
class BitRuleDiscriminate(MicroSkill):
    name = "bit_rule_discriminate"
    puzzle_type = "bit_manipulation"
    description = "Which if any of these 3 ops produced this output? Sometimes none did."
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_discriminate.jsonl"

    # HIGH GRADIENT: 3 candidates that produce SIMILAR outputs.
    # Sometimes one matches, sometimes NONE match. Model must check each honestly.
    # Teaches: don't force-pick when nothing fits.

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        x = rng.randint(1, 254)
        bits = format(x, "08b")

        # Pick 3 random single operations
        ops = rng.sample(src_names, 3)

        # Compute what each produces
        outputs = {op: format(SHIFT_OPS[op](x), "08b") for op in ops}

        # 60% one of them is correct, 40% none match
        include_correct = rng.random() < 0.6

        if include_correct:
            # Pick one as the "true" operation, use its output
            correct_op = rng.choice(ops)
            target = outputs[correct_op]
        else:
            # Generate an output that NONE of them produce
            # But make it CLOSE to one of them (1-2 bits off)
            base_op = rng.choice(ops)
            base_out = SHIFT_OPS[base_op](x)
            # Flip 1-2 bits
            flipped = base_out
            for _ in range(rng.randint(1, 2)):
                flipped ^= (1 << rng.randint(0, 7))
            target = format(flipped, "08b")
            # Make sure it doesn't accidentally match any
            if target in outputs.values():
                return None
            correct_op = None

        labels = ["A", "B", "C"]
        options = "\n".join(f"  {labels[i]}) {ops[i]}" for i in range(3))

        think_lines = []
        for i, op in enumerate(ops):
            result = outputs[op]
            match = result == target
            think_lines.append(f"{labels[i]}) {shift_str(op, bits)}")
            think_lines.append(f"   = {result} {'= ' + target + ' → MATCH MATCH' if match else '≠ ' + target + ' → MISMATCH'}")

        if correct_op:
            answer = labels[ops.index(correct_op)]
            think_lines.append(f"Answer: {answer}")
        else:
            answer = "None"
            think_lines.append("None of them match.")

        return {
            "user": f"x={bits}, output={target}. Which of these produced it, if any?\n{options}",
            "think": "\n".join(think_lines),
            "answer": answer,
        }


@register
class BitRuleDiscriminateMulti(MicroSkill):
    name = "bit_rule_discriminate_multi"
    puzzle_type = "bit_manipulation"
    description = "Given 3 input->output pairs, which of 3 rules fits ALL of them?"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_disc_multi.jsonl"

    # Like bit_rule_discriminate but with MULTIPLE examples — closer to real puzzles
    # The wrong rules might fit 1-2 examples but fail on the 3rd

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        gate_names = list(GATE_OPS.keys())

        # Build 3 rules: 2 sources + gate each
        rule_specs = []
        for _ in range(3):
            sa, sb = rng.sample(src_names, 2)
            gate = rng.choice(gate_names)
            fn = lambda x, sa=sa, sb=sb, gate=gate: GATE_OPS[gate](SHIFT_OPS[sa](x), SHIFT_OPS[sb](x))
            desc = f"{sa} {gate} {sb}"
            rule_specs.append((desc, fn))

        correct_idx = rng.randint(0, 2)
        correct_fn = rule_specs[correct_idx][1]

        # Generate 3 examples using correct rule
        inputs = rng.sample(range(256), 3)
        examples = [(format(x, "08b"), format(correct_fn(x), "08b")) for x in inputs]

        # Verify wrong rules don't match ALL examples
        for i in range(3):
            if i == correct_idx:
                continue
            wrong_fn = rule_specs[i][1]
            all_match = all(format(wrong_fn(int(inp, 2)), "08b") == out for inp, out in examples)
            if all_match:
                return None  # ambiguous, skip

        # Build trace
        labels = ["A", "B", "C"]
        ex_str = "\n".join(f"  {inp} -> {out}" for inp, out in examples)
        options = "\n".join(f"  {labels[i]}) {rule_specs[i][0]}" for i in range(3))

        think_lines = []
        for i in range(3):
            fn = rule_specs[i][1]
            fits = []
            for j, (inp, expected) in enumerate(examples):
                computed = format(fn(int(inp, 2)), "08b")
                fits.append("→ MATCH" if computed == expected else "→ MISMATCH")
            all_fit = all(f == "→ MATCH" for f in fits)
            think_lines.append(f"{labels[i]}) {rule_specs[i][0]}: {' '.join(fits)} {'-> ALL MATCH' if all_fit else '-> FAIL'}")

        return {
            "user": f"Which rule fits ALL examples?\nExamples:\n{ex_str}\nRules:\n{options}",
            "think": "\n".join(think_lines),
            "answer": labels[correct_idx],
        }


# ============================================================
# WEIGHT + POOL SIZE CONFIGURATION
# ============================================================
# Set once here rather than on each class. Edit this table to rebalance.

_WEIGHTS = {
    # BIT — highest priority (14% accuracy, biggest gap)
    "bit_rule_discriminate":      10.0,   # core inference — boost   # THE core inference task
    "bit_rule_discriminate_multi":10.0,   # multi-example version — boost   # Multi-example version
    "bit_rule_check":              4.0,   # Does this rule fit?
    "bit_distinguish":            10.0,   # model defaults to xor/and/or when wrong — needs discrimination   # Which example separates rules?
    "bit_nojump":                  4.0,   # Full puzzle step-by-step
    "bit_shift":                   3.0,   # Foundation: execute shifts
    "bit_gate":                    3.0,   # Foundation: execute gates
    "bit_compose2":                3.0,   # Two-step chains
    "bit_compose3":                3.0,   # Three-step chains
    "bit_full_verify":             3.0,   # Verify against all examples
    "bit_similarity":              3.0,   # Score candidate sources
    "bit_impossible":              2.0,   # Spot impossible claims
    "bit_spot_error":              2.0,   # Find exact error position
    "bit_which_op":                2.0,   # Identify single operation
    "bit_counterfactual":          2.0,   # What can't be right
    "bit_error_detect":            2.0,   # General error detection
    "bit_popcount":                1.5,   # Popcount awareness
    "bit_properties":              1.0,   # True/false about ops
    "bit_step_by_step":            1.0,   # Detailed explanation
    "bit_reverse_find":            1.0,   # Find input from output
    "bit_two_step_id":             1.0,   # ID shift in composition
    "bit_edge_shift_vs_rotate":    1.0,   # Edge case
    "bit_edge_zeros":              1.0,   # Edge case
    "bit_edge_gate":               1.0,   # Edge case
    "general_string_diff":         1.0,   # Foundation

    # ENCRYPTION — second priority (89%, want 95%)
    "enc_extract_mapping":        12.0,   # 72% of enc failures are mapping extraction — boost hard
    "enc_apply_mapping":          10.0,   # Apply table to decode — upstream of all other enc steps
    "enc_forced_mapping":          2.0,   # What's forced
    "enc_bijection":               2.0,   # Is it valid
    "enc_most_constrained":        2.0,   # CSP ordering
    "enc_can_fit":                 2.0,   # Constraint checking
    "enc_impossible":              2.0,   # Spot violations
    "enc_not_forced":              2.0,   # Counterfactual
    "enc_why_wrong":               2.0,   # Explain violations
    "enc_propagation":             1.5,   # Chain effects
    "enc_pattern_fill":            1.5,   # Vocabulary + pattern
    "enc_vocab":                   1.5,   # Know the word list
    "enc_repeated_letters":        1.0,   # Repeat constraints
    "enc_reverse_decrypt":         1.0,   # Reverse direction
    "str_count":                   1.0,   # Foundation
    "str_compare":                 1.0,   # Foundation

    # TRANSFORMATION — third priority
    "trans_op_from_examples":      2.0,   # Core inference task
    "trans_base":                  1.5,   # Base encoding
    "trans_sign":                  1.5,   # Sign conventions
    "trans_parse":                 1.0,   # Parse equations
    "trans_chain":                 1.0,   # Chained ops
    "trans_reverse":               1.0,   # Reversal
    "trans_symbol_edit":           1.0,   # Edit detection
    "trans_rev_chain":             1.0,   # Modifier chains
    "trans_impossible":            1.0,   # Spot impossible

    # GRAV/UNIT/NUMCONV — maintenance
    "arith_round":                 0.5,
    "arith_long_multiply":         0.5,
    "arith_long_divide":           0.5,
    "arith_hundredths":            0.5,
    "numconv_base":                0.5,
}

_MAX_POOLS = {
    # Skills with huge sample spaces get big pools
    "bit_shift": 10000, "bit_gate": 10000, "bit_compose2": 10000,
    "bit_compose3": 10000, "bit_rule_check": 10000, "bit_nojump": 10000,
    "bit_rule_discriminate": 10000, "bit_rule_discriminate_multi": 10000,
    "bit_full_verify": 10000, "bit_similarity": 10000,
    "bit_distinguish": 5000, "bit_impossible": 5000,
    "enc_extract_mapping": 5000, "enc_apply_mapping": 5000,
    "enc_forced_mapping": 5000, "enc_propagation": 5000,
    "enc_most_constrained": 5000, "enc_bijection": 5000,
    "trans_op_from_examples": 5000, "trans_base": 5000,
    # Smaller sample spaces
    "enc_vocab": 2000, "enc_repeated_letters": 2000,
}

# Apply weights and pool sizes
from generators.microskill_framework import REGISTRY as _REG
for _name, _weight in _WEIGHTS.items():
    if _name in _REG:
        _REG[_name].weight = _weight
for _name, _pool in _MAX_POOLS.items():
    if _name in _REG:
        _REG[_name].max_pool = _pool


@register
class EncVocabAudit(MicroSkill):
    name = "enc_vocab_audit"
    puzzle_type = "encryption"
    description = "Student recited word list — find impostors and missing words"
    output_dir = "data/encryption/pool/generated/ms_enc_vocab_audit.jsonl"
    weight = 3.0    # high — forces exact vocabulary knowledge
    max_pool = 5000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None
        from collections import defaultdict
        by_len = defaultdict(list)
        for w in sorted(vocab): by_len[len(w)].append(w)

        # Pick 1-2 length groups and combine for more variety
        available_lens = [l for l in by_len if len(by_len[l]) >= 3]
        n_lens = rng.randint(1, 2)
        chosen_lens = rng.sample(available_lens, min(n_lens, len(available_lens)))
        length = chosen_lens[0]  # primary for labeling
        real_words = []
        for l in chosen_lens:
            real_words.extend(by_len[l])
        real_words = sorted(real_words)

        # Build student's attempt: some real words + some impostors + some missing
        n_show = min(len(real_words), rng.randint(3, min(12, len(real_words))))
        shown_real = sorted(rng.sample(real_words, n_show))

        # Add 1-3 impostors (common English words NOT in Alice's list)
        impostor_pool = [
            # 3 letters
            "dog", "run", "big", "hot", "red", "old", "new", "try", "ask", "say",
            # 4 letters
            "phone", "fish", "bike", "cake", "home", "park", "wall", "milk", "sand",
            "desk", "lamp", "coat", "shoe", "wine", "tree", "moon", "rain", "salt",
            "bone", "soup", "frog", "duck", "ball", "kite", "rope", "bell", "drum",
            # 5 letters
            "email", "hello", "world", "pizza", "happy", "think", "never", "maybe",
            "water", "house", "light", "music", "green", "quick", "brown", "jumps",
            "brain", "cloud", "storm", "dance", "ocean", "stone", "flame", "night",
            "stars", "bread", "chair", "table", "clock", "paper", "plant", "glass",
            "truck", "beach", "swing", "candy", "float", "spoon", "grape", "mango",
            # 6 letters
            "planet", "rocket", "coffee", "window", "orange", "purple", "yellow",
            "flower", "butter", "cookie", "frozen", "bottle", "engine", "dinner",
            "simple", "gentle", "modern", "pocket", "pirate", "zombie", "noodle",
            "parrot", "pencil", "silver", "turtle", "rabbit", "kitten", "banana",
            # 7 letters
            "morning", "kitchen", "blanket", "captain", "chicken", "dolphin",
            "diamond", "factory", "giraffe", "holiday", "imagine", "justice",
            "kingdom", "lantern", "machine", "network", "ostrich", "penguin",
            "rainbow", "station", "thunder", "unicorn", "vampire", "whistle",
            # 8 letters
            "airplane", "birthday", "campfire", "darkness", "elephant", "favorite",
            "goldfish", "handsome", "icecream", "jokester", "keyboard", "lemonade",
            "mushroom", "notebook", "omission", "platypus", "question", "sandwich",
        ]
        impostors = [w for w in impostor_pool if len(w) == length and w not in real_words]
        n_impostors = min(rng.randint(1, 2), len(impostors))
        chosen_impostors = sorted(rng.sample(impostors, n_impostors)) if impostors else []

        # Find missing words
        missing = sorted(set(real_words) - set(shown_real))
        n_missing_to_show = min(3, len(missing))
        missing_sample = missing[:n_missing_to_show]

        # Build student's list (real + impostors, sorted)
        student_list = sorted(shown_real + chosen_impostors)

        student_str = ", ".join(student_list)
        len_label = "/".join(str(l) for l in sorted(chosen_lens))
        think_lines = [f"Student's {len_label}-letter words: {student_str}"]
        think_lines.append(f"\nImpostors (not in Alice's vocabulary):")
        if chosen_impostors:
            for w in chosen_impostors:
                think_lines.append(f"  '{w}' — NOT an Alice word")
        else:
            think_lines.append("  (none found)")
        think_lines.append(f"\nMissing from student's list:")
        if missing_sample:
            for w in missing_sample:
                think_lines.append(f"  '{w}' — Alice word, student forgot it")
        else:
            think_lines.append("  (none missing)")

        imp_str = ", ".join(chosen_impostors) if chosen_impostors else "none"
        miss_str = ", ".join(missing_sample) if missing_sample else "none"

        return {
            "user": f"A student tried to recite Alice's {length}-letter words:\n  {student_str}\nWhich words don't belong? Which are missing?",
            "think": "\n".join(think_lines),
            "answer": f"Impostors: {imp_str}. Missing: {miss_str}",
        }


# ============================================================
# BIT: IDENTIFICATION TRAINING (teaches internal pattern recognition)
# ============================================================

@register
class BitFamilyFromPopcount(MicroSkill):
    name = "bit_family_from_popcount"
    puzzle_type = "bit_manipulation"
    description = "Given popcount stats across examples, identify the boolean family"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_family_pop.jsonl"
    weight = 5.0
    max_pool = 10000

    FAMILIES = {
        "XOR": lambda a, b: (a ^ b) & 0xFF,
        "AND": lambda a, b: (a & b) & 0xFF,
        "OR": lambda a, b: (a | b) & 0xFF,
        "OR_XNOR": lambda a, b: (b | (~(a ^ b) & 0xFF)) & 0xFF,  # C | xnor(A,B)
        "AND_NOT": lambda a, b: (a & (~b & 0xFF)) & 0xFF,
    }

    SIGNATURES = {
        "XOR": "output ones-count fluctuates around input ones-count, no consistent increase",
        "AND": "output ones-count always <= both inputs",
        "OR": "output ones-count always >= both inputs",
        "OR_XNOR": "output ones-count is high — often 6-8 even for moderate inputs",
        "AND_NOT": "output ones-count <= input A, often much lower",
    }

    def generate_one(self, rng, difficulty="medium"):
        family = rng.choice(list(self.FAMILIES.keys()))
        gate_fn = self.FAMILIES[family]

        # Generate examples with two random shifts
        src_names = list(SHIFT_OPS.keys())
        sa, sb = rng.sample(src_names, 2)

        examples = []
        for _ in range(5):
            x = rng.randint(0, 255)
            a = SHIFT_OPS[sa](x)
            b = SHIFT_OPS[sb](x)
            out = gate_fn(a, b)
            examples.append((format(x, "08b"), format(out, "08b")))

        in_pops = [bin(int(i, 2)).count('1') for i, _ in examples]
        out_pops = [bin(int(o, 2)).count('1') for _, o in examples]

        ex_str = "\n".join(f"  {i} -> {o}  (ones: {ip} in, {op} out)" 
                          for (i, o), ip, op in zip(examples, in_pops, out_pops))

        avg_in = sum(in_pops) / len(in_pops)
        avg_out = sum(out_pops) / len(out_pops)
        always_ge = all(op >= ip for ip, op in zip(in_pops, out_pops))
        always_le = all(op <= ip for ip, op in zip(in_pops, out_pops))

        # Build options — correct + 2 wrong
        options = [family]
        wrongs = [f for f in self.FAMILIES if f != family]
        options.extend(rng.sample(wrongs, 2))
        rng.shuffle(options)
        labels = ["A", "B", "C"]
        correct_label = labels[options.index(family)]
        opt_str = "\n".join(f"  {labels[j]}) {options[j]}" for j in range(3))

        think_lines = [f"Ones-count analysis:"]
        think_lines.append(f"  Avg input: {avg_in:.1f}, Avg output: {avg_out:.1f}")
        if always_ge:
            think_lines.append(f"  Output ones-count always >= input -> OR-type")
        elif always_le:
            think_lines.append(f"  Output ones-count always <= input -> AND-type")
        else:
            think_lines.append(f"  Output ones-count varies relative to input")
        think_lines.append(f"  {self.SIGNATURES[family]}")
        think_lines.append(f"  -> {family}")

        return {
            "user": f"What boolean family produced these?\n{ex_str}\n{opt_str}",
            "think": "\n".join(think_lines),
            "answer": correct_label,
        }


@register
class BitEliminateFamily(MicroSkill):
    name = "bit_eliminate_family"
    puzzle_type = "bit_manipulation"
    description = "Given ACTUAL examples, make an observation and eliminate families"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_eliminate.jsonl"
    weight = 4.0
    max_pool = 10000

    FAMILIES = {
        "XOR": lambda a, b: (a ^ b) & 0xFF,
        "AND": lambda a, b: (a & b) & 0xFF,
        "OR": lambda a, b: (a | b) & 0xFF,
        "OR_XNOR": lambda a, b: (b | (~(a ^ b) & 0xFF)) & 0xFF,
        "AND_NOT": lambda a, b: (a & (~b & 0xFF)) & 0xFF,
    }

    def generate_one(self, rng, difficulty="medium"):
        # Generate REAL examples from a random rule
        src_names = list(SHIFT_OPS.keys())
        sa, sb = rng.sample(src_names, 2)
        family = rng.choice(list(self.FAMILIES.keys()))
        gate_fn = self.FAMILIES[family]

        examples = []
        for _ in range(5):
            x = rng.randint(0, 255)
            a = SHIFT_OPS[sa](x); b = SHIFT_OPS[sb](x)
            out = gate_fn(a, b)
            examples.append((format(x, "08b"), format(out, "08b")))

        in_ones = [bin(int(i,2)).count('1') for i,_ in examples]
        out_ones = [bin(int(o,2)).count('1') for _,o in examples]
        avg_out = sum(out_ones) / len(out_ones)

        # Find a TRUE observation about these examples
        observations = []
        if all(oo > io for io, oo in zip(in_ones, out_ones)):
            observations.append(("Output has MORE ones than input in every example",
                                ["AND", "XOR", "AND_NOT"], ["OR", "OR_XNOR"],
                                "AND/XOR/AND_NOT can only keep or reduce 1-bits"))
        if all(oo < io for io, oo in zip(in_ones, out_ones)):
            observations.append(("Output has FEWER ones than input in every example",
                                ["OR", "OR_XNOR"], ["AND", "AND_NOT", "XOR"],
                                "OR/OR_XNOR can only keep or add 1-bits"))
        if avg_out >= 6:
            observations.append((f"Output is dense (avg {avg_out:.1f} ones)",
                                ["AND", "AND_NOT"], ["OR", "OR_XNOR", "XOR"],
                                "AND produces sparse outputs"))
        if avg_out <= 2:
            observations.append((f"Output is sparse (avg {avg_out:.1f} ones)",
                                ["OR", "OR_XNOR"], ["AND", "AND_NOT", "XOR"],
                                "OR produces dense outputs"))
        if all(io == oo for io, oo in zip(in_ones, out_ones)):
            observations.append(("Ones count preserved in every example",
                                ["AND", "OR", "OR_XNOR"], ["XOR"],
                                "Only XOR-type preserves ones count"))
        # Position-specific
        for pos in range(8):
            if all(examples[j][1][7-pos] == '0' for j in range(5)):
                observations.append((f"Output bit {pos} is always 0",
                                    [], [],
                                    f"Position {pos} is forced to 0 by the rule"))
                break
        for pos in range(8):
            if all(examples[j][1][7-pos] == '1' for j in range(5)):
                observations.append((f"Output bit {pos} is always 1",
                                    [], [],
                                    f"Position {pos} is forced to 1 by the rule"))
                break

        if not observations:
            return None

        obs_text, eliminates, keeps, reason = rng.choice(observations)
        ex_str = "\n".join(f"  {i} -> {o}  ({bin(int(i,2)).count('1')} ones -> {bin(int(o,2)).count('1')} ones)" for i, o in examples)

        think = f"Examples:\n{ex_str}\n\n"
        think += f'Observation: "{obs_text}"\n'
        if eliminates:
            think += f'Eliminates: {", ".join(eliminates)}\n'
            think += f'Reason: {reason}\n'
            think += f'Remaining: {", ".join(keeps)}'
        else:
            think += f'Note: {reason}'

        answer = ", ".join(eliminates) if eliminates else obs_text
        return {
            "user": f"What can you observe about these examples? What families does it eliminate?\n{ex_str}",
            "think": think,
            "answer": answer,
        }


@register
class BitCorrelatePositions(MicroSkill):
    name = "bit_correlate_positions"
    puzzle_type = "bit_manipulation"
    description = "When input[i]=1, output[j] is always 1. What shift does that suggest?"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_correlate.jsonl"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        # Pick a real shift and demonstrate the correlation
        op = rng.choice(list(SHIFT_OPS.keys()))

        # Generate examples and find a strong bit correlation
        examples = [(rng.randint(0, 255),) for _ in range(6)]
        examples = [(x, SHIFT_OPS[op](x)) for (x,) in examples]

        # Find a position pair (i, j) where input[i]=1 → output[j]=1
        # For a pure shift by k, input[i] → output[i-k] (or wrapping for rotation)
        import re
        m = re.match(r'(shl|shr|rol|ror)(\d+)', op)
        kind, k = m.group(1), int(m.group(2))

        # Pick an input bit position
        i = rng.randint(0, 7)

        if kind == "shr":
            j = i + k if i + k < 8 else None
        elif kind == "shl":
            j = i - k if i - k >= 0 else None
        elif kind == "rol":
            j = (i - k) % 8
        elif kind == "ror":
            j = (i + k) % 8

        if j is None:
            return None

        # Verify the correlation on examples
        # For shift: input[i] directly becomes output[j]
        # But with a gate, it's more complex. Keep it simple — teach pure shift correlation first.
        correlations = []
        for x, out in examples:
            in_bit = (x >> (7 - i)) & 1
            out_bit = (out >> (7 - j)) & 1
            correlations.append((in_bit, out_bit))

        # Check if input[i]=1 always means output[j]=1
        ones_in = [(ib, ob) for ib, ob in correlations if ib == 1]
        if not ones_in:
            return None
        always_one = all(ob == 1 for _, ob in ones_in)

        ex_str = "\n".join(f"  {format(x,'08b')} -> {format(o,'08b')}  input[{i}]={(x>>(7-i))&1} output[{j}]={(o>>(7-j))&1}"
                          for x, o in examples)

        if always_one:
            think = f"When input[{i}]=1, output[{j}] is ALWAYS 1\n"
            think += f"Bit moves from position {i} to position {j}\n"
            think += f"Shift distance: {abs(i-j)} positions {'right' if j > i else 'left'}\n"
            think += f"Suggests: {op}"
            answer = op
        else:
            think = f"When input[{i}]=1, output[{j}] is NOT always 1\n"
            think += f"No direct shift correlation at this position pair"
            answer = "No direct correlation"

        return {
            "user": f"Do these examples show a correlation between input[{i}] and output[{j}]?\n{ex_str}",
            "think": think,
            "answer": answer,
        }


@register
class BitNarrowSources(MicroSkill):
    name = "bit_narrow_sources"
    puzzle_type = "bit_manipulation"
    description = "Which single shift makes input MOST SIMILAR to output? Score candidates"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_narrow.jsonl"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        # Pick a real rule and generate one example
        src_names = list(SHIFT_OPS.keys())
        gate_names = list(GATE_OPS.keys())
        sa, sb = rng.sample(src_names, 2)
        gate = rng.choice(gate_names)

        x = rng.randint(1, 254)
        bits = format(x, "08b")
        out = format(GATE_OPS[gate](SHIFT_OPS[sa](x), SHIFT_OPS[sb](x)), "08b")

        # Score 4 candidate shifts against output
        candidates = rng.sample(src_names, 4)
        # Make sure the best source is included
        if sa not in candidates:
            candidates[0] = sa

        scores = []
        for c in candidates:
            shifted = format(SHIFT_OPS[c](x), "08b")
            match = sum(1 for k in range(8) if shifted[k] == out[k])
            scores.append((c, shifted, match))

        scores.sort(key=lambda t: -t[2])
        best = scores[0]

        think_lines = [f"Input: {bits}, Output: {out}", ""]
        for c, shifted, match in scores:
            bar = "".join("→ MATCH" if shifted[k] == out[k] else "→ MISMATCH" for k in range(8))
            marker = " ← best" if c == best[0] else ""
            think_lines.append(f"  {c:6s}({bits}) = {shifted}  {match}/8 match  {bar}{marker}")
        think_lines.append(f"\nMost similar: {best[0]} ({best[2]}/8)")

        return {
            "user": f"Which shift makes {bits} most similar to {out}?\nCandidates: {', '.join(candidates)}",
            "think": "\n".join(think_lines),
            "answer": f"{best[0]} ({best[2]}/8)",
        }


@register 
class BitVisualPattern(MicroSkill):
    name = "bit_visual_pattern"
    puzzle_type = "bit_manipulation"
    description = "Describe the pattern you see across these examples (open-ended)"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_visual.jsonl"
    weight = 3.0
    max_pool = 5000

    def generate_one(self, rng, difficulty="medium"):
        # Generate examples from a real rule and describe the pattern
        src_names = list(SHIFT_OPS.keys())
        gate_names = list(GATE_OPS.keys())
        
        pattern_type = rng.choice(["simple_shift", "xor_pair", "or_heavy", "and_sparse"])
        
        if pattern_type == "simple_shift":
            op = rng.choice(src_names)
            examples = []
            for _ in range(4):
                x = rng.randint(1, 254)
                examples.append((format(x, "08b"), format(SHIFT_OPS[op](x), "08b")))
            
            import re
            m = re.match(r'(shl|shr|rol|ror)(\d+)', op)
            kind, k = m.group(1), int(m.group(2))
            
            if kind == "shr":
                pattern_desc = f"Output always starts with {k} zeros. Looks like a right shift by {k}."
            elif kind == "shl":
                pattern_desc = f"Output always ends with {k} zeros. Looks like a left shift by {k}."
            elif kind == "rol":
                pattern_desc = f"Same bits as input, rotated left. Popcount preserved in every example."
            else:
                pattern_desc = f"Same bits as input, rotated right. Popcount preserved in every example."
        
        elif pattern_type == "xor_pair":
            sa, sb = rng.sample(src_names, 2)
            examples = []
            for _ in range(4):
                x = rng.randint(1, 254)
                out = SHIFT_OPS[sa](x) ^ SHIFT_OPS[sb](x)
                examples.append((format(x, "08b"), format(out & 0xFF, "08b")))
            pattern_desc = "Output ones-count varies. No consistent zeros at fixed positions. Looks like XOR of two shifted views."
        
        elif pattern_type == "or_heavy":
            sa, sb = rng.sample(src_names, 2)
            examples = []
            for _ in range(4):
                x = rng.randint(1, 254)
                a = SHIFT_OPS[sa](x); b = SHIFT_OPS[sb](x)
                out = (b | (~(a ^ b) & 0xFF)) & 0xFF  # OR_XNOR
                examples.append((format(x, "08b"), format(out, "08b")))
            pattern_desc = "Output is very dense — lots of 1s. Output ones-count is consistently higher than input. Suggests OR or XNOR family."
        
        else:  # and_sparse
            sa, sb = rng.sample(src_names, 2)
            examples = []
            for _ in range(4):
                x = rng.randint(1, 254)
                out = SHIFT_OPS[sa](x) & SHIFT_OPS[sb](x)
                examples.append((format(x, "08b"), format(out & 0xFF, "08b")))
            pattern_desc = "Output is sparse — few 1s. Output ones-count consistently lower than input. Suggests AND family."

        ex_str = "\n".join(f"  {i} -> {o}" for i, o in examples)
        
        in_pops = [bin(int(i,2)).count('1') for i,_ in examples]
        out_pops = [bin(int(o,2)).count('1') for _,o in examples]
        
        think = f"Examples:\n{ex_str}\n\n"
        think += f"Input ones-counts: {in_pops}\n"
        think += f"Output ones-counts: {out_pops}\n"
        think += f"Pattern: {pattern_desc}"

        return {
            "user": f"What pattern do you see in these examples?\n{ex_str}",
            "think": think,
            "answer": pattern_desc,
        }


# Update weights for new skills
_WEIGHTS.update({
    "bit_family_from_popcount": 5.0,
    "bit_eliminate_family": 4.0,
    "bit_correlate_positions": 4.0,
    "bit_narrow_sources": 5.0,
    "bit_visual_pattern": 3.0,
    "enc_vocab_audit": 3.0,
})
for _name, _weight in _WEIGHTS.items():
    if _name in _REG:
        _REG[_name].weight = _weight


# ============================================================
# BIT: ADVANCED IDENTIFICATION + INFERENCE SKILLS
# ============================================================

@register
class BitGateFromKnownSources(MicroSkill):
    name = "bit_gate_from_known_sources"
    puzzle_type = "bit_manipulation"
    description = "Sources A,B are known. Which gate matches the output?"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_gate_known.jsonl"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        sa, sb = rng.sample(src_names, 2)
        correct_gate = rng.choice(list(GATE_OPS.keys()))

        # Generate 3 examples
        examples = []
        for _ in range(3):
            x = rng.randint(1, 254)
            a = SHIFT_OPS[sa](x); b = SHIFT_OPS[sb](x)
            out = GATE_OPS[correct_gate](a, b)
            examples.append((format(x,"08b"), format(a,"08b"), format(b,"08b"), format(out,"08b")))

        gates = list(GATE_OPS.keys())
        options = [correct_gate] + rng.sample([g for g in gates if g != correct_gate], 2)
        rng.shuffle(options)
        labels = ["A", "B", "C"]
        correct_label = labels[options.index(correct_gate)]

        opt_str = "\n".join(f"  {labels[j]}) {options[j]}" for j in range(3))

        think_lines = []
        for j, gate in enumerate(options):
            matches = 0
            for x_bits, a_bits, b_bits, out_bits in examples:
                computed = format(GATE_OPS[gate](int(a_bits,2), int(b_bits,2)), "08b")
                if computed == out_bits: matches += 1
            mark = "→ MATCH ALL" if matches == 3 else f"→ MISMATCH {matches}/3"
            think_lines.append(f"{labels[j]}) {gate}: {mark}")

        return {
            "user": f"A={sa}(x), B={sb}(x). Given these:\n" +
                    "\n".join(f"  x={x} A={a} B={b} out={o}" for x,a,b,o in examples) +
                    f"\nWhich gate?\n{opt_str}",
            "think": "\n".join(think_lines),
            "answer": correct_label,
        }


@register
class BitCountAcrossExamples(MicroSkill):
    name = "bit_count_across_examples"
    puzzle_type = "bit_manipulation"
    description = "Count how many examples have output bit N = 1 (vertical scanning)"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_count_across.jsonl"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        # Generate some examples
        n_ex = rng.randint(5, 8)
        examples = [(format(rng.randint(0,255),"08b"), format(rng.randint(0,255),"08b")) for _ in range(n_ex)]
        pos = rng.randint(0, 7)

        count_1 = sum(1 for _, out in examples if out[7-pos] == '1')
        count_0 = n_ex - count_1

        ex_str = "\n".join(f"  {inp} -> {out}  (bit {pos} = {out[7-pos]})" for inp, out in examples)
        think = f"Scanning output bit {pos} across {n_ex} examples:\n{ex_str}\n{count_1} ones, {count_0} zeros"

        return {
            "user": f"In these examples, how many have output bit {pos} = 1?\n" +
                    "\n".join(f"  {inp} -> {out}" for inp, out in examples),
            "think": think,
            "answer": str(count_1),
        }


@register
class BitRankRules(MicroSkill):
    name = "bit_rank_rules"
    puzzle_type = "bit_manipulation"
    description = "Rank 3 rules by how many examples they match. 9/9 beats 7/9."
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_rank.jsonl"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        gate_names = list(GATE_OPS.keys())

        # Pick the correct rule
        sa, sb = rng.sample(src_names, 2)
        gate = rng.choice(gate_names)
        def compute(x):
            return GATE_OPS[gate](SHIFT_OPS[sa](x), SHIFT_OPS[sb](x))

        n_ex = rng.randint(6, 9)
        inputs = rng.sample(range(256), n_ex)
        examples = [(format(x,"08b"), format(compute(x),"08b")) for x in inputs]

        # Build 3 rules with different match counts
        rules = [(f"{sa} {gate} {sb}", compute, n_ex)]  # correct: matches all

        for _ in range(2):
            wa, wb = rng.sample(src_names, 2)
            wg = rng.choice(gate_names)
            def wrong_fn(x, wa=wa, wb=wb, wg=wg):
                return GATE_OPS[wg](SHIFT_OPS[wa](x), SHIFT_OPS[wb](x))
            matches = sum(1 for x in inputs if format(wrong_fn(x),"08b") == format(compute(x),"08b"))
            rules.append((f"{wa} {wg} {wb}", wrong_fn, matches))

        rng.shuffle(rules)
        labels = ["A", "B", "C"]

        think_lines = []
        for j, (desc, fn, matches) in enumerate(rules):
            think_lines.append(f"{labels[j]}) {desc}: matches {matches}/{n_ex}")

        ranked = sorted(range(3), key=lambda j: -rules[j][2])
        ranking = " > ".join(labels[j] for j in ranked)
        think_lines.append(f"Ranking: {ranking}")
        best = labels[ranked[0]]

        ex_str = "\n".join(f"  {inp} -> {out}" for inp, out in examples[:4])
        opt_str = "\n".join(f"  {labels[j]}) {rules[j][0]}" for j in range(3))

        return {
            "user": f"Which rule fits the MOST examples?\nExamples:\n{ex_str}\n  ...\nRules:\n{opt_str}",
            "think": "\n".join(think_lines),
            "answer": best,
        }


@register
class BitSecondSource(MicroSkill):
    name = "bit_second_source"
    puzzle_type = "bit_manipulation"
    description = "First source is known. Find the second source that completes the rule."
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_second.jsonl"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        sa, sb = rng.sample(src_names, 2)
        gate = rng.choice(list(GATE_OPS.keys()))

        x = rng.randint(1, 254)
        bits = format(x, "08b")
        a = SHIFT_OPS[sa](x); b = SHIFT_OPS[sb](x)
        out = GATE_OPS[gate](a, b)
        out_bits = format(out, "08b")
        a_bits = format(a, "08b")

        # Test 3 candidates for second source
        candidates = [sb] + rng.sample([s for s in src_names if s != sb and s != sa], 2)
        rng.shuffle(candidates)
        labels = ["A", "B", "C"]
        correct_label = labels[candidates.index(sb)]

        think_lines = [f"Known: first source = {sa}, gate = {gate}", f"x = {bits}, A = {a_bits}, output = {out_bits}", ""]
        for j, cand in enumerate(candidates):
            b_test = format(SHIFT_OPS[cand](x), "08b")
            computed = format(GATE_OPS[gate](a, SHIFT_OPS[cand](x)), "08b")
            match = sum(1 for k in range(8) if computed[k] == out_bits[k])
            mark = "→ MATCH MATCH" if computed == out_bits else f"→ MISMATCH {match}/8"
            think_lines.append(f"{labels[j]}) B={cand}: {b_test}, {gate} = {computed} {mark}")

        return {
            "user": f"A={sa}(x), gate={gate}. x={bits}, output={out_bits}.\nWhat's the second source B?\n" +
                    "\n".join(f"  {labels[j]}) {candidates[j]}" for j in range(3)),
            "think": "\n".join(think_lines),
            "answer": correct_label,
        }


@register
class BitTraceAudit(MicroSkill):
    name = "bit_trace_audit"
    puzzle_type = "bit_manipulation"
    description = "Is this trace step correct? Check one specific computation."
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_audit.jsonl"
    weight = 3.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        op = rng.choice(list(SHIFT_OPS.keys()))
        x = rng.randint(0, 255)
        bits = format(x, "08b")
        correct = format(SHIFT_OPS[op](x), "08b")

        is_correct = rng.random() < 0.5
        if is_correct:
            claimed = correct
            think = f"Check: {op}({bits})\n{shift_str(op, bits)}\nClaimed {claimed} = actual {correct} → MATCH"
            answer = "Correct"
        else:
            pos = rng.randint(0, 7)
            claimed = list(correct)
            claimed[pos] = '1' if claimed[pos] == '0' else '0'
            claimed = "".join(claimed)
            think = f"Check: {op}({bits})\n{shift_str(op, bits)}\nClaimed {claimed} ≠ actual {correct}\nBit {pos} is wrong"
            answer = f"Wrong at bit {pos}. Should be {correct}"

        return {
            "user": f'Is this correct? "{op}({bits}) = {claimed}"',
            "think": think,
            "answer": answer,
        }


@register
class BitConfidentOrNot(MicroSkill):
    name = "bit_confident_or_not"
    puzzle_type = "bit_manipulation"
    description = "Rule matches all examples. But does another rule ALSO match? Teaches uncertainty."
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_confident.jsonl"
    weight = 3.0
    max_pool = 5000

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        gate_names = list(GATE_OPS.keys())

        sa, sb = rng.sample(src_names, 2)
        gate = rng.choice(gate_names)
        def compute(x):
            return GATE_OPS[gate](SHIFT_OPS[sa](x), SHIFT_OPS[sb](x))

        inputs = rng.sample(range(256), 5)
        examples = [(format(x,"08b"), format(compute(x),"08b")) for x in inputs]

        # Check if an alternative rule also matches all examples
        alt_sa, alt_sb = rng.sample(src_names, 2)
        alt_gate = rng.choice(gate_names)
        def alt_compute(x):
            return GATE_OPS[alt_gate](SHIFT_OPS[alt_sa](x), SHIFT_OPS[alt_sb](x))

        alt_matches = all(format(alt_compute(x),"08b") == format(compute(x),"08b") for x in inputs)
        is_same_rule = (sa == alt_sa and sb == alt_sb and gate == alt_gate)

        if alt_matches and not is_same_rule:
            think = f"Rule 1: {sa} {gate} {sb} — matches all 5 → MATCH\n"
            think += f"Rule 2: {alt_sa} {alt_gate} {alt_sb} — ALSO matches all 5 → MATCH\n"
            think += f"Two different rules both fit. NOT confident — ambiguous."
            answer = "Not confident — multiple rules fit"
        else:
            think = f"Rule 1: {sa} {gate} {sb} — matches all 5 → MATCH\n"
            think += f"Checked alternatives — none match all 5.\n"
            think += f"Confident — unique rule."
            answer = "Confident — unique rule"

        ex_str = "\n".join(f"  {i} -> {o}" for i, o in examples)
        return {
            "user": f"This rule matches all 5 examples. Should you be confident?\nRule: {sa} {gate} {sb}\n{ex_str}",
            "think": think,
            "answer": answer,
        }


@register
class BitConstantPositions(MicroSkill):
    name = "bit_constant_positions"
    puzzle_type = "bit_manipulation"
    description = "Which output bit positions are ALWAYS 0 or ALWAYS 1 across examples?"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_constant.jsonl"
    weight = 3.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        n_ex = rng.randint(5, 8)
        # Use a real rule so the constants are meaningful
        src_names = list(SHIFT_OPS.keys())
        sa, sb = rng.sample(src_names, 2)
        gate = rng.choice(list(GATE_OPS.keys()))
        def compute(x):
            return GATE_OPS[gate](SHIFT_OPS[sa](x), SHIFT_OPS[sb](x))

        inputs = rng.sample(range(256), n_ex)
        outputs = [format(compute(x), "08b") for x in inputs]

        # Find constant positions
        always_0 = []
        always_1 = []
        for pos in range(8):
            bits = [out[7-pos] for out in outputs]
            if all(b == '0' for b in bits):
                always_0.append(pos)
            elif all(b == '1' for b in bits):
                always_1.append(pos)

        ex_str = "\n".join(f"  {format(x,'08b')} -> {outputs[j]}" for j, x in enumerate(inputs))

        think_lines = ["Scanning each bit position across all outputs:"]
        for pos in range(8):
            bits = [out[7-pos] for out in outputs]
            vals = set(bits)
            if len(vals) == 1:
                think_lines.append(f"  Bit {pos}: always {bits[0]}")
            else:
                think_lines.append(f"  Bit {pos}: varies ({bits.count('0')} zeros, {bits.count('1')} ones)")

        a0 = ", ".join(str(p) for p in always_0) if always_0 else "none"
        a1 = ", ".join(str(p) for p in always_1) if always_1 else "none"
        answer = f"Always 0: [{a0}], Always 1: [{a1}]"

        return {
            "user": f"Which output bit positions are constant across all examples?\n{ex_str}",
            "think": "\n".join(think_lines),
            "answer": answer,
        }


@register
class BitBatchPipeline(MicroSkill):
    name = "bit_batch_pipeline"
    puzzle_type = "bit_manipulation"
    description = "Execute full rule pipeline on 3 different inputs (builds execution fluency)"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_batch.jsonl"
    weight = 3.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        sa, sb = rng.sample(src_names, 2)
        gate = rng.choice(list(GATE_OPS.keys()))

        inputs = rng.sample(range(1, 255), 3)
        think_lines = [f"Rule: A={sa}, B={sb}, {gate}\n"]
        results = []

        for x in inputs:
            bits = format(x, "08b")
            a = format(SHIFT_OPS[sa](x), "08b")
            b = format(SHIFT_OPS[sb](x), "08b")
            r, g_lines = gate_position_by_position(a, b, gate)
            think_lines.append(f"x={bits}: A={a}, B={b}")
            think_lines.append(f"  {gate}:")
            think_lines.extend(f"  {l}" for l in g_lines)
            think_lines.append(f"  = {r}\n")
            results.append(r)

        return {
            "user": f"Execute A={sa}, B={sb}, {gate} on: {', '.join(format(x,'08b') for x in inputs)}",
            "think": "\n".join(think_lines),
            "answer": ", ".join(results),
        }


@register
class BitGateFromProperties(MicroSkill):
    name = "bit_gate_from_properties"
    puzzle_type = "bit_manipulation"
    description = "Output has more/fewer 1s than inputs — which gate?"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_gate_prop.jsonl"
    weight = 3.0
    max_pool = 5000

    def generate_one(self, rng, difficulty="medium"):
        a = rng.randint(0, 255); b = rng.randint(0, 255)
        a_bits = format(a, "08b"); b_bits = format(b, "08b")
        a_ones = bin(a).count('1'); b_ones = bin(b).count('1')

        gate = rng.choice(list(GATE_OPS.keys()))
        out = GATE_OPS[gate](a, b)
        out_bits = format(out, "08b")
        out_ones = bin(out).count('1')

        if out_ones > max(a_ones, b_ones):
            property_desc = f"Output has {out_ones} ones, MORE than A ({a_ones}) or B ({b_ones})"
        elif out_ones < min(a_ones, b_ones):
            property_desc = f"Output has {out_ones} ones, FEWER than A ({a_ones}) and B ({b_ones})"
        else:
            property_desc = f"Output has {out_ones} ones, between A ({a_ones}) and B ({b_ones})"

        options = [gate] + rng.sample([g for g in GATE_OPS if g != gate], 2)
        rng.shuffle(options)
        labels = ["A", "B", "C"]
        correct_label = labels[options.index(gate)]

        think = f"A={a_bits} ({a_ones} ones), B={b_bits} ({b_ones} ones)\n"
        think += f"Output={out_bits} ({out_ones} ones)\n"
        think += f"{property_desc}\n"
        for j, g in enumerate(options):
            r = GATE_OPS[g](a, b)
            think += f"{labels[j]}) {g} would give {bin(r).count('1')} ones {'→ MATCH' if r == out else '→ MISMATCH'}\n"

        return {
            "user": f"A={a_bits}, B={b_bits}, output={out_bits}.\n{property_desc}\nWhich gate?\n" +
                    "\n".join(f"  {labels[j]}) {options[j]}" for j in range(3)),
            "think": think,
            "answer": correct_label,
        }


@register
class BitSourceConsistency(MicroSkill):
    name = "bit_source_consistency"
    puzzle_type = "bit_manipulation"
    description = "shr2 matches examples 1-3. Does it also match example 4?"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_consistency.jsonl"
    weight = 3.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        sa, sb = rng.sample(src_names, 2)
        gate = rng.choice(list(GATE_OPS.keys()))
        def compute(x):
            return GATE_OPS[gate](SHIFT_OPS[sa](x), SHIFT_OPS[sb](x))

        # Generate 4 examples
        inputs = rng.sample(range(256), 4)
        examples = [(format(x,"08b"), format(compute(x),"08b")) for x in inputs]

        # Pick a candidate source to test
        if rng.random() < 0.5:
            test_src = sa  # correct
        else:
            test_src = rng.choice([s for s in src_names if s != sa])  # wrong

        # Check similarity of test_src(x) to output for each example
        think_lines = [f"Testing: does {test_src} contribute to the output?\n"]
        matches_per_ex = []
        for j, (inp, out) in enumerate(examples):
            x = int(inp, 2)
            shifted = format(SHIFT_OPS[test_src](x), "08b")
            match = sum(1 for k in range(8) if shifted[k] == out[k])
            matches_per_ex.append(match)
            think_lines.append(f"  Ex {j+1}: {test_src}({inp})={shifted} vs {out}: {match}/8 match")

        avg_match = sum(matches_per_ex) / len(matches_per_ex)
        consistent = all(m >= 5 for m in matches_per_ex)

        if consistent:
            think_lines.append(f"\nAvg {avg_match:.1f}/8 — consistently high across all examples. Likely a real source.")
            answer = "Yes — consistent"
        else:
            think_lines.append(f"\nAvg {avg_match:.1f}/8 — inconsistent. Probably not a source.")
            answer = "No — inconsistent"

        return {
            "user": f"Does {test_src} consistently match the output across these examples?\n" +
                    "\n".join(f"  {inp} -> {out}" for inp, out in examples),
            "think": "\n".join(think_lines),
            "answer": answer,
        }


# Update weights for new skills
_WEIGHTS.update({
    "bit_gate_from_known_sources": 5.0,
    "bit_count_across_examples": 4.0,
    "bit_rank_rules": 4.0,
    "bit_second_source": 4.0,
    "bit_trace_audit": 3.0,
    "bit_confident_or_not": 3.0,
    "bit_constant_positions": 3.0,
    "bit_batch_pipeline": 3.0,
    "bit_gate_from_properties": 3.0,
    "bit_source_consistency": 3.0,
})
for _name, _weight in _WEIGHTS.items():
    if _name in _REG:
        _REG[_name].weight = _weight


# ============================================================
# BIT: SPOT THE INVARIANT
# ============================================================

@register
class BitSpotInvariant(MicroSkill):
    name = "bit_spot_invariant"
    puzzle_type = "bit_manipulation"
    description = "What pattern holds across ALL examples? Spot the invariant."
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_invariant.jsonl"
    weight = 4.0
    max_pool = 10000

    def _check_invariants(self, examples):
        """Check which invariants hold across all (input, output) examples."""
        holds = []

        ins = [int(i, 2) for i, _ in examples]
        outs = [int(o, 2) for _, o in examples]
        in_bits = [i for i, _ in examples]
        out_bits = [o for _, o in examples]

        # Ones count invariants
        in_ones = [bin(x).count('1') for x in ins]
        out_ones = [bin(x).count('1') for x in outs]

        if all(io == oo for io, oo in zip(in_ones, out_ones)):
            holds.append(("ones count stays the same", "rotation or XOR-like"))
        if all(oo > io for io, oo in zip(in_ones, out_ones)):
            holds.append(("ones count always increases", "OR-family adds 1-bits"))
        if all(oo < io for io, oo in zip(in_ones, out_ones)):
            holds.append(("ones count always decreases", "AND-family removes 1-bits"))
        if all(oo >= 6 for oo in out_ones):
            holds.append(("output always has 6+ ones", "dense output — OR_XNOR likely"))
        if all(oo <= 2 for oo in out_ones):
            holds.append(("output always has 2 or fewer ones", "sparse output — AND likely"))

        # Positional invariants
        for pos in range(8):
            out_at_pos = [o[7-pos] for o in out_bits]
            if all(b == '0' for b in out_at_pos):
                holds.append((f"bit {pos} is always 0", f"shift zeros that position"))
            if all(b == '1' for b in out_at_pos):
                holds.append((f"bit {pos} is always 1", f"OR forces that position to 1"))

        # Last bit
        if all(o[-1] == '0' for o in out_bits):
            holds.append(("last bit always 0", "left shift involved (appends zeros)"))
        if all(o[0] == '0' for o in out_bits):
            holds.append(("first bit always 0", "right shift involved (prepends zeros)"))

        # Palindrome
        if all(o == o[::-1] for o in out_bits):
            holds.append(("output is always a palindrome", "symmetric rule"))

        # Same as input
        if all(i == o for i, o in zip(in_bits, out_bits)):
            holds.append(("output equals input", "identity or cancelling operations"))

        # Output is complement of input
        if all(format((~int(i,2)) & 0xFF, '08b') == o for i, o in examples):
            holds.append(("output is NOT of input", "complement operation"))

        return holds

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        gate_names = list(GATE_OPS.keys())

        # Try rules until we find one with at least 1 interesting invariant
        for _ in range(20):
            sa, sb = rng.sample(src_names, 2)
            gate = rng.choice(gate_names)
            def compute(x, sa=sa, sb=sb, gate=gate):
                return GATE_OPS[gate](SHIFT_OPS[sa](x), SHIFT_OPS[sb](x))

            inputs = rng.sample(range(256), 6)
            examples = [(format(x, "08b"), format(compute(x), "08b")) for x in inputs]
            invariants = self._check_invariants(examples)

            if invariants:
                break
        else:
            return None

        ex_str = "\n".join(f"  {i} -> {o}" for i, o in examples)

        # Pick one invariant to ask about, plus 2 false ones
        correct = rng.choice(invariants)

        false_invariants = [
            ("ones count stays the same", ""),
            ("ones count always increases", ""),
            ("ones count always decreases", ""),
            ("output always has 6+ ones", ""),
            ("output always has 2 or fewer ones", ""),
            ("last bit always 0", ""),
            ("first bit always 0", ""),
            ("output is always a palindrome", ""),
            ("output equals input", ""),
        ]
        false_options = [(desc, _) for desc, _ in false_invariants
                        if desc != correct[0] and not any(desc == h[0] for h in invariants)]

        if len(false_options) < 2:
            return None

        options = [correct] + rng.sample(false_options, 2)
        rng.shuffle(options)
        labels = ["A", "B", "C"]
        correct_label = labels[options.index(correct)]

        opt_str = "\n".join(f"  {labels[j]}) {options[j][0]}" for j in range(3))

        think_lines = []
        in_ones = [bin(int(i,2)).count('1') for i,_ in examples]
        out_ones = [bin(int(o,2)).count('1') for _,o in examples]
        think_lines.append(f"Input ones: {in_ones}")
        think_lines.append(f"Output ones: {out_ones}")
        for j, (desc, _) in enumerate(options):
            is_true = any(desc == h[0] for h in invariants)
            think_lines.append(f"{labels[j]}) \"{desc}\" — {'TRUE → MATCH' if is_true else 'FALSE → MISMATCH'}")
        think_lines.append(f"\n{correct[0]} -> {correct[1]}")

        return {
            "user": f"Which pattern holds across ALL these examples?\n{ex_str}\n{opt_str}",
            "think": "\n".join(think_lines),
            "answer": correct_label,
        }


@register
class BitSpotInvariantOpen(MicroSkill):
    name = "bit_spot_invariant_open"
    puzzle_type = "bit_manipulation"
    description = "List ALL invariants you can observe (open-ended, high gradient)"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_invariant_open.jsonl"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        # Reuse the invariant checker from BitSpotInvariant
        src_names = list(SHIFT_OPS.keys())
        gate_names = list(GATE_OPS.keys())

        for _ in range(20):
            sa, sb = rng.sample(src_names, 2)
            gate = rng.choice(gate_names)
            def compute(x, sa=sa, sb=sb, gate=gate):
                return GATE_OPS[gate](SHIFT_OPS[sa](x), SHIFT_OPS[sb](x))

            inputs = rng.sample(range(256), 6)
            examples = [(format(x, "08b"), format(compute(x), "08b")) for x in inputs]

            # Check invariants inline
            ins = [int(i, 2) for i, _ in examples]
            outs = [int(o, 2) for _, o in examples]
            in_bits = [i for i, _ in examples]
            out_bits = [o for _, o in examples]
            in_ones = [bin(x).count('1') for x in ins]
            out_ones = [bin(x).count('1') for x in outs]

            observations = []

            if all(io == oo for io, oo in zip(in_ones, out_ones)):
                observations.append("ones count preserved (same in every example)")
            elif all(oo > io for io, oo in zip(in_ones, out_ones)):
                observations.append("ones count always increases")
            elif all(oo < io for io, oo in zip(in_ones, out_ones)):
                observations.append("ones count always decreases")

            for pos in range(8):
                vals = [o[7-pos] for o in out_bits]
                if all(b == '0' for b in vals):
                    observations.append(f"output bit {pos} is always 0")
                elif all(b == '1' for b in vals):
                    observations.append(f"output bit {pos} is always 1")

            if all(o[-1] == '0' for o in out_bits):
                observations.append("last bit always 0 (left shift likely)")
            if all(o[0] == '0' for o in out_bits):
                observations.append("first bit always 0 (right shift likely)")

            avg_out = sum(out_ones) / len(out_ones)
            if avg_out >= 6:
                observations.append(f"output is dense (avg {avg_out:.1f} ones)")
            elif avg_out <= 2:
                observations.append(f"output is sparse (avg {avg_out:.1f} ones)")

            if observations:
                break
        else:
            return None

        ex_str = "\n".join(f"  {i} -> {o}" for i, o in examples)

        think = f"Input ones: {in_ones}\nOutput ones: {out_ones}\n\nInvariants found:\n"
        for obs in observations:
            think += f"  - {obs}\n"

        return {
            "user": f"List every pattern you can observe across these examples:\n{ex_str}",
            "think": think,
            "answer": "; ".join(observations),
        }


# Update weights
_WEIGHTS.update({
    "bit_spot_invariant": 4.0,
    "bit_spot_invariant_open": 4.0,
})
for _name, _weight in _WEIGHTS.items():
    if _name in _REG:
        _REG[_name].weight = _weight


@register
class BitInvariantChecklist(MicroSkill):
    name = "bit_invariant_checklist"
    puzzle_type = "bit_manipulation"
    description = "Check each invariant from a fixed list: yes or no for each"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_checklist.jsonl"
    weight = 5.0
    max_pool = 10000

    CHECKLIST = [
        "ones count preserved",
        "ones count increases",
        "ones count decreases",
        "output dense (avg 6+ ones)",
        "output sparse (avg 2- ones)",
        "first bit always 0",
        "last bit always 0",
        "first bit always 1",
        "last bit always 1",
        "output is palindrome",
    ]

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        sa, sb = rng.sample(src_names, 2)
        gate = rng.choice(list(GATE_OPS.keys()))
        def compute(x):
            return GATE_OPS[gate](SHIFT_OPS[sa](x), SHIFT_OPS[sb](x))

        inputs = rng.sample(range(256), 6)
        examples = [(format(x,"08b"), format(compute(x),"08b")) for x in inputs]

        in_ones = [bin(int(i,2)).count('1') for i,_ in examples]
        out_ones = [bin(int(o,2)).count('1') for _,o in examples]
        out_bits = [o for _,o in examples]
        avg_out = sum(out_ones)/len(out_ones)

        checks = {
            "ones count preserved": all(io == oo for io,oo in zip(in_ones, out_ones)),
            "ones count increases": all(oo > io for io,oo in zip(in_ones, out_ones)),
            "ones count decreases": all(oo < io for io,oo in zip(in_ones, out_ones)),
            "output dense (avg 6+ ones)": avg_out >= 6,
            "output sparse (avg 2- ones)": avg_out <= 2,
            "first bit always 0": all(o[0] == '0' for o in out_bits),
            "last bit always 0": all(o[-1] == '0' for o in out_bits),
            "first bit always 1": all(o[0] == '1' for o in out_bits),
            "last bit always 1": all(o[-1] == '1' for o in out_bits),
            "output is palindrome": all(o == o[::-1] for o in out_bits),
        }

        # Pick a random subset of 5 to check (keeps it focused)
        items = rng.sample(self.CHECKLIST, 5)

        ex_str = "\n".join(f"  {i} -> {o}" for i,o in examples)
        checklist_str = "\n".join(f"  {j+1}. {item}" for j, item in enumerate(items))

        think_lines = [f"Ones in: {in_ones}, Ones out: {out_ones}, Avg out: {avg_out:.1f}\n"]
        yes_items = []
        for j, item in enumerate(items):
            result = checks[item]
            think_lines.append(f"  {j+1}. {item}: {'YES' if result else 'NO'}")
            if result:
                yes_items.append(str(j+1))

        answer = ", ".join(yes_items) if yes_items else "None"

        return {
            "user": f"Check each — which hold for ALL examples?\n{ex_str}\n\nChecklist:\n{checklist_str}",
            "think": "\n".join(think_lines),
            "answer": answer,
        }


_WEIGHTS["bit_invariant_checklist"] = 5.0
if "bit_invariant_checklist" in _REG:
    _REG["bit_invariant_checklist"].weight = 5.0


@register
class BitCompareToTarget(MicroSkill):
    name = "bit_compare_to_target"
    puzzle_type = "bit_manipulation"
    description = "Which of a-f has most/least in common with X? Any ties? Teaches precise comparison."
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_compare.jsonl"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        target = format(rng.randint(0, 255), "08b")
        
        # Generate 6 candidates with varying similarity
        candidates = []
        for _ in range(6):
            x = rng.randint(0, 255)
            candidates.append(format(x, "08b"))
        
        # For harder difficulty, make some very close (1-2 bits off)
        if difficulty == "hard":
            t = int(target, 2)
            # Make 2 candidates that are 1 bit off
            for j in range(min(2, len(candidates))):
                close = t ^ (1 << rng.randint(0, 7))
                candidates[j] = format(close, "08b")
        
        labels = ["a", "b", "c", "d", "e", "f"]
        
        # Compute matches for each
        scores = []
        for j, cand in enumerate(candidates):
            matches = sum(1 for i in range(8) if target[i] == cand[i])
            scores.append((labels[j], cand, matches))
        
        # Find most, least, ties
        max_score = max(s[2] for s in scores)
        min_score = min(s[2] for s in scores)
        most = [s for s in scores if s[2] == max_score]
        least = [s for s in scores if s[2] == min_score]
        
        has_tie_most = len(most) > 1
        has_tie_least = len(least) > 1
        
        cand_str = "\n".join(f"  {labels[j]}) {candidates[j]}" for j in range(6))
        
        think_lines = [f"Target: {target}\n"]
        for label, cand, matches in scores:
            bar = "".join("→ MATCH" if target[i] == cand[i] else "→ MISMATCH" for i in range(8))
            think_lines.append(f"  {label}) {cand}  {bar}  {matches}/8")
        
        think_lines.append(f"\nMost in common: {', '.join(s[0] for s in most)} ({max_score}/8)" + 
                          (f" — TIE" if has_tie_most else ""))
        think_lines.append(f"Least in common: {', '.join(s[0] for s in least)} ({min_score}/8)" +
                          (f" — TIE" if has_tie_least else ""))
        
        most_str = ", ".join(s[0] for s in most)
        least_str = ", ".join(s[0] for s in least)
        tie_str = "yes" if has_tie_most or has_tie_least else "no"
        
        return {
            "user": f"Compared to X = {target}, which has MOST in common? LEAST? Any ties?\n{cand_str}",
            "think": "\n".join(think_lines),
            "answer": f"Most: {most_str} ({max_score}/8). Least: {least_str} ({min_score}/8). Ties: {tie_str}",
        }


_WEIGHTS["bit_compare_to_target"] = 5.0
if "bit_compare_to_target" in _REG:
    _REG["bit_compare_to_target"].weight = 5.0


@register
class BitLeastError(MicroSkill):
    name = "bit_least_error"
    puzzle_type = "bit_manipulation"
    description = "Which candidate answer has the fewest wrong bits compared to the target?"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_least_error.jsonl"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        target = format(rng.randint(0, 255), "08b")

        n_cands = rng.randint(4, 6)
        candidates = []
        t_int = int(target, 2)

        if difficulty == "hard":
            # Make candidates very close — 1-3 bits off each
            for _ in range(n_cands):
                flips = rng.randint(1, 3)
                c = t_int
                for _ in range(flips):
                    c ^= (1 << rng.randint(0, 7))
                candidates.append(format(c & 0xFF, "08b"))
        else:
            for _ in range(n_cands):
                candidates.append(format(rng.randint(0, 255), "08b"))

        labels = [chr(65 + j) for j in range(n_cands)]

        scores = []
        for j, cand in enumerate(candidates):
            wrong = sum(1 for i in range(8) if target[i] != cand[i])
            scores.append((labels[j], cand, wrong))

        min_wrong = min(s[2] for s in scores)
        best = [s for s in scores if s[2] == min_wrong]

        think_lines = [f"Target: {target}\n"]
        for label, cand, wrong in scores:
            bar = "".join("→ MATCH" if target[i] == cand[i] else "→ MISMATCH" for i in range(8))
            marker = " ← best" if wrong == min_wrong else ""
            think_lines.append(f"  {label}) {cand}  {bar}  {wrong} wrong{marker}")

        if len(best) > 1:
            think_lines.append(f"\nTie: {', '.join(s[0] for s in best)} all have {min_wrong} wrong")

        best_str = ", ".join(s[0] for s in best)

        return {
            "user": f"Target: {target}. Which has the FEWEST wrong bits?\n" +
                    "\n".join(f"  {labels[j]}) {candidates[j]}" for j in range(n_cands)),
            "think": "\n".join(think_lines),
            "answer": f"{best_str} ({min_wrong} wrong)",
        }


@register
class BitWhichCloser(MicroSkill):
    name = "bit_which_closer"
    puzzle_type = "bit_manipulation"
    description = "Is A or B closer to the target? By how many bits?"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_closer.jsonl"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        target = format(rng.randint(0, 255), "08b")
        t_int = int(target, 2)

        if difficulty == "hard":
            # Both very close — 1 vs 2 bits off
            a = t_int ^ (1 << rng.randint(0, 7))
            b = t_int ^ (1 << rng.randint(0, 7)) ^ (1 << rng.randint(0, 7))
        else:
            a = rng.randint(0, 255)
            b = rng.randint(0, 255)

        a_bits = format(a & 0xFF, "08b")
        b_bits = format(b & 0xFF, "08b")

        a_wrong = sum(1 for i in range(8) if target[i] != a_bits[i])
        b_wrong = sum(1 for i in range(8) if target[i] != b_bits[i])

        a_bar = "".join("→ MATCH" if target[i] == a_bits[i] else "→ MISMATCH" for i in range(8))
        b_bar = "".join("→ MATCH" if target[i] == b_bits[i] else "→ MISMATCH" for i in range(8))

        think = f"Target: {target}\n"
        think += f"A: {a_bits}  {a_bar}  {a_wrong} wrong\n"
        think += f"B: {b_bits}  {b_bar}  {b_wrong} wrong\n"

        if a_wrong < b_wrong:
            think += f"A is closer by {b_wrong - a_wrong} bits"
            answer = f"A ({a_wrong} vs {b_wrong} wrong)"
        elif b_wrong < a_wrong:
            think += f"B is closer by {a_wrong - b_wrong} bits"
            answer = f"B ({b_wrong} vs {a_wrong} wrong)"
        else:
            think += f"Tie — both {a_wrong} wrong"
            answer = f"Tie ({a_wrong} wrong each)"

        return {
            "user": f"Target: {target}. Which is closer, A or B?\n  A) {a_bits}\n  B) {b_bits}",
            "think": think,
            "answer": answer,
        }


@register
class BitFixOneBit(MicroSkill):
    name = "bit_fix_one_bit"
    puzzle_type = "bit_manipulation"
    description = "This answer is 1 bit off from correct. Which bit needs flipping?"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_fix1.jsonl"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        correct = format(rng.randint(0, 255), "08b")
        pos = rng.randint(0, 7)
        wrong = list(correct)
        wrong[pos] = "1" if wrong[pos] == "0" else "0"
        wrong = "".join(wrong)

        think = f"Correct: {correct}\nWrong:   {wrong}\n"
        think += f"         {''.join('→ MATCH' if correct[i] == wrong[i] else '→ MISMATCH' for i in range(8))}\n"
        think += f"Position {pos}: wrong has '{wrong[pos]}', should be '{correct[pos]}'"

        return {
            "user": f"This answer is 1 bit off. The correct answer is {correct}. The wrong answer is {wrong}. Which position needs flipping?",
            "think": think,
            "answer": f"Position {pos} (flip '{wrong[pos]}' to '{correct[pos]}')",
        }


@register
class BitPredictOutput(MicroSkill):
    name = "bit_predict_output"
    puzzle_type = "bit_manipulation"
    description = "Given a rule and input, predict just ONE specific output bit position"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_predict1.jsonl"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        sa, sb = rng.sample(src_names, 2)
        gate = rng.choice(list(GATE_OPS.keys()))

        x = rng.randint(0, 255)
        bits = format(x, "08b")
        a = SHIFT_OPS[sa](x)
        b = SHIFT_OPS[sb](x)
        out = GATE_OPS[gate](a, b)

        pos = rng.randint(0, 7)
        a_bit = format(a, "08b")[7-pos]
        b_bit = format(b, "08b")[7-pos]
        out_bit = format(out, "08b")[7-pos]

        think = f"Rule: A={sa}, B={sb}, {gate}\n"
        think += f"x = {bits}\n"
        think += f"A = {sa}({bits}) → A[{pos}] = {a_bit}\n"
        think += f"B = {sb}({bits}) → B[{pos}] = {b_bit}\n"

        if "^" in gate and "~" not in gate:
            think += f"{gate}: {a_bit} ≠ {b_bit} → {'1' if a_bit != b_bit else '0'}" if a_bit != b_bit else f"{gate}: {a_bit} = {b_bit} → 0"
        elif "&" in gate and "~" not in gate:
            think += f"{gate}: {a_bit} and {b_bit} both 1? {'yes → 1' if a_bit == '1' and b_bit == '1' else 'no → 0'}"
        elif "|" in gate:
            think += f"{gate}: {a_bit} or {b_bit} is 1? {'yes → 1' if a_bit == '1' or b_bit == '1' else 'no → 0'}"
        else:
            think += f"output bit {pos} = {out_bit}"

        return {
            "user": f"Rule: A={sa}, B={sb}, {gate}. Input: {bits}. What is output bit {pos}?",
            "think": think,
            "answer": out_bit,
        }


_WEIGHTS.update({
    "bit_least_error": 5.0,
    "bit_which_closer": 4.0,
    "bit_fix_one_bit": 5.0,
    "bit_predict_output": 4.0,
})
for _name, _weight in _WEIGHTS.items():
    if _name in _REG:
        _REG[_name].weight = _weight


# ============================================================
# BIT: MULTIPLE VIEWS / REPRESENTATIONS
# ============================================================

@register
class BitWhereAreOnes(MicroSkill):
    name = "bit_where_ones"
    puzzle_type = "bit_manipulation"
    description = "List 1-positions for 2 values, then find which positions they SHARE"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_where_ones.jsonl"
    weight = 3.0
    max_pool = 5000

    def generate_one(self, rng, difficulty="medium"):
        x = rng.randint(0, 255)
        y = rng.randint(0, 255)
        xb = format(x, "08b")
        yb = format(y, "08b")
        x_ones = [7-i for i in range(8) if xb[i] == "1"]
        y_ones = [7-i for i in range(8) if yb[i] == "1"]
        shared = sorted(set(x_ones) & set(y_ones))

        think = f"X = {xb}: ones at positions {x_ones}\n"
        think += f"Y = {yb}: ones at positions {y_ones}\n"
        think += f"Shared 1-positions: {shared if shared else 'none'}\n"
        think += f"X has {len(x_ones)} ones, Y has {len(y_ones)} ones, {len(shared)} shared"

        return {
            "user": f"Where are the 1-bits in X={xb} and Y={yb}? Which positions do they share?",
            "think": think,
            "answer": f"Shared: {shared if shared else 'none'}",
        }


@register
class BitAndAcrossExamples(MicroSkill):
    name = "bit_and_across"
    puzzle_type = "bit_manipulation"
    description = "AND all example outputs together — reveals bits that are ALWAYS 1"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_and_across.jsonl"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        # Use a real rule so the result is meaningful
        src_names = list(SHIFT_OPS.keys())
        sa, sb = rng.sample(src_names, 2)
        gate = rng.choice(list(GATE_OPS.keys()))
        def compute(x):
            return GATE_OPS[gate](SHIFT_OPS[sa](x), SHIFT_OPS[sb](x))

        n = rng.randint(4, 6)
        inputs = rng.sample(range(256), n)
        outputs = [compute(x) for x in inputs]
        out_bits = [format(o, "08b") for o in outputs]

        # AND all outputs
        and_result = outputs[0]
        for o in outputs[1:]:
            and_result &= o
        and_bits = format(and_result & 0xFF, "08b")

        always_one = [str(7-i) for i in range(8) if and_bits[i] == "1"]

        think = "AND all outputs:\n"
        for ob in out_bits:
            think += f"  {ob}\n"
        think += f"  {''.join('─' for _ in range(8))}\n"
        think += f"  {and_bits}  (AND)\n\n"
        think += f"Positions always 1: {', '.join(always_one) if always_one else 'none'}"

        return {
            "user": f"AND all these outputs together. Which bits are ALWAYS 1?\n" +
                    "\n".join(f"  {format(x,'08b')} -> {ob}" for x, ob in zip(inputs, out_bits)),
            "think": think,
            "answer": ", ".join(always_one) if always_one else "none",
        }


@register
class BitOrAcrossExamples(MicroSkill):
    name = "bit_or_across"
    puzzle_type = "bit_manipulation"
    description = "OR all example outputs together — reveals bits that are EVER 1"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_or_across.jsonl"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        sa, sb = rng.sample(src_names, 2)
        gate = rng.choice(list(GATE_OPS.keys()))
        def compute(x):
            return GATE_OPS[gate](SHIFT_OPS[sa](x), SHIFT_OPS[sb](x))

        n = rng.randint(4, 6)
        inputs = rng.sample(range(256), n)
        outputs = [compute(x) for x in inputs]
        out_bits = [format(o, "08b") for o in outputs]

        or_result = 0
        for o in outputs:
            or_result |= o
        or_bits = format(or_result & 0xFF, "08b")

        never_one = [str(7-i) for i in range(8) if or_bits[i] == "0"]

        think = "OR all outputs:\n"
        for ob in out_bits:
            think += f"  {ob}\n"
        think += f"  {''.join('─' for _ in range(8))}\n"
        think += f"  {or_bits}  (OR)\n\n"
        think += f"Positions NEVER 1 (always 0): {', '.join(never_one) if never_one else 'none'}"

        return {
            "user": f"OR all these outputs. Which bits are NEVER 1?\n" +
                    "\n".join(f"  {format(x,'08b')} -> {ob}" for x, ob in zip(inputs, out_bits)),
            "think": think,
            "answer": ", ".join(never_one) if never_one else "none",
        }


@register
class BitConstantVsVariable(MicroSkill):
    name = "bit_constant_vs_variable"
    puzzle_type = "bit_manipulation"
    description = "Which output bits are constant (same in all examples) vs variable?"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_const_var.jsonl"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        sa, sb = rng.sample(src_names, 2)
        gate = rng.choice(list(GATE_OPS.keys()))
        def compute(x):
            return GATE_OPS[gate](SHIFT_OPS[sa](x), SHIFT_OPS[sb](x))

        n = rng.randint(5, 7)
        inputs = rng.sample(range(256), n)
        outputs = [compute(x) for x in inputs]
        out_bits = [format(o, "08b") for o in outputs]

        and_r = outputs[0]
        or_r = 0
        for o in outputs:
            and_r &= o
            or_r |= o
        and_bits = format(and_r & 0xFF, "08b")
        or_bits = format(or_r & 0xFF, "08b")

        constant = []
        variable = []
        for i in range(8):
            pos = 7 - i
            if and_bits[i] == or_bits[i]:
                constant.append(f"{pos}={'1' if and_bits[i] == '1' else '0'}")
            else:
                variable.append(str(pos))

        think = "Outputs:\n"
        for ob in out_bits:
            think += f"  {ob}\n"
        think += f"\nAND: {and_bits}\nOR:  {or_bits}\n\n"
        think += f"Constant (AND=OR): {', '.join(constant) if constant else 'none'}\n"
        think += f"Variable (AND≠OR): positions {', '.join(variable) if variable else 'none'}"

        return {
            "user": f"Which output bits are constant vs variable across these examples?\n" +
                    "\n".join(f"  {format(x,'08b')} -> {ob}" for x, ob in zip(inputs, out_bits)),
            "think": think,
            "answer": f"Constant: {', '.join(constant) if constant else 'none'}. Variable: {', '.join(variable) if variable else 'none'}",
        }


@register
class BitIsRotation(MicroSkill):
    name = "bit_is_rotation"
    puzzle_type = "bit_manipulation"
    description = "Is X a rotated version of Y? If so, by how many positions?"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_is_rot.jsonl"
    weight = 3.0
    max_pool = 5000

    def generate_one(self, rng, difficulty="medium"):
        from generators.microskill_framework import rol
        x = rng.randint(1, 254)
        x_bits = format(x, "08b")

        is_rotation = rng.random() < 0.6
        if is_rotation:
            k = rng.randint(1, 7)
            y = rol(x, k)
            y_bits = format(y, "08b")
            think = f"x = {x_bits}\ny = {y_bits}\n\nCheck each rotation of x:\n"
            for r in range(1, 8):
                rotated = format(rol(x, r), "08b")
                match = "→ MATCH MATCH" if rotated == y_bits else ""
                think += f"  rol{r}: {rotated} {match}\n"
            think += f"\ny is x rotated left by {k}"
            answer = f"Yes — rol{k}"
        else:
            # Make y NOT a rotation (different popcount guarantees this)
            y = rng.randint(0, 255)
            while bin(y).count('1') == bin(x).count('1'):
                y = rng.randint(0, 255)
            y_bits = format(y, "08b")
            x_ones = bin(x).count('1')
            y_ones = bin(y).count('1')
            think = f"x = {x_bits} ({x_ones} ones)\ny = {y_bits} ({y_ones} ones)\n"
            think += f"Different ones count ({x_ones} vs {y_ones}) → cannot be a rotation"
            answer = "No — different ones count"

        return {
            "user": f"Is {y_bits} a rotated version of {x_bits}?",
            "think": think,
            "answer": answer,
        }


@register
class BitNibbleView(MicroSkill):
    name = "bit_nibble_view"
    puzzle_type = "bit_manipulation"
    description = "Split byte into nibbles. What changed between input and output nibbles?"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_nibble.jsonl"
    weight = 2.0
    max_pool = 5000

    def generate_one(self, rng, difficulty="medium"):
        x = rng.randint(0, 255)
        # Apply a simple operation
        op = rng.choice(list(SHIFT_OPS.keys()))
        y = SHIFT_OPS[op](x)

        x_bits = format(x, "08b")
        y_bits = format(y, "08b")
        x_hi, x_lo = x_bits[:4], x_bits[4:]
        y_hi, y_lo = y_bits[:4], y_bits[4:]

        think = f"Input:  {x_bits} = [{x_hi}][{x_lo}]\n"
        think += f"Output: {y_bits} = [{y_hi}][{y_lo}]\n\n"
        think += f"High nibble: {x_hi} → {y_hi} {'(same)' if x_hi == y_hi else '(changed)'}\n"
        think += f"Low nibble:  {x_lo} → {y_lo} {'(same)' if x_lo == y_lo else '(changed)'}"

        hi_same = x_hi == y_hi
        lo_same = x_lo == y_lo
        if hi_same and lo_same:
            answer = "Both nibbles unchanged"
        elif hi_same:
            answer = "Only low nibble changed"
        elif lo_same:
            answer = "Only high nibble changed"
        else:
            answer = "Both nibbles changed"

        return {
            "user": f"Split into nibbles. What changed?\n  Input:  {x_bits}\n  Output: {y_bits}",
            "think": think,
            "answer": answer,
        }


@register
class BitComplementView(MicroSkill):
    name = "bit_complement_view"
    puzzle_type = "bit_manipulation"
    description = "What's the NOT/complement? Is the output the complement of a shifted input?"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_complement.jsonl"
    weight = 2.0
    max_pool = 5000

    def generate_one(self, rng, difficulty="medium"):
        qtype = rng.choice(["compute_not", "is_complement"])

        if qtype == "compute_not":
            x = rng.randint(0, 255)
            bits = format(x, "08b")
            comp = format((~x) & 0xFF, "08b")
            think = f"NOT({bits}):\n  {' '.join(bits)}\n  {' '.join(comp)}  (flip every bit)"
            return {
                "user": f"What is NOT({bits})?",
                "think": think,
                "answer": comp,
            }
        else:
            x = rng.randint(1, 254)
            x_bits = format(x, "08b")
            is_comp = rng.random() < 0.5
            if is_comp:
                y = (~x) & 0xFF
                y_bits = format(y, "08b")
                think = f"{x_bits}\n{y_bits}\nEvery bit is flipped → yes, complement"
                answer = "Yes"
            else:
                y = rng.randint(0, 255)
                y_bits = format(y, "08b")
                comp = format((~x) & 0xFF, "08b")
                matches = sum(1 for i in range(8) if y_bits[i] == comp[i])
                think = f"{x_bits}\n{y_bits}\nNOT(x) would be {comp}\n{y_bits} vs {comp}: {matches}/8 match → {'yes' if matches == 8 else 'no'}"
                answer = "Yes" if y == (~x) & 0xFF else "No"

            return {
                "user": f"Is {y_bits} the complement of {x_bits}?",
                "think": think,
                "answer": answer,
            }


# Update weights
_WEIGHTS.update({
    "bit_where_ones": 3.0,
    "bit_and_across": 4.0,
    "bit_or_across": 4.0,
    "bit_constant_vs_variable": 4.0,
    "bit_is_rotation": 3.0,
    "bit_nibble_view": 2.0,
    "bit_complement_view": 2.0,
})
for _name, _weight in _WEIGHTS.items():
    if _name in _REG:
        _REG[_name].weight = _weight


# ============================================================
# BIT: TARGET THE TWO CORE FAILURES
# ============================================================

@register
class BitGridVsExpected(MicroSkill):
    name = "bit_grid_vs_expected"
    puzzle_type = "bit_manipulation"
    description = "You computed X in the grid. The example says Y. Do they match? FORCES honest comparison."
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_grid_vs.jsonl"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        sa, sb = rng.sample(src_names, 2)
        gate = rng.choice(list(GATE_OPS.keys()))

        x = rng.randint(1, 254)
        bits = format(x, "08b")
        a = SHIFT_OPS[sa](x)
        b = SHIFT_OPS[sb](x)
        computed = GATE_OPS[gate](a, b)
        computed_bits = format(computed, "08b")

        # Sometimes the "expected" matches, sometimes it doesn't
        matches = rng.random() < 0.5
        if matches:
            expected_bits = computed_bits
        else:
            # Flip 1-2 bits to create a near-miss
            expected = computed
            for _ in range(rng.randint(1, 2)):
                expected ^= (1 << rng.randint(0, 7))
            expected_bits = format(expected & 0xFF, "08b")

        a_bits = format(a, "08b")
        b_bits = format(b, "08b")

        # Show the computation grid
        _, g_lines = gate_position_by_position(a_bits, b_bits, gate)

        think = f"Rule: A={sa}, B={sb}, {gate}\n"
        think += f"x = {bits}\n"
        think += f"A = {a_bits}\n"
        think += f"B = {b_bits}\n"
        think += f"{gate}:\n" + "\n".join(g_lines) + "\n\n"
        think += f"Grid result:    {computed_bits}\n"
        think += f"Expected output: {expected_bits}\n\n"

        if matches:
            think += f"Compare position by position:\n"
            think += f"  {' '.join(computed_bits)}\n"
            think += f"  {' '.join(expected_bits)}\n"
            think += f"  {''.join('→ MATCH' for _ in range(8))}\n"
            think += f"ALL MATCH → MATCH — rule is correct for this example"
            answer = "Match → MATCH"
        else:
            diff_positions = [i for i in range(8) if computed_bits[i] != expected_bits[i]]
            think += f"Compare position by position:\n"
            think += f"  {' '.join(computed_bits)}\n"
            think += f"  {' '.join(expected_bits)}\n"
            think += f"  {''.join('→ MATCH' if computed_bits[i] == expected_bits[i] else '→ MISMATCH' for i in range(8))}\n"
            think += f"MISMATCH at positions {diff_positions} → MISMATCH — rule is WRONG"
            answer = f"Mismatch → MISMATCH at {diff_positions}"

        return {
            "user": f"You applied {gate} with A={sa}, B={sb} on x={bits}.\nYour grid computed: {computed_bits}\nThe example output is: {expected_bits}\nDo they match?",
            "think": think,
            "answer": answer,
        }


@register
class BitHonestRuleTest(MicroSkill):
    name = "bit_honest_rule_test"
    puzzle_type = "bit_manipulation"
    description = "Compute 3 candidate rules step by step. Compare EACH to example. Which actually matches?"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_honest_test.jsonl"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        gate_names = list(GATE_OPS.keys())

        # Pick the real rule and compute the real output
        sa, sb = rng.sample(src_names, 2)
        gate = rng.choice(gate_names)
        x = rng.randint(1, 254)
        bits = format(x, "08b")
        real_output = format(GATE_OPS[gate](SHIFT_OPS[sa](x), SHIFT_OPS[sb](x)), "08b")

        # Build 3 candidate rules: 1 correct + 2 wrong
        rules = [(sa, sb, gate)]
        for _ in range(2):
            wa, wb = rng.sample(src_names, 2)
            wg = rng.choice(gate_names)
            rules.append((wa, wb, wg))
        rng.shuffle(rules)

        labels = ["A", "B", "C"]
        correct_label = None

        think_lines = [f"Input: {bits}, Expected output: {real_output}\n"]

        for j, (ra, rb, rg) in enumerate(rules):
            a = SHIFT_OPS[ra](x)
            b = SHIFT_OPS[rb](x)
            out = GATE_OPS[rg](a, b)
            a_bits = format(a, "08b")
            b_bits = format(b, "08b")
            out_bits = format(out, "08b")

            _, g_lines = gate_position_by_position(a_bits, b_bits, rg)

            think_lines.append(f"{labels[j]}) A={ra}, B={rb}, {rg}")
            think_lines.append(f"  A={a_bits}, B={b_bits}")
            think_lines.append(f"  {rg}:")
            think_lines.extend(f"  {l}" for l in g_lines)
            think_lines.append(f"  Result: {out_bits}")

            # HONEST comparison
            if out_bits == real_output:
                think_lines.append(f"  vs expected {real_output}: MATCH → MATCH")
                correct_label = labels[j]
            else:
                diff = sum(1 for i in range(8) if out_bits[i] != real_output[i])
                think_lines.append(f"  vs expected {real_output}: {diff} bits wrong → MISMATCH")
            think_lines.append("")

        if correct_label is None:
            # None matched (all wrong) — rare but possible
            think_lines.append("None of the rules match this example.")
            answer = "None"
        else:
            think_lines.append(f"Rule {correct_label} matches.")
            answer = correct_label

        options = "\n".join(f"  {labels[j]}) A={rules[j][0]}, B={rules[j][1]}, {rules[j][2]}" for j in range(3))

        return {
            "user": f"Input: {bits}. Output: {real_output}.\nCompute EACH rule and compare honestly:\n{options}",
            "think": "\n".join(think_lines),
            "answer": answer,
        }


_WEIGHTS.update({
    "bit_grid_vs_expected": 5.0,
    "bit_honest_rule_test": 5.0,
})
for _name, _weight in _WEIGHTS.items():
    if _name in _REG:
        _REG[_name].weight = _weight


@register
class BitJustChooseRule(MicroSkill):
    name = "bit_just_choose_rule"
    puzzle_type = "bit_manipulation"
    description = "Examples + 3 rules with precomputed outputs. Just CHOOSE. No computation hiding."
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_just_choose.jsonl"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        src_names = list(SHIFT_OPS.keys())
        gate_names = list(GATE_OPS.keys())

        # Pick the correct rule
        sa, sb = rng.sample(src_names, 2)
        gate = rng.choice(gate_names)
        def correct_fn(x):
            return GATE_OPS[gate](SHIFT_OPS[sa](x), SHIFT_OPS[sb](x))

        # Generate 3 examples
        inputs = rng.sample(range(256), 3)
        examples = [(format(x, "08b"), format(correct_fn(x), "08b")) for x in inputs]

        # Build 3 rules: 1 correct + 2 wrong
        rules = [(sa, sb, gate, correct_fn)]
        for _ in range(2):
            wa, wb = rng.sample(src_names, 2)
            wg = rng.choice(gate_names)
            def wrong_fn(x, wa=wa, wb=wb, wg=wg):
                return GATE_OPS[wg](SHIFT_OPS[wa](x), SHIFT_OPS[wb](x))
            rules.append((wa, wb, wg, wrong_fn))
        rng.shuffle(rules)

        labels = ["A", "B", "C"]
        correct_label = None

        # Precompute outputs for each rule on each example
        rule_lines = []
        think_lines = []
        for j, (ra, rb, rg, fn) in enumerate(rules):
            outputs = [format(fn(x), "08b") for x in inputs]
            rule_lines.append(f"  {labels[j]}) {ra} {rg} {rb} gives: {', '.join(outputs)}")

            # Compare to actual
            match_count = sum(1 for k in range(3) if outputs[k] == examples[k][1])
            if match_count == 3:
                think_lines.append(f"{labels[j]}) {', '.join(outputs)} vs {', '.join(e[1] for e in examples)} → ALL MATCH → MATCH")
                correct_label = labels[j]
            else:
                think_lines.append(f"{labels[j]}) {', '.join(outputs)} vs {', '.join(e[1] for e in examples)} → {match_count}/3 → MISMATCH")

        if correct_label is None:
            return None

        ex_str = "\n".join(f"  {inp} -> {out}" for inp, out in examples)

        return {
            "user": f"Examples:\n{ex_str}\n\nPrecomputed outputs for each rule:\n" + "\n".join(rule_lines) + "\n\nWhich rule matches ALL examples?",
            "think": "\n".join(think_lines),
            "answer": correct_label,
        }


_WEIGHTS["bit_just_choose_rule"] = 5.0
if "bit_just_choose_rule" in _REG:
    _REG["bit_just_choose_rule"].weight = 5.0


# ============================================================
# DEMOTE PATTERN ANALYSIS — these teach "guess quickly" which
# conflicts with the falsification discipline we need.
# ============================================================
for _demote_name, _new_w in [
    # Pattern analysis — demoted (teach guessing, not falsification)
    ("bit_similarity", 1.0),
    ("bit_popcount", 0.5),
    ("bit_count_across_examples", 1.0),
    ("bit_spot_invariant", 1.5),
    ("bit_spot_invariant_open", 1.5),
    ("bit_invariant_checklist", 1.5),
    ("bit_compare_to_target", 1.5),
    ("bit_where_ones", 1.0),
    ("bit_and_across", 1.0),
    ("bit_or_across", 1.0),
    ("bit_constant_vs_variable", 1.0),
    # Bit skills that don't target top failures — reduce
    ("bit_complement_view", 1.0),
    ("bit_nibble_view", 1.0),
    ("bit_impossible", 1.0),
    ("bit_spot_error", 1.0),
    ("bit_visual_pattern", 1.5),
    ("bit_is_rotation", 1.5),
    # Bit skills that DO target top failures — boost
    ("bit_full_verify", 5.0),
    ("bit_confident_or_not", 4.0),
    ("bit_rule_check", 5.0),
    # 3-input family identification — model is 2% on these, max priority
    ("bit_family3_execute", 10.0),
    ("bit_family3_verify", 10.0),
    ("bit_family3_discriminate", 10.0),
    ("bit_compose3", 8.0),           # was 3.0 — only other 3-input skill
    # Rule selection — the actual decision point
    ("bit_first_fail", 8.0),         # was 5.0
    ("bit_survivor_set", 8.0),       # was 5.0
    ("bit_reject_and_backtrack", 8.0),# was 5.0
    # Demote execution — model can already compute, just picks wrong rules
    ("bit_shift", 1.0),              # was 3.0
    ("bit_gate", 1.0),               # was 3.0
    ("bit_step_by_step", 0.5),       # was 1.0
    ("bit_batch_pipeline", 1.0),     # was 3.0
    # Encryption — boost length/vocab skills (34% of enc errors)
    ("enc_vocab", 3.0),              # was 1.5 — word length matching
    ("enc_pattern_fill", 3.0),       # was 1.5 — what words fit this pattern
    ("str_count", 3.0),              # was 1.0 — count characters (length check)
    ("enc_can_fit", 3.0),            # was 2.0 — does word fit with blocked letters
    ("enc_most_constrained", 3.0),   # was 2.0 — pick hardest word first
    # Transformation — boost op identification (297 numeric errors)
    ("trans_op_from_examples", 4.0), # was 2.0 — which op fits ALL examples
    ("trans_parse", 2.0),            # was 1.0 — parse equation parts
    ("trans_base", 3.0),             # was 1.5 — base identification
    ("trans_sign", 2.0),             # was 1.5 — sign conventions
]:
    _WEIGHTS[_demote_name] = _new_w
    if _demote_name in _REG:
        _REG[_demote_name].weight = _new_w


# ============================================================
# NEW: FALSIFICATION / ILLEGAL-MOVE SKILLS
# Raw bit computation helpers (framework's shift_str/gate_position_by_position
# return formatted strings with descriptions, these return raw 8-bit strings)
# ============================================================

def _raw_shift(bits, op):
    """Apply shift/rotate, return raw 8-bit string."""
    import re
    m = re.match(r'(shl|shr|rol|ror)(\d+)', op)
    if not m:
        return bits
    kind, k = m.group(1), int(m.group(2))
    if kind == "shr":
        return ("0" * k + bits)[:8]
    elif kind == "shl":
        return (bits[k:] + "0" * k)[:8]
    elif kind == "rol":
        return bits[k:] + bits[:k]
    elif kind == "ror":
        return bits[-k:] + bits[:-k]
    return bits

def _raw_gate(a, b, gate):
    """Apply gate position-by-position, return raw 8-bit string."""
    result = []
    for i in range(8):
        ai, bi = int(a[i]), int(b[i])
        if gate == "xor":
            result.append(str(ai ^ bi))
        elif gate == "xnor":
            result.append(str(1 - (ai ^ bi)))
        elif gate == "and":
            result.append(str(ai & bi))
        elif gate == "or":
            result.append(str(ai | bi))
        elif gate == "and_not":
            result.append(str(ai & (1 - bi)))
        elif gate == "or_not":
            result.append(str((1 - ai) | bi))
        else:
            result.append("0")
    return "".join(result)

def _shift_recipe(op):
    """Human-readable recipe for a shift/rotate."""
    import re
    m = re.match(r'(shl|shr|rol|ror)(\d+)', op)
    if not m:
        return op
    kind, k = m.group(1), int(m.group(2))
    recipes = {
        "shl": f"drop first {k}, append {k} zeros",
        "shr": f"prepend {k} zeros, drop last {k}",
        "rol": f"move first {k} to end",
        "ror": f"move last {k} to front",
    }
    return recipes.get(kind, op)

_GATE_DESC = {"xor": "diff→1 same→0", "xnor": "same→1 diff→0",
              "and": "both 1→1", "or": "either 1→1",
              "and_not": "A=1,B=0→1", "or_not": "A=0 or B=0→1"}

_STD_OPS = [f"{op}{k}" for op in ["shl","shr","rol","ror"] for k in range(1,5)]
_STD_GATES = ["xor", "and", "or", "xnor", "and_not", "or_not"]


def _make_examples(rng, a_op, b_op, gate, n=5):
    """Generate n (input, output) examples for a rule."""
    inputs = list(set(rng.randint(0, 255) for _ in range(n + 3)))[:n]
    while len(inputs) < n:
        inputs.append(rng.randint(0, 255))
    examples = []
    for x in inputs:
        inp = format(x, '08b')
        a = _raw_shift(inp, a_op)
        b = _raw_shift(inp, b_op)
        out = _raw_gate(a, b, gate)
        examples.append((inp, out))
    return examples


@register
class BitFirstFail(MicroSkill):
    """Given a tempting rule that fits most examples, find the FIRST one that kills it."""
    name = "bit_first_fail"
    puzzle_type = "bit_manipulation"
    description = "Find the first example that falsifies a tempting candidate rule"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        a_op = rng.choice(_STD_OPS)
        b_op = rng.choice(_STD_OPS)
        gate = rng.choice(_STD_GATES)

        examples = _make_examples(rng, a_op, b_op, gate, 5)

        # Try ALL wrong gates, pick the one with best partial match
        best_wrong = None
        best_matches = -1
        best_first_fail = None
        for wg in _STD_GATES:
            if wg == gate:
                continue
            matches = 0
            first_fail = None
            for i, (inp, expected) in enumerate(examples):
                a = _raw_shift(inp, a_op)
                b = _raw_shift(inp, b_op)
                if _raw_gate(a, b, wg) == expected:
                    matches += 1
                elif first_fail is None:
                    first_fail = i
            # Want: some matches AND some failures, prefer more matches (tempting)
            if first_fail is not None and matches > 0 and matches > best_matches:
                best_wrong = wg
                best_matches = matches
                best_first_fail = first_fail

        if best_wrong is None:
            return None

        wrong_gate = best_wrong
        first_fail_idx = best_first_fail

        wrong_rule = f"A={a_op}, B={b_op}, A {wrong_gate} B"
        ex_str = "\n".join(f"  {inp} -> {out}" for inp, out in examples)

        prompt = (
            f"Someone proposes this rule: {wrong_rule}\n"
            f"Test it against each example IN ORDER. Which is the FIRST one that fails?\n\n"
            f"Examples:\n{ex_str}"
        )

        think_lines = [f"Testing: {wrong_rule}", ""]
        for i, (inp, expected) in enumerate(examples):
            a = _raw_shift(inp, a_op)
            b = _raw_shift(inp, b_op)
            wrong_out = _raw_gate(a, b, wrong_gate)
            ok = wrong_out == expected
            think_lines.append(f"Ex {i+1}: x={inp}")
            think_lines.append(f"  A={a_op}({inp}): {_shift_recipe(a_op)} → {a}")
            think_lines.append(f"  B={b_op}({inp}): {_shift_recipe(b_op)} → {b}")
            think_lines.append(f"  {wrong_gate}: {_GATE_DESC.get(wrong_gate, wrong_gate)}")
            think_lines.append(f"    {' '.join(a)}")
            think_lines.append(f"    {' '.join(b)}")
            think_lines.append(f"    {' '.join(wrong_out)}")
            think_lines.append(f"  = {wrong_out} vs expected {expected}  {'→ MATCH' if ok else '→ MISMATCH FAIL'}")
            if not ok:
                think_lines.append(f"\nFirst failure: Example {i+1}")
                think_lines.append(f"Rule REJECTED.")
                break

        return {"user": prompt, "think": "\n".join(think_lines), "answer": f"Example {first_fail_idx + 1}"}


@register
class BitAntiCopy(MicroSkill):
    """Prompt shows a WRONG expected output. Model must compute honestly and spot the mismatch."""
    name = "bit_anti_copy"
    puzzle_type = "bit_manipulation"
    description = "Detect that the shown expected output is WRONG — forces honest computation"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        a_op = rng.choice(_STD_OPS)
        b_op = rng.choice(_STD_OPS)
        gate = rng.choice(_STD_GATES)

        inp = format(rng.randint(0, 255), '08b')
        a = _raw_shift(inp, a_op)
        b = _raw_shift(inp, b_op)
        real_out = _raw_gate(a, b, gate)

        # Flip 1-2 bits
        wrong_list = list(real_out)
        flip_pos = rng.sample(range(8), rng.randint(1, 2))
        for p in flip_pos:
            wrong_list[p] = '1' if wrong_list[p] == '0' else '0'
        wrong_out = ''.join(wrong_list)
        if wrong_out == real_out:
            return None

        prompt = (
            f"Rule: A={a_op}, B={b_op}, A {gate} B\n"
            f"Someone claims: x={inp} → output={wrong_out}\n\n"
            f"Compute the actual output. Is the claim correct?"
        )

        think_lines = [
            f"Rule: A={a_op}, B={b_op}, A {gate} B",
            f"",
            f"x = {inp}",
            f"A = {a_op}({inp}): {_shift_recipe(a_op)} → {a}",
            f"B = {b_op}({inp}): {_shift_recipe(b_op)} → {b}",
            f"A {gate} B: {_GATE_DESC.get(gate, gate)}",
            f"  {' '.join(a)}",
            f"  {' '.join(b)}",
            f"  {' '.join(real_out)}",
            f"= {real_out}",
            f"",
            f"Claimed: {wrong_out}",
            f"Actual:  {real_out}",
        ]
        diffs = [i for i in range(8) if real_out[i] != wrong_out[i]]
        think_lines.append(f"MISMATCH at position(s) {', '.join(str(d) for d in diffs)}")
        think_lines.append(f"The claim is WRONG.")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": f"WRONG — actual is {real_out}"}


@register
class BitSurvivorSet(MicroSkill):
    """Start with 3 candidate rules, eliminate after each example. One survives."""
    name = "bit_survivor_set"
    puzzle_type = "bit_manipulation"
    description = "Candidate elimination: test 3 rules against examples, reject as they fail"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        a_op = rng.choice(_STD_OPS)
        b_op = rng.choice(_STD_OPS)
        real_gate = rng.choice(_STD_GATES)
        other_gates = [g for g in _STD_GATES if g != real_gate]
        wrong1, wrong2 = rng.sample(other_gates, 2)

        examples = _make_examples(rng, a_op, b_op, real_gate, 5)

        # Assign gates to labels and shuffle
        combined = list(zip(["A", "B", "C"], [wrong1, real_gate, wrong2]))
        rng.shuffle(combined)
        labels_s, gates_s = zip(*combined)

        # Check wrong rules actually fail somewhere
        first_fails = {}
        for label, g in zip(labels_s, gates_s):
            for i, (inp, expected) in enumerate(examples):
                computed = _raw_gate(_raw_shift(inp, a_op), _raw_shift(inp, b_op), g)
                if computed != expected:
                    first_fails[label] = i
                    break

        survivor = [l for l, g in zip(labels_s, gates_s) if l not in first_fails]
        if len(survivor) != 1 or len(first_fails) < 2:
            return None
        survivor = survivor[0]

        rule_strs = [f"  {l}) A={a_op}, B={b_op}, A {g} B" for l, g in zip(labels_s, gates_s)]
        ex_str = "\n".join(f"  {inp} -> {out}" for inp, out in examples)

        prompt = (
            f"Three candidate rules:\n" + "\n".join(rule_strs) + "\n\n"
            f"Examples:\n{ex_str}\n\n"
            f"Test each rule against examples in order. Eliminate on first failure. Which survives?"
        )

        think_lines = [f"Sources: A={a_op}, B={b_op}", ""]
        alive = set(labels_s)

        for i, (inp, expected) in enumerate(examples):
            think_lines.append(f"Example {i+1}: x={inp} → expected {expected}")
            a_val = _raw_shift(inp, a_op)
            b_val = _raw_shift(inp, b_op)

            for label, g in zip(labels_s, gates_s):
                if label not in alive:
                    continue
                computed = _raw_gate(a_val, b_val, g)
                ok = computed == expected
                think_lines.append(f"  {label}) {g}({a_val},{b_val}) = {computed}  {'→ MATCH' if ok else '→ MISMATCH ELIMINATED'}")
                if not ok:
                    alive.discard(label)

            if len(alive) == 1:
                think_lines.append(f"\nOnly {list(alive)[0]} survives. Done.")
                break
            think_lines.append(f"  Alive: {', '.join(sorted(alive))}")
            think_lines.append("")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": survivor}


@register
class BitRejectAndBacktrack(MicroSkill):
    """Model checks a rule, finds it fails, explicitly rejects, tries the next."""
    name = "bit_reject_and_backtrack"
    puzzle_type = "bit_manipulation"
    description = "Check → fail → reject → try next. The core falsification loop."
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        a_op = rng.choice(_STD_OPS)
        b_op = rng.choice(_STD_OPS)
        real_gate = rng.choice(_STD_GATES)
        wrong_gate = rng.choice([g for g in _STD_GATES if g != real_gate])

        examples = _make_examples(rng, a_op, b_op, real_gate, 4)

        # Wrong rule must fail on at least one
        first_wrong_fail = None
        for i, (inp, expected) in enumerate(examples):
            computed = _raw_gate(_raw_shift(inp, a_op), _raw_shift(inp, b_op), wrong_gate)
            if computed != expected and first_wrong_fail is None:
                first_wrong_fail = i

        if first_wrong_fail is None:
            return None

        ex_str = "\n".join(f"  {inp} -> {out}" for inp, out in examples)
        prompt = (
            f"Rule uses A={a_op}, B={b_op}, gate=?\n"
            f"Try {wrong_gate} first. If it fails, try {real_gate}.\n\n"
            f"Examples:\n{ex_str}"
        )

        think_lines = [f"Sources: A={a_op}, B={b_op}", ""]
        think_lines.append(f"Attempt 1: gate = {wrong_gate}")

        for i, (inp, expected) in enumerate(examples):
            a = _raw_shift(inp, a_op)
            b = _raw_shift(inp, b_op)
            computed = _raw_gate(a, b, wrong_gate)
            ok = computed == expected
            think_lines.append(f"  Ex {i+1}: {wrong_gate}({a},{b}) = {computed} vs {expected} {'→ MATCH' if ok else '→ MISMATCH'}")
            if not ok:
                think_lines.append(f"  FAIL at example {i+1}. Reject {wrong_gate}.")
                think_lines.append("")
                break

        think_lines.append(f"Attempt 2: gate = {real_gate}")
        all_ok = True
        for i, (inp, expected) in enumerate(examples):
            a = _raw_shift(inp, a_op)
            b = _raw_shift(inp, b_op)
            computed = _raw_gate(a, b, real_gate)
            ok = computed == expected
            think_lines.append(f"  Ex {i+1}: {real_gate}({a},{b}) = {computed} vs {expected} {'→ MATCH' if ok else '→ MISMATCH'}")
            if not ok:
                all_ok = False

        if not all_ok:
            return None

        think_lines.append(f"ALL MATCH. Gate = {real_gate}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": real_gate}


# ============================================================
# 3-INPUT FAMILY SKILLS — model is 2% on these, need direct practice
# ============================================================

ALL_FAMILIES = ['or_xnor', 'gated_xnor_nand', 'ch', 'maj', 'tt121', 't1']
FAMILY_DESCRIPTIONS = {
    'or_xnor': 'C | XNOR(A,B): where A=B → 1, where A≠B → C',
    'gated_xnor_nand': '(C|XNOR(A,B)) & NAND(A,B,C): like or_xnor but 111→0',
    'ch': 'where A=1: B, where A=0: C (choose)',
    'maj': 'majority: 2+ inputs are 1 → 1',
    'tt121': 'where A=0: XNOR(B,C), where A=1: NAND(B,C)',
    't1': '~(A^B^C) | (~A & ~B & C)',
}
# Extended source range to match competition (1-7)
_STD_OPS_EXTENDED = [f"{op}{k}" for op in ["shl","shr","rol","ror"] for k in range(1,8)]

def _family_fn(a, b, c, family):
    """Compute 3-input family function for one bit position."""
    if family == 'or_xnor':
        return c | (1 - (a ^ b))
    elif family == 'gated_xnor_nand':
        return (c | (1 - (a ^ b))) & (1 - (a & b & c))
    elif family == 'ch':
        return (a & b) | ((1 - a) & c)
    elif family == 'maj':
        return (a & b) | (a & c) | (b & c)
    elif family == 'tt121':
        return ((1 - a) & (1 - (b ^ c))) | (a & (1 - (b & c)))
    elif family == 't1':
        return (1 - (a ^ b ^ c)) | ((1 - a) & (1 - b) & c)
    return 0

def _apply_family(a_str, b_str, c_str, family):
    """Apply 3-input family position by position."""
    return ''.join(str(_family_fn(int(a_str[i]), int(b_str[i]), int(c_str[i]), family) & 1) for i in range(8))


@register
class BitFamily3Execute(MicroSkill):
    """Execute a 3-input family — no description given, just family name and inputs.
    Model must know what each family computes from training, not from a description."""
    name = "bit_family3_execute"
    puzzle_type = "bit_manipulation"
    description = "Compute 3-input family from name alone (no description crutch)"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        family = rng.choice(ALL_FAMILIES)
        ops = _STD_OPS_EXTENDED
        a_op, b_op, c_op = rng.sample(ops, 3)

        x = rng.randint(0, 255)
        x_str = format(x, '08b')
        a = _raw_shift(x_str, a_op)
        b = _raw_shift(x_str, b_op)
        c = _raw_shift(x_str, c_op)
        result = _apply_family(a, b, c, family)

        # No description — model must know what the family computes
        templates = [
            f"x={x_str}\nA={a_op}({x_str})={a}\nB={b_op}({x_str})={b}\nC={c_op}({x_str})={c}\n\nCompute {family}(A,B,C).",
            f"Given A={a}, B={b}, C={c}\nWhat is {family}(A,B,C)?",
            f"A={a} B={b} C={c}\nApply {family} to (A,B,C). Output?",
            f"Sources: A={a}, B={b}, C={c}\n{family}(A,B,C) = ?",
        ]
        prompt = rng.choice(templates)

        think_lines = [
            f"GRID(A,B,C,{family}):",
            f"{' '.join(a)}",
            f"{' '.join(b)}",
            f"{' '.join(c)}",
            f"{' '.join(result)}",
            f"={result}",
        ]

        return {"user": prompt, "think": "\n".join(think_lines), "answer": result}


@register
class BitFamily3FromExamples(MicroSkill):
    """The real inference task: given sources + examples, identify which family fits.
    Tests ALL 6 families, rejects on first failure, exactly like inference."""
    name = "bit_family3_verify"  # keep old name for weight compatibility
    puzzle_type = "bit_manipulation"
    description = "Given sources and examples, test all 6 families, find the one that fits"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        real_family = rng.choice(ALL_FAMILIES)
        ops = _STD_OPS_EXTENDED
        a_op, b_op, c_op = rng.sample(ops, 3)

        # Generate 3 examples
        examples = []
        for _ in range(3):
            x = rng.randint(0, 255)
            x_str = format(x, '08b')
            a = _raw_shift(x_str, a_op)
            b = _raw_shift(x_str, b_op)
            c = _raw_shift(x_str, c_op)
            out = _apply_family(a, b, c, real_family)
            examples.append((x_str, a, b, c, out))

        # Present like inference: sources + examples, no family name
        example_block = "\n".join(f"  {x_str} → {out}" for x_str, _, _, _, out in examples)
        suffixes = [
            "Test each family against all examples. Which one fits?",
            "Which of the 6 families (or_xnor, gated_xnor_nand, ch, maj, tt121, t1) matches?",
            "Identify the family. Reject any that fail on any example.",
            "Try all families. Which survives all examples?",
        ]
        prompt = (
            f"Sources: A={a_op}, B={b_op}, C={c_op}\n"
            f"Examples:\n{example_block}\n\n{rng.choice(suffixes)}"
        )

        # Trace: test each family, reject on first failure
        think_lines = []
        families_to_test = list(ALL_FAMILIES)
        rng.shuffle(families_to_test)

        for fam in families_to_test:
            think_lines.append(f"Try {fam}:")
            all_ok = True
            for x_str, a, b, c, expected in examples:
                computed = _apply_family(a, b, c, fam)
                ok = computed == expected
                think_lines.append(f"  {x_str}: {computed} vs {expected} {'→ MATCH' if ok else '→ MISMATCH'}")
                if not ok:
                    think_lines.append(f"  REJECT")
                    all_ok = False
                    break
            if all_ok:
                think_lines.append(f"  ALL MATCH → MATCH → {fam}")
                break
            think_lines.append("")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": real_family}


@register
class BitFamily3TwoSurvivors(MicroSkill):
    """Two families both match example 1. Which one fails on example 2?
    This is the critical disambiguation step at inference."""
    name = "bit_family3_discriminate"  # keep old name for weight compatibility
    puzzle_type = "bit_manipulation"
    description = "Two families pass one example — find which fails on the next"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        ops = _STD_OPS_EXTENDED
        a_op, b_op, c_op = rng.sample(ops, 3)

        # Find two families that agree on one input but disagree on another
        for _ in range(100):
            real_fam = rng.choice(ALL_FAMILIES)
            wrong_fam = rng.choice([f for f in ALL_FAMILIES if f != real_fam])

            x1 = rng.randint(0, 255)
            x1_str = format(x1, '08b')
            a1, b1, c1 = _raw_shift(x1_str, a_op), _raw_shift(x1_str, b_op), _raw_shift(x1_str, c_op)
            out_real_1 = _apply_family(a1, b1, c1, real_fam)
            out_wrong_1 = _apply_family(a1, b1, c1, wrong_fam)

            if out_real_1 != out_wrong_1:
                continue  # need them to AGREE on example 1

            # They agree on example 1. Find example 2 where they disagree.
            for _ in range(20):
                x2 = rng.randint(0, 255)
                x2_str = format(x2, '08b')
                a2, b2, c2 = _raw_shift(x2_str, a_op), _raw_shift(x2_str, b_op), _raw_shift(x2_str, c_op)
                out_real_2 = _apply_family(a2, b2, c2, real_fam)
                out_wrong_2 = _apply_family(a2, b2, c2, wrong_fam)

                if out_real_2 != out_wrong_2:
                    # Found it
                    prompt = (
                        f"Sources: A={a_op}, B={b_op}, C={c_op}\n\n"
                        f"Example 1: {x1_str} → {out_real_1}\n"
                        f"Both {real_fam} and {wrong_fam} produce {out_real_1} on this input.\n\n"
                        f"Example 2: {x2_str} → {out_real_2}\n"
                        f"Which family survives?"
                    )

                    think_lines = [
                        f"Test example 2: x={x2_str}",
                        f"A={a_op}({x2_str})={a2}",
                        f"B={b_op}({x2_str})={b2}",
                        f"C={c_op}({x2_str})={c2}",
                        f"",
                        f"{real_fam}: {out_real_2} vs expected {out_real_2} → MATCH",
                        f"{wrong_fam}: {out_wrong_2} vs expected {out_real_2} → MISMATCH REJECT",
                        f"",
                        f"Survivor: {real_fam}",
                    ]

                    return {"user": prompt, "think": "\n".join(think_lines), "answer": real_fam}

        return None  # couldn't find agreeing pair


@register
class BitWhatProducedThis(MicroSkill):
    """Given A, B, output — is output = A&B, A|B, A^B, A&~B, or none?"""
    name = "bit_what_produced"
    puzzle_type = "bit_manipulation"
    description = "Identify which operation produced the output from given inputs"
    weight = 8.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        # Pick two random 8-bit values
        a = format(rng.randint(0, 255), '08b')
        b = format(rng.randint(0, 255), '08b')

        # Pick the real operation
        ops = {
            'A & B': _raw_gate(a, b, 'and'),
            'A | B': _raw_gate(a, b, 'or'),
            'A ^ B': _raw_gate(a, b, 'xor'),
            'A & ~B': _raw_gate(a, b, 'and_not'),
            'XNOR(A,B)': _raw_gate(a, b, 'xnor'),
        }
        real_op = rng.choice(list(ops.keys()))
        output = ops[real_op]

        # Present all options
        labels = list("ABCDE")
        options = list(ops.keys())
        rng.shuffle(options)
        correct_label = labels[options.index(real_op)]

        opt_lines = []
        for label, op in zip(labels, options):
            result = ops[op]
            opt_lines.append(f"  {label}) {op} = {result}")

        prompt = (
            f"A = {a}\n"
            f"B = {b}\n"
            f"Output = {output}\n\n"
            f"Which operation produced this output?\n"
            + "\n".join(opt_lines)
        )

        think_lines = [f"Output = {output}", ""]
        for label, op in zip(labels, options):
            result = ops[op]
            match = result == output
            think_lines.append(f"{label}) {op} = {result} {'→ MATCH MATCH' if match else '→ MISMATCH'}")

        think_lines.append(f"\nAnswer: {correct_label}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": correct_label}


@register
class BitFamilyPermutation(MicroSkill):
    """Does family(A,B,C) give the same as family(B,A,C)? Teaches permutation awareness."""
    name = "bit_family_permutation"
    puzzle_type = "bit_manipulation"
    description = "Check if reordering sources changes the family output"
    weight = 8.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        family = rng.choice(ALL_FAMILIES)
        ops = _STD_OPS_EXTENDED
        a_op, b_op, c_op = rng.sample(ops, 3)

        x = rng.randint(0, 255)
        x_str = format(x, '08b')
        a = _raw_shift(x_str, a_op)
        b = _raw_shift(x_str, b_op)
        c = _raw_shift(x_str, c_op)

        # Original order
        out_abc = _apply_family(a, b, c, family)

        # Pick a permutation
        perms = [
            ('A,B,C', a, b, c),
            ('B,A,C', b, a, c),
            ('A,C,B', a, c, b),
            ('C,B,A', c, b, a),
            ('B,C,A', b, c, a),
            ('C,A,B', c, a, b),
        ]

        perm_name, pa, pb, pc = rng.choice(perms[1:])  # skip original
        out_perm = _apply_family(pa, pb, pc, family)

        same = out_abc == out_perm

        prompt = (
            f"Family: {family}\n"
            f"Sources: A={a_op}, B={b_op}, C={c_op}\n"
            f"x={x_str}\n\n"
            f"A={a}, B={b}, C={c}\n\n"
            f"Original order (A,B,C): output = {out_abc}\n"
            f"Reordered ({perm_name}): output = ?\n\n"
            f"Does reordering change the output?"
        )

        think_lines = [
            f"{family} with original (A,B,C): {out_abc}",
            f"{family} with ({perm_name}): computing...",
            f"GRID({perm_name},{family}):",
            f"{' '.join(pa)}",
            f"{' '.join(pb)}",
            f"{' '.join(pc)}",
            f"{' '.join(out_perm)}",
            f"={out_perm}",
            f"",
            f"{'Same → MATCH' if same else 'Different → MISMATCH'}: {out_abc} vs {out_perm}",
        ]

        answer = f"{'Same' if same else 'Different'}: {out_perm}"
        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


@register
class BitFullPipeline3(MicroSkill):
    """The complete inference pipeline as a micro-skill: sources given, 3 examples,
    identify family from ALL candidates, verify, compute query. Closest to inference."""
    name = "bit_how_many_sources"  # reuse name for weight compat
    puzzle_type = "bit_manipulation"
    description = "Full 3-input pipeline: test all families against examples, pick winner, compute query"
    weight = 6.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        real_family = rng.choice(ALL_FAMILIES)
        ops = _STD_OPS_EXTENDED
        a_op, b_op, c_op = rng.sample(ops, 3)

        # Generate 3 check examples + 1 query
        inputs = [rng.randint(0, 255) for _ in range(4)]
        examples = []
        for x in inputs[:3]:
            x_str = format(x, '08b')
            a = _raw_shift(x_str, a_op)
            b = _raw_shift(x_str, b_op)
            c = _raw_shift(x_str, c_op)
            out = _apply_family(a, b, c, real_family)
            examples.append((x_str, a, b, c, out))

        query_x = format(inputs[3], '08b')
        qa = _raw_shift(query_x, a_op)
        qb = _raw_shift(query_x, b_op)
        qc = _raw_shift(query_x, c_op)
        answer = _apply_family(qa, qb, qc, real_family)

        # Prompt: looks like inference (sources given but no family)
        prompt = (
            f"Sources: A={a_op}, B={b_op}, C={c_op}\n"
            f"Examples:\n"
        )
        for x_str, _, _, _, out in examples:
            prompt += f"  {x_str} → {out}\n"
        prompt += f"\nQuery: {query_x} → ?"

        # Trace: test families, find winner, compute query
        think_lines = []

        # Test each family on example 1 first
        survivors = []
        families_order = list(ALL_FAMILIES)
        rng.shuffle(families_order)

        think_lines.append("Test families on example 1:")
        x1, a1, b1, c1, exp1 = examples[0]
        for fam in families_order:
            computed = _apply_family(a1, b1, c1, fam)
            ok = computed == exp1
            think_lines.append(f"  {fam}: {computed} {'→ MATCH' if ok else '→ MISMATCH'}")
            if ok:
                survivors.append(fam)

        # Narrow with example 2
        if len(survivors) > 1:
            think_lines.append(f"\nSurvivors: {', '.join(survivors)}")
            think_lines.append(f"Test on example 2:")
            x2, a2, b2, c2, exp2 = examples[1]
            new_survivors = []
            for fam in survivors:
                computed = _apply_family(a2, b2, c2, fam)
                ok = computed == exp2
                think_lines.append(f"  {fam}: {computed} {'→ MATCH' if ok else '→ MISMATCH'}")
                if ok:
                    new_survivors.append(fam)
            survivors = new_survivors

        if len(survivors) == 1:
            think_lines.append(f"\nFamily: {survivors[0]}")
        elif len(survivors) > 1:
            think_lines.append(f"\nMultiple match, picking: {real_family}")
        else:
            think_lines.append(f"\nNone matched (error)")
            return None

        # Compute query
        think_lines.append(f"\nQuery: x={query_x}")
        think_lines.append(f"A={a_op}({query_x})={qa}")
        think_lines.append(f"B={b_op}({query_x})={qb}")
        think_lines.append(f"C={c_op}({query_x})={qc}")
        chosen = survivors[0] if len(survivors) == 1 else real_family
        think_lines.append(f"GRID(A,B,C,{chosen}):")
        think_lines.append(f"{' '.join(qa)}")
        think_lines.append(f"{' '.join(qb)}")
        think_lines.append(f"{' '.join(qc)}")
        think_lines.append(f"{' '.join(answer)}")
        think_lines.append(f"={answer}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


@register
class BitCompositionIdentify(MicroSkill):
    """Given A, B, and an output: is it A op B, just A, just B, or none?
    Teaches the model to distinguish single-source vs two-source vs other."""
    name = "bit_compose_identify"
    puzzle_type = "bit_manipulation"
    description = "Identify whether output came from A op B, just A, just B, or none of these"
    weight = 6.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        a = format(rng.randint(0, 255), '08b')
        b = format(rng.randint(0, 255), '08b')

        # Pick what the output actually is
        options = {
            'A & B': _raw_gate(a, b, 'and'),
            'A | B': _raw_gate(a, b, 'or'),
            'A ^ B': _raw_gate(a, b, 'xor'),
            'just A': a,
            'just B': b,
            '~A': ''.join('1' if c == '0' else '0' for c in a),
        }

        real_key = rng.choice(list(options.keys()))
        output = options[real_key]

        # Build multiple-choice options (always include the real one + 3 distractors)
        all_keys = list(options.keys())
        distractors = [k for k in all_keys if k != real_key]
        rng.shuffle(distractors)
        choices = [real_key] + distractors[:3]
        rng.shuffle(choices)

        prompt = (
            f"A = {a}\n"
            f"B = {b}\n"
            f"Output = {output}\n\n"
            f"Which produced the output?\n"
        )
        for i, ch in enumerate(choices):
            prompt += f"  ({chr(65+i)}) {ch}\n"

        # Trace: test each option
        think_lines = []
        answer_letter = None
        for i, ch in enumerate(choices):
            expected = options[ch]
            match = expected == output
            think_lines.append(f"({chr(65+i)}) {ch} = {expected} {'→ MATCH' if match else '→ MISMATCH'}")
            if match:
                answer_letter = chr(65+i)

        answer_str = f"({answer_letter}) {real_key}"
        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer_str}


@register
class BitScanCompute(MicroSkill):
    """Compute the Scan preamble statistics: ones count + diff from input for each example.
    This is the exact first step of our compact trace format."""
    name = "bit_scan_compute"
    puzzle_type = "bit_manipulation"
    description = "Compute ones count and input/output diff for input-output pairs"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        n_examples = rng.randint(2, 4)
        examples = []
        for _ in range(n_examples):
            inp = format(rng.randint(0, 255), '08b')
            out = format(rng.randint(0, 255), '08b')
            examples.append((inp, out))

        prompt = "Compute ones count and diff for each pair:\n"
        for inp, out in examples:
            prompt += f"  {inp} → {out}\n"

        think_lines = []
        for inp, out in examples:
            in_ones = inp.count('1')
            out_ones = out.count('1')
            diff = out_ones - in_ones
            same = sum(1 for a, b in zip(inp, out) if a == b)
            diff_s = f"+{diff}" if diff >= 0 else str(diff)
            think_lines.append(f"{inp}→{out} ones={out_ones} diff={diff_s} same={same}/8")

        answer = "\n".join(think_lines)
        return {"user": prompt, "think": answer, "answer": answer}


@register
class BitBookendVerify(MicroSkill):
    """Given query input + computed output, verify bookend stats: ones, delta, match.
    This is the final verification step in our compact trace."""
    name = "bit_bookend_verify"
    puzzle_type = "bit_manipulation"
    description = "Verify bookend statistics (ones/delta/match) for a query result"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        query = format(rng.randint(0, 255), '08b')
        answer = format(rng.randint(0, 255), '08b')
        q_ones = query.count('1')
        a_ones = answer.count('1')
        delta = a_ones - q_ones
        match = sum(1 for a, b in zip(answer, query) if a == b)
        delta_s = f"+{delta}" if delta >= 0 else str(delta)

        prompt = (
            f"Query: x={query} ones={q_ones}\n"
            f"Computed output: {answer}\n\n"
            f"Verify: ones=? delta=? match=?/8"
        )

        think = f"ones={a_ones} delta={delta_s} match={match}/8"
        return {"user": prompt, "think": think, "answer": think}


@register
class BitTwoStepCompose(MicroSkill):
    """Apply shift, then gate with another shifted value. Two explicit steps.
    This is the core composition the model must learn for 2-source puzzles."""
    name = "bit_compose_two_step"
    puzzle_type = "bit_manipulation"
    description = "Shift two sources, then apply gate — the core 2-source pipeline"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        ops = _STD_OPS_EXTENDED
        a_op, b_op = rng.sample(ops, 2)
        gate = rng.choice(['xor', 'and', 'or', 'xnor', 'and_not'])

        x = rng.randint(0, 255)
        x_str = format(x, '08b')
        a = _raw_shift(x_str, a_op)
        b = _raw_shift(x_str, b_op)
        result = _raw_gate(a, b, gate)

        gate_display = {
            'xor': 'A ^ B', 'and': 'A & B', 'or': 'A | B',
            'xnor': '~(A ^ B)', 'and_not': 'A & ~B',
        }[gate]

        prompt = (
            f"x = {x_str}\n"
            f"Step 1: A = {a_op}(x), B = {b_op}(x)\n"
            f"Step 2: output = {gate_display}\n\n"
            f"Compute the output."
        )

        recipe_a = _shift_recipe(a_op)
        recipe_b = _shift_recipe(b_op)
        think_lines = [
            f"A = {a_op}({x_str})",
            f"  {recipe_a} → {a}",
            f"B = {b_op}({x_str})",
            f"  {recipe_b} → {b}",
            f"output = {gate_display}:",
            f"  {' '.join(a)}",
            f"  {' '.join(b)}",
            f"  {' '.join(result)}",
            f"= {result}",
        ]

        return {"user": prompt, "think": "\n".join(think_lines), "answer": result}


@register
class BitVerifyAgainstGiven(MicroSkill):
    """THE critical failure mode: model picks a rule, checks pass on self-selected
    examples, but fails on the actual prompt examples. This skill gives SPECIFIC
    input-output pairs and asks: does this rule reproduce them?

    50% of the time the rule is WRONG and the model must catch it."""
    name = "bit_verify_given"
    puzzle_type = "bit_manipulation"
    description = "Given a rule and prompt examples, verify the rule reproduces each example"
    weight = 10.0  # highest priority — targets 89.5% of bit failures
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        ops = _STD_OPS_EXTENDED
        a_op, b_op = rng.sample(ops, 2)
        real_gate = rng.choice(['xor', 'and', 'or', 'xnor', 'and_not'])

        # Generate 3 examples with the REAL rule
        examples = []
        for _ in range(3):
            x = rng.randint(0, 255)
            x_str = format(x, '08b')
            a = _raw_shift(x_str, a_op)
            b = _raw_shift(x_str, b_op)
            out = _raw_gate(a, b, real_gate)
            examples.append((x_str, out))

        # 50% of the time, propose the WRONG rule
        is_wrong = rng.random() < 0.5
        if is_wrong:
            wrong_gates = [g for g in ['xor', 'and', 'or', 'xnor', 'and_not'] if g != real_gate]
            proposed_gate = rng.choice(wrong_gates)
        else:
            proposed_gate = real_gate

        gate_names = {
            'xor': 'xor(A,B)', 'and': 'and(A,B)', 'or': 'or(A,B)',
            'xnor': 'xnor(A,B)', 'and_not': 'and_not(A,B)',
        }

        prompt = (
            f"Proposed rule: A={a_op}, B={b_op}, output={gate_names[proposed_gate]}\n\n"
            f"Verify against these examples:\n"
        )
        for x_str, out in examples:
            prompt += f"  {x_str} → {out}\n"
        prompt += f"\nDoes the rule match ALL examples?"

        # Trace: compute each example with proposed rule
        think_lines = []
        all_match = True
        first_fail = None
        for x_str, expected in examples:
            a = _raw_shift(x_str, a_op)
            b = _raw_shift(x_str, b_op)
            computed = _raw_gate(a, b, proposed_gate)
            match = computed == expected
            if not match and first_fail is None:
                first_fail = x_str
                all_match = False
            think_lines.append(f"x={x_str}: A={a} B={b}")
            think_lines.append(f"  {gate_names[proposed_gate]}={computed} vs {expected} {'→ MATCH' if match else '→ MISMATCH MISMATCH'}")
            if not match:
                think_lines.append(f"  REJECT — rule does not match this example")
                break

        if all_match:
            think_lines.append(f"\nAll examples match → MATCH — rule is consistent")
            answer = "MATCH"
        else:
            think_lines.append(f"\nRule REJECTED — fails on {first_fail}")
            answer = "REJECT"

        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


@register
class BitVerifyAgainstGiven3(MicroSkill):
    """Same as BitVerifyAgainstGiven but for 3-input families."""
    name = "bit_verify_given_3"
    puzzle_type = "bit_manipulation"
    description = "Given a 3-input family and prompt examples, verify the family reproduces each"
    weight = 10.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        ops = _STD_OPS_EXTENDED
        a_op, b_op, c_op = rng.sample(ops, 3)
        real_family = rng.choice(ALL_FAMILIES)

        examples = []
        for _ in range(3):
            x = rng.randint(0, 255)
            x_str = format(x, '08b')
            a = _raw_shift(x_str, a_op)
            b = _raw_shift(x_str, b_op)
            c = _raw_shift(x_str, c_op)
            out = _apply_family(a, b, c, real_family)
            examples.append((x_str, out))

        is_wrong = rng.random() < 0.5
        if is_wrong:
            proposed = rng.choice([f for f in ALL_FAMILIES if f != real_family])
        else:
            proposed = real_family

        prompt = (
            f"Proposed: A={a_op}, B={b_op}, C={c_op}, family={proposed}\n\n"
            f"Verify against examples:\n"
        )
        for x_str, out in examples:
            prompt += f"  {x_str} → {out}\n"
        prompt += f"\nDoes the family match ALL examples?"

        think_lines = []
        all_match = True
        for x_str, expected in examples:
            a = _raw_shift(x_str, a_op)
            b = _raw_shift(x_str, b_op)
            c = _raw_shift(x_str, c_op)
            computed = _apply_family(a, b, c, proposed)
            match = computed == expected
            think_lines.append(f"x={x_str}: {proposed}({a},{b},{c})={computed} vs {expected} {'→ MATCH' if match else '→ MISMATCH'}")
            if not match:
                all_match = False
                think_lines.append(f"  REJECT")
                break

        if all_match:
            think_lines.append(f"All match → MATCH")
            answer = "MATCH"
        else:
            think_lines.append(f"Rule REJECTED")
            answer = "REJECT"

        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


# ============================================================
# NEW: ENCRYPTION — LENGTH VERIFICATION
# ============================================================

@register
class EncLengthCheck(MicroSkill):
    """Cipher word has N letters → plaintext MUST have N letters. Reject wrong-length candidates."""
    name = "enc_length_check"
    puzzle_type = "encryption"
    description = "Verify cipher word length = plaintext word length, reject mismatches"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab:
            return None

        # Pick a target word length
        target_len = rng.choice([3, 4, 5, 6, 7, 8])
        candidates = [w for w in vocab if len(w) == target_len]
        wrong_len_words = [w for w in vocab if len(w) != target_len and abs(len(w) - target_len) <= 2]

        if len(candidates) < 2 or len(wrong_len_words) < 2:
            return None

        correct = rng.choice(candidates)
        wrongs = rng.sample(wrong_len_words, min(3, len(wrong_len_words)))

        # Build a cipher string of target_len random letters
        cipher = ''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(target_len))

        options = wrongs + [correct]
        rng.shuffle(options)
        labels = list("ABCDE")[:len(options)]

        opt_lines = [f"  {l}) {w} ({len(w)} letters)" for l, w in zip(labels, options)]
        correct_label = labels[options.index(correct)]

        prompt = (
            f"Cipher word '{cipher}' has {target_len} letters.\n"
            f"Which plaintext candidate has the correct length?\n\n"
            + "\n".join(opt_lines)
        )

        think_lines = [f"Cipher '{cipher}' = {target_len} letters. Plaintext must also be {target_len} letters.", ""]
        for l, w in zip(labels, options):
            ok = len(w) == target_len
            think_lines.append(f"{l}) {w} = {len(w)} letters {'→ MATCH matches' if ok else '→ MISMATCH wrong length'}")

        think_lines.append(f"\nOnly {correct_label}) has {target_len} letters.")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": correct_label}


@register
class EncWordLengthFromCipher(MicroSkill):
    """Given cipher text, count letters per word and state required plaintext lengths."""
    name = "enc_word_lengths"
    puzzle_type = "encryption"
    description = "Count cipher word lengths — plaintext words MUST match these lengths"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab:
            return None

        # Pick 3-5 words
        n_words = rng.randint(3, 5)
        words = rng.sample([w for w in vocab if 3 <= len(w) <= 8], min(n_words, len(vocab)))
        if len(words) < n_words:
            return None

        # Make a simple substitution cipher
        alphabet = list('abcdefghijklmnopqrstuvwxyz')
        shuffled = list(alphabet)
        rng.shuffle(shuffled)
        mapping = dict(zip(alphabet, shuffled))

        cipher_words = [''.join(mapping.get(c, c) for c in w) for w in words]
        cipher_text = ' '.join(cipher_words)

        prompt = (
            f"Encrypted text: {cipher_text}\n"
            f"Count the letters in each cipher word. What lengths must the plaintext words be?"
        )

        think_lines = ["Counting cipher word lengths:", ""]
        lengths = []
        for i, cw in enumerate(cipher_words):
            think_lines.append(f"  Word {i+1}: '{cw}' = {len(cw)} letters")
            lengths.append(str(len(cw)))

        answer = ", ".join(lengths)
        think_lines.append(f"\nPlaintext words must have lengths: {answer}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


@register
class EncRejectWrongLength(MicroSkill):
    """49% of enc failures are wrong-length words. This skill shows a decrypted
    word candidate and asks: does its length match the cipher word? Reject if not."""
    name = "enc_reject_wrong_len"
    puzzle_type = "encryption"
    description = "Reject plaintext word if length doesn't match cipher word"
    weight = 6.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None

        target_len = rng.choice([3, 4, 5, 6, 7, 8])
        correct_words = [w for w in vocab if len(w) == target_len]
        wrong_words = [w for w in vocab if abs(len(w) - target_len) in [1, 2]]
        if not correct_words or not wrong_words: return None

        # 50% show a wrong candidate, 50% show a correct one
        if rng.random() < 0.5:
            candidate = rng.choice(wrong_words)
            is_wrong = True
        else:
            candidate = rng.choice(correct_words)
            is_wrong = False

        cipher = ''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(target_len))

        prompt = (
            f"Cipher word: '{cipher}' ({target_len} letters)\n"
            f"Proposed plaintext: '{candidate}' ({len(candidate)} letters)\n\n"
            f"Accept or reject?"
        )

        if is_wrong:
            think = (f"Cipher has {target_len} letters. "
                    f"'{candidate}' has {len(candidate)} letters. "
                    f"{len(candidate)} ≠ {target_len} → REJECT")
            answer = "REJECT"
        else:
            think = (f"Cipher has {target_len} letters. "
                    f"'{candidate}' has {len(candidate)} letters. "
                    f"{len(candidate)} = {target_len} → ACCEPT")
            answer = "ACCEPT"

        return {"user": prompt, "think": think, "answer": answer}


@register
class EncVocabFillGaps(MicroSkill):
    """20 failures have underscores (incomplete decrypt). This skill gives a
    partial word with gaps and the vocabulary, asks which word fits."""
    name = "enc_vocab_fill"
    puzzle_type = "encryption"
    description = "Given partial decryption with _ gaps, find the vocabulary word that fits"
    weight = 6.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None

        # Pick a word and blank out 1-3 letters
        word = rng.choice([w for w in vocab if len(w) >= 4])
        n_gaps = rng.randint(1, min(3, len(word) - 2))
        gap_positions = rng.sample(range(len(word)), n_gaps)
        partial = list(word)
        for p in gap_positions:
            partial[p] = '_'
        partial_str = ''.join(partial)

        # Find all vocab words that fit this pattern
        matches = []
        for w in vocab:
            if len(w) != len(word): continue
            if all(partial[i] == '_' or partial[i] == w[i] for i in range(len(w))):
                matches.append(w)

        if len(matches) == 0: return None

        prompt = (
            f"Partially decrypted word: '{partial_str}' ({len(word)} letters)\n"
            f"Known letters: {', '.join(f'pos {i}={word[i]}' for i in range(len(word)) if partial[i] != '_')}\n"
            f"Which vocabulary word fits?"
        )

        think_lines = [f"Pattern: {partial_str}"]
        think_lines.append(f"Length: {len(word)}")
        think_lines.append(f"Matches from vocabulary: {', '.join(matches[:5])}")
        if len(matches) == 1:
            think_lines.append(f"Only one match: {matches[0]}")
        else:
            think_lines.append(f"{len(matches)} candidates — need more letters to disambiguate")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": word}


@register
class EncCharByCharDecrypt(MicroSkill):
    """Decrypt char-by-char, not whole words. The language prior hijacks
    whole-word decryption. This skill forces letter-by-letter mapping lookup."""
    name = "enc_char_decrypt"
    puzzle_type = "encryption"
    description = "Decrypt a cipher word one letter at a time using a mapping table"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None

        word = rng.choice([w for w in vocab if 4 <= len(w) <= 8])
        # Create a random cipher mapping
        plain_letters = sorted(set(word))
        cipher_letters = rng.sample('abcdefghijklmnopqrstuvwxyz', len(plain_letters))
        mapping = dict(zip(plain_letters, cipher_letters))
        rev_mapping = {v: k for k, v in mapping.items()}

        cipher_word = ''.join(mapping[c] for c in word)

        # Show mapping table and ask for decryption
        table_str = ", ".join(f"{c}→{p}" for c, p in sorted(rev_mapping.items()))

        prompt = (
            f"Mapping: {table_str}\n"
            f"Decrypt: '{cipher_word}'\n"
            f"Go letter by letter."
        )

        think_lines = []
        result = []
        for i, c in enumerate(cipher_word):
            p = rev_mapping[c]
            think_lines.append(f"  {c} → {p}")
            result.append(p)

        answer = ''.join(result)
        think_lines.append(f"= {answer}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


@register
class EncBijectionReject(MicroSkill):
    """43% of enc failures are wrong vocab words (right length). Often the model
    picks a word that violates bijection — two different cipher letters mapping
    to the same plain letter. This skill shows a proposed mapping and asks
    if it violates bijection."""
    name = "enc_bijection_reject"
    puzzle_type = "encryption"
    description = "Check if a proposed cipher→plain mapping violates bijection (two-to-one)"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        # Create a valid mapping
        letters = list('abcdefghijklmnopqrstuvwxyz')
        n = rng.randint(6, 12)
        cipher_letters = rng.sample(letters, n)
        plain_letters = rng.sample(letters, n)
        mapping = dict(zip(cipher_letters, plain_letters))

        # 50% introduce a bijection violation
        if rng.random() < 0.5:
            # Pick two cipher letters and map them to the same plain letter
            if len(cipher_letters) >= 3:
                victim = rng.choice(cipher_letters[:3])
                target = rng.choice(cipher_letters[3:])
                mapping[target] = mapping[victim]  # violation!
                has_violation = True
                violating = (victim, target, mapping[victim])
            else:
                has_violation = False
                violating = None
        else:
            has_violation = False
            violating = None

        table = ", ".join(f"{c}→{p}" for c, p in sorted(mapping.items()))
        prompt = f"Mapping: {table}\nIs this a valid bijection (one-to-one)?"

        if has_violation:
            v_c1, v_c2, v_p = violating
            think = (f"Check: {v_c1}→{v_p} and {v_c2}→{v_p}. "
                    f"Two cipher letters map to '{v_p}' → VIOLATION. Not bijective.")
            answer = "VIOLATION"
        else:
            think = "Each cipher letter maps to a unique plain letter. Bijection → MATCH"
            answer = "VALID"

        return {"user": prompt, "think": think, "answer": answer}


# ============================================================
# NEW: TRANSFORMATION — OPERATOR ELIMINATION
# ============================================================

@register
class EncConfusableWords(MicroSkill):
    """Model confuses student↔strange, rabbit↔ribbon, draws↔drags etc.
    This skill presents confusable same-length vocab words and asks which
    one matches a given partial decryption pattern."""
    name = "enc_confusable"
    puzzle_type = "encryption"
    description = "Distinguish confusable same-length vocabulary words by letter pattern"
    weight = 5.0
    max_pool = 10000

    # Pairs the model actually confuses (from eval failure analysis)
    CONFUSABLE_GROUPS = [
        ["student", "strange", "studies"],
        ["rabbit", "ribbon", "riddle", "barrel"],
        ["draws", "drags", "drops"],
        ["queen", "under"],
        ["garden", "around"],
        ["dragon", "around"],
        ["silver", "silent"],
        ["village", "blanket"],
        ["explores", "embraces", "delivers"],
        ["mountain", "fountain"],
        ["book", "door"],
        ["palace", "castle"],
        ["clever", "cipher"],
        ["dreams", "desert"],
        ["watches", "catches"],
        ["princess", "treasure"],
        ["teacher", "treasure"],
    ]

    def generate_one(self, rng, difficulty="medium"):
        group = rng.choice(self.CONFUSABLE_GROUPS)
        if len(group) < 2:
            return None

        correct = rng.choice(group)
        wrong = rng.choice([w for w in group if w != correct])

        # Reveal some letters as if from a cipher mapping
        n_reveal = max(2, len(correct) // 2)
        positions = rng.sample(range(len(correct)), n_reveal)
        partial = list('_' * len(correct))
        for p in positions:
            partial[p] = correct[p]
        partial_str = ''.join(partial)

        prompt = (
            f"Partial decryption: '{partial_str}' ({len(correct)} letters)\n"
            f"Candidates: '{correct}' or '{wrong}'?\n"
            f"Which matches the revealed letters?"
        )

        think_lines = [f"Pattern: {partial_str}"]
        for p in sorted(positions):
            think_lines.append(f"  pos {p}: must be '{correct[p]}'")

        # Check wrong word against pattern
        mismatch = None
        for p in sorted(positions):
            if p < len(wrong) and wrong[p] != correct[p]:
                mismatch = p
                break

        if mismatch is not None:
            think_lines.append(f"'{wrong}' has '{wrong[mismatch]}' at pos {mismatch}, need '{correct[mismatch]}' → REJECT")
        think_lines.append(f"Answer: {correct}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": correct}


@register
class GravMagnitudeCheck(MicroSkill):
    """3 grav failures are decimal point errors (9.21→0.92, 24.48→244.82).
    Model gets right digits but wrong magnitude. This skill teaches
    order-of-magnitude sanity checking."""
    name = "grav_magnitude"
    puzzle_type = "gravitational"
    description = "Check if computed distance has the right order of magnitude"
    weight = 4.0
    max_pool = 5000

    def generate_one(self, rng, difficulty="medium"):
        # Generate a gravity problem with rate and target
        g = rng.uniform(4.0, 12.0)  # random gravity constant
        rate = g / 2  # 0.5g
        t = rng.uniform(1.0, 10.0)
        d_correct = rate * t * t

        # Create plausible wrong answers (off by factor of 10)
        d_wrong1 = d_correct * 10
        d_wrong2 = d_correct / 10

        prompt = (
            f"Rate = {rate:.4f} (from d/t²)\n"
            f"Target t = {t:.2f}\n"
            f"t² = {t*t:.4f}\n"
            f"Result = rate × t² = ?\n\n"
            f"Is the answer closer to {d_correct:.2f}, {d_wrong1:.2f}, or {d_wrong2:.2f}?"
        )

        think = (
            f"rate × t² = {rate:.4f} × {t*t:.4f} = {d_correct:.4f}\n"
            f"Rounded: {d_correct:.2f}\n"
            f"Sanity: t={t:.2f}, rate≈{rate:.1f}, so d≈{rate:.0f}×{t*t:.0f}≈{rate*t*t:.0f}"
        )

        return {"user": prompt, "think": think, "answer": f"{d_correct:.2f}"}


@register
class TransOpEliminate(MicroSkill):
    """Test candidate operators against examples, eliminate on first failure."""
    name = "trans_op_eliminate"
    puzzle_type = "transformation"
    description = "Test candidate operators against ALL examples, reject failures"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        # Simple arithmetic ops
        ops = {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "absdiff": lambda a, b: abs(a - b),
        }

        real_op_name = rng.choice(list(ops.keys()))
        real_op = ops[real_op_name]

        # Generate 3-4 examples
        n_ex = rng.randint(3, 4)
        examples = []
        for _ in range(n_ex):
            a = rng.randint(1, 50)
            b = rng.randint(1, 50)
            result = real_op(a, b)
            examples.append((a, b, result))

        # Pick 2 wrong ops
        wrong_names = rng.sample([n for n in ops if n != real_op_name], 2)

        all_ops = wrong_names + [real_op_name]
        rng.shuffle(all_ops)

        ex_str = "\n".join(f"  {a} ? {b} = {r}" for a, b, r in examples)
        op_str = ", ".join(all_ops)

        prompt = (
            f"Which operator fits ALL examples? Candidates: {op_str}\n\n"
            f"Examples:\n{ex_str}"
        )

        think_lines = ["Testing each operator against all examples:", ""]
        answer = None
        for op_name in all_ops:
            op_fn = ops[op_name]
            think_lines.append(f"Try {op_name}:")
            all_match = True
            for a, b, expected in examples:
                computed = op_fn(a, b)
                ok = computed == expected
                think_lines.append(f"  {a} {op_name} {b} = {computed} vs {expected} {'→ MATCH' if ok else '��'}")
                if not ok:
                    think_lines.append(f"  REJECT {op_name}")
                    all_match = False
                    break
            if all_match:
                think_lines.append(f"  ALL MATCH → MATCH")
                answer = op_name
            think_lines.append("")

        if answer is None:
            return None

        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


@register
class EncCascadeCheck(MicroSkill):
    """If a mapping changes, re-check all earlier resolved words for conflicts."""
    name = "enc_cascade_check"
    puzzle_type = "encryption"
    description = "Detect cascade: new mapping conflicts with earlier word, must re-resolve"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab:
            return None

        # Pick 2 words that share a letter
        target_len = rng.choice([4, 5, 6])
        candidates = [w for w in vocab if len(w) == target_len]
        if len(candidates) < 5:
            return None

        word1, word2 = rng.sample(candidates, 2)
        shared = set(word1) & set(word2)
        if not shared:
            return None

        shared_letter = rng.choice(list(shared))

        # Create a mapping
        alphabet = list('abcdefghijklmnopqrstuvwxyz')
        shuffled = list(alphabet)
        rng.shuffle(shuffled)
        mapping = dict(zip(alphabet, shuffled))

        cipher1 = ''.join(mapping[c] for c in word1)
        cipher2 = ''.join(mapping[c] for c in word2)

        # Find a wrong word2 candidate that would conflict
        wrong_candidates = [w for w in candidates if w != word2 and w != word1]
        conflict_word = None
        for wc in wrong_candidates:
            for i, (cc, wl) in enumerate(zip(cipher2, wc)):
                if cc == mapping[shared_letter] and wl != shared_letter:
                    conflict_word = wc
                    break
            if conflict_word:
                break

        if not conflict_word:
            return None

        prompt = (
            f"We know: cipher '{cipher1}' = '{word1}'\n"
            f"This gives mapping: {mapping[shared_letter]}→{shared_letter}\n\n"
            f"Now cipher '{cipher2}' could be '{conflict_word}' or '{word2}'.\n"
            f"Which one is consistent with the existing mapping?"
        )

        think_lines = [
            f"From '{cipher1}'='{word1}': {mapping[shared_letter]}→{shared_letter}",
            f"",
            f"Test '{conflict_word}' for '{cipher2}':",
        ]

        for i, (cc, wl) in enumerate(zip(cipher2, conflict_word)):
            if cc == mapping[shared_letter] and wl != shared_letter:
                think_lines.append(f"  Position {i}: cipher '{cc}' must map to '{shared_letter}' (established)")
                think_lines.append(f"  But '{conflict_word}' needs '{cc}'→'{wl}' — CONFLICT")
                think_lines.append(f"  Reject '{conflict_word}'")
                break

        think_lines.append(f"")
        think_lines.append(f"Test '{word2}' for '{cipher2}':")
        think_lines.append(f"  All mappings consistent → MATCH")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": word2}


# ============================================================
# TERMINAL WEIGHT TABLE — SINGLE SOURCE OF TRUTH
#
# This table is applied LAST and overrides all class defaults
# and intermediate _WEIGHTS blocks. Edit ONLY THIS TABLE to
# change the training curriculum weights.
#
# Weight guide:
#   8-10 = top priority, targets primary failure mode
#   5-6  = critical inference/falsification skill
#   3-4  = important supporting skill
#   1-2  = standard/maintenance skill
#   0.5  = low priority / niche
# ============================================================
_FINAL_WEIGHTS = {
    # ── BIT: HONEST VERIFICATION (88% of failures are fake_verify!) ──
    "bit_verify_given":           15.0,   # #1 failure: fake verify. Max weight.
    "bit_verify_given_3":         15.0,   # same for 3-input families
    "bit_grid_vs_expected":       12.0,   # "is this GRID correct?" — direct audit
    "bit_honest_rule_test":       12.0,   # honest rule testing — anti-fake-verify
    "bit_anti_copy":              10.0,   # prevent copying expected output as GRID
    "bit_trace_audit":            10.0,   # full trace audit — "is this check valid?"
    "bit_just_choose_rule":       12.0,   # choose among precomputed — directly targets wrong_rule   # selection from precomputed outputs

    # ── BIT: RULE SELECTION / FALSIFICATION ──
    "bit_survivor_set":            8.0,
    "bit_reject_and_backtrack":    8.0,
    "bit_first_fail":              8.0,
    "bit_how_many_sources":        6.0,
    "bit_compose_identify":        6.0,
    "bit_rule_discriminate":      10.0,   # core inference — boost
    "bit_rule_discriminate_multi":10.0,   # multi-example version — boost
    "bit_rule_check":              5.0,
    "bit_full_verify":             5.0,
    "bit_family_from_popcount":    5.0,
    "bit_gate_from_known_sources": 5.0,
    "bit_least_error":             5.0,
    "bit_fix_one_bit":             5.0,
    "bit_compose_two_step":        5.0,
    "bit_narrow_sources":          5.0,

    # ── BIT: FAMILY-3 (demoted — model fails on simple gates, not exotic families) ──
    "bit_family3_execute":         5.0,
    "bit_family3_verify":          5.0,
    "bit_family3_discriminate":    5.0,
    "bit_compose3":                5.0,
    "bit_what_produced":           5.0,
    "bit_family_permutation":      5.0,

    # ── BIT: SUPPORTING SKILLS ──
    "bit_distinguish":            10.0,   # model defaults to xor/and/or when wrong — needs discrimination
    "bit_nojump":                  4.0,
    "bit_eliminate_family":        10.0,   # same — teach elimination
    "bit_correlate_positions":     4.0,
    "bit_rank_rules":              4.0,
    "bit_second_source":           4.0,
    "bit_confident_or_not":        4.0,
    "bit_which_closer":            4.0,
    "bit_predict_output":          4.0,
    "bit_scan_compute":            4.0,
    "bit_bookend_verify":          4.0,
    "bit_compose2":                3.0,
    "bit_constant_positions":      3.0,
    "bit_gate_from_properties":    3.0,
    "bit_source_consistency":      3.0,

    # ── BIT: LOW PRIORITY (pattern analysis demoted, execution demoted) ──
    "bit_error_detect":            2.0,
    "bit_which_op":                2.0,
    "bit_counterfactual":          2.0,
    "bit_visual_pattern":          1.5,
    "bit_spot_invariant":          1.5,
    "bit_spot_invariant_open":     1.5,
    "bit_invariant_checklist":     1.5,
    "bit_compare_to_target":       1.5,
    "bit_is_rotation":             1.5,
    "bit_shift":                   1.0,
    "bit_gate":                    1.0,
    "bit_similarity":              1.0,
    "bit_edge_shift_vs_rotate":    1.0,
    "bit_edge_zeros":              1.0,
    "bit_edge_gate":               1.0,
    "general_string_diff":         1.0,
    "bit_properties":              1.0,
    "bit_reverse_find":            1.0,
    "bit_two_step_id":             1.0,
    "bit_impossible":              1.0,
    "bit_spot_error":              1.0,
    "bit_count_across_examples":   1.0,
    "bit_batch_pipeline":          1.0,
    "bit_where_ones":              1.0,
    "bit_and_across":              1.0,
    "bit_or_across":               1.0,
    "bit_constant_vs_variable":    1.0,
    "bit_nibble_view":             1.0,
    "bit_complement_view":         1.0,
    "bit_ones_batch":              3.0,   # batch: count ones in 8-bit strings
    "bit_hamming_batch":           3.0,   # batch: hamming distance for pairs
    "bit_step_by_step":            0.5,
    "bit_popcount":                0.5,

    # ── ENCRYPTION ──
    "enc_reject_wrong_len":       12.0,   # 49% of enc failures are wrong-length (bumped from 8)
    "enc_vocab_fill":              8.0,   # 20 failures have incomplete decrypt with _
    "enc_confusable":              6.0,   # student↔strange, rabbit↔ribbon confusion
    "enc_char_decrypt":            6.0,   # char-by-char prevents language prior hijack
    "enc_bijection_reject":       10.0,   # 43% pick wrong vocab word — bijection violations (bumped from 6)
    "enc_length_check":            5.0,
    "enc_word_lengths":            5.0,
    "enc_length_delta":            6.0,   # count letters in word pairs, compute delta
    "enc_length_delta_v2":         5.0,   # reversed/scrambled variant — non-trivial deltas
    "enc_cascade_check":           4.0,
    "enc_gap_resolve":             0.0,   # DISABLED: 77-word vocab has no ambiguous gap patterns
    "enc_gap_exclude":             0.0,   # DISABLED: same reason
    "enc_gap_batch":               0.0,   # DISABLED: same reason

    # ── ENCRYPTION: ROUND 4 MASTERY SKILLS ──
    "enc_table_fill_row":         10.0,   # fill mapping table row from cipher+plain pair
    "enc_detect_swap":            12.0,   # find swapped pair in nearly-correct table (#1 failure mode)
    "enc_word_from_table":        10.0,   # decode word char-by-char from positional table
    "enc_letter_match":            5.0,   # batch: count matching letters per pair
    "enc_pattern_match":           5.0,   # batch: match partial patterns to vocab
    "enc_extract_mapping":        12.0,   # 72% of enc failures are mapping extraction — BOOST
    "enc_apply_mapping":          10.0,   # Apply table — upstream of all other enc steps — BOOST
    "enc_vocab":                   3.0,
    "enc_pattern_fill":            3.0,
    "enc_vocab_audit":             3.0,
    "enc_can_fit":                 3.0,
    "enc_most_constrained":        3.0,
    "str_count":                   3.0,
    "enc_forced_mapping":          2.0,
    "enc_bijection":               2.0,
    "enc_impossible":              2.0,
    "enc_not_forced":              2.0,
    "enc_why_wrong":               2.0,
    "enc_propagation":             1.5,
    "enc_repeated_letters":        1.0,
    "enc_reverse_decrypt":         1.0,
    "str_compare":                 1.0,

    # ── TRANSFORMATION ──
    "trans_scan_mini":             7.0,   # full mini scan-reject-lock pipeline
    "trans_crack_cipher":          6.0,   # crack symbol→digit mapping
    "trans_scan_reject":           6.0,   # scan candidates, reject on mismatch
    "trans_multi_op":              5.0,   # same or different operation for two symbols?
    "trans_verify_op":             5.0,   # verify proposed op against examples
    "trans_encode":                5.0,   # encode numeric result back to cipher
    "trans_compute_batch":         4.0,   # batch: compute op results for operand pairs
    "trans_op_eliminate":           5.0,
    "trans_digit_order":           4.0,   # AB_CD vs BA_DC ordering
    "trans_format_detect":         4.0,   # raw vs rev vs abs vs dsum
    "trans_op_from_examples":      4.0,
    "trans_base":                  3.0,
    "trans_parse":                 2.0,
    "trans_sign":                  2.0,
    "trans_chain":                 5.0,   # chain ops are remaining numeric gap (R3 feedback)
    "trans_reverse":               1.0,
    "trans_symbol_edit":           1.0,
    "trans_rev_chain":             1.0,
    "trans_impossible":            1.0,

    # ── TRANSFORMATION: ROUND 3 COVERAGE SKILLS ──
    "trans_style_pick":            5.0,   # identify output format from result + expected
    "trans_alias_op":              5.0,   # recognize decorative operator aliasing
    "trans_output_length":         4.0,   # predict output digit count from operands + op
    "trans_unseen_transfer":       5.0,   # transfer known op to unseen operator symbol
    "trans_open_encode":           4.0,   # encode with fresh symbols for unmapped digits

    # ── TRANSFORMATION: ROUND 3.1 CONTROL SKILLS ──
    "trans_detect_op_pos":         6.0,   # identify operator position in 5-char equations
    "trans_regime_vote":           6.0,   # infer output regime from support examples
    "trans_close_miss_reject":     6.0,   # reject closest rival using discriminating example
    "trans_candidate_rank":        7.0,   # choose best candidate among 3 survivors (#1 num failure)

    # ── TRANSFORMATION: ROUND 3.1 COMPLEMENTARY ──
    "trans_mapping_extend":        6.0,   # extend partial mapping from aligned decode
    "trans_format_scope":          5.0,   # operator-scoped format in multi-op rows
    "trans_operator_partition":    5.0,   # partition 3 operators into alias groups

    "trans_style_then_compute":    6.0,   # end-to-end: infer style + compute + render
    "trans_task_identify":         8.0,   # identify numeric vs cipher-digit vs non-cipher symbolic

    # ── BIT: ROUND 3.1 ──
    "bit_witness_pick":            6.0,   # pick example that best discriminates two rules

    # ── GRAV/UNIT/NUMCONV — maintenance ──
    "grav_magnitude":              1.0,   # 3 grav failures are decimal point errors
    "arith_round":                 0.5,
    "arith_long_multiply":         0.5,
    "arith_long_divide":           0.5,
    "arith_hundredths":            0.5,
    "numconv_base":                0.5,
}

# Apply terminal weights and validate
from generators.microskill_framework import REGISTRY as _FINAL_REG
_missing_weights = []
for _name, _cls in _FINAL_REG.items():
    if _name in _FINAL_WEIGHTS:
        _cls.weight = _FINAL_WEIGHTS[_name]
    else:
        _missing_weights.append(_name)

if _missing_weights:
    import warnings
    warnings.warn(
        f"Skills without terminal weight (using class default): {_missing_weights}",
        stacklevel=1,
    )


# ============================================================
# NEW: TRANSFORMATION — SCAN/REJECT/LOCK SKILLS
# ============================================================

@register
class TransScanReject(MicroSkill):
    """Show candidate operations being tested against examples and rejected.
    Frequency-ordered brute-force scan with immediate verification.
    Model learns what rejection looks like."""
    name = "trans_scan_reject"
    puzzle_type = "transformation"
    description = "Test candidate operations against examples, reject on mismatch"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        ops = {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "absdiff": lambda a, b: abs(a - b),
        }
        real_op = rng.choice(list(ops.keys()))
        a1, b1 = rng.randint(10, 99), rng.randint(10, 99)
        result1 = ops[real_op](a1, b1)

        a2, b2 = rng.randint(10, 99), rng.randint(10, 99)
        result2 = ops[real_op](a2, b2)

        prompt = (
            f"Examples:\n"
            f"  {a1}, {b1} → {result1}\n"
            f"  {a2}, {b2} → {result2}\n\n"
            f"Scan operations. Which one matches both examples?"
        )

        # Show scan with rejections
        think_lines = []
        scan_order = list(ops.keys())
        rng.shuffle(scan_order)

        for op_name in scan_order:
            fn = ops[op_name]
            v1 = fn(a1, b1)
            ok1 = v1 == result1
            if ok1:
                v2 = fn(a2, b2)
                ok2 = v2 == result2
                if ok2:
                    think_lines.append(f"#{scan_order.index(op_name)+1}: {op_name}({a1},{b1})={v1} → MATCH {op_name}({a2},{b2})={v2} → MATCH LOCK")
                    break
                else:
                    think_lines.append(f"#{scan_order.index(op_name)+1}: {op_name}({a1},{b1})={v1} → MATCH {op_name}({a2},{b2})={v2} vs {result2} → MISMATCH")
            else:
                think_lines.append(f"#{scan_order.index(op_name)+1}: {op_name}({a1},{b1})={v1} vs {result1} → MISMATCH")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": real_op}


@register
class TransVerifyOp(MicroSkill):
    """Given an operation and examples, verify it matches. Directly targets
    the 252 numeric failures where model picks wrong operation."""
    name = "trans_verify_op"
    puzzle_type = "transformation"
    description = "Verify a proposed operation against given examples"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        ops = {"add": lambda a,b: a+b, "sub": lambda a,b: a-b,
               "mul": lambda a,b: a*b, "absdiff": lambda a,b: abs(a-b)}
        real_op = rng.choice(list(ops.keys()))
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        result = ops[real_op](a, b)

        # 50% propose wrong op
        if rng.random() < 0.5:
            proposed = rng.choice([o for o in ops if o != real_op])
            is_wrong = True
        else:
            proposed = real_op
            is_wrong = False

        computed = ops[proposed](a, b)

        prompt = (
            f"Example: {a}, {b} → {result}\n"
            f"Proposed operation: {proposed}\n"
            f"Does {proposed}({a}, {b}) = {result}?"
        )

        think = f"{proposed}({a}, {b}) = {computed}"
        if computed == result:
            think += f" = {result} → MATCH MATCH"
            answer = "MATCH"
        else:
            think += f" ≠ {result} → MISMATCH REJECT"
            answer = "REJECT"

        return {"user": prompt, "think": think, "answer": answer}


@register
class TransDigitOrder(MicroSkill):
    """Identify operand ordering: is it AB,CD or BA,DC or AB,DC or BA,CD?
    Critical for cipher-digit solving."""
    name = "trans_digit_order"
    puzzle_type = "transformation"
    description = "Identify which digit ordering (AB_CD, BA_DC, etc.) produces the right operands"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        a, b, c, d = [rng.randint(0, 9) for _ in range(4)]
        orderings = {
            "AB_CD": (a*10+b, c*10+d),
            "BA_DC": (b*10+a, d*10+c),
            "AB_DC": (a*10+b, d*10+c),
            "BA_CD": (b*10+a, c*10+d),
        }
        real_order = rng.choice(list(orderings.keys()))
        L, R = orderings[real_order]

        prompt = (
            f"Digits: A={a}, B={b}, C={c}, D={d}\n"
            f"The operands are L={L}, R={R}.\n"
            f"Which ordering was used? AB_CD, BA_DC, AB_DC, or BA_CD?"
        )

        think_lines = []
        for name, (tL, tR) in orderings.items():
            match = tL == L and tR == R
            think_lines.append(f"  {name}: L={tL}, R={tR} {'→ MATCH' if match else ''}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": real_order}


@register
class TransFormatDetect(MicroSkill):
    """Detect output format: raw, reversed, absolute value, digit sum.
    The model needs to identify the format applied to the numeric result."""
    name = "trans_format_detect"
    puzzle_type = "transformation"
    description = "Identify output format (raw, rev, abs, dsum) from numeric result"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        val = rng.randint(-999, 9999)
        formats = {
            "raw": str(val),
            "rev": str(val)[::-1] if val >= 0 else '-' + str(val)[1:][::-1],
            "abs": str(abs(val)),
        }
        if val != 0:
            formats["dsum"] = str(sum(int(c) for c in str(abs(val))))

        real_fmt = rng.choice(list(formats.keys()))
        output = formats[real_fmt]

        prompt = (
            f"Numeric result: {val}\n"
            f"Displayed as: {output}\n"
            f"What format was applied? (raw, rev, abs, dsum)"
        )

        think_lines = []
        for name, fval in formats.items():
            match = fval == output
            think_lines.append(f"  {name}: {fval} {'→ MATCH' if match else ''}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": real_fmt}


@register
class EncLengthDelta(MicroSkill):
    """Two columns of words, compute length difference for each pair.
    Drills letter-counting which is the #1 encryption failure mode."""
    name = "enc_length_delta"
    puzzle_type = "encryption"
    description = "Given two columns of words, compute the letter-count delta for each pair"
    weight = 6.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab or len(vocab) < 10:
            return None

        n_pairs = rng.randint(3, 6)
        words_a = rng.sample(list(vocab), n_pairs)
        words_b = rng.sample([w for w in vocab if w not in words_a], n_pairs)

        col_a = "\n".join(f"  {i+1}. {w}" for i, w in enumerate(words_a))
        col_b = "\n".join(f"  {i+1}. {w}" for i, w in enumerate(words_b))

        prompt = (
            f"Column A:\n{col_a}\n\n"
            f"Column B:\n{col_b}\n\n"
            f"For each pair, what is len(A) - len(B)?"
        )

        think_lines = []
        answers = []
        for i, (wa, wb) in enumerate(zip(words_a, words_b)):
            la, lb = len(wa), len(wb)
            delta = la - lb
            sign = f"+{delta}" if delta > 0 else str(delta)
            think_lines.append(f"{i+1}. {wa}({la}) - {wb}({lb}) = {sign}")
            answers.append(str(delta))

        answer = ", ".join(answers)
        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


@register
class EncLengthDeltaVariant(MicroSkill):
    """Variant: Column B words are transformations of Column A (reversed,
    shuffled from same list, or from a different part of vocab).
    Forces model to count letters on TRANSFORMED words, not memorize."""
    name = "enc_length_delta_v2"
    puzzle_type = "encryption"
    description = "Length delta with reversed/scrambled words — prevents memorization"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab or len(vocab) < 10:
            return None

        n_pairs = rng.randint(3, 6)
        words_a = rng.sample(list(vocab), n_pairs)

        # Variants for Column B
        variant = rng.choices(
            ["shuffled", "reverse_other", "other_vocab", "scramble_other"],
            weights=[30, 25, 25, 20]
        )[0]

        words_b = []
        variant_label = ""
        if variant == "shuffled":
            # B = same words as A but shuffled, some reversed
            words_b = list(words_a)
            rng.shuffle(words_b)
            if words_b == words_a and len(words_a) > 1:
                words_b[0], words_b[-1] = words_b[-1], words_b[0]
            # Randomly reverse some words in B (wolf→flow)
            for i in range(len(words_b)):
                if rng.random() < 0.4:
                    words_b[i] = words_b[i][::-1]
            variant_label = "(Column B = Column A shuffled, some reversed)"
        elif variant == "reverse_other":
            # B[i] = reverse of A[j] where j != i — different lengths!
            shifted = words_a[1:] + words_a[:1]
            words_b = [w[::-1] for w in shifted]
            variant_label = "(B = reversed from another word in A)"
        elif variant == "scramble_other":
            # Mix: some scrambled from different A, some from other vocab
            pool = [w for w in vocab if w not in words_a]
            shifted = words_a[1:] + words_a[:1]
            for i in range(n_pairs):
                if rng.random() < 0.5:
                    chars = list(shifted[i])
                    rng.shuffle(chars)
                    words_b.append(''.join(chars))
                elif pool:
                    words_b.append(pool.pop(rng.randrange(len(pool))))
                else:
                    chars = list(shifted[i])
                    rng.shuffle(chars)
                    words_b.append(''.join(chars))
            variant_label = "(mixed)"
        else:
            pool = [w for w in vocab if w not in words_a]
            words_b = rng.sample(pool, min(n_pairs, len(pool)))
            variant_label = ""

        if len(words_b) < n_pairs:
            return None

        col_a = "\n".join(f"  {i+1}. {w}" for i, w in enumerate(words_a))
        col_b = "\n".join(f"  {i+1}. {w}" for i, w in enumerate(words_b))

        prompt = (
            f"Column A:\n{col_a}\n\n"
            f"Column B: {variant_label}\n{col_b}\n\n"
            f"For each pair, what is len(A) - len(B)?"
        )

        think_lines = []
        answers = []
        for i, (wa, wb) in enumerate(zip(words_a, words_b)):
            la, lb = len(wa), len(wb)
            delta = la - lb
            sign = f"+{delta}" if delta > 0 else str(delta)
            think_lines.append(f"{i+1}. {wa}({la}) - {wb}({lb}) = {sign}")
            answers.append(str(delta))

        answer = ", ".join(answers)
        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


# ============================================================
# COLUMN-COMPARISON BATCH SKILLS
# High gradient density: each item in the list is independently verifiable
# ============================================================

@register
class EncLetterMatch(MicroSkill):
    """Column A: cipher words with a mapping. Column B: candidate plaintext words.
    Answer: how many letters match for each pair."""
    name = "enc_letter_match"
    puzzle_type = "encryption"
    description = "Count matching letters when applying cipher mapping to candidate words"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None

        # Create a partial mapping (6-10 letters mapped)
        n_mapped = rng.randint(6, 10)
        letters = list('abcdefghijklmnopqrstuvwxyz')
        plain_letters = rng.sample(letters, n_mapped)
        cipher_letters = rng.sample(letters, n_mapped)
        mapping = dict(zip(cipher_letters, plain_letters))

        n_pairs = rng.randint(3, 5)
        words = rng.sample([w for w in vocab if 4 <= len(w) <= 8], n_pairs)

        # For each word, create a cipher version and count matches
        rows = []
        for word in words:
            # Encrypt the word (some letters may not be in mapping → use random)
            rev_map = {v: k for k, v in mapping.items()}
            cipher = []
            for c in word:
                if c in rev_map:
                    cipher.append(rev_map[c])
                else:
                    cipher.append(rng.choice(letters))
            cipher_word = ''.join(cipher)

            # Pick a candidate (sometimes correct, sometimes wrong)
            if rng.random() < 0.4:
                candidate = word
            else:
                candidate = rng.choice([w for w in vocab if len(w) == len(word) and w != word])

            # Count matches: apply mapping to cipher, compare with candidate
            matches = 0
            for cc, pc in zip(cipher_word, candidate):
                if cc in mapping and mapping[cc] == pc:
                    matches += 1

            rows.append((cipher_word, candidate, matches, len(word)))

        map_str = ", ".join(f"{c}→{p}" for c, p in sorted(mapping.items())[:8])
        col_a = "\n".join(f"  {i+1}. {cw}" for i, (cw, _, _, _) in enumerate(rows))
        col_b = "\n".join(f"  {i+1}. {cd}" for i, (_, cd, _, _) in enumerate(rows))

        prompt = (
            f"Mapping: {map_str}\n\n"
            f"Cipher words:\n{col_a}\n\n"
            f"Candidates:\n{col_b}\n\n"
            f"For each pair, how many letters match using the mapping?"
        )

        think_lines = []
        answers = []
        for i, (cw, cd, m, l) in enumerate(rows):
            think_lines.append(f"{i+1}. {cw} vs {cd}: {m}/{l} match")
            answers.append(f"{m}/{l}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": ", ".join(answers)}


@register
class EncPatternMatch(MicroSkill):
    """Column A: partial decryptions with _ gaps.
    Column B: vocab candidates.
    Answer: which candidate fits each pattern."""
    name = "enc_pattern_match"
    puzzle_type = "encryption"
    description = "Match partial decryption patterns to vocabulary words"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None

        n_items = rng.randint(3, 5)
        words = rng.sample([w for w in vocab if 4 <= len(w) <= 8], n_items)

        rows = []
        for word in words:
            # Create partial with 1-3 gaps
            n_gaps = rng.randint(1, min(3, len(word) - 2))
            gap_pos = rng.sample(range(len(word)), n_gaps)
            partial = list(word)
            for p in gap_pos:
                partial[p] = '_'
            partial_str = ''.join(partial)

            # Pick 2-3 candidates (including the correct one)
            same_len = [w for w in vocab if len(w) == len(word) and w != word]
            if not same_len:
                continue
            wrong = rng.sample(same_len, min(2, len(same_len)))
            options = [word] + wrong
            rng.shuffle(options)

            rows.append((partial_str, word, options))

        if len(rows) < 3:
            return None

        patterns = "\n".join(f"  {i+1}. {p}" for i, (p, _, _) in enumerate(rows))
        options_str = "\n".join(
            f"  {i+1}. {' / '.join(opts)}" for i, (_, _, opts) in enumerate(rows)
        )

        prompt = (
            f"Patterns:\n{patterns}\n\n"
            f"Candidates:\n{options_str}\n\n"
            f"Which candidate fits each pattern?"
        )

        think_lines = []
        answers = []
        for i, (partial, correct, options) in enumerate(rows):
            think_lines.append(f"{i+1}. {partial} → {correct}")
            answers.append(correct)

        return {"user": prompt, "think": "\n".join(think_lines), "answer": ", ".join(answers)}


@register
class BitOnesCountBatch(MicroSkill):
    """Batch ones-counting: list of 8-bit strings, count ones for each.
    Drills the exact computation used in Scan preamble and bookend."""
    name = "bit_ones_batch"
    puzzle_type = "bit_manipulation"
    description = "Count ones in a batch of 8-bit strings"
    weight = 3.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        n = rng.randint(4, 8)
        strings = [format(rng.randint(0, 255), '08b') for _ in range(n)]

        col = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(strings))
        prompt = f"Count the ones in each:\n{col}"

        think_lines = []
        answers = []
        for i, s in enumerate(strings):
            c = s.count('1')
            think_lines.append(f"{i+1}. {s} → {c}")
            answers.append(str(c))

        return {"user": prompt, "think": "\n".join(think_lines), "answer": ", ".join(answers)}


@register
class BitHammingBatch(MicroSkill):
    """Batch hamming distance: pairs of 8-bit strings.
    Drills the comparison used in Check verification."""
    name = "bit_hamming_batch"
    puzzle_type = "bit_manipulation"
    description = "Compute hamming distance for pairs of 8-bit strings"
    weight = 3.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        n = rng.randint(3, 6)
        pairs = [(format(rng.randint(0, 255), '08b'), format(rng.randint(0, 255), '08b'))
                 for _ in range(n)]

        col_a = "\n".join(f"  {i+1}. {a}" for i, (a, _) in enumerate(pairs))
        col_b = "\n".join(f"  {i+1}. {b}" for i, (_, b) in enumerate(pairs))

        prompt = f"Column A:\n{col_a}\n\nColumn B:\n{col_b}\n\nHamming distance for each pair?"

        think_lines = []
        answers = []
        for i, (a, b) in enumerate(pairs):
            h = sum(1 for x, y in zip(a, b) if x != y)
            think_lines.append(f"{i+1}. {a} vs {b} → {h}")
            answers.append(str(h))

        return {"user": prompt, "think": "\n".join(think_lines), "answer": ", ".join(answers)}


@register
class TransComputeBatch(MicroSkill):
    """Batch arithmetic: list of (L, R, op) triples, compute each.
    Drills the exact computation transformation puzzles need."""
    name = "trans_compute_batch"
    puzzle_type = "transformation"
    description = "Compute operation results for a batch of operand pairs"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        ops = {"add": lambda a,b: a+b, "sub": lambda a,b: a-b,
               "mul": lambda a,b: a*b, "absdiff": lambda a,b: abs(a-b)}

        n = rng.randint(3, 6)
        items = []
        for _ in range(n):
            op_name = rng.choice(list(ops.keys()))
            L = rng.randint(10, 99)
            R = rng.randint(10, 99)
            result = ops[op_name](L, R)
            items.append((L, R, op_name, result))

        rows = "\n".join(f"  {i+1}. {op}({L}, {R})" for i, (L, R, op, _) in enumerate(items))
        prompt = f"Compute each:\n{rows}"

        think_lines = []
        answers = []
        for i, (L, R, op, result) in enumerate(items):
            think_lines.append(f"{i+1}. {op}({L}, {R}) = {result}")
            answers.append(str(result))

        return {"user": prompt, "think": "\n".join(think_lines), "answer": ", ".join(answers)}


@register
class EncGapResolve(MicroSkill):
    """Given a gap pattern like 'dra_s' that matches multiple vocab words,
    use a cipher mapping constraint to pick the right one.
    Targets the EXACT failure mode: draws↔drags, catches↔watches, etc."""
    name = "enc_gap_resolve"
    puzzle_type = "encryption"
    description = "Resolve ambiguous gap patterns using cipher mapping constraints"
    weight = 7.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None

        # Find an ambiguous gap pattern
        candidates = []
        for word in vocab:
            if len(word) < 4: continue
            for gap_pos in range(len(word)):
                pattern = list(word)
                gap_letter = pattern[gap_pos]
                pattern[gap_pos] = '_'
                matches = [w for w in vocab if len(w) == len(word)
                           and all(p == '_' or p == c for p, c in zip(pattern, w))]
                if 2 <= len(matches) <= 4:
                    candidates.append((word, ''.join(pattern), gap_pos, gap_letter, matches))

        if not candidates: return None
        word, pattern, gap_pos, gap_letter, matches = rng.choice(candidates)
        wrong_words = [m for m in matches if m != word]

        # Create a cipher mapping that disambiguates
        # The gap letter in cipher maps to the correct plain letter
        alphabet = list('abcdefghijklmnopqrstuvwxyz')
        cipher_letter = rng.choice(alphabet)

        prompt = (
            f"Partial decryption: '{pattern}'\n"
            f"Position {gap_pos} is unknown (cipher letter '{cipher_letter}').\n"
            f"Possible words: {', '.join(matches)}\n\n"
            f"Constraint: cipher '{cipher_letter}' → plain '{gap_letter}'\n"
            f"Which word fits?"
        )

        think_lines = [f"Gap at position {gap_pos}. Cipher '{cipher_letter}' → '{gap_letter}'."]
        for m in matches:
            needed = m[gap_pos]
            if needed == gap_letter:
                think_lines.append(f"  '{m}' needs '{needed}' at pos {gap_pos} → '{needed}' = '{gap_letter}' → MATCH")
            else:
                think_lines.append(f"  '{m}' needs '{needed}' at pos {gap_pos} → '{needed}' ≠ '{gap_letter}' → MISMATCH")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": word}


@register
class EncGapMultiple(MicroSkill):
    """Multiple gap patterns at once. Column format:
    1. dra_s → ? (mapping says _→w)
    2. _old → ? (mapping says _→c)
    Batch version of gap resolution."""
    name = "enc_gap_batch"
    puzzle_type = "encryption"
    description = "Resolve multiple gap patterns in batch using mapping constraints"
    weight = 6.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None

        # Collect ambiguous patterns
        all_patterns = []
        for word in vocab:
            if len(word) < 4: continue
            for gap_pos in range(len(word)):
                pattern = list(word)
                gap_letter = pattern[gap_pos]
                pattern[gap_pos] = '_'
                matches = [w for w in vocab if len(w) == len(word)
                           and all(p == '_' or p == c for p, c in zip(pattern, w))]
                if 2 <= len(matches) <= 4:
                    all_patterns.append((word, ''.join(pattern), gap_pos, gap_letter, matches))

        if len(all_patterns) < 3: return None

        n_items = rng.randint(3, 5)
        chosen = rng.sample(all_patterns, min(n_items, len(all_patterns)))

        rows = []
        for word, pattern, gap_pos, gap_letter, matches in chosen:
            cipher_letter = rng.choice('abcdefghijklmnopqrstuvwxyz')
            rows.append((pattern, cipher_letter, gap_letter, word, matches))

        prompt_lines = []
        for i, (pat, cl, gl, _, _) in enumerate(rows):
            prompt_lines.append(f"  {i+1}. '{pat}' (cipher '{cl}' → '{gl}')")

        prompt = "Resolve each gap:\n" + "\n".join(prompt_lines)

        think_lines = []
        answers = []
        for i, (pat, cl, gl, word, matches) in enumerate(rows):
            think_lines.append(f"{i+1}. '{pat}' + '{gl}' → {word}")
            answers.append(word)

        return {"user": prompt, "think": "\n".join(think_lines), "answer": ", ".join(answers)}


@register
class EncGapExcludeLetters(MicroSkill):
    """Gap pattern + excluded letters (already used in mapping).
    'dra_s' — which word fits if letters a,g,t are already taken?
    Directly simulates the bijection constraint during decryption."""
    name = "enc_gap_exclude"
    puzzle_type = "encryption"
    description = "Fill gap pattern excluding already-mapped letters (bijection)"
    weight = 7.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None

        # Find ambiguous pattern
        candidates = []
        for word in vocab:
            if len(word) < 4: continue
            for gap_pos in range(len(word)):
                pattern = list(word)
                gap_letter = pattern[gap_pos]
                pattern[gap_pos] = '_'
                matches = [w for w in vocab if len(w) == len(word)
                           and all(p == '_' or p == c for p, c in zip(pattern, w))]
                if 2 <= len(matches) <= 5:
                    candidates.append((word, ''.join(pattern), gap_pos, gap_letter, matches))

        if not candidates: return None
        word, pattern, gap_pos, gap_letter, matches = rng.choice(candidates)

        # Create excluded letters: letters that are "already used" in the cipher mapping
        # Include the wrong candidates' gap letters to make exclusion meaningful
        wrong_gap_letters = set(m[gap_pos] for m in matches if m != word)
        # Add some random excluded letters for realism
        extra_excluded = rng.sample([c for c in 'abcdefghijklmnopqrstuvwxyz' 
                                     if c != gap_letter and c not in wrong_gap_letters], 
                                    rng.randint(2, 5))
        excluded = sorted(wrong_gap_letters | set(extra_excluded))

        # Which matches survive after exclusion?
        surviving = [m for m in matches if m[gap_pos] not in excluded]

        prompt = (
            f"Pattern: '{pattern}'\n"
            f"Already used (cannot appear): {', '.join(excluded)}\n"
            f"Possible words: {', '.join(matches)}\n"
            f"Which word(s) fit?"
        )

        think_lines = [f"Gap at position {gap_pos}. Excluded: {', '.join(excluded)}"]
        for m in matches:
            needed = m[gap_pos]
            if needed in excluded:
                think_lines.append(f"  '{m}' needs '{needed}' → EXCLUDED → MISMATCH")
            else:
                think_lines.append(f"  '{m}' needs '{needed}' → available → MATCH")

        if len(surviving) == 1:
            answer = surviving[0]
        elif surviving:
            answer = ", ".join(surviving)
        else:
            answer = "none"

        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


# ============================================================
# TRANSFORMATION — CIPHER-DIGIT PIPELINE SKILLS
# ============================================================

@register
class TransCrackCipher(MicroSkill):
    """Given cipher examples AB⊕CD=result (all symbols), deduce the
    symbol→digit mapping. The core first step of cipher-digit solving."""
    name = "trans_crack_cipher"
    puzzle_type = "transformation"
    description = "Crack a symbol→digit cipher from example equations"
    weight = 6.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        # Create a random bijective cipher (10 symbols → 0-9)
        symbols = rng.sample(list('!@#$%^&*()-_=+[]:,.<>?/~`'), 10)
        mapping = dict(zip(symbols, range(10)))
        rev_map = {v: k for k, v in mapping.items()}

        # Pick an operation
        ops = {"add": lambda a,b: a+b, "mul": lambda a,b: a*b, "sub": lambda a,b: a-b}
        op_name = rng.choice(list(ops.keys()))
        op_fn = ops[op_name]

        # Pick an operator symbol (position 2)
        op_sym = rng.choice([s for s in '!@#$%^&*' if s not in symbols])

        # Generate 2-3 examples
        examples = []
        for _ in range(rng.randint(2, 3)):
            a, b, c, d = [rng.randint(0, 9) for _ in range(4)]
            L, R = b*10+a, d*10+c  # BA_DC
            result = op_fn(L, R)
            result_str = str(result)

            # Encode
            lhs = rev_map[a] + rev_map[b] + op_sym + rev_map[c] + rev_map[d]
            rhs = ''.join(rev_map[int(digit)] for digit in result_str if digit.isdigit())
            if '-' in result_str:
                continue  # skip negatives for simplicity
            examples.append((lhs, rhs, a, b, c, d, result))

        if len(examples) < 2:
            return None

        # Show examples, ask for mapping
        ex_str = "\n".join(f"  {lhs} = {rhs}" for lhs, rhs, *_ in examples)
        prompt = (
            f"Cipher equations (operator at position 2):\n{ex_str}\n\n"
            f"Crack the symbol→digit mapping."
        )

        # Trace: show how each symbol gets assigned
        think_lines = ["Working from examples:"]
        shown = {}
        for lhs, rhs, a, b, c, d, result in examples:
            digits_str = f"{a}{b}{c}{d}"
            think_lines.append(f"  {lhs}={rhs}: digits {digits_str}→{result}")
            for sym, dig in [(lhs[0],a),(lhs[1],b),(lhs[3],c),(lhs[4],d)]:
                if sym not in shown:
                    shown[sym] = dig
                    think_lines.append(f"    {sym}={dig}")
            for i, rc in enumerate(rhs):
                rd = int(str(result)[i]) if i < len(str(result)) else None
                if rd is not None and rc not in shown:
                    shown[rc] = rd
                    think_lines.append(f"    {rc}={rd}")

        map_str = " ".join(f"{s}={d}" for s, d in sorted(shown.items(), key=lambda x: x[1]))
        think_lines.append(f"\nMapping: {map_str}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": map_str}


@register
class TransEncodeResult(MicroSkill):
    """Given a numeric answer and a symbol→digit mapping, encode back to symbols.
    The final step of cipher-digit solving."""
    name = "trans_encode"
    puzzle_type = "transformation"
    description = "Encode a numeric result back to cipher symbols using mapping"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        # Random mapping
        symbols = rng.sample(list('!@#$%^&*()-_=+[]:,.<>?/~`'), 10)
        mapping = dict(zip(range(10), symbols))

        # Random number to encode
        val = rng.randint(1, 9999)
        val_str = str(val)

        map_display = " ".join(f"{d}={s}" for d, s in sorted(mapping.items()))
        prompt = (
            f"Mapping: {map_display}\n"
            f"Encode the number {val} to cipher symbols, digit by digit."
        )

        think_lines = []
        encoded = []
        for c in val_str:
            d = int(c)
            s = mapping[d]
            think_lines.append(f"  {d} → {s}")
            encoded.append(s)

        answer = ''.join(encoded)
        think_lines.append(f"= {answer}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


@register
class TransMultiOp(MicroSkill):
    """Two operators in examples — do they mean the same or different operations?
    Operator characters are decorative, sometimes adversarial."""
    name = "trans_multi_op"
    puzzle_type = "transformation"
    description = "Determine if two operator symbols represent the same or different operations"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        ops = {"add": lambda a,b: a+b, "sub": lambda a,b: a-b,
               "mul": lambda a,b: a*b, "absdiff": lambda a,b: abs(a-b)}

        # 50% same operation (decorative), 50% different
        op1_name = rng.choice(list(ops.keys()))

        if rng.random() < 0.5:
            op2_name = op1_name  # same op, different symbol
            is_same = True
        else:
            op2_name = rng.choice([o for o in ops if o != op1_name])
            is_same = False

        sym1 = rng.choice(['+', '-', '*', '/', '|', '^', '&'])
        sym2 = rng.choice([s for s in ['+', '-', '*', '/', '|', '^', '&'] if s != sym1])

        # Generate examples
        a1, b1 = rng.randint(10, 99), rng.randint(10, 99)
        r1 = ops[op1_name](a1, b1)
        a2, b2 = rng.randint(10, 99), rng.randint(10, 99)
        r2 = ops[op2_name](a2, b2)

        prompt = (
            f"Example 1: {a1}{sym1}{b1} = {r1}\n"
            f"Example 2: {a2}{sym2}{b2} = {r2}\n\n"
            f"Do '{sym1}' and '{sym2}' represent the same operation?"
        )

        think_lines = [
            f"'{sym1}' example: {a1}{sym1}{b1}={r1}",
            f"  {op1_name}({a1},{b1})={ops[op1_name](a1,b1)} {'→ MATCH' if ops[op1_name](a1,b1)==r1 else '→ MISMATCH'}",
            f"'{sym2}' example: {a2}{sym2}{b2}={r2}",
        ]

        if is_same:
            think_lines.append(f"  {op1_name}({a2},{b2})={ops[op1_name](a2,b2)} {'→ MATCH' if ops[op1_name](a2,b2)==r2 else '→ MISMATCH'}")
            think_lines.append(f"Same operation: {op1_name}")
            answer = f"SAME ({op1_name})"
        else:
            think_lines.append(f"  {op2_name}({a2},{b2})={ops[op2_name](a2,b2)} → MATCH")
            think_lines.append(f"Different: '{sym1}'={op1_name}, '{sym2}'={op2_name}")
            answer = f"DIFFERENT ({op1_name} vs {op2_name})"

        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


@register
class TransScanMini(MicroSkill):
    """Full mini scan-reject-lock pipeline as a micro-skill.
    Closest to the actual inference task for transformation."""
    name = "trans_scan_mini"
    puzzle_type = "transformation"
    description = "Complete mini scan-reject-lock: find operation from examples, apply to query"
    weight = 7.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        ops = {"add": lambda a,b: a+b, "sub": lambda a,b: a-b,
               "mul": lambda a,b: a*b, "absdiff": lambda a,b: abs(a-b)}
        orders = {"BA_DC": lambda a,b,c,d: (b*10+a, d*10+c),
                  "AB_CD": lambda a,b,c,d: (a*10+b, c*10+d)}
        fmts = {"raw": str, "rev": lambda v: str(v)[::-1]}

        real_order = rng.choice(list(orders.keys()))
        real_op = rng.choice(list(ops.keys()))
        real_fmt = rng.choice(list(fmts.keys()))

        def compute(a, b, c, d):
            L, R = orders[real_order](a, b, c, d)
            val = ops[real_op](L, R)
            return fmts[real_fmt](val)

        # 2 examples + 1 query
        examples = []
        for _ in range(2):
            a, b, c, d = [rng.randint(1, 9) for _ in range(4)]
            result = compute(a, b, c, d)
            sym = rng.choice(['+', '-', '*', '|'])
            examples.append((a, b, c, d, sym, result))

        qa, qb, qc, qd = [rng.randint(1, 9) for _ in range(4)]
        q_sym = rng.choice(['+', '-', '*', '|'])
        q_answer = compute(qa, qb, qc, qd)

        ex_str = "\n".join(
            f"  {a}{b}{sym}{c}{d} = {result}" for a, b, c, d, sym, result in examples
        )
        prompt = f"Examples:\n{ex_str}\n\nQuery: {qa}{qb}{q_sym}{qc}{qd} = ?"

        # Scan trace
        think_lines = ["Scan:"]
        scan_num = 0
        a1, b1, c1, d1, _, exp1 = examples[0]
        a2, b2, c2, d2, _, exp2 = examples[1]

        # Show 1-2 wrong combos then the right one
        wrong_combos = [(o, op, f) for o in orders for op in ops for f in fmts
                        if (o, op, f) != (real_order, real_op, real_fmt)]
        rng.shuffle(wrong_combos)

        for o, op, f in wrong_combos[:2]:
            scan_num += 1
            L, R = orders[o](a1, b1, c1, d1)
            val = ops[op](L, R)
            fval = fmts[f](val)
            think_lines.append(f"  #{scan_num}: {o}|{op}|{f} → {fval} vs {exp1} → MISMATCH")

        scan_num += 1
        L1, R1 = orders[real_order](a1, b1, c1, d1)
        v1 = ops[real_op](L1, R1)
        f1 = fmts[real_fmt](v1)
        L2, R2 = orders[real_order](a2, b2, c2, d2)
        v2 = ops[real_op](L2, R2)
        f2 = fmts[real_fmt](v2)
        think_lines.append(f"  #{scan_num}: {real_order}|{real_op}|{real_fmt} → {f1} → MATCH VER: {f2} → MATCH LOCK")

        think_lines.append("")
        qL, qR = orders[real_order](qa, qb, qc, qd)
        qv = ops[real_op](qL, qR)
        qf = fmts[real_fmt](qv)
        think_lines.append(f"Query: L={qL} R={qR} {real_op}={qv} {real_fmt}={qf}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": q_answer}


# ============================================================
# TRANSFORMATION — MISSING COVERAGE SKILLS (from Round 3 feedback)
# ============================================================

@register
class TransStylePick(MicroSkill):
    """Given a numeric result, which output format matches the expected answer?
    Directly targets the 51% of seen-op misses that are pure style errors."""
    name = "trans_style_pick"
    puzzle_type = "transformation"
    description = "Identify output format (raw/rev/abs/opsign/tailsign) from result + expected"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        val = rng.randint(-999, 9999)
        s = str(val)
        rev = s[::-1] if val >= 0 else '-' + s[1:][::-1]
        formats = {
            "raw": s,
            "rev": rev,
            "abs": str(abs(val)),
        }
        if abs(val) > 0:
            formats["dsum"] = str(sum(int(c) for c in str(abs(val))))

        real_fmt = rng.choice(list(formats.keys()))
        output = formats[real_fmt]

        # Add one op-prefix variant
        op = rng.choice(['+', '-', '*', '/', '|'])
        formats[f"opprefix({op})"] = op + str(abs(val))

        options = list(formats.items())
        rng.shuffle(options)

        prompt = (
            f"Numeric result: {val}\n"
            f"Displayed output: {output}\n"
            f"Which format?\n"
            + "\n".join(f"  {name}: {fval}" for name, fval in options)
        )

        think = f"Result {val} displayed as '{output}' → format = {real_fmt}"
        return {"user": prompt, "think": think, "answer": real_fmt}


@register
class TransAliasOperator(MicroSkill):
    """Two operator symbols in examples produce the same results.
    Teaches: operators can be decorative (same operation, different symbol)."""
    name = "trans_alias_op"
    puzzle_type = "transformation"
    description = "Recognize that two different operator symbols use the same operation"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        ops = {"add": lambda a,b: a+b, "sub": lambda a,b: a-b,
               "mul": lambda a,b: a*b}
        real_op = rng.choice(list(ops.keys()))
        fn = ops[real_op]

        sym1 = rng.choice(['+', '-', '*', '/', '|', '^'])
        sym2 = rng.choice([s for s in ['@', '#', '!', '&', '<', '>'] if s != sym1])

        examples = []
        for sym in [sym1, sym1, sym2, sym2]:
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            examples.append(f"  {a}{sym}{b} = {fn(a,b)}")

        prompt = (
            f"Examples:\n" + "\n".join(examples) +
            f"\n\nDo '{sym1}' and '{sym2}' perform the same operation?"
        )

        think_lines = [
            f"'{sym1}' examples: {real_op}",
            f"'{sym2}' examples: {real_op}",
            f"Same operation → MATCH — operators are aliases"
        ]

        return {"user": prompt, "think": "\n".join(think_lines), "answer": f"ALIAS ({real_op})"}


@register
class TransOutputLengthPredict(MicroSkill):
    """Given operands and operation, predict the output LENGTH.
    Helps model not emit 4 symbols when answer should be 2."""
    name = "trans_output_length"
    puzzle_type = "transformation"
    description = "Predict output digit count from operands and operation"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        ops = {"add": a+b, "sub": a-b, "mul": a*b, "absdiff": abs(a-b), "cat": int(f"{a}{b}")}
        op_name = rng.choice(list(ops.keys()))
        result = ops[op_name]

        prompt = f"{op_name}({a}, {b}) = {result}\nHow many digits in the output?"
        n_digits = len(str(abs(result)))
        think = f"{op_name}({a},{b}) = {result}, |{result}| has {n_digits} digits"
        return {"user": prompt, "think": think, "answer": str(n_digits)}


@register 
class TransUnseenOpTransfer(MicroSkill):
    """Query uses an operator not seen in examples. Which existing combo applies?
    Teaches: infer unseen operator behavior from context."""
    name = "trans_unseen_transfer"
    puzzle_type = "transformation"
    description = "Transfer a known operation to an unseen operator symbol"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        ops = {"add": lambda a,b: a+b, "sub": lambda a,b: a-b, "mul": lambda a,b: a*b}
        real_op = rng.choice(list(ops.keys()))
        fn = ops[real_op]

        seen_sym = rng.choice(['+', '-', '*'])
        unseen_sym = rng.choice([s for s in ['/', '|', '^', '@', '#'] if s != seen_sym])

        # Show examples with seen_sym
        examples = []
        for _ in range(3):
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            examples.append(f"  {a}{seen_sym}{b} = {fn(a,b)}")

        qa, qb = rng.randint(10, 99), rng.randint(10, 99)
        q_answer = fn(qa, qb)

        prompt = (
            f"Examples (operator '{seen_sym}'):\n" + "\n".join(examples) +
            f"\n\nQuery uses '{unseen_sym}': {qa}{unseen_sym}{qb} = ?\n"
            f"Assume '{unseen_sym}' behaves like '{seen_sym}'."
        )

        think = f"'{seen_sym}' is {real_op}. Assume '{unseen_sym}' = {real_op}.\n{real_op}({qa},{qb}) = {q_answer}"
        return {"user": prompt, "think": think, "answer": str(q_answer)}


@register
class TransOpenWorldEncode(MicroSkill):
    """Given a mapping with gaps, encode a number that needs a fresh symbol.
    Teaches: assign unused symbols to unmapped digits deterministically."""
    name = "trans_open_encode"
    puzzle_type = "transformation"
    description = "Encode a number when some digits need fresh (unseen) symbols"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        # Create a partial mapping (7-8 digits mapped, 2-3 missing)
        all_digits = list(range(10))
        mapped_count = rng.randint(7, 8)
        mapped_digits = rng.sample(all_digits, mapped_count)
        unmapped_digits = [d for d in all_digits if d not in mapped_digits]

        safe_syms = list('!@#$%^&*()-_=+[]:,.<>?/~`')
        symbols = rng.sample(safe_syms, mapped_count)
        mapping = dict(zip(mapped_digits, symbols))
        
        # Number that needs at least one unmapped digit
        val = rng.randint(10, 9999)
        digits = [int(c) for c in str(val)]
        needs_fresh = [d for d in digits if d not in mapping]
        
        if not needs_fresh:
            # Force an unmapped digit
            if unmapped_digits:
                val = int(str(unmapped_digits[0]) + str(rng.randint(10, 99)))
                digits = [int(c) for c in str(val)]
                needs_fresh = [d for d in digits if d not in mapping]
        
        if not needs_fresh:
            return None

        map_str = " ".join(f"{d}={s}" for d, s in sorted(mapping.items()))
        unused_syms = [s for s in safe_syms if s not in symbols][:5]

        prompt = (
            f"Known mapping: {map_str}\n"
            f"Unused symbols: {', '.join(unused_syms[:3])}\n"
            f"Encode the number {val}. Assign fresh symbols for unmapped digits."
        )

        think_lines = []
        encoded = []
        fresh_map = dict(mapping)
        fresh_idx = 0
        for d in digits:
            if d in fresh_map:
                encoded.append(fresh_map[d])
                think_lines.append(f"  {d} → {fresh_map[d]} (known)")
            else:
                fresh_sym = unused_syms[fresh_idx] if fresh_idx < len(unused_syms) else '?'
                fresh_map[d] = fresh_sym
                encoded.append(fresh_sym)
                think_lines.append(f"  {d} → {fresh_sym} (fresh)")
                fresh_idx += 1

        answer = ''.join(encoded)
        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


# ============================================================
# ROUND 3.1 CONTROL SKILLS — ranking, rejection, detection
# ============================================================

@register
class TransDetectOpPos(MicroSkill):
    """Identify which position in a 5-char symbolic LHS is the operator slot.
    Foundational for cipher-digit: model must learn structure before solving."""
    name = "trans_detect_op_pos"
    puzzle_type = "transformation"
    description = "Identify operator position in 5-char symbolic equations"
    weight = 6.0
    max_pool = 8000

    def generate_one(self, rng, difficulty="medium"):
        safe = list('!@#$%^&*()[]{}<>?/|~`\\:;,.')
        digit_syms = rng.sample(safe, min(12, len(safe)))
        # Pick operator position (usually 2, sometimes 0 or 1 for variety)
        op_pos = rng.choices([2, 0, 1, 3], weights=[70, 10, 10, 10])[0]
        op_syms = rng.sample([s for s in safe if s not in digit_syms], min(3, len(safe) - len(digit_syms)))
        if not op_syms:
            return None

        examples = []
        for _ in range(rng.randint(3, 5)):
            chars = rng.sample(digit_syms, 4)
            op = rng.choice(op_syms)
            lhs = list(chars)
            lhs.insert(op_pos, op)
            rhs_len = rng.randint(2, 4)
            rhs = ''.join(rng.sample(digit_syms, min(rhs_len, len(digit_syms))))
            examples.append(f"  {''.join(lhs[:5])} = {rhs}")

        prompt = "Examples:\n" + "\n".join(examples) + \
                 "\n\nWhich position (0-4) is the operator slot in the 5-character left side?"

        # Reasoning: check what varies at each position
        think_lines = []
        for pos in range(5):
            if pos == op_pos:
                think_lines.append(f"Pos {pos}: operator symbols {sorted(set(op_syms))}")
            else:
                think_lines.append(f"Pos {pos}: digit symbols (many distinct)")
        think_lines.append(f"Operator slot = {op_pos}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": str(op_pos)}


@register
class TransRegimeVote(MicroSkill):
    """Given support examples, infer the output regime (raw/rev/abs/opprefix/dsum).
    Directly targets the 51% of seen-op misses that are pure style errors."""
    name = "trans_regime_vote"
    puzzle_type = "transformation"
    description = "Infer shared output format regime from support examples"
    weight = 6.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        regimes = {
            "raw": lambda v, op_c: str(v),
            "rev": lambda v, op_c: (str(v)[::-1] if v >= 0 else '-' + str(v)[1:][::-1]),
            "abs": lambda v, op_c: str(abs(v)),
            "opprefix": lambda v, op_c: op_c + str(abs(v)),
            "dsum": lambda v, op_c: str(sum(int(c) for c in str(abs(v)))),
        }
        regime_name = rng.choice(list(regimes.keys()))
        render = regimes[regime_name]

        ops = {"add": lambda a,b: a+b, "sub": lambda a,b: a-b,
               "mul": lambda a,b: a*b}
        op_name = rng.choice(list(ops.keys()))
        calc = ops[op_name]
        op_char = rng.choice(['+', '-', '*', '/', '|', '^'])

        rows = []
        think_lines = []
        for i in range(rng.randint(2, 3)):
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            raw = calc(a, b)
            shown = render(raw, op_char)
            rows.append(f"  {a}{op_char}{b} = {shown}")
            think_lines.append(f"Ex{i+1}: {op_name}({a},{b})={raw}, shown={shown}")

        think_lines.append(f"All examples fit regime = {regime_name}")

        prompt = "Support examples:\n" + "\n".join(rows) + \
                 "\n\nWhat output format regime do these examples use?\n" + \
                 "Choices: raw, rev, abs, opprefix, dsum"
        return {"user": prompt, "think": "\n".join(think_lines), "answer": regime_name}


@register
class TransCloseMissReject(MicroSkill):
    """Reject the closest plausible rival using a discriminating example.
    Teaches: sub vs absdiff, muladd1 vs mulsub1, raw vs rev — the real confusables."""
    name = "trans_close_miss_reject"
    puzzle_type = "transformation"
    description = "Reject the most confusable rival candidate using support evidence"
    weight = 6.0
    max_pool = 10000

    # Confusable pairs that actually cause errors in competition data
    RIVAL_PAIRS = [
        ("sub", "absdiff", lambda a,b: a-b, lambda a,b: abs(a-b)),
        ("add", "add1", lambda a,b: a+b, lambda a,b: a+b+1),
        ("mul", "muladd1", lambda a,b: a*b, lambda a,b: a*b+1),
        ("mul", "mulsub1", lambda a,b: a*b, lambda a,b: a*b-1),
        ("add", "sub", lambda a,b: a+b, lambda a,b: a-b),
        ("cat", "rcat", lambda a,b: int(f"{a}{b}"), lambda a,b: int(f"{b}{a}")),
    ]

    def generate_one(self, rng, difficulty="medium"):
        name_a, name_b, fn_a, fn_b = rng.choice(self.RIVAL_PAIRS)

        # Generate two examples, use the CORRECT op (randomly A or B)
        correct_name = rng.choice([name_a, name_b])
        correct_fn = fn_a if correct_name == name_a else fn_b
        wrong_name = name_b if correct_name == name_a else name_a
        wrong_fn = fn_b if correct_name == name_a else fn_a

        a1, b1 = rng.randint(10, 99), rng.randint(10, 99)
        a2, b2 = rng.randint(10, 99), rng.randint(10, 99)

        try:
            r1 = correct_fn(a1, b1)
            r2 = correct_fn(a2, b2)
            w1 = wrong_fn(a1, b1)
            w2 = wrong_fn(a2, b2)
        except (ValueError, ZeroDivisionError):
            return None

        # At least one example must disagree
        if r1 == w1 and r2 == w2:
            return None

        op = rng.choice(['+', '-', '*', '|'])
        prompt = (
            f"Examples:\n"
            f"  {a1}{op}{b1} = {r1}\n"
            f"  {a2}{op}{b2} = {r2}\n\n"
            f"Two candidates:\n"
            f"  A = {name_a}\n"
            f"  B = {name_b}\n\n"
            f"Which candidate should be rejected?"
        )

        think_lines = [
            f"A: {name_a}({a1},{b1})={fn_a(a1,b1)} vs {r1} {'→ MATCH' if fn_a(a1,b1)==r1 else '→ MISMATCH'}"
            f" ; {name_a}({a2},{b2})={fn_a(a2,b2)} vs {r2} {'→ MATCH' if fn_a(a2,b2)==r2 else '→ MISMATCH'}",
            f"B: {name_b}({a1},{b1})={fn_b(a1,b1)} vs {r1} {'→ MATCH' if fn_b(a1,b1)==r1 else '→ MISMATCH'}"
            f" ; {name_b}({a2},{b2})={fn_b(a2,b2)} vs {r2} {'→ MATCH' if fn_b(a2,b2)==r2 else '→ MISMATCH'}",
        ]

        wrong_label = 'A' if wrong_name == name_a else 'B'
        think_lines.append(f"Reject {wrong_label} ({wrong_name}).")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": f"Reject {wrong_label} ({wrong_name})"}


@register
class TransCandidateRank(MicroSkill):
    """Given 3 surviving candidates, pick the one that fits all support examples.
    Directly targets wrong-combo selection — the #1 numeric failure mode."""
    name = "trans_candidate_rank"
    puzzle_type = "transformation"
    description = "Choose best candidate among 3 surviving arithmetic+style options"
    weight = 7.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        orders = {
            "AB_CD": lambda a,b,c,d: (a*10+b, c*10+d),
            "BA_DC": lambda a,b,c,d: (b*10+a, d*10+c),
            "AB_DC": lambda a,b,c,d: (a*10+b, d*10+c),
            "BA_CD": lambda a,b,c,d: (b*10+a, c*10+d),
        }
        ops = {"add": lambda l,r: l+r, "sub": lambda l,r: l-r,
               "mul": lambda l,r: l*r, "absdiff": lambda l,r: abs(l-r)}
        fmts = {
            "raw": lambda v: str(v),
            "rev": lambda v: str(v)[::-1] if v >= 0 else '-'+str(v)[1:][::-1],
            "abs": lambda v: str(abs(v)),
        }

        # Pick the correct combo
        real_order = rng.choice(list(orders.keys()))
        real_op = rng.choice(list(ops.keys()))
        real_fmt = rng.choice(list(fmts.keys()))

        # Generate 2 examples
        for _ in range(50):
            a1, b1 = rng.randint(1, 9), rng.randint(1, 9)
            c1, d1 = rng.randint(1, 9), rng.randint(1, 9)
            L1, R1 = orders[real_order](a1, b1, c1, d1)
            v1 = ops[real_op](L1, R1)
            f1 = fmts[real_fmt](v1)

            a2, b2 = rng.randint(1, 9), rng.randint(1, 9)
            c2, d2 = rng.randint(1, 9), rng.randint(1, 9)
            L2, R2 = orders[real_order](a2, b2, c2, d2)
            v2 = ops[real_op](L2, R2)
            f2 = fmts[real_fmt](v2)

            # Build 2 wrong candidates that fail on at least one example
            wrong_combos = []
            for wo in rng.sample(list(orders.keys()), len(orders)):
                for wop in rng.sample(list(ops.keys()), len(ops)):
                    for wf in rng.sample(list(fmts.keys()), len(fmts)):
                        if (wo, wop, wf) == (real_order, real_op, real_fmt):
                            continue
                        wL1, wR1 = orders[wo](a1, b1, c1, d1)
                        wv1 = ops[wop](wL1, wR1)
                        wf1 = fmts[wf](wv1)
                        wL2, wR2 = orders[wo](a2, b2, c2, d2)
                        wv2 = ops[wop](wL2, wR2)
                        wf2 = fmts[wf](wv2)
                        # Wrong must fail on at least one example
                        if wf1 != f1 or wf2 != f2:
                            wrong_combos.append((wo, wop, wf, wf1, wf2))
                        if len(wrong_combos) >= 2:
                            break
                    if len(wrong_combos) >= 2:
                        break
                if len(wrong_combos) >= 2:
                    break

            if len(wrong_combos) < 2:
                continue

            sym = rng.choice(['+', '-', '*', '|'])
            ex1_str = f"{a1}{b1}{sym}{c1}{d1} = {f1}"
            ex2_str = f"{a2}{b2}{sym}{c2}{d2} = {f2}"

            labels = ['A', 'B', 'C']
            combos_list = [(real_order, real_op, real_fmt)] + \
                          [(w[0], w[1], w[2]) for w in wrong_combos[:2]]
            rng.shuffle(combos_list)
            correct_label = labels[combos_list.index((real_order, real_op, real_fmt))]

            cand_strs = [f"  {labels[i]}: {c[0]}|{c[1]}|{c[2]}" for i, c in enumerate(combos_list)]

            prompt = (
                f"Support:\n  {ex1_str}\n  {ex2_str}\n\n"
                f"Candidates:\n" + "\n".join(cand_strs) + "\n\n"
                f"Which candidate fits both examples?"
            )

            think_lines = []
            for i, (o, op, f) in enumerate(combos_list):
                tL1, tR1 = orders[o](a1, b1, c1, d1)
                tv1 = ops[op](tL1, tR1)
                tf1 = fmts[f](tv1)
                match1 = "→ MATCH" if tf1 == f1 else "→ MISMATCH"
                tL2, tR2 = orders[o](a2, b2, c2, d2)
                tv2 = ops[op](tL2, tR2)
                tf2 = fmts[f](tv2)
                match2 = "→ MATCH" if tf2 == f2 else "→ MISMATCH"
                think_lines.append(
                    f"{labels[i]}: Ex1 {op}({tL1},{tR1})={tv1} {f}={tf1} {match1}"
                    f" ; Ex2 {op}({tL2},{tR2})={tv2} {f}={tf2} {match2}")

            think_lines.append(f"Only {correct_label} fits both. Lock {correct_label}.")
            return {"user": prompt, "think": "\n".join(think_lines), "answer": correct_label}

        return None


@register
class BitWitnessPick(MicroSkill):
    """Given two candidate rules and 3 examples, which example best discriminates?
    Teaches witness selection — the key to breaking wrong-rule persistence."""
    name = "bit_witness_pick"
    puzzle_type = "bit_manipulation"
    description = "Pick the example that best discriminates between two candidate rules"
    weight = 6.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        ops = {
            "xor": lambda a, b: a ^ b,
            "and": lambda a, b: a & b,
            "or": lambda a, b: a | b,
            "xnor": lambda a, b: 1 - (a ^ b),
            "and_not": lambda a, b: a & (1 - b),
            "or_not": lambda a, b: a | (1 - b),
        }
        shifts = {
            "shr1": lambda x: x[1:] + '0',
            "shl1": lambda x: '0' + x[:-1],
            "ror1": lambda x: x[-1] + x[:-1],
            "rol1": lambda x: x[1:] + x[0],
            "shr2": lambda x: x[2:] + '00',
            "shl2": lambda x: '00' + x[:-2],
        }

        def apply_shift(x_str, shift_name):
            return shifts[shift_name](x_str)

        def apply_gate(a_str, b_str, gate_name):
            fn = ops[gate_name]
            return ''.join(str(fn(int(a), int(b))) for a, b in zip(a_str, b_str))

        # Pick two rules that differ only in gate
        shift_a = rng.choice(list(shifts.keys()))
        shift_b = rng.choice(list(shifts.keys()))
        gate_pair = rng.choice([("xor", "and"), ("xor", "or"), ("and", "or"),
                                 ("xnor", "or"), ("xnor", "xor"), ("and", "and_not")])
        gate_real, gate_wrong = gate_pair

        # Generate 3 examples
        examples = []
        for _ in range(3):
            x = rng.randint(0, 255)
            x_str = format(x, '08b')
            a = apply_shift(x_str, shift_a)
            b = apply_shift(x_str, shift_b)
            out_real = apply_gate(a, b, gate_real)
            out_wrong = apply_gate(a, b, gate_wrong)
            examples.append((x_str, out_real, out_wrong, a, b))

        # Find best witness: the example with biggest difference between rules
        best_idx = 0
        best_diff = 0
        for i, (x_str, out_real, out_wrong, a, b) in enumerate(examples):
            diff = sum(r != w for r, w in zip(out_real, out_wrong))
            if diff > best_diff:
                best_diff = diff
                best_idx = i

        prompt = (
            f"Two candidate rules:\n"
            f"  Rule 1: A={shift_a}(x), B={shift_b}(x), output={gate_real}(A,B)\n"
            f"  Rule 2: A={shift_a}(x), B={shift_b}(x), output={gate_wrong}(A,B)\n\n"
            f"Examples with known outputs:\n"
        )
        for i, (x_str, out_real, _, _, _) in enumerate(examples):
            prompt += f"  Ex{i+1}: {x_str} → {out_real}\n"
        prompt += f"\nWhich example best distinguishes Rule 1 from Rule 2?"

        think_lines = []
        for i, (x_str, out_real, out_wrong, a, b) in enumerate(examples):
            diff = sum(r != w for r, w in zip(out_real, out_wrong))
            r1_match = "→ MATCH" if out_real == out_real else "→ MATCH"
            r2_val = out_wrong
            r2_match = "→ MATCH" if out_wrong == out_real else f"→ MISMATCH ({diff} bits differ)"
            think_lines.append(
                f"Ex{i+1}: Rule1={out_real} Rule2={out_wrong} — {diff} bits differ")
        think_lines.append(f"Ex{best_idx+1} has most disagreement ({best_diff} bits). Best witness.")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": f"Ex{best_idx+1}"}


# ============================================================
# ROUND 3.1 COMPLEMENTARY SKILLS (from reviewer scaffolds)
# ============================================================

@register
class TransMappingExtend(MicroSkill):
    """Given a partial cipher mapping and one aligned decode, what new bindings are forced?
    Narrower than full crack — teaches incremental mapping propagation."""
    name = "trans_mapping_extend"
    puzzle_type = "transformation"
    description = "Extend partial symbol→digit mapping from an aligned decode"
    weight = 6.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        safe = list('!@#$%^&*()[]{}<>?/|~`')
        syms = rng.sample(safe, 10)
        full_map = {syms[d]: d for d in range(10)}
        rev_map = {d: syms[d] for d in range(10)}

        # Pick a 4-digit number to decode
        digits = [rng.randint(0, 9) for _ in range(4)]
        cipher = ''.join(rev_map[d] for d in digits)

        # Hide 1-2 of the mappings
        unique_digits = list(dict.fromkeys(digits))
        if len(unique_digits) < 2:
            return None
        n_hide = min(rng.randint(1, 2), len(unique_digits) - 1)
        hidden = set(rng.sample(unique_digits, n_hide))

        known_pairs = []
        new_pairs = []
        for d in range(10):
            if d not in hidden:
                known_pairs.append(f"{rev_map[d]}={d}")

        think_lines = []
        for sym, d in zip(cipher, digits):
            if d in hidden:
                think_lines.append(f"  {sym} aligns with {d} → NEW: {sym}={d}")
                pair = f"{sym}={d}"
                if pair not in new_pairs:
                    new_pairs.append(pair)
            else:
                think_lines.append(f"  {sym}={d} (known)")

        if not new_pairs:
            return None

        prompt = (
            f"Known mapping: {' '.join(known_pairs)}\n"
            f"Aligned decode:\n  cipher: {cipher}\n  digits: {''.join(str(d) for d in digits)}\n"
            f"What new bindings are forced?"
        )
        return {"user": prompt, "think": "\n".join(think_lines), "answer": ", ".join(new_pairs)}


@register
class TransFormatScope(MicroSkill):
    """Identify Format[op] for a specific operator in a multi-operator row.
    Targets the ambiguity where global Format: is wrong for the query operator."""
    name = "trans_format_scope"
    puzzle_type = "transformation"
    description = "Identify operator-scoped format in multi-operator rows"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        target_sym = rng.choice(['@', '#', '!', '&', '|'])
        other_sym = rng.choice([s for s in ['+', '-', '*', '^', '/'] if s != target_sym])

        fmts = {
            "raw": lambda v, op: str(v),
            "rev": lambda v, op: (str(v)[::-1] if v >= 0 else '-' + str(v)[1:][::-1]),
            "abs": lambda v, op: str(abs(v)),
            "opprefix": lambda v, op: op + str(abs(v)),
        }
        target_fmt = rng.choice(list(fmts.keys()))
        other_fmt = rng.choice([f for f in fmts if f != target_fmt])

        rows = []
        think_lines = []

        # Target operator examples (sub so negatives reveal format)
        for i in range(2):
            a = rng.randint(10, 39)
            b = rng.randint(50, 99)
            raw_val = a - b  # negative
            shown = fmts[target_fmt](raw_val, target_sym)
            rows.append(f"  {a}{target_sym}{b} = {shown}")
            think_lines.append(f"  {a}{target_sym}{b}: sub={raw_val}, shown={shown} → {target_fmt}")

        # Other operator examples (add, positive)
        for i in range(2):
            a = rng.randint(10, 49)
            b = rng.randint(10, 49)
            raw_val = a + b
            shown = fmts[other_fmt](raw_val, other_sym)
            rows.append(f"  {a}{other_sym}{b} = {shown}")

        rng.shuffle(rows)
        think_lines.append(f"Format[{target_sym}] = {target_fmt}")

        prompt = (
            "Examples:\n" + "\n".join(rows) +
            f"\n\nWhat is Format[{target_sym}]?\nOptions: raw, rev, abs, opprefix"
        )
        return {"user": prompt, "think": "\n".join(think_lines), "answer": target_fmt}


@register
class TransOperatorPartition(MicroSkill):
    """Partition 3 operator symbols into alias groups based on examples.
    Row-level version of alias detection — beyond pairwise."""
    name = "trans_operator_partition"
    puzzle_type = "transformation"
    description = "Partition 3 operators into alias groups by shared behavior"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        sym_a, sym_b, sym_c = rng.sample(['+', '-', '*', '@', '#', '!', '&', '|'], 3)
        ops = {"add": lambda a,b: a+b, "sub": lambda a,b: a-b,
               "mul": lambda a,b: a*b}

        # Two operators share one op, third is different
        shared_op = rng.choice(list(ops.keys()))
        diff_op = rng.choice([o for o in ops if o != shared_op])

        # sym_a and sym_b are aliases, sym_c is different
        rows = []
        for sym, op_name in [(sym_a, shared_op), (sym_a, shared_op),
                              (sym_b, shared_op), (sym_b, shared_op),
                              (sym_c, diff_op), (sym_c, diff_op)]:
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            result = ops[op_name](a, b)
            rows.append(f"  {a}{sym}{b} = {result}")
        rng.shuffle(rows)

        group_alias = ','.join(sorted([sym_a, sym_b]))
        answer = f"{group_alias} | {sym_c}"

        think_lines = [
            f"'{sym_a}' examples fit {shared_op}",
            f"'{sym_b}' examples fit {shared_op}",
            f"'{sym_c}' examples fit {diff_op}",
            f"Partition: {answer}",
        ]

        prompt = (
            "Examples:\n" + "\n".join(rows) +
            "\n\nGroup the operator symbols by shared behavior.\n"
            "Return like: sym1,sym2 | sym3"
        )
        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


@register
class TransStyleThenCompute(MicroSkill):
    """End-to-end: infer style from support examples, then compute and render query answer.
    Bridges the gap between style classification and actual execution."""
    name = "trans_style_then_compute"
    puzzle_type = "transformation"
    description = "Infer style from examples then compute+render query answer"
    weight = 6.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_transform import _make_operands, _calc, _fmt, COMBO_DISPLAY
        orders = {"AB_CD": "AB,CD", "BA_DC": "BA,DC", "AB_DC": "AB,DC", "BA_CD": "BA,CD"}
        ops = {"add": lambda L,R: L+R, "sub": lambda L,R: L-R,
               "mul": lambda L,R: L*R, "absdiff": lambda L,R: abs(L-R)}

        order = rng.choice(list(orders.keys()))
        op = rng.choice(list(ops.keys()))
        fmt = rng.choice(["raw", "rev", "abs", "dsum"])
        op_char = rng.choice(['+', '-', '*', '/', '|'])

        def compute(a, b):
            L, R = _make_operands(a//10, a%10, b//10, b%10, order)
            val = _calc(L, R, op)
            if val is None: return None, None, None
            shown = _fmt(val, fmt, op_char=op_char)
            return L, R, val, shown

        examples = []
        for _ in range(20):
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            result = compute(a, b)
            if result[0] is None or len(str(result[3])) > 5: continue
            examples.append((a, b, result[0], result[1], result[2], result[3]))
            if len(examples) == 2: break
        if len(examples) < 2: return None

        qa, qb = rng.randint(10, 99), rng.randint(10, 99)
        qresult = compute(qa, qb)
        if qresult[0] is None: return None
        qL, qR, qval, qshown = qresult

        prompt = (
            "Examples:\n"
            + "\n".join(f"  {a}{op_char}{b} = {shown}" for a, b, _, _, _, shown in examples)
            + f"\n\nQuery: {qa}{op_char}{qb} = ?"
        )

        think_lines = []
        for a, b, L, R, val, shown in examples:
            think_lines.append(f"  {a}{op_char}{b}: {orders[order]} L={L} R={R} {op}={val} {fmt}={shown}")
        think_lines.append(f"Style = {fmt}")
        think_lines.append(f"Query: {orders[order]} L={qL} R={qR} {op}({qL},{qR})={qval} {fmt}={qshown}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": str(qshown)}


# ============================================================
# ROUND 5: SYMBOLIC TASK IDENTIFICATION
# ============================================================

@register
class TransTaskIdentify(MicroSkill):
    """Given a transformation puzzle's examples, identify the task type:
    numeric (visible digits), cipher-digit (all symbols, bijective cipher),
    or non-cipher symbolic (string/pattern operation).
    Directly targets the 2% symbolic accuracy — model applies cipher-digit to everything."""
    name = "trans_task_identify"
    puzzle_type = "transformation"
    description = "Identify transformation subtype: numeric vs cipher-digit vs non-cipher symbolic"
    weight = 8.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        task_type = rng.choice(["numeric", "cipher_digit", "cipher_digit", "cipher_digit"])
        # Weight toward cipher_digit since that's what we need to teach

        if task_type == "numeric":
            # Visible digits + operator
            op = rng.choice(['+', '-', '*', '/', '|'])
            examples = []
            for _ in range(3):
                a, b = rng.randint(10, 99), rng.randint(10, 99)
                examples.append(f"  {a}{op}{b} = {a+b}")
            lhs_sample = f"{rng.randint(10,99)}{op}{rng.randint(10,99)}"
            prompt = "Examples:\n" + "\n".join(examples) + f"\nQuery: {lhs_sample} = ?\n\nWhat type of transformation is this?"
            think = "LHS has visible digits and an operator symbol. Output is numeric.\nTask = numeric"
            answer = "numeric"

        else:  # cipher_digit
            safe = list('!@#$%^&*()[]{}<>?/|~`')
            syms = rng.sample(safe, min(10, len(safe)))
            op_syms = rng.sample([s for s in safe if s not in syms], min(2, len(safe) - len(syms)))
            if not op_syms: return None
            examples = []
            for _ in range(3):
                lhs = ''.join(rng.sample(syms, 4)[:2] + [rng.choice(op_syms)] + rng.sample(syms, 4)[2:4])
                rhs = ''.join(rng.sample(syms, rng.randint(2, 4)))
                examples.append(f"  {lhs} = {rhs}")
            query_lhs = ''.join(rng.sample(syms, 2) + [rng.choice(op_syms)] + rng.sample(syms, 2))
            prompt = "Examples:\n" + "\n".join(examples) + f"\nQuery: {query_lhs} = ?\n\nWhat type of transformation is this?"
            think = "LHS is 5 symbols with no visible digits. All characters are symbols.\nEach position maps to a digit via bijective cipher.\nTask = cipher_digit"
            answer = "cipher_digit"

        return {"user": prompt, "think": think, "answer": answer}


# ============================================================
# ROUND 4: ENCRYPTION MASTERY SKILLS
# ============================================================

@register
class EncTableFillRow(MicroSkill):
    """Fill one row of the positional mapping table from a cipher→plain word pair.
    Teaches exact position→letter extraction — the core enc skill."""
    name = "enc_table_fill_row"
    puzzle_type = "encryption"
    description = "Extract mapping entries from cipher+plain word pair"
    weight = 10.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None
        word = rng.choice([w for w in vocab if len(w) >= 4])
        alpha = list('abcdefghijklmnopqrstuvwxyz')
        rng.shuffle(alpha)
        c2p = dict(zip(alpha, 'abcdefghijklmnopqrstuvwxyz'))
        p2c = {v: k for k, v in c2p.items()}
        cipher_word = ''.join(p2c[ch] for ch in word)

        prompt = (
            f"Cipher word: {cipher_word}\n"
            f"Plain word: {word}\n"
            f"List all cipher=plain mappings from this pair."
        )
        seen = set()
        think_lines = []
        mappings = []
        for cc, pp in zip(cipher_word, word):
            if cc not in seen:
                think_lines.append(f"  {cc}={pp}")
                mappings.append(f"{cc}={pp}")
                seen.add(cc)

        answer = " ".join(mappings)
        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


@register
class EncDetectSwap(MicroSkill):
    """Given a mapping table with ONE swapped pair, find the error.
    Directly targets 63% of enc failures: adjacent letter swaps."""
    name = "enc_detect_swap"
    puzzle_type = "encryption"
    description = "Find the swapped pair in a nearly-correct mapping"
    weight = 12.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        alpha = list('abcdefghijklmnopqrstuvwxyz')
        plain = list('abcdefghijklmnopqrstuvwxyz')
        rng.shuffle(plain)
        c2p = dict(zip(alpha, plain))

        swap_pos = rng.randint(0, 24)
        c1, c2 = alpha[swap_pos], alpha[swap_pos + 1]
        bad_c2p = dict(c2p)
        bad_c2p[c1], bad_c2p[c2] = bad_c2p[c2], bad_c2p[c1]

        vocab = load_vocab()
        if not vocab: return None
        verify_words = [w for w in vocab if c2p[c1] in w or c2p[c2] in w]
        if not verify_words: return None
        verify_word = rng.choice(verify_words)
        p2c = {v: k for k, v in c2p.items()}
        cipher_word = ''.join(p2c[ch] for ch in verify_word)

        map_lines = _enc_build_flat_map(bad_c2p)

        prompt = (
            "This mapping has ONE swapped pair (two adjacent letters reversed).\n"
            + "\n".join(map_lines) + "\n\n"
            f"Verify with: {cipher_word} should decode to {verify_word}\n"
            f"Which two letters are swapped?"
        )

        bad_decode = ''.join(bad_c2p[c] for c in cipher_word)
        think_lines = [
            f"Decode {cipher_word} with map: {bad_decode}",
            f"Expected: {verify_word}",
        ]
        mismatches = [(i, bad_decode[i], verify_word[i]) for i in range(len(bad_decode)) if bad_decode[i] != verify_word[i]]
        for pos, got, expected in mismatches:
            think_lines.append(f"  pos {pos}: got '{got}' expected '{expected}' — cipher '{cipher_word[pos]}'")
        think_lines.append(f"Swapped: {c1}↔{c2} (should be {c1}={c2p[c1]}, {c2}={c2p[c2]})")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": f"{c1}↔{c2}"}


@register
class EncWordFromTable(MicroSkill):
    """Given the mapping table and a cipher word, decode it character by character.
    The core execution skill — prevents language prior from overriding."""
    name = "enc_word_from_table"
    puzzle_type = "encryption"
    description = "Decode a cipher word using the flat map + vocab verify"
    weight = 10.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None
        word = rng.choice([w for w in vocab if len(w) >= 4])

        alpha = list('abcdefghijklmnopqrstuvwxyz')
        plain = list('abcdefghijklmnopqrstuvwxyz')
        rng.shuffle(plain)
        c2p = dict(zip(alpha, plain))
        p2c = {v: k for k, v in c2p.items()}
        cipher_word = ''.join(p2c[ch] for ch in word)

        map_lines = _enc_build_flat_map(c2p)

        prompt = (
            "\n".join(map_lines) + "\n\n"
            f"Decode: {cipher_word}"
        )

        think_lines = []
        for c in cipher_word:
            think_lines.append(f"  {c}={c2p[c]}")
        think_lines.append(f"= {word}")
        think_lines.append(f"{word} in vocab ✓")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": word}


# ============================================================
# ENC: FLAT MAP HELPERS (new oversolve format)
# ============================================================

def _enc_table_row_for_letter(c):
    """Return (row_label, position_in_row) for cipher letter c.
    DEPRECATED — kept for backward compat. New skills use flat map."""
    ranges = [('a','e'), ('f','j'), ('k','o'), ('p','t'), ('u','z')]
    for start, end in ranges:
        si, ei = ord(start), ord(end)
        if si <= ord(c) <= ei:
            return f"{start}-{end}", ord(c) - si
    return None, None


def _enc_build_flat_map(c2p, known_set=None):
    """Build 2-line flat map: 'Map: a=? b=t c=u ...'"""
    def _fmt(c):
        if known_set and c not in known_set:
            return f"{c}=?"
        return f"{c}={c2p.get(c, '?')}"
    line1 = " ".join(_fmt(chr(i)) for i in range(ord('a'), ord('n')))
    line2 = " ".join(_fmt(chr(i)) for i in range(ord('n'), ord('z') + 1))
    return [f"Map: {line1}", f"     {line2}"]


def _enc_build_table_lines(c2p, known_set=None):
    """Build explicit-pair table lines — DEPRECATED, use _enc_build_flat_map."""
    return _enc_build_flat_map(c2p, known_set)


@register
class EncTableLookupSingle(MicroSkill):
    """Given a flat map and ONE cipher letter, find its plain letter."""
    name = "enc_table_lookup_single"
    puzzle_type = "encryption"
    description = "Look up one cipher letter in the flat mapping"
    weight = 15.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        alpha = list('abcdefghijklmnopqrstuvwxyz')
        plain = list('abcdefghijklmnopqrstuvwxyz')
        rng.shuffle(plain)
        c2p = dict(zip(alpha, plain))

        n_known = rng.randint(12, 22)
        known_set = set(rng.sample(alpha, n_known))
        map_lines = _enc_build_flat_map(c2p, known_set)

        target = rng.choice(sorted(known_set))

        prompt = (
            "\n".join(map_lines) + "\n\n"
            f"What plain letter does cipher '{target}' map to?"
        )

        think = f"From map: {target}={c2p[target]}"

        return {"user": prompt, "think": think, "answer": c2p[target]}


@register
class EncDecodeWithRowRefs(MicroSkill):
    """Decode a full cipher word using the flat map."""
    name = "enc_decode_with_rows"
    puzzle_type = "encryption"
    description = "Decode cipher word letter by letter using flat map"
    weight = 15.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None
        word = rng.choice([w for w in vocab if len(w) >= 4])

        alpha = list('abcdefghijklmnopqrstuvwxyz')
        plain = list('abcdefghijklmnopqrstuvwxyz')
        rng.shuffle(plain)
        c2p = dict(zip(alpha, plain))
        p2c = {v: k for k, v in c2p.items()}
        cipher_word = ''.join(p2c[ch] for ch in word)

        map_lines = _enc_build_flat_map(c2p)

        prompt = (
            "\n".join(map_lines) + "\n\n"
            f"Decode each letter of: {cipher_word}"
        )

        think_lines = []
        for c in cipher_word:
            think_lines.append(f"  {c}={c2p[c]}")
        think_lines.append(f"= {word}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": word}


@register
class EncDecodeAndVocabMatch(MicroSkill):
    """Decode cipher word with unknowns, then exhaustively test all same-length
    vocab words with per-letter match/reject. The core oversolve skill."""
    name = "enc_decode_vocab_match"
    puzzle_type = "encryption"
    description = "Decode with unknowns, exhaustive candidate test with rejection reasons"
    weight = 12.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None
        word = rng.choice([w for w in vocab if len(w) >= 4])

        alpha = list('abcdefghijklmnopqrstuvwxyz')
        plain = list('abcdefghijklmnopqrstuvwxyz')
        rng.shuffle(plain)
        c2p = dict(zip(alpha, plain))
        p2c = {v: k for k, v in c2p.items()}
        cipher_word = ''.join(p2c[ch] for ch in word)

        # Make some mappings unknown (1-3 letters in the word)
        word_cipher_letters = list(set(cipher_word))
        n_unknown = min(rng.randint(1, 3), len(word_cipher_letters))
        unknowns = set(rng.sample(word_cipher_letters, n_unknown))

        # Build partial decode
        partial = []
        for c in cipher_word:
            if c in unknowns:
                partial.append('?')
            else:
                partial.append(c2p[c])
        partial_str = ''.join(partial)

        from collections import defaultdict
        by_len = defaultdict(list)
        for w in sorted(vocab): by_len[len(w)].append(w)

        # Build the partial map (excluding unknowns)
        partial_c2p = {c: p for c, p in c2p.items() if c not in unknowns}
        partial_p2c = {v: k for k, v in partial_c2p.items()}
        map_lines = _enc_build_flat_map(partial_c2p)

        prompt = (
            f"{len(word)}-letter vocab: {', '.join(by_len[len(word)])}\n\n"
            + "\n".join(map_lines) + "\n\n"
            f"Decode: {cipher_word}"
        )

        # Build exhaustive test trace (like the new trace format)
        think_lines = []
        decode_parts = " ".join(f"{c}={partial_c2p.get(c, '?')}" for c in cipher_word)
        think_lines.append(f"{cipher_word} ({len(cipher_word)}): {decode_parts} = {partial_str}")
        think_lines.append(f"Test {len(word)}-letter vocab:")

        for candidate in by_len[len(word)]:
            reject_reason = None
            new_maps = []
            for pos, (cc, wc) in enumerate(zip(cipher_word, candidate)):
                if cc in partial_c2p:
                    if partial_c2p[cc] != wc:
                        reject_reason = f"pos {pos}: {partial_c2p[cc]}≠{wc}"
                        break
                else:
                    if wc in partial_p2c and partial_p2c[wc] != cc:
                        reject_reason = f"{cc}={wc} but {wc} mapped from {partial_p2c[wc]}"
                        break
                    tent = {c: p for c, p in new_maps}
                    if cc in tent:
                        if tent[cc] != wc:
                            reject_reason = f"{cc} maps to both {tent[cc]} and {wc}"
                            break
                    else:
                        tent_ps = {p for _, p in new_maps}
                        if wc in tent_ps:
                            other = [c for c, p in new_maps if p == wc][0]
                            reject_reason = f"{wc} already assigned to {other}"
                            break
                        new_maps.append((cc, wc))
            if reject_reason:
                think_lines.append(f"  {candidate}: {reject_reason} → no")
            else:
                new_str = " ".join(f"{c}={p}(new)" for c, p in new_maps)
                think_lines.append(f"  {candidate}: all match, {new_str} → YES")
        think_lines.append(f"Accept: {word}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": word}


@register
class EncVerifyDecodeAgainstTable(MicroSkill):
    """Given a proposed decode and the flat map, verify if it's correct."""
    name = "enc_verify_decode"
    puzzle_type = "encryption"
    description = "Verify a proposed cipher→plain decode against the flat map"
    weight = 10.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None
        word = rng.choice([w for w in vocab if len(w) >= 4])

        alpha = list('abcdefghijklmnopqrstuvwxyz')
        plain = list('abcdefghijklmnopqrstuvwxyz')
        rng.shuffle(plain)
        c2p = dict(zip(alpha, plain))
        p2c = {v: k for k, v in c2p.items()}
        cipher_word = ''.join(p2c[ch] for ch in word)

        map_lines = _enc_build_flat_map(c2p)

        # 50% correct proposal, 50% wrong (different vocab word)
        is_correct = rng.random() < 0.5
        if is_correct:
            proposed = word
        else:
            same_len = [w for w in vocab if len(w) == len(word) and w != word]
            if not same_len: return None
            proposed = rng.choice(same_len)

        prompt = (
            "\n".join(map_lines) + "\n\n"
            f"Cipher: {cipher_word}\n"
            f"Proposed decode: {proposed}\n"
            f"Is this correct? Check each letter."
        )

        think_lines = []
        all_ok = True
        for i, (c, p) in enumerate(zip(cipher_word, proposed)):
            actual = c2p[c]
            if actual == p:
                think_lines.append(f"  {c}={actual}, proposed '{p}' → MATCH")
            else:
                think_lines.append(f"  {c}={actual}, proposed '{p}' → MISMATCH")
                all_ok = False

        if all_ok:
            think_lines.append("All match → correct")
            answer = f"Correct: {proposed}"
        else:
            think_lines.append(f"Mismatch → wrong, correct decode is {word}")
            answer = f"Wrong. Correct: {word}"

        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


# ============================================================
# ENC: CONTROL-FLOW SKILLS (targets hallucinated words + 0-match failures)
# ============================================================

@register
class EncZeroMatchRecheck(MicroSkill):
    """Partial pattern gives 0 vocab matches → identify suspect lookup → redo.
    Directly targets 53% hallucinated_word failures where model writes junk on 0 matches."""
    name = "enc_zero_match_recheck"
    puzzle_type = "encryption"
    description = "When 0 vocab words match partial, find and fix the wrong lookup"
    weight = 15.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None
        word = rng.choice([w for w in vocab if len(w) >= 4])

        alpha = list('abcdefghijklmnopqrstuvwxyz')
        plain = list('abcdefghijklmnopqrstuvwxyz')
        rng.shuffle(plain)
        c2p = dict(zip(alpha, plain))
        p2c = {v: k for k, v in c2p.items()}
        cipher_word = ''.join(p2c[ch] for ch in word)

        # Introduce ONE wrong mapping to create a 0-match scenario
        cipher_letters = list(set(cipher_word))
        if not cipher_letters: return None
        bad_letter = rng.choice(cipher_letters)
        correct_plain = c2p[bad_letter]
        # Pick a wrong plain letter (swap with another mapping)
        other_letters = [c for c in cipher_letters if c != bad_letter]
        if not other_letters: return None
        swap_with = rng.choice(other_letters)
        wrong_plain = c2p[swap_with]

        bad_c2p = dict(c2p)
        bad_c2p[bad_letter] = wrong_plain

        map_lines = _enc_build_flat_map(bad_c2p)

        # Build the wrong partial
        bad_partial = ''.join(bad_c2p.get(c, '?') for c in cipher_word)

        from collections import defaultdict
        by_len = defaultdict(list)
        for w in sorted(vocab): by_len[len(w)].append(w)

        prompt = (
            f"{len(word)}-letter words: {', '.join(by_len[len(word)])}\n\n"
            + "\n".join(map_lines) + "\n\n"
            f"Decode {cipher_word}: partial = {bad_partial}\n"
            f"0 matches in vocab. Find the wrong lookup and fix it."
        )

        think_lines = [
            f"Partial '{bad_partial}' has 0 vocab matches.",
            f"Check each letter of {cipher_word}:",
        ]
        for c in cipher_word:
            val = bad_c2p[c]
            if c == bad_letter:
                think_lines.append(f"  {c}={val} ← SUSPECT (should be {c}={correct_plain})")
            else:
                think_lines.append(f"  {c}={val} → MATCH")
        correct_partial = ''.join(c2p.get(c, '_') for c in cipher_word)
        think_lines.append(f"Fix: {bad_letter}={correct_plain}")
        think_lines.append(f"Corrected partial: {correct_partial} → {word}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": word}


@register
class EncCandidateListPick(MicroSkill):
    """Given partial pattern + list of same-length candidates, pick by consistency.
    Teaches constrained selection from vocab instead of open generation."""
    name = "enc_candidate_list_pick"
    puzzle_type = "encryption"
    description = "Pick the correct vocab word from candidates matching a partial pattern"
    weight = 12.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None

        from collections import defaultdict
        by_len = defaultdict(list)
        for w in sorted(vocab): by_len[len(w)].append(w)

        # Pick a word length with multiple candidates
        good_lens = [l for l, words in by_len.items() if len(words) >= 3]
        if not good_lens: return None
        length = rng.choice(good_lens)
        candidates = by_len[length]

        # Pick the target word
        word = rng.choice(candidates)

        # Create a partial pattern with 1-3 unknowns
        n_unknown = min(rng.randint(1, 3), len(word) - 1)
        unknown_positions = set(rng.sample(range(len(word)), n_unknown))
        partial = ''.join('_' if i in unknown_positions else word[i] for i in range(len(word)))

        # Find matching candidates
        matches = []
        for w in candidates:
            if all(partial[i] == '_' or partial[i] == w[i] for i in range(len(w))):
                matches.append(w)

        if len(matches) < 2:
            # Need at least 2 to make it interesting — adjust partial
            partial = ''.join('_' if i in unknown_positions or i == 0 else word[i] for i in range(len(word)))
            matches = [w for w in candidates if all(partial[i] == '_' or partial[i] == w[i] for i in range(len(w)))]

        if len(matches) < 1: return None

        prompt = (
            f"Partial pattern: {partial}\n"
            f"Candidates: {', '.join(matches)}\n"
            f"Which word fits? The unknown letters must be consistent with the cipher mapping."
        )

        if len(matches) == 1:
            think = f"Only one candidate matches '{partial}': {matches[0]}"
        else:
            think = f"Pattern '{partial}' matches: {', '.join(matches)}. Answer: {word}"

        return {"user": prompt, "think": think, "answer": word}


@register
class EncNonVocabReject(MicroSkill):
    """Given a proposed decryption word, check if it's in the 77-word vocab. Reject if not.
    Directly targets 53% hallucinated_word failures."""
    name = "enc_non_vocab_reject"
    puzzle_type = "encryption"
    description = "Check if a proposed word is in the 77-word vocab — reject imposters"
    weight = 15.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None

        # 50% valid vocab word, 50% imposter
        is_valid = rng.random() < 0.5

        if is_valid:
            word = rng.choice(vocab)
            think = f"'{word}' — checking 77-word vocab... YES, it is in the list."
            answer = f"Valid: {word}"
        else:
            imposters = [
                # Actual model hallucinations (from failure annotations DB)
                "farmer", "baker", "writer", "ranger", "brother", "singer",
                "dagger", "throne", "compass", "hunter", "sailor", "trader",
                "cipher", "drummer", "rover", "temple", "spider", "viper",
                "fountain", "volcano", "dolphin", "warrior", "fencer", "drone",
                "saddle", "scarlet", "vendor", "trophy", "window", "wonder",
                "stolen", "sister", "simple", "monkey", "journal", "furnace",
                # Near-misses of vocab words (spelling/form variants)
                "catches", "fetches", "launches", "touches", "reaches", "crushes",
                "writings", "dragons", "knights", "rabbits", "turtles", "wizards",
                "castles", "forests", "gardens", "islands", "palaces", "towers",
                "studying", "reading", "writing", "drawing", "dreaming", "watching",
                "created", "explored", "followed", "imagined", "discovered",
                "brightest", "curious", "cleverly", "darkly", "golden",
                "princesses", "mountains", "treasures", "villages", "mysteries",
                # Semantically similar but not in vocab
                "prince", "knight", "castle", "bridge", "gentle", "bitter",
                "broken", "frozen", "modern", "outside", "emperor", "diamond",
            ]
            # Remove any that ARE actually in vocab
            imposters = [w for w in imposters if w not in set(vocab)]
            word = rng.choice(imposters)
            # Show same-length vocab words for comparison
            from collections import defaultdict
            by_len = defaultdict(list)
            for w in sorted(vocab): by_len[len(w)].append(w)
            same_len = by_len.get(len(word), [])
            think = f"'{word}' — checking 77-word vocab... NO. Not in the list."
            if same_len:
                think += f"\n{len(word)}-letter vocab words: {', '.join(same_len)}"
            answer = f"REJECT: '{word}' is not in the 77-word vocabulary"

        prompt = f"Is '{word}' in Alice's 77-word competition vocabulary?"

        return {"user": prompt, "think": think, "answer": answer}


# ============================================================
# BIT: GRID AUDIT (directly targets fake_verify — 88% of bit failures)
# ============================================================

@register
class BitGridAudit(MicroSkill):
    """Given A, B, gate, and claimed GRID result — is it correct?
    50/50 right/wrong. Model must actually verify, not just say → MATCH."""
    name = "bit_grid_audit"
    puzzle_type = "bit_manipulation"
    description = "Audit a claimed GRID computation — correct or wrong?"
    weight = 15.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_compact import apply_shift, apply_gate, fmt
        gate = rng.choice(["xor", "and", "or", "xnor"])
        src_a = rng.choice(["shr1", "shr2", "shl1", "shl2", "rol1", "rol2"])
        src_b = rng.choice(["shr3", "shl3", "rol3", "ror1", "ror2", "shr4"])
        x = rng.randint(0, 255)
        a = fmt(apply_shift(x, src_a))
        b = fmt(apply_shift(x, src_b))
        correct_result = apply_gate(a, b, gate)

        is_correct = rng.random() < 0.5
        if is_correct:
            claimed = correct_result
        else:
            # Flip 1-2 bits
            claimed_list = list(correct_result)
            n_flip = rng.randint(1, 2)
            positions = rng.sample(range(8), n_flip)
            for p in positions:
                claimed_list[p] = '1' if claimed_list[p] == '0' else '0'
            claimed = ''.join(claimed_list)

        prompt = (
            f"x = {format(x, '08b')}\n"
            f"A = {src_a}(x) = {a}\n"
            f"B = {src_b}(x) = {b}\n"
            f"Claimed: {gate}(A,B) = {claimed}\n"
            f"Is this GRID result correct?"
        )

        if is_correct:
            think = f"Compute {gate} bit by bit:\n"
            for i in range(8):
                av, bv = int(a[i]), int(b[i])
                if gate == "xor": r = av ^ bv
                elif gate == "and": r = av & bv
                elif gate == "or": r = av | bv
                elif gate == "xnor": r = 1 - (av ^ bv)
                think += f"  bit{i}: {gate}({av},{bv})={r}\n"
            think += f"Result: {correct_result} — matches claimed. → MATCH Correct."
            answer = "Correct"
        else:
            think = f"Compute {gate} bit by bit:\n"
            first_wrong = None
            for i in range(8):
                av, bv = int(a[i]), int(b[i])
                if gate == "xor": r = av ^ bv
                elif gate == "and": r = av & bv
                elif gate == "or": r = av | bv
                elif gate == "xnor": r = 1 - (av ^ bv)
                if str(r) != claimed[i] and first_wrong is None:
                    first_wrong = i
                    think += f"  bit{i}: {gate}({av},{bv})={r} but claimed {claimed[i]} → MISMATCH WRONG\n"
                else:
                    think += f"  bit{i}: {gate}({av},{bv})={r}\n"
            think += f"Result: {correct_result} ≠ {claimed}. First error at bit {first_wrong}."
            answer = f"Wrong. First error at bit {first_wrong}. Correct: {correct_result}"

        return {"user": prompt, "think": think, "answer": answer}


@register
class BitTerminalReject(MicroSkill):
    """A rule fails both checks — model must output 'Rule rejected' not an answer.
    Teaches that failed verification = STOP, don't force an answer."""
    name = "bit_terminal_reject"
    puzzle_type = "bit_manipulation"
    description = "Rule fails checks — reject it instead of forcing an answer"
    weight = 10.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_compact import apply_shift, apply_gate, fmt
        real_gate = rng.choice(["xor", "and", "or"])
        wrong_gate = rng.choice([g for g in ["xor", "and", "or", "xnor"] if g != real_gate])
        src_a = rng.choice(["shr1", "shl1", "rol1", "shr2"])
        src_b = rng.choice(["shl2", "rol2", "shr3", "ror1"])

        examples = []
        for _ in range(3):
            x = rng.randint(0, 255)
            a = fmt(apply_shift(x, src_a))
            b = fmt(apply_shift(x, src_b))
            out = apply_gate(a, b, real_gate)
            examples.append((format(x, '08b'), out))

        # Show wrong rule being checked
        x0, out0 = examples[0]
        a0 = fmt(apply_shift(int(x0, 2), src_a))
        b0 = fmt(apply_shift(int(x0, 2), src_b))
        wrong_result = apply_gate(a0, b0, wrong_gate)

        prompt = (
            f"Examples:\n"
            + '\n'.join(f"  {x} → {out}" for x, out in examples) +
            f"\n\nProposed rule: {wrong_gate}({src_a}(x), {src_b}(x))\n"
            f"Check Ex1: {wrong_gate}({a0},{b0}) = {wrong_result}\n"
            f"Expected: {out0}\n"
            f"What should you do?"
        )

        think = (
            f"{wrong_result} vs {out0}\n"
            f"{'→ MATCH Match' if wrong_result == out0 else '→ MISMATCH MISMATCH'}\n"
        )
        if wrong_result != out0:
            think += f"Rule {wrong_gate} fails on Ex1. REJECT — do not use this rule."
            answer = f"Rule rejected. {wrong_gate} gives {wrong_result} but expected {out0}."
        else:
            # Check ex2
            x1, out1 = examples[1]
            a1 = fmt(apply_shift(int(x1, 2), src_a))
            b1 = fmt(apply_shift(int(x1, 2), src_b))
            wr1 = apply_gate(a1, b1, wrong_gate)
            think += f"Check Ex2: {wr1} vs {out1} — {'→ MATCH' if wr1 == out1 else '→ MISMATCH MISMATCH'}\n"
            if wr1 != out1:
                think += f"Rule {wrong_gate} fails on Ex2. REJECT."
                answer = f"Rule rejected. {wrong_gate} fails on Ex2."
            else:
                think += f"Rule passes both checks."
                answer = f"Rule accepted."

        return {"user": prompt, "think": think, "answer": answer}


@register
class TransVerifyLock(MicroSkill):
    """Given a locked combo and two examples — does the combo actually work?
    Targets trans:wrong_combo (28 failures at step 5800)."""
    name = "trans_verify_lock"
    puzzle_type = "transformation"
    description = "Verify a locked combo reproduces both examples — accept or reject"
    weight = 10.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_transform import _make_operands, _calc, _fmt, COMBO_DISPLAY
        orders = ["BA_DC", "AB_CD", "AB_DC", "BA_CD"]
        ops = ["mul", "add", "sub", "absdiff", "add1", "muladd1"]
        fmts = ["rev", "raw", "abs"]

        real_combo = (rng.choice(orders), rng.choice(ops), rng.choice(fmts))

        examples = []
        for _ in range(2):
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            L, R = _make_operands(a//10, a%10, b//10, b%10, real_combo[0])
            val = _calc(L, R, real_combo[1])
            fval = _fmt(val, real_combo[2])
            if fval is None or len(str(fval)) > 5:
                return None
            examples.append((a, b, str(fval), L, R, val))

        # 50% correct combo, 50% wrong (change one axis)
        is_correct = rng.random() < 0.5
        if is_correct:
            test_combo = real_combo
        else:
            axis = rng.randint(0, 2)
            test_combo = list(real_combo)
            if axis == 0:
                test_combo[0] = rng.choice([o for o in orders if o != real_combo[0]])
            elif axis == 1:
                test_combo[1] = rng.choice([o for o in ops if o != real_combo[1]])
            else:
                test_combo[2] = rng.choice([f for f in fmts if f != real_combo[2]])
            test_combo = tuple(test_combo)

        rd = COMBO_DISPLAY.get(test_combo[0], test_combo[0])
        prompt_lines = [f"Locked combo: {rd}|{test_combo[1]}|{test_combo[2]}"]
        for i, (a, b, fval, _, _, _) in enumerate(examples):
            prompt_lines.append(f"Ex{i+1}: {a}+{b} = {fval}")
        prompt_lines.append("Does this combo reproduce both examples?")

        think_lines = []
        all_ok = True
        for i, (a, b, expected, _, _, _) in enumerate(examples):
            tL, tR = _make_operands(a//10, a%10, b//10, b%10, test_combo[0])
            tval = _calc(tL, tR, test_combo[1])
            tfval = _fmt(tval, test_combo[2]) if tval is not None else None
            ok = str(tfval) == expected
            think_lines.append(f"Ex{i+1}: {rd} L={tL} R={tR} {test_combo[1]}({tL},{tR})={tval} {test_combo[2]}={tfval} vs {expected} {'→ MATCH' if ok else '→ MISMATCH'}")
            if not ok:
                all_ok = False

        if all_ok:
            answer = "Combo verified → MATCH"
        else:
            answer = "Combo REJECTED → MISMATCH — does not reproduce examples"

        return {"user": '\n'.join(prompt_lines), "think": '\n'.join(think_lines), "answer": answer}


@register
class TransTraceAudit(MicroSkill):
    """Given a complete claimed transformation solution, is it valid?
    Checks: does the locked combo reproduce the examples? Does the query use the locked combo?
    Directly targets wrong_combo and fake verification in transformation."""
    name = "trans_trace_audit"
    puzzle_type = "transformation"
    description = "Audit a claimed transformation solution — valid or invalid?"
    weight = 12.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_transform import _make_operands, _calc, _fmt, COMBO_DISPLAY
        orders = ["BA_DC", "AB_CD", "AB_DC", "BA_CD"]
        ops = ["mul", "add", "sub", "absdiff", "add1"]
        fmts = ["rev", "raw", "abs"]

        real_combo = (rng.choice(orders), rng.choice(ops), rng.choice(fmts))
        op_char = rng.choice(['+', '-', '*', '/'])

        examples = []
        for _ in range(3):
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            L, R = _make_operands(a//10, a%10, b//10, b%10, real_combo[0])
            val = _calc(L, R, real_combo[1])
            fval = _fmt(val, real_combo[2], op_char=op_char)
            if fval is None or len(str(fval)) > 5:
                return None
            examples.append((a, b, str(fval)))

        qa, qb = rng.randint(10, 99), rng.randint(10, 99)
        qL, qR = _make_operands(qa//10, qa%10, qb//10, qb%10, real_combo[0])
        qval = _calc(qL, qR, real_combo[1])
        qfval = _fmt(qval, real_combo[2], op_char=op_char)
        if qfval is None:
            return None

        rd = COMBO_DISPLAY.get(real_combo[0], real_combo[0])

        # 50% valid solution, 50% invalid (wrong combo in query)
        is_valid = rng.random() < 0.5
        if is_valid:
            shown_combo = real_combo
            shown_answer = str(qfval)
        else:
            # Use wrong order or wrong op for query
            wrong_order = rng.choice([o for o in orders if o != real_combo[0]])
            wL, wR = _make_operands(qa//10, qa%10, qb//10, qb%10, wrong_order)
            wval = _calc(wL, wR, real_combo[1])
            wfval = _fmt(wval, real_combo[2], op_char=op_char)
            if wfval is None or str(wfval) == str(qfval):
                return None
            shown_combo = (wrong_order, real_combo[1], real_combo[2])
            shown_answer = str(wfval)

        wd = COMBO_DISPLAY.get(shown_combo[0], shown_combo[0])

        prompt = (
            f"Claimed solution for {qa}{op_char}{qb}:\n"
            f"  Lock: {rd}|{real_combo[1]}|{real_combo[2]}\n"
            f"  Ex1: {examples[0][0]}{op_char}{examples[0][1]} = {examples[0][2]} → MATCH\n"
            f"  Query: {qa}{op_char}{qb}\n"
            f"    {wd}: L={_make_operands(qa//10,qa%10,qb//10,qb%10,shown_combo[0])[0]} "
            f"R={_make_operands(qa//10,qa%10,qb//10,qb%10,shown_combo[0])[1]}\n"
            f"    Answer: {shown_answer}\n"
            f"Is this solution valid?"
        )

        if is_valid:
            think = f"Lock is {rd}|{real_combo[1]}|{real_combo[2]}. Query uses same combo. Answer {shown_answer} is correct."
            answer = "Valid → MATCH"
        else:
            think = (
                f"Lock is {rd}|{real_combo[1]}|{real_combo[2]} but query uses {wd} (different order). "
                f"Correct answer with {rd} would be {qfval}, not {shown_answer}."
            )
            answer = f"Invalid → MISMATCH — query used wrong order. Correct: {qfval}"

        return {"user": prompt, "think": think, "answer": answer}


# ============================================================
# TRANSFORMATION: OPERAND ASSEMBLY + STYLE (R10 feedback — highest ROI)
# ============================================================

@register
class TransOperandAssembly(MicroSkill):
    """Given input digits and an order code, compute L and R.
    Directly targets wrong_operand_order failures."""
    name = "trans_operand_assembly"
    puzzle_type = "transformation"
    description = "Compute L and R from input digits and operand order"
    weight = 15.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_transform import _make_operands, COMBO_DISPLAY
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        orders = ["BA_DC", "AB_CD", "AB_DC", "BA_CD"]
        order = rng.choice(orders)
        L, R = _make_operands(a // 10, a % 10, b // 10, b % 10, order)
        od = COMBO_DISPLAY[order]

        a_str = str(a)
        b_str = str(b)

        prompt = f"Input: {a}/{b}\nOrder: {od}\nWhat are L and R?"

        if order == "BA_DC":
            think = f"{a_str[1]}{a_str[0]}={L}, {b_str[1]}{b_str[0]}={R}"
        elif order == "AB_CD":
            think = f"{a_str}={L}, {b_str}={R}"
        elif order == "AB_DC":
            think = f"{a_str}={L}, {b_str[1]}{b_str[0]}={R}"
        elif order == "BA_CD":
            think = f"{a_str[1]}{a_str[0]}={L}, {b_str}={R}"

        return {"user": prompt, "think": think, "answer": f"L={L} R={R}"}


@register
class TransOrderTest(MicroSkill):
    """Given an example with expected result, determine which operand order is correct.
    Tests: try AB,CD and BA,DC, see which produces the expected result."""
    name = "trans_order_test"
    puzzle_type = "transformation"
    description = "Determine correct operand order by testing both on an example"
    weight = 15.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_transform import _make_operands, _calc, _fmt, COMBO_DISPLAY
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        op = rng.choice(["add", "sub", "mul", "absdiff"])
        fmt = rng.choice(["raw", "rev", "abs"])
        correct_order = rng.choice(["BA_DC", "AB_CD", "AB_DC", "BA_CD"])

        L, R = _make_operands(a // 10, a % 10, b // 10, b % 10, correct_order)
        val = _calc(L, R, op)
        fval = _fmt(val, fmt)
        if fval is None:
            return None

        # Test wrong order
        wrong_orders = [o for o in ["BA_DC", "AB_CD", "AB_DC", "BA_CD"] if o != correct_order]
        wrong_order = rng.choice(wrong_orders)
        wL, wR = _make_operands(a // 10, a % 10, b // 10, b % 10, wrong_order)
        wval = _calc(wL, wR, op)
        wfval = _fmt(wval, fmt)

        if str(wfval) == str(fval):
            return None  # Both orders give same result, not useful

        od = COMBO_DISPLAY[correct_order]
        wd = COMBO_DISPLAY[wrong_order]

        prompt = (
            f"Example: {a}/{b} = {fval}\n"
            f"Operation: {op}|{fmt}\n"
            f"Which order: {od} or {wd}?"
        )

        think = (
            f"Try {od}: L={L} R={R} → {op}({L},{R})={val} {fmt}={fval} → MATCH\n"
            f"Try {wd}: L={wL} R={wR} → {op}({wL},{wR})={wval} {fmt}={wfval} → MISMATCH"
        )

        return {"user": prompt, "think": think, "answer": f"{od}"}


@register
class TransStyleApply(MicroSkill):
    """Given a raw numeric result and a format, apply the format.
    Directly targets wrong_format failures."""
    name = "trans_style_apply"
    puzzle_type = "transformation"
    description = "Apply a format modifier (rev, abs, opsign, tailsign) to a raw result"
    weight = 12.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_transform import _fmt
        val = rng.randint(-99, 999)
        op_char = rng.choice(['+', '-', '*', '/', '|', '^'])
        fmt = rng.choice(["raw", "rev", "abs", "opsign", "opsign_always", "tailsign", "tailsign_always"])

        result = _fmt(val, fmt, op_char=op_char)
        if result is None:
            return None

        prompt = f"Raw result: {val}\nFormat: {fmt}\nOperator: {op_char}\nApply the format."

        if fmt == "raw":
            think = f"{val} → raw = {result}"
        elif fmt == "rev":
            think = f"{val} → reverse digits = {result}"
        elif fmt == "abs":
            think = f"{val} → absolute value = {result}"
        elif fmt == "opsign":
            if val < 0:
                think = f"{val} is negative → prefix with '{op_char}': {result}"
            else:
                think = f"{val} is positive → no prefix: {result}"
        elif fmt == "opsign_always":
            think = f"Always prefix with '{op_char}': {result}"
        elif fmt == "tailsign":
            if val < 0:
                think = f"{val} is negative → suffix with '{op_char}': {result}"
            else:
                think = f"{val} is positive → no suffix: {result}"
        elif fmt == "tailsign_always":
            think = f"Always suffix with '{op_char}': {result}"
        else:
            think = f"{val} → {fmt} = {result}"

        return {"user": prompt, "think": think, "answer": str(result)}


@register
class BitGateTrivia(MicroSkill):
    """Quick gate identification: given A, B, and result — which gate?
    Targets the #1 confusion: model defaults to xor/and/or.
    Includes hard gates (xnor, nand, nor, and_not) alongside easy ones."""
    name = "bit_gate_trivia"
    puzzle_type = "bit_manipulation"
    description = "Given A, B, result — identify which gate (xor vs and vs or vs xnor vs nand vs nor)"
    weight = 15.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        gates = {
            'xor': lambda a, b: a ^ b,
            'and': lambda a, b: a & b,
            'or': lambda a, b: a | b,
            'xnor': lambda a, b: ~(a ^ b) & 0xFF,
            'nand': lambda a, b: ~(a & b) & 0xFF,
            'nor': lambda a, b: ~(a | b) & 0xFF,
        }
        correct_gate = rng.choice(list(gates.keys()))
        a = rng.randint(0, 255)
        b = rng.randint(0, 255)
        result = gates[correct_gate](a, b)

        a_str = format(a, '08b')
        b_str = format(b, '08b')
        r_str = format(result, '08b')

        # Pick 2 wrong gates as distractors
        wrong = rng.sample([g for g in gates if g != correct_gate], 2)
        options = [correct_gate] + wrong
        rng.shuffle(options)
        labels = ['A', 'B', 'C']

        prompt = (
            f"A = {a_str}\nB = {b_str}\nResult = {r_str}\n\n"
            f"Which gate produced this?\n"
            + '\n'.join(f"  {labels[i]}) {options[i]}" for i in range(3))
        )

        # Verify by computing each
        think_lines = []
        for i, opt in enumerate(options):
            computed = format(gates[opt](a, b), '08b')
            match = "→ MATCH" if computed == r_str else "→ MISMATCH"
            think_lines.append(f"{labels[i]}) {opt}({a_str},{b_str})={computed} {match}")

        correct_label = labels[options.index(correct_gate)]
        think_lines.append(f"Answer: {correct_label}) {correct_gate}")

        return {"user": prompt, "think": '\n'.join(think_lines), "answer": f"{correct_label}) {correct_gate}"}


@register
class BitGateSurvivor(MicroSkill):
    """Given 5 candidate gates and 2 input/output examples — which gates survive both?
    Teaches systematic elimination, not just picking the first plausible gate."""
    name = "bit_gate_survivor"
    puzzle_type = "bit_manipulation"
    description = "Given 5 gates and 2 examples, list which gates are consistent with both"
    weight = 15.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_compact import apply_shift, apply_gate, fmt
        gates = ['xor', 'and', 'or', 'xnor', 'nand', 'nor']
        src_pool = ['shr1', 'shr2', 'shl1', 'shl2', 'rol1', 'rol2', 'ror1', 'ror2']

        real_gate = rng.choice(gates)
        src_a = rng.choice(src_pool)
        src_b = rng.choice([s for s in src_pool if s != src_a])

        # Generate 2 examples
        examples = []
        for _ in range(2):
            x = rng.randint(0, 255)
            a = fmt(apply_shift(x, src_a))
            b = fmt(apply_shift(x, src_b))
            out = apply_gate(a, b, real_gate)
            examples.append((format(x, '08b'), a, b, out))

        # Pick 5 candidate gates (must include the real one)
        candidates = [real_gate]
        others = [g for g in gates if g != real_gate]
        candidates.extend(rng.sample(others, min(4, len(others))))
        rng.shuffle(candidates)

        # Test each candidate on both examples
        survivors = []
        think_lines = []
        for gate in candidates:
            passes = True
            for x_str, a_str, b_str, expected in examples:
                computed = apply_gate(a_str, b_str, gate)
                if computed != expected:
                    passes = False
                    think_lines.append(f"  {gate}: {gate}({a_str},{b_str})={computed} vs {expected} → MISMATCH")
                    break
            if passes:
                survivors.append(gate)
                think_lines.append(f"  {gate}: passes both → MATCH")

        prompt = (
            f"Sources: A={src_a}(x), B={src_b}(x)\n"
            f"Ex1: x={examples[0][0]} → {examples[0][3]}\n"
            f"Ex2: x={examples[1][0]} → {examples[1][3]}\n\n"
            f"Candidates: {', '.join(candidates)}\n"
            f"Which gates are consistent with BOTH examples?"
        )

        answer = ', '.join(survivors) if survivors else 'none'
        return {"user": prompt, "think": '\n'.join(think_lines), "answer": answer}


@register
class TransComboSurvivor(MicroSkill):
    """Given 4 candidate combos and 2 examples, which combos survive both?
    Targets wrong_combo failures — model picks combo that matches Ex1 but fails Ex2."""
    name = "trans_combo_survivor"
    puzzle_type = "transformation"
    description = "Given candidate combos and examples, list which survive verification on both"
    weight = 15.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_transform import _make_operands, _calc, _fmt, COMBO_DISPLAY
        orders = ["BA_DC", "AB_CD", "AB_DC", "BA_CD"]
        ops = ["mul", "add", "sub", "absdiff", "add1"]
        fmts = ["rev", "raw", "abs"]

        real_combo = (rng.choice(orders), rng.choice(ops), rng.choice(fmts))

        # Generate 2 examples
        examples = []
        for _ in range(2):
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            L, R = _make_operands(a//10, a%10, b//10, b%10, real_combo[0])
            val = _calc(L, R, real_combo[1])
            fval = _fmt(val, real_combo[2])
            if fval is None or len(str(fval)) > 5:
                return None
            examples.append((a, b, str(fval)))

        # Generate 3 wrong combos (close misses — differ by 1 axis)
        wrong_combos = []
        for o in orders:
            for op in ops:
                for f in fmts:
                    if (o, op, f) == real_combo: continue
                    diff = (o != real_combo[0]) + (op != real_combo[1]) + (f != real_combo[2])
                    if diff == 1:
                        wrong_combos.append((o, op, f))
        if len(wrong_combos) < 3:
            return None
        candidates = [real_combo] + rng.sample(wrong_combos, 3)
        rng.shuffle(candidates)

        # Test each
        prompt_lines = [
            f"Ex1: {examples[0][0]}/{examples[0][1]} = {examples[0][2]}",
            f"Ex2: {examples[1][0]}/{examples[1][1]} = {examples[1][2]}",
            "",
            "Candidates:",
        ]
        think_lines = []
        survivors = []
        for i, (o, op, f) in enumerate(candidates):
            od = COMBO_DISPLAY.get(o, o)
            prompt_lines.append(f"  {i+1}) order={od} op={op} style={f}")
            # Test on both examples
            all_ok = True
            for a, b, exp in examples:
                L, R = _make_operands(a//10, a%10, b//10, b%10, o)
                val = _calc(L, R, op)
                fval = _fmt(val, f)
                if str(fval) != exp:
                    think_lines.append(f"{i+1}) {od}|{op}|{f}: {op}({L},{R})={val} {f}={fval} vs {exp} → MISMATCH")
                    all_ok = False
                    break
            if all_ok:
                think_lines.append(f"{i+1}) {od}|{op}|{f}: → MATCH both")
                survivors.append(i+1)

        prompt_lines.append("")
        prompt_lines.append("Which combos match BOTH examples?")

        answer = ', '.join(str(s) for s in survivors)
        return {"user": '\n'.join(prompt_lines), "think": '\n'.join(think_lines), "answer": answer}


@register
class TransEncodeLength(MicroSkill):
    """Given a numeric result and a base, how many symbols does the encoding have?
    Targets wrong_length failures (42% of cipher errors)."""
    name = "trans_encode_length"
    puzzle_type = "transformation"
    description = "Predict encoding length: how many symbols for a given number in a given base"
    weight = 12.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        base = rng.randint(6, 10)
        val = rng.randint(0, base**4 - 1)  # up to 4 digits in base

        # Encode
        if val == 0:
            encoded_len = 1
        else:
            encoded_len = 0
            v = val
            while v > 0:
                v //= base
                encoded_len += 1

        prompt = f"Number: {val}\nBase: {base}\nHow many digits when encoded in base {base}?"
        think = f"{val} in base {base}: "

        if val == 0:
            think += "0 → 1 digit"
        else:
            digits = []
            v = val
            while v > 0:
                digits.append(v % base)
                v //= base
            digits.reverse()
            think += f"{''.join(str(d) for d in digits)} → {len(digits)} digits"

        return {"user": prompt, "think": think, "answer": str(encoded_len)}


@register
class EncSentenceVocabCheck(MicroSkill):
    """Check entire decrypted sentence for OOV words. 50% have one imposter."""
    name = "enc_sentence_vocab_check"
    puzzle_type = "encryption"
    description = "Check entire sentence — every word must be in 77-word vocab"
    weight = 12.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None
        n_words = rng.randint(3, 5)
        words = rng.sample(list(vocab), min(n_words, len(vocab)))
        is_valid = rng.random() < 0.5
        if not is_valid:
            imposters = ["farmer","baker","writer","ranger","catches","fetches","launches",
                         "dragons","knights","rabbits","castles","forests","throne","compass"]
            imposters = [w for w in imposters if w not in set(vocab)]
            if not imposters: return None
            idx = rng.randint(0, len(words)-1)
            words[idx] = rng.choice(imposters)
        sentence = ' '.join(words)
        prompt = f"Decrypted: {sentence}\nAll words in Alice's vocabulary?"
        if is_valid:
            answer = "Yes — all valid"
        else:
            bad = [w for w in words if w not in set(vocab)]
            answer = f"No — '{bad[0]}' is not in vocabulary"
        return {"user": prompt, "think": "", "answer": answer}


# ============================================================
# R13 NEW SKILLS — Diff-based verification + Transformation audits
# ============================================================

@register
class BitDiffCheck(MicroSkill):
    """Compute XOR diff between two 8-bit strings to verify PASS/FAIL."""
    name = "bit_diff_check"
    puzzle_type = "bit_manipulation"
    description = "Compute XOR diff of two 8-bit strings, decide PASS or FAIL"
    weight = 15.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        a = format(rng.randint(0, 255), '08b')
        # 50% match, 50% mismatch
        if rng.random() < 0.5:
            b = a  # exact match
        else:
            b = format(rng.randint(0, 255), '08b')
        diff = ''.join('0' if x == y else '1' for x, y in zip(a, b))
        verdict = "PASS" if diff == '00000000' else "FAIL"
        prompt = f"Computed output: {a}\nExpected output: {b}\nCompute the XOR diff and state PASS or FAIL."
        think = f"diff = {a} XOR {b}\n"
        think += ' '.join(a) + "\n"
        think += ' '.join(b) + "\n"
        think += ' '.join(diff) + "\n"
        think += f"diff={diff}"
        if diff == '00000000':
            think += " (all zeros) → PASS"
        else:
            n_diff = diff.count('1')
            think += f" ({n_diff} bits differ) → FAIL"
        answer = f"diff={diff} → {verdict}"
        return {"user": prompt, "think": think, "answer": answer}


@register
class TransOrderAudit(MicroSkill):
    """Given digits and two orderings, compute both results and identify which matches."""
    name = "trans_order_audit"
    puzzle_type = "transformation"
    description = "Test two operand orderings on an example, identify which matches"
    weight = 15.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_transform import _make_operands, _calc, _fmt, COMBO_DISPLAY
        orders = ["AB_CD", "BA_DC", "AB_DC", "BA_CD"]
        ops = ["mul", "add", "sub"]
        op = rng.choice(ops)
        correct_order = rng.choice(orders)
        wrong_order = rng.choice([o for o in orders if o != correct_order])

        a, b = rng.randint(10, 99), rng.randint(10, 99)
        cL, cR = _make_operands(a//10, a%10, b//10, b%10, correct_order)
        cval = _calc(cL, cR, op)
        if cval is None: return None
        expected = str(cval)

        wL, wR = _make_operands(a//10, a%10, b//10, b%10, wrong_order)
        wval = _calc(wL, wR, op)
        if wval is None: return None

        cd = COMBO_DISPLAY.get(correct_order, correct_order)
        wd = COMBO_DISPLAY.get(wrong_order, wrong_order)

        prompt = (f"Digits: {a},{b}. Operation: {op}. Expected result: {expected}.\n"
                  f"Which ordering is correct?\n"
                  f"A) {wd}: L={wL} R={wR}\n"
                  f"B) {cd}: L={cL} R={cR}")

        # Randomize which option is correct
        if rng.random() < 0.5:
            prompt = (f"Digits: {a},{b}. Operation: {op}. Expected result: {expected}.\n"
                      f"Which ordering is correct?\n"
                      f"A) {cd}: L={cL} R={cR}\n"
                      f"B) {wd}: L={wL} R={wR}")
            think = (f"A) {cd}: {op}({cL},{cR})={cval} vs {expected} → match\n"
                     f"B) {wd}: {op}({wL},{wR})={wval} vs {expected} → {'match' if str(wval) == expected else 'no match'}")
            answer = f"A) {cd}"
        else:
            think = (f"A) {wd}: {op}({wL},{wR})={wval} vs {expected} → {'match' if str(wval) == expected else 'no match'}\n"
                     f"B) {cd}: {op}({cL},{cR})={cval} vs {expected} → match")
            answer = f"B) {cd}"

        return {"user": prompt, "think": think, "answer": answer}


@register
class TransStyleAudit(MicroSkill):
    """Given a raw result and operator, apply different styles and identify correct one."""
    name = "trans_style_audit"
    puzzle_type = "transformation"
    description = "Apply two style transforms to same raw result, identify which matches expected"
    weight = 15.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        raw = rng.randint(-999, 999)
        op_char = rng.choice(['+', '-', '*', '/'])
        styles = ["raw", "rev", "abs"]
        correct_style = rng.choice(styles)
        wrong_style = rng.choice([s for s in styles if s != correct_style])

        def apply_style(val, style, op_c):
            s = str(abs(val)) if style == "abs" else str(val)
            if style == "rev":
                s = str(val)
                neg = s.startswith('-')
                digits = s.lstrip('-')
                s = digits[::-1]
                if neg:
                    s = '-' + s
            return s

        correct_result = apply_style(raw, correct_style, op_char)
        wrong_result = apply_style(raw, wrong_style, op_char)

        prompt = (f"Raw computation result: {raw}\n"
                  f"Expected formatted output: {correct_result}\n"
                  f"Which style was applied?\n"
                  f"A) {wrong_style} → {wrong_result}\n"
                  f"B) {correct_style} → {correct_result}")

        if rng.random() < 0.5:
            prompt = (f"Raw computation result: {raw}\n"
                      f"Expected formatted output: {correct_result}\n"
                      f"Which style was applied?\n"
                      f"A) {correct_style} → {correct_result}\n"
                      f"B) {wrong_style} → {wrong_result}")
            answer = f"A) {correct_style}"
        else:
            answer = f"B) {correct_style}"

        think = (f"raw={raw}\n"
                 f"{correct_style}({raw})={correct_result} → matches expected\n"
                 f"{wrong_style}({raw})={wrong_result} → does not match")

        return {"user": prompt, "think": think, "answer": answer}


# ============================================================
# R13 NEW SKILLS — Style regime inheritance + Mapping audit
# ============================================================

@register
class TransRegimeInherit(MicroSkill):
    """One operator establishes the style regime; unseen operator inherits it."""
    name = "trans_regime_inherit"
    puzzle_type = "transformation"
    description = "Given locked style for one operator, infer what style unseen operator should use"
    weight = 12.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        styles = ["raw", "rev", "abs"]
        locked_style = rng.choice(styles)
        ops_shown = rng.sample(['+', '-', '*', '/'], 2)
        op_query = rng.choice([c for c in ['+', '-', '*', '/', '|', '(', ')'] if c not in ops_shown])
        orders = ["AB,CD", "BA,DC", "AB,DC", "BA,CD"]
        locked_order = rng.choice(orders)

        prompt = (f"Locked operators from support:\n"
                  f"  Lock[{ops_shown[0]}]: order={locked_order} op=add style={locked_style}\n"
                  f"  Lock[{ops_shown[1]}]: order={locked_order} op=mul style={locked_style}\n"
                  f"\nQuery operator '{op_query}' is not in support.\n"
                  f"What style regime should it use?")

        think = (f"Both locked operators use style={locked_style} and order={locked_order}.\n"
                 f"The row has a consistent regime.\n"
                 f"Unseen operator '{op_query}' should inherit: order={locked_order}, style={locked_style}")

        answer = f"style={locked_style}, order={locked_order}"
        return {"user": prompt, "think": think, "answer": answer}


@register
class TransMappingAudit(MicroSkill):
    """Check if a cipher digit mapping is valid (bijective, no duplicates)."""
    name = "trans_mapping_audit"
    puzzle_type = "transformation"
    description = "Check cipher mapping for errors: duplicates, missing digits, non-bijective"
    weight = 10.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        symbols = list('!@#$%^&*()[]{}|\\/<>~`')
        rng.shuffle(symbols)
        syms = symbols[:10]
        digits = list(range(10))

        is_valid = rng.random() < 0.4
        if is_valid:
            mapping = dict(zip(syms, digits))
            prompt_lines = [f"Mapping: {' '.join(f'{s}={d}' for s, d in mapping.items())}"]
            prompt = '\n'.join(prompt_lines) + "\nIs this mapping valid (bijective, all 10 digits present)?"
            answer = "Yes — bijective, all digits 0-9 present exactly once"
            think = f"10 symbols, 10 digits, each digit appears once. Valid."
        else:
            # Introduce an error
            error_type = rng.choice(['duplicate_digit', 'missing_digit'])
            mapping = dict(zip(syms, digits))
            if error_type == 'duplicate_digit':
                # Make two symbols map to same digit
                victim_idx = rng.randint(1, 9)
                source_idx = rng.randint(0, victim_idx - 1)
                dup_digit = mapping[syms[source_idx]]
                orig_digit = mapping[syms[victim_idx]]
                mapping[syms[victim_idx]] = dup_digit
                prompt_lines = [f"Mapping: {' '.join(f'{s}={d}' for s, d in mapping.items())}"]
                prompt = '\n'.join(prompt_lines) + "\nIs this mapping valid (bijective, all 10 digits present)?"
                answer = f"No — digit {dup_digit} appears twice ({syms[source_idx]} and {syms[victim_idx]}), digit {orig_digit} is missing"
                think = (f"Check each digit: {dup_digit} appears for both {syms[source_idx]} and {syms[victim_idx]}.\n"
                         f"Digit {orig_digit} has no symbol. NOT bijective.")
            else:
                # Swap to make gap
                victim_idx = rng.randint(0, 9)
                orig_digit = mapping[syms[victim_idx]]
                bad_digit = rng.randint(10, 15)  # obviously wrong
                mapping[syms[victim_idx]] = bad_digit
                prompt_lines = [f"Mapping: {' '.join(f'{s}={d}' for s, d in mapping.items())}"]
                prompt = '\n'.join(prompt_lines) + "\nIs this mapping valid (bijective, all 10 digits present)?"
                answer = f"No — {syms[victim_idx]}={bad_digit} is out of range (must be 0-9), digit {orig_digit} is missing"
                think = f"{syms[victim_idx]} maps to {bad_digit} which is not a valid digit (0-9). Missing digit {orig_digit}."

        return {"user": prompt, "think": think, "answer": answer}


@register
class BitWitnessSelect(MicroSkill):
    """Given multiple examples, pick the one that best discriminates two gate candidates."""
    name = "bit_witness_select"
    puzzle_type = "bit_manipulation"
    description = "Pick which example best discriminates two confusable gates"
    weight = 12.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_compact import apply_shift, apply_gate, fmt
        gates = ["xor", "or", "and", "xnor", "and_not", "or_not"]
        gate_a, gate_b = rng.sample(gates, 2)
        src = rng.choice(["shr1", "shr2", "shl1", "shl2", "rol1", "ror1"])

        # Generate 3 examples
        examples = []
        best_idx = -1
        best_diff = 0
        for i in range(3):
            x = rng.randint(0, 255)
            x_str = format(x, '08b')
            a = fmt(apply_shift(x, src))
            b = fmt(apply_shift(x, rng.choice(["shr3", "shl3", "rol2", "ror2"])))
            out_a = apply_gate(a, b, gate_a)
            out_b = apply_gate(a, b, gate_b)
            diff_count = sum(1 for ca, cb in zip(out_a, out_b) if ca != cb)
            examples.append((x_str, out_a, out_b, diff_count))
            if diff_count > best_diff:
                best_diff = diff_count
                best_idx = i

        if best_diff == 0:
            return None  # No discriminating example

        prompt_lines = [f"Two candidate gates: {gate_a} vs {gate_b}"]
        for i, (x, oa, ob, d) in enumerate(examples):
            prompt_lines.append(f"Ex{i+1}: x={x} → {gate_a}={oa}, {gate_b}={ob}")
        prompt_lines.append("Which example best discriminates the two gates?")
        prompt = '\n'.join(prompt_lines)

        think_lines = []
        for i, (x, oa, ob, d) in enumerate(examples):
            think_lines.append(f"Ex{i+1}: {d} bits differ")
        think_lines.append(f"Ex{best_idx+1} has most differences ({best_diff} bits)")

        answer = f"Ex{best_idx+1} — {best_diff} bits differ"
        return {"user": prompt, "think": '\n'.join(think_lines), "answer": answer}


# ============================================================
# R13 Brainstorm: FAIL-STOP drill + Gate truth tables
# ============================================================

@register
class BitFailStopDrill(MicroSkill):
    """Drill the FAIL→REJECT decision. Given diff, decide: proceed or reject?
    50/50 PASS/FAIL. The model must learn that diff≠00000000 means STOP."""
    name = "bit_fail_stop_drill"
    puzzle_type = "bit_manipulation"
    description = "Given XOR diff, decide REJECT (diff≠0) or PROCEED (diff=0) — teaches STOP discipline"
    weight = 20.0  # highest weight — targets 75% of bit failures
    max_pool = 20000

    def generate_one(self, rng, difficulty="medium"):
        computed = format(rng.randint(0, 255), '08b')
        is_pass = rng.random() < 0.5
        if is_pass:
            expected = computed
        else:
            # Flip 1-4 bits
            expected_int = int(computed, 2) ^ rng.randint(1, 255)
            expected = format(expected_int & 0xFF, '08b')

        diff = ''.join('0' if a == b else '1' for a, b in zip(computed, expected))

        prompt = (f"Witness check result:\n"
                  f"  computed={computed}\n"
                  f"  expected={expected}\n"
                  f"  diff={diff}\n"
                  f"Should you PROCEED to Query or REJECT this candidate?")

        if diff == '00000000':
            think = f"diff={diff} — all zeros. Computed matches expected."
            answer = "PROCEED — diff=00000000, candidate is correct"
        else:
            n_diff = diff.count('1')
            think = f"diff={diff} — {n_diff} bits differ. Computed does NOT match expected."
            answer = f"REJECT — diff≠00000000, candidate fails. STOP and try different rule."

        return {"user": prompt, "think": think, "answer": answer}


@register
class BitGateTruthTable(MicroSkill):
    """Given a gate name, fill in the truth table. Forces actual computation."""
    name = "bit_gate_truth_table"
    puzzle_type = "bit_manipulation"
    description = "Fill in truth table for a named gate — forces computation not pattern matching"
    weight = 12.0
    max_pool = 10000

    GATES = {
        "and":  lambda a, b: a & b,
        "or":   lambda a, b: a | b,
        "xor":  lambda a, b: a ^ b,
        "nand": lambda a, b: 1 - (a & b),
        "nor":  lambda a, b: 1 - (a | b),
        "xnor": lambda a, b: 1 - (a ^ b),
    }

    def generate_one(self, rng, difficulty="medium"):
        gate = rng.choice(list(self.GATES.keys()))
        fn = self.GATES[gate]

        # Generate 4-8 random bit pairs
        n = rng.randint(4, 8)
        pairs = [(rng.randint(0, 1), rng.randint(0, 1)) for _ in range(n)]

        prompt_lines = [f"Gate: {gate}", "Compute the output for each input pair:"]
        for a, b in pairs:
            prompt_lines.append(f"  {gate}({a},{b}) = ?")
        prompt = '\n'.join(prompt_lines)

        think_lines = []
        results = []
        for a, b in pairs:
            r = fn(a, b)
            results.append(r)
            think_lines.append(f"{gate}({a},{b}) = {r}")

        answer = ' '.join(str(r) for r in results)
        return {"user": prompt, "think": '\n'.join(think_lines), "answer": answer}


@register
class BitShiftSourceBacktrack(MicroSkill):
    """Given a failed shift+gate combo, identify which shift source to change."""
    name = "bit_shift_source_backtrack"
    puzzle_type = "bit_manipulation"
    description = "After gate fails with shift A, identify that changing the SHIFT (not gate) could help"
    weight = 10.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_compact import apply_shift, apply_gate, fmt
        gates = ["xor", "and", "or"]
        shifts = ["shr1", "shr2", "shl1", "shl2", "rol1", "ror1"]

        gate = rng.choice(gates)
        correct_shifts = rng.sample(shifts, 2)
        wrong_shift = rng.choice([s for s in shifts if s != correct_shifts[0]])

        x = rng.randint(1, 254)
        x_str = format(x, '08b')

        # Correct output
        a_correct = fmt(apply_shift(x, correct_shifts[0]))
        b = fmt(apply_shift(x, correct_shifts[1]))
        correct_out = apply_gate(a_correct, b, gate)

        # Wrong output (wrong first shift)
        a_wrong = fmt(apply_shift(x, wrong_shift))
        wrong_out = apply_gate(a_wrong, b, gate)

        if wrong_out == correct_out:
            return None  # no difference

        prompt = (f"x={x_str}\n"
                  f"Tried: A={wrong_shift}(x)={a_wrong}, B={correct_shifts[1]}(x)={b}\n"
                  f"{gate}(A,B)={wrong_out}\n"
                  f"Expected: {correct_out}\n"
                  f"The gate {gate} is correct. What should change?")

        think = (f"Gate {gate} is confirmed correct.\n"
                 f"B={correct_shifts[1]} is fine.\n"
                 f"A={wrong_shift} gives {a_wrong}, but we need A such that {gate}(A,{b})={correct_out}.\n"
                 f"Try A={correct_shifts[0]}: {a_correct} → {gate}({a_correct},{b})={correct_out} → MATCH")

        answer = f"Change A from {wrong_shift} to {correct_shifts[0]}"
        return {"user": prompt, "think": think, "answer": answer}


# ============================================================
# Gate confusion + two-pass verify + gate identification
# ============================================================

@register
class BitGateConfusionDrill(MicroSkill):
    """Show two confusable gates on the SAME inputs, highlight where they differ."""
    name = "bit_gate_confusion_drill"
    puzzle_type = "bit_manipulation"
    description = "Compare two confusable gates bit-by-bit, identify where they differ"
    weight = 15.0
    max_pool = 10000

    PAIRS = [("xor", "or"), ("and", "nand"), ("xor", "xnor"), ("and", "or"), ("or", "nor")]
    GATE_FNS = {
        "and": lambda a, b: a & b, "or": lambda a, b: a | b, "xor": lambda a, b: a ^ b,
        "nand": lambda a, b: 1-(a&b), "nor": lambda a, b: 1-(a|b), "xnor": lambda a, b: 1-(a^b),
    }

    def generate_one(self, rng, difficulty="medium"):
        g1, g2 = rng.choice(self.PAIRS)
        # Pick inputs where they differ
        a = format(rng.randint(1, 254), '08b')
        b = format(rng.randint(1, 254), '08b')
        out1 = ''.join(str(self.GATE_FNS[g1](int(x), int(y))) for x, y in zip(a, b))
        out2 = ''.join(str(self.GATE_FNS[g2](int(x), int(y))) for x, y in zip(a, b))
        diff = ''.join('0' if x == y else '1' for x, y in zip(out1, out2))
        if diff == '00000000':
            return None  # no difference on these inputs

        n_diff = diff.count('1')
        prompt = (f"A={a}\nB={b}\n"
                  f"Compare {g1} vs {g2} on these inputs. Where do they differ?")

        think_lines = [
            f"{g1}(A,B) = {out1}",
            f"{g2}(A,B) = {out2}",
            f"diff = {diff} ({n_diff} positions differ)",
        ]
        # Show which positions
        diff_pos = [str(7-i) for i, c in enumerate(diff) if c == '1']
        think_lines.append(f"Differ at bit positions: {', '.join(diff_pos)}")

        # Show WHY they differ at those positions
        for i, c in enumerate(diff):
            if c == '1':
                ai, bi = int(a[i]), int(b[i])
                think_lines.append(f"  pos {7-i}: A={ai} B={bi} → {g1}={self.GATE_FNS[g1](ai,bi)} {g2}={self.GATE_FNS[g2](ai,bi)}")

        answer = f"{g1}={out1}, {g2}={out2}, differ at {n_diff} positions"
        return {"user": prompt, "think": '\n'.join(think_lines), "answer": answer}


@register
class BitGateFromOutputs(MicroSkill):
    """Given A, B, and output bits — identify which gate produced the output."""
    name = "bit_gate_from_outputs"
    puzzle_type = "bit_manipulation"
    description = "Given A, B, output bits, identify the gate by testing each candidate"
    weight = 15.0
    max_pool = 10000

    GATES = {
        "and": lambda a, b: a & b, "or": lambda a, b: a | b, "xor": lambda a, b: a ^ b,
        "nand": lambda a, b: 1-(a&b), "nor": lambda a, b: 1-(a|b), "xnor": lambda a, b: 1-(a^b),
    }

    def generate_one(self, rng, difficulty="medium"):
        correct_gate = rng.choice(list(self.GATES.keys()))
        fn = self.GATES[correct_gate]
        # Generate 3-4 bit positions to test
        n = rng.randint(3, 5)
        positions = []
        for _ in range(n):
            a, b = rng.randint(0, 1), rng.randint(0, 1)
            positions.append((a, b, fn(a, b)))

        prompt_lines = ["Given these A, B, output values, which gate is it?"]
        for i, (a, b, o) in enumerate(positions):
            prompt_lines.append(f"  A={a} B={b} → {o}")
        prompt_lines.append(f"Test: and, or, xor, nand, nor, xnor")
        prompt = '\n'.join(prompt_lines)

        think_lines = []
        for gate_name, gate_fn in self.GATES.items():
            matches = all(gate_fn(a, b) == o for a, b, o in positions)
            results = [str(gate_fn(a, b)) for a, b, _ in positions]
            expected = [str(o) for _, _, o in positions]
            if matches:
                think_lines.append(f"{gate_name}: {' '.join(results)} vs {' '.join(expected)} → MATCH")
            else:
                think_lines.append(f"{gate_name}: {' '.join(results)} vs {' '.join(expected)} → no")

        answer = correct_gate
        return {"user": prompt, "think": '\n'.join(think_lines), "answer": answer}


# ============================================================
# Trans op discrimination + cipher decode drills
# ============================================================

@register
class TransOpDiscriminate(MicroSkill):
    """Given two candidate ops on same operands, compute both, identify which matches."""
    name = "trans_op_discriminate"
    puzzle_type = "transformation"
    description = "Compute two different ops on same operands, identify which matches expected output"
    weight = 15.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_transform import _make_operands, _calc, _fmt, COMBO_DISPLAY
        ops = ["add", "sub", "mul", "absdiff", "add1", "muladd1"]
        correct_op = rng.choice(ops)
        wrong_op = rng.choice([o for o in ops if o != correct_op])
        order = rng.choice(["AB_CD", "BA_DC"])

        a, b = rng.randint(10, 99), rng.randint(10, 99)
        L, R = _make_operands(a//10, a%10, b//10, b%10, order)
        correct_val = _calc(L, R, correct_op)
        wrong_val = _calc(L, R, wrong_op)
        if correct_val is None or wrong_val is None:
            return None
        if str(correct_val) == str(wrong_val):
            return None

        od = COMBO_DISPLAY.get(order, order)
        prompt = (f"Operands: {a},{b} with order {od} → L={L} R={R}\n"
                  f"Expected output: {correct_val}\n"
                  f"Which op? A) {correct_op} B) {wrong_op}")

        if rng.random() < 0.5:
            prompt = (f"Operands: {a},{b} with order {od} → L={L} R={R}\n"
                      f"Expected output: {correct_val}\n"
                      f"Which op? A) {wrong_op} B) {correct_op}")
            think = f"A) {wrong_op}({L},{R})={wrong_val} vs {correct_val} → no\nB) {correct_op}({L},{R})={correct_val} vs {correct_val} → match"
            answer = f"B) {correct_op}"
        else:
            think = f"A) {correct_op}({L},{R})={correct_val} vs {correct_val} → match\nB) {wrong_op}({L},{R})={wrong_val} vs {correct_val} → no"
            answer = f"A) {correct_op}"

        return {"user": prompt, "think": think, "answer": answer}


@register
class TransCipherDecodeStep(MicroSkill):
    """Practice decoding cipher symbols to digits using a mapping table."""
    name = "trans_cipher_decode_step"
    puzzle_type = "transformation"
    description = "Decode cipher symbols to digits using mapping, then compute operands"
    weight = 12.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        symbols = list('!@#$%^&*()[]{}|/<>~`')
        rng.shuffle(symbols)
        syms = symbols[:10]
        mapping = dict(zip(syms, range(10)))

        # Generate a 4-symbol cipher string
        cipher = [rng.choice(syms) for _ in range(4)]
        digits = [mapping[s] for s in cipher]

        from generators.trace_transform import _make_operands, COMBO_DISPLAY
        order = rng.choice(["AB_CD", "BA_DC"])
        L, R = _make_operands(*digits, order)

        map_str = ' '.join(f'{s}={d}' for s, d in sorted(mapping.items(), key=lambda x: x[1]))
        cipher_str = ''.join(cipher)
        digit_str = ''.join(str(d) for d in digits)

        prompt = (f"Mapping: {map_str}\n"
                  f"Cipher: {cipher_str}\n"
                  f"Decode to digits, then compute L,R with order {COMBO_DISPLAY.get(order, order)}")

        think = (f"Decode: {' '.join(f'{c}={mapping[c]}' for c in cipher)}\n"
                 f"Digits: {digit_str}\n"
                 f"Order {COMBO_DISPLAY.get(order, order)}: L={L} R={R}")

        answer = f"digits={digit_str} L={L} R={R}"
        return {"user": prompt, "think": think, "answer": answer}


@register
class BitIdentityDetect(MicroSkill):
    """Given scan stats showing output=input, identify as identity."""
    name = "bit_identity_detect"
    puzzle_type = "bit_manipulation"
    description = "From scan showing output=input on all examples, identify identity rule"
    weight = 12.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        # Generate examples where output = shifted(input)
        from generators.trace_compact import apply_shift, fmt
        shift = rng.choice(["shr1", "shr2", "shl1", "shl2", "rol1", "rol2", "ror1", "ror2"])

        examples = []
        for _ in range(4):
            x = rng.randint(1, 254)
            inp = format(x, '08b')
            out = fmt(apply_shift(x, shift))
            examples.append((inp, out))

        # 50% identity (output = input), 50% shifted
        is_identity = rng.random() < 0.5
        if is_identity:
            examples = [(inp, inp) for inp, _ in examples]
            correct = "identity: output = x"
        else:
            correct = f"shifted: output = {shift}(x)"

        prompt_lines = ["Scan:"]
        all_match = all(inp == out for inp, out in examples)
        for i, (inp, out) in enumerate(examples):
            match = "SAME" if inp == out else "DIFF"
            prompt_lines.append(f"  Ex{i+1}: {inp}→{out} [{match}]")
        prompt_lines.append("Is this identity (output=input) or a shifted operation?")
        prompt = '\n'.join(prompt_lines)

        think = f"Check if all outputs equal inputs: {'YES' if all_match else 'NO'}"
        answer = correct
        return {"user": prompt, "think": think, "answer": answer}


# ============================================================
# Rapid-fire gate drills — high volume, random order
# ============================================================

@register
class BitGateRapidFire(MicroSkill):
    """Compute 6-10 gate operations on the SAME pair of 8-bit inputs, random order."""
    name = "bit_gate_rapid_fire"
    puzzle_type = "bit_manipulation"
    description = "Given A and B, compute 6-10 different gates in random order"
    weight = 18.0  # very high — teaches gate execution at volume
    max_pool = 20000

    GATES = {
        "and":  lambda a, b: ''.join(str(int(x)&int(y)) for x,y in zip(a,b)),
        "or":   lambda a, b: ''.join(str(int(x)|int(y)) for x,y in zip(a,b)),
        "xor":  lambda a, b: ''.join(str(int(x)^int(y)) for x,y in zip(a,b)),
        "nand": lambda a, b: ''.join(str(1-(int(x)&int(y))) for x,y in zip(a,b)),
        "nor":  lambda a, b: ''.join(str(1-(int(x)|int(y))) for x,y in zip(a,b)),
        "xnor": lambda a, b: ''.join(str(1-(int(x)^int(y))) for x,y in zip(a,b)),
    }

    def generate_one(self, rng, difficulty="medium"):
        a = format(rng.randint(1, 254), '08b')
        b = format(rng.randint(1, 254), '08b')
        gates = list(self.GATES.keys())
        rng.shuffle(gates)
        n = rng.randint(6, len(gates))
        gates = gates[:n]

        prompt = f"A={a}\nB={b}\nCompute each gate:"
        for i, g in enumerate(gates, 1):
            prompt += f"\n{i}. {g}(A,B) = ?"

        think_lines = []
        answers = []
        for i, g in enumerate(gates, 1):
            result = self.GATES[g](a, b)
            think_lines.append(f"{i}. {g}(A,B) = {result}")
            answers.append(result)

        return {"user": prompt, "think": '\n'.join(think_lines), "answer": ' '.join(answers)}


@register
class BitShiftRapidFire(MicroSkill):
    """Compute 5-8 different shifts on the SAME input, random order."""
    name = "bit_shift_rapid_fire"
    puzzle_type = "bit_manipulation"
    description = "Given x, compute 5-8 different shifts/rotates in random order"
    weight = 15.0
    max_pool = 20000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_compact import apply_shift, fmt
        x = rng.randint(1, 254)
        x_str = format(x, '08b')

        all_shifts = [f"{op}{k}" for op in ["shl","shr","rol","ror"] for k in range(1,5)]
        rng.shuffle(all_shifts)
        n = rng.randint(5, 8)
        shifts = all_shifts[:n]

        prompt = f"x={x_str}\nCompute each:"
        for i, s in enumerate(shifts, 1):
            prompt += f"\n{i}. {s}(x) = ?"

        think_lines = []
        answers = []
        for i, s in enumerate(shifts, 1):
            result = fmt(apply_shift(x, s))
            think_lines.append(f"{i}. {s}({x_str}) = {result}")
            answers.append(result)

        return {"user": prompt, "think": '\n'.join(think_lines), "answer": ' '.join(answers)}


@register
class BitDiffRapidFire(MicroSkill):
    """Compute XOR diff for 5-8 pairs, decide PASS/FAIL for each. Random mix."""
    name = "bit_diff_rapid_fire"
    puzzle_type = "bit_manipulation"
    description = "5-8 pairs of computed/expected, compute diff and PASS/FAIL for each"
    weight = 18.0  # very high — directly targets fake_verify (65% of failures)
    max_pool = 20000

    def generate_one(self, rng, difficulty="medium"):
        n = rng.randint(5, 8)
        pairs = []
        for _ in range(n):
            computed = format(rng.randint(0, 255), '08b')
            if rng.random() < 0.4:
                expected = computed  # PASS
            else:
                expected = format(rng.randint(0, 255), '08b')  # likely FAIL
            pairs.append((computed, expected))

        prompt = "For each pair, compute XOR diff and state PASS or FAIL:"
        for i, (c, e) in enumerate(pairs, 1):
            prompt += f"\n{i}. computed={c} expected={e}"

        think_lines = []
        answers = []
        for i, (c, e) in enumerate(pairs, 1):
            diff = ''.join('0' if a==b else '1' for a,b in zip(c, e))
            verdict = "PASS" if diff == '00000000' else "FAIL"
            think_lines.append(f"{i}. diff={diff} → {verdict}")
            answers.append(verdict)

        return {"user": prompt, "think": '\n'.join(think_lines), "answer": ' '.join(answers)}


@register
class BitFullPipelineDrill(MicroSkill):
    """Full pipeline: shift → gate → diff → verdict. Single example, forced sequential."""
    name = "bit_full_pipeline_drill"
    puzzle_type = "bit_manipulation"
    description = "Shift two inputs, apply gate, compute diff vs expected, state verdict"
    weight = 15.0
    max_pool = 20000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_compact import apply_shift, apply_gate, fmt
        x = rng.randint(1, 254)
        x_str = format(x, '08b')
        expected = format(rng.randint(0, 255), '08b')

        shifts = ["shr1","shr2","shl1","shl2","rol1","rol2","ror1","ror2"]
        gates = ["xor","and","or","xnor","nand","nor"]
        s1, s2 = rng.sample(shifts, 2)
        gate = rng.choice(gates)

        a = fmt(apply_shift(x, s1))
        b = fmt(apply_shift(x, s2))
        result = apply_gate(a, b, gate)
        diff = ''.join('0' if x==y else '1' for x,y in zip(result, expected))
        verdict = "PASS" if diff == '00000000' else "FAIL"

        prompt = (f"x={x_str}, expected output={expected}\n"
                  f"Rule: A={s1}(x), B={s2}(x), output={gate}(A,B)\n"
                  f"Execute step by step:")

        think = (f"1. A = {s1}({x_str}) = {a}\n"
                 f"2. B = {s2}({x_str}) = {b}\n"
                 f"3. output = {gate}({a},{b}) = {result}\n"
                 f"4. diff = {result} XOR {expected} = {diff}\n"
                 f"5. verdict: {'PASS (diff=00000000)' if verdict == 'PASS' else f'FAIL (diff≠00000000)'}")

        answer = f"output={result} diff={diff} → {verdict}"
        return {"user": prompt, "think": think, "answer": answer}


# ============================================================
# MEGA DRILLS — 20-30 computations back to back, max pressure
# ============================================================

@register
class BitMegaGateDrill(MicroSkill):
    """20-30 gate computations on different inputs. Pure volume execution pressure."""
    name = "bit_mega_gate_drill"
    puzzle_type = "bit_manipulation"
    description = "20-30 gate(A,B) computations back to back — massive execution practice"
    weight = 20.0  # highest tier — directly builds computation muscle
    max_pool = 20000

    GATES = {
        "and":  lambda a, b: ''.join(str(int(x)&int(y)) for x,y in zip(a,b)),
        "or":   lambda a, b: ''.join(str(int(x)|int(y)) for x,y in zip(a,b)),
        "xor":  lambda a, b: ''.join(str(int(x)^int(y)) for x,y in zip(a,b)),
        "nand": lambda a, b: ''.join(str(1-(int(x)&int(y))) for x,y in zip(a,b)),
        "nor":  lambda a, b: ''.join(str(1-(int(x)|int(y))) for x,y in zip(a,b)),
        "xnor": lambda a, b: ''.join(str(1-(int(x)^int(y))) for x,y in zip(a,b)),
    }

    def generate_one(self, rng, difficulty="medium"):
        n = rng.randint(20, 30)
        gate_names = list(self.GATES.keys())

        prompt = "Compute each gate operation:\n"
        think_lines = []
        answers = []
        for i in range(1, n+1):
            a = format(rng.randint(0, 255), '08b')
            b = format(rng.randint(0, 255), '08b')
            g = rng.choice(gate_names)
            result = self.GATES[g](a, b)
            prompt += f"{i}. {g}({a},{b}) = ?\n"
            think_lines.append(f"{i}. {g}({a},{b}) = {result}")
            answers.append(result)

        return {"user": prompt, "think": '\n'.join(think_lines), "answer": '\n'.join(answers)}


@register
class BitMegaDiffDrill(MicroSkill):
    """20-25 XOR diff computations + PASS/FAIL verdicts. Teaches honest verification at scale."""
    name = "bit_mega_diff_drill"
    puzzle_type = "bit_manipulation"
    description = "20-25 diff computations — teaches the model what real diffs look like"
    weight = 20.0  # highest tier — directly fights fake_verify (65% of failures)
    max_pool = 20000

    def generate_one(self, rng, difficulty="medium"):
        n = rng.randint(20, 25)

        prompt = "For each pair, compute XOR diff and state PASS (00000000) or FAIL:\n"
        think_lines = []
        answers = []
        n_pass = 0
        for i in range(1, n+1):
            computed = format(rng.randint(0, 255), '08b')
            # ~30% PASS, ~70% FAIL — model needs to see more FAILs than PASSes
            if rng.random() < 0.30:
                expected = computed
            else:
                expected = format(rng.randint(0, 255), '08b')
            diff = ''.join('0' if a==b else '1' for a,b in zip(computed, expected))
            verdict = "PASS" if diff == '00000000' else "FAIL"
            if verdict == "PASS": n_pass += 1
            prompt += f"{i}. {computed} vs {expected}\n"
            think_lines.append(f"{i}. diff={diff} → {verdict}")
            answers.append(f"{diff} {verdict}")

        return {"user": prompt, "think": '\n'.join(think_lines),
                "answer": f"{n_pass} PASS, {n-n_pass} FAIL"}


@register
class BitMegaShiftGatePipeline(MicroSkill):
    """15-20 full shift→gate→diff pipelines. The complete computation chain repeated."""
    name = "bit_mega_pipeline_drill"
    puzzle_type = "bit_manipulation"
    description = "15-20 full shift+gate+diff pipelines back to back"
    weight = 18.0
    max_pool = 20000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_compact import apply_shift, apply_gate, fmt
        n = rng.randint(15, 20)
        shifts = ["shr1","shr2","shl1","shl2","rol1","rol2","ror1","ror2"]
        gates = ["xor","and","or","xnor","nand","nor"]

        prompt = "Execute each pipeline: shift A, shift B, gate, diff vs expected:\n"
        think_lines = []
        for i in range(1, n+1):
            x = rng.randint(1, 254)
            x_str = format(x, '08b')
            expected = format(rng.randint(0, 255), '08b')
            s1, s2 = rng.sample(shifts, 2)
            gate = rng.choice(gates)

            a = fmt(apply_shift(x, s1))
            b = fmt(apply_shift(x, s2))
            result = apply_gate(a, b, gate)
            diff = ''.join('0' if x==y else '1' for x,y in zip(result, expected))
            verdict = "PASS" if diff == '00000000' else "FAIL"

            prompt += f"{i}. x={x_str} {s1}/{s2}/{gate} expected={expected}\n"
            think_lines.append(
                f"{i}. A={a} B={b} {gate}={result} diff={diff} → {verdict}")

        return {"user": prompt, "think": '\n'.join(think_lines), "answer": f"{n} pipelines computed"}


@register
class BitMegaGRIDDrill(MicroSkill):
    """25 full GRID computations — A row, B row, gate row. Pure bit-by-bit execution."""
    name = "bit_mega_grid_drill"
    puzzle_type = "bit_manipulation"
    description = "25 full 3-row GRID computations (A, B, gate output) back to back"
    weight = 25.0
    max_pool = 20000

    GATES = {
        "and":  lambda a, b: str(int(a)&int(b)),
        "or":   lambda a, b: str(int(a)|int(b)),
        "xor":  lambda a, b: str(int(a)^int(b)),
        "nand": lambda a, b: str(1-(int(a)&int(b))),
        "nor":  lambda a, b: str(1-(int(a)|int(b))),
        "xnor": lambda a, b: str(1-(int(a)^int(b))),
    }

    def generate_one(self, rng, difficulty="medium"):
        n = rng.randint(20, 25)
        gate_names = list(self.GATES.keys())

        prompt = "Compute each GRID (show A row, B row, output row):\n"
        think_lines = []
        for i in range(1, n+1):
            a = format(rng.randint(0, 255), '08b')
            b = format(rng.randint(0, 255), '08b')
            g = rng.choice(gate_names)
            fn = self.GATES[g]
            out = ''.join(fn(x, y) for x, y in zip(a, b))
            prompt += f"{i}. {g}(A,B) where A={a} B={b}\n"
            think_lines.append(f"{i}. GRID({g}):")
            think_lines.append(f"   {' '.join(a)}")
            think_lines.append(f"   {' '.join(b)}")
            think_lines.append(f"   {' '.join(out)}")
            think_lines.append(f"   ={out}")

        return {"user": prompt, "think": '\n'.join(think_lines), "answer": f"{n} GRIDs computed"}


@register
class BitDecisionChainDrill(MicroSkill):
    """20-25 decision points: given a diff, decide STOP or PROCEED. Pure routing practice."""
    name = "bit_decision_chain_drill"
    puzzle_type = "bit_manipulation"
    description = "25 decision points back to back: diff → STOP or PROCEED?"
    weight = 25.0
    max_pool = 20000

    def generate_one(self, rng, difficulty="medium"):
        n = rng.randint(20, 25)
        prompt = "For each witness result, what is the correct action?\n"
        think_lines = []
        answers = []
        for i in range(1, n+1):
            diff = format(rng.randint(0, 255), '08b')
            # 30% are all zeros (PASS)
            if rng.random() < 0.30:
                diff = '00000000'

            if diff == '00000000':
                action = "PROCEED"
                reason = "diff=00000000, candidate matches"
            else:
                action = "STOP"
                reason = f"diff={diff}≠00000000, candidate fails"

            prompt += f"{i}. diff={diff} → ?\n"
            think_lines.append(f"{i}. {reason} → {action}")
            answers.append(action)

        n_stop = answers.count("STOP")
        n_go = answers.count("PROCEED")
        return {"user": prompt, "think": '\n'.join(think_lines),
                "answer": f"{n_stop} STOP, {n_go} PROCEED"}


@register
class BitFullDecisionDrill(MicroSkill):
    """15 scenarios: given a candidate result + expected, compute diff, decide, and state next action."""
    name = "bit_full_decision_drill"
    puzzle_type = "bit_manipulation"
    description = "15 full scenarios: compute → diff → decide → next action (STOP+retry or LOCK+query)"
    weight = 20.0
    max_pool = 20000

    def generate_one(self, rng, difficulty="medium"):
        n = rng.randint(12, 15)
        prompt = "For each: compute diff, decide STOP or PROCEED, state what comes NEXT:\n"
        think_lines = []
        for i in range(1, n+1):
            computed = format(rng.randint(0, 255), '08b')
            if rng.random() < 0.35:
                expected = computed  # PASS
            else:
                expected = format(rng.randint(0, 255), '08b')

            diff = ''.join('0' if a==b else '1' for a,b in zip(computed, expected))

            prompt += f"{i}. computed={computed} expected={expected}\n"

            if diff == '00000000':
                think_lines.append(f"{i}. diff={diff} → PROCEED → run Witness 2 (or LOCK if already 2 PASS)")
            else:
                think_lines.append(f"{i}. diff={diff} → STOP → reject this candidate, try Candidate N+1")

        return {"user": prompt, "think": '\n'.join(think_lines),
                "answer": f"{n} decisions made"}


@register
class TransMegaOpDrill(MicroSkill):
    """25 op computations: given L,R compute add/sub/mul/absdiff etc. back to back."""
    name = "trans_mega_op_drill"
    puzzle_type = "transformation"
    description = "25 arithmetic op computations on different operands — pure execution"
    weight = 25.0
    max_pool = 20000

    OPS = {
        "add": lambda l,r: l+r,
        "sub": lambda l,r: l-r,
        "mul": lambda l,r: l*r,
        "absdiff": lambda l,r: abs(l-r),
        "add1": lambda l,r: l+r+1,
        "muladd1": lambda l,r: l*r+1,
    }

    def generate_one(self, rng, difficulty="medium"):
        n = rng.randint(20, 25)
        op_names = list(self.OPS.keys())
        prompt = "Compute each operation:\n"
        think_lines = []
        for i in range(1, n+1):
            L = rng.randint(1, 99)
            R = rng.randint(1, 99)
            op = rng.choice(op_names)
            result = self.OPS[op](L, R)
            prompt += f"{i}. {op}({L},{R}) = ?\n"
            think_lines.append(f"{i}. {op}({L},{R}) = {result}")
        return {"user": prompt, "think": '\n'.join(think_lines), "answer": f"{n} ops computed"}


@register
class TransMegaOrderDrill(MicroSkill):
    """25 operand assembly drills: digits a,b,c,d + order → L,R."""
    name = "trans_mega_order_drill"
    puzzle_type = "transformation"
    description = "25 operand assemblies: digits + order → L,R back to back"
    weight = 20.0
    max_pool = 20000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_transform import _make_operands, COMBO_DISPLAY
        n = rng.randint(20, 25)
        orders = ["AB_CD", "BA_DC", "AB_DC", "BA_CD"]
        prompt = "For each, assemble L and R from the digits:\n"
        think_lines = []
        for i in range(1, n+1):
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            order = rng.choice(orders)
            L, R = _make_operands(a//10, a%10, b//10, b%10, order)
            od = COMBO_DISPLAY.get(order, order)
            prompt += f"{i}. digits={a},{b} order={od} → L=? R=?\n"
            think_lines.append(f"{i}. {od}: L={L} R={R}")
        return {"user": prompt, "think": '\n'.join(think_lines), "answer": f"{n} assemblies"}


@register
class EncMegaVocabDrill(MicroSkill):
    """25 vocab checks: is this word in the 77-word Alice vocabulary?"""
    name = "enc_mega_vocab_drill"
    puzzle_type = "encryption"
    description = "25 rapid vocab checks — is each word in the 77-word vocabulary?"
    weight = 20.0
    max_pool = 20000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None
        vocab_list = sorted(vocab)
        imposters = ["farmer","baker","writer","ranger","catches","fetches","launches",
                     "dragons","knights","rabbits","castles","forests","throne","compass",
                     "painting","journey","lantern","chamber","enchant","whisper",
                     "seeking","guarded","ancient","finding","written","kingdom"]
        imposters = [w for w in imposters if w not in vocab]

        n = rng.randint(20, 25)
        prompt = "Is each word in Alice's 77-word vocabulary? Answer YES or NO:\n"
        think_lines = []
        answers = []
        for i in range(1, n+1):
            if rng.random() < 0.5:
                word = rng.choice(vocab_list)
                ans = "YES"
            else:
                word = rng.choice(imposters) if imposters else rng.choice(vocab_list)
                ans = "NO" if word not in vocab else "YES"
            prompt += f"{i}. {word}\n"
            think_lines.append(f"{i}. {word} → {ans}")
            answers.append(ans)

        n_yes = answers.count("YES")
        return {"user": prompt, "think": '\n'.join(think_lines),
                "answer": f"{n_yes} YES, {n-n_yes} NO"}


@register
class BitMegaShiftDrill(MicroSkill):
    """25 shift/rotate computations on different inputs. Pure positional execution."""
    name = "bit_mega_shift_drill"
    puzzle_type = "bit_manipulation"
    description = "25 shift/rotate computations back to back"
    weight = 20.0
    max_pool = 20000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_compact import apply_shift, fmt
        n = rng.randint(20, 25)
        all_shifts = [f"{op}{k}" for op in ["shl","shr","rol","ror"] for k in range(1,5)]
        prompt = "Compute each shift/rotate:\n"
        think_lines = []
        for i in range(1, n+1):
            x = rng.randint(1, 254)
            x_str = format(x, '08b')
            s = rng.choice(all_shifts)
            result = fmt(apply_shift(x, s))
            prompt += f"{i}. {s}({x_str}) = ?\n"
            think_lines.append(f"{i}. {s}({x_str}) = {result}")
        return {"user": prompt, "think": '\n'.join(think_lines), "answer": f"{n} shifts computed"}


@register
class EncMegaDecodeDrill(MicroSkill):
    """20-25 cipher→plain word decodes: given mapping + cipher word, decode letter by letter, match to vocab."""
    name = "enc_mega_decode_drill"
    puzzle_type = "encryption"
    description = "20-25 full cipher word decodes: flat map lookup → vocab verify"
    weight = 25.0
    max_pool = 20000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None
        vocab_list = sorted(vocab)

        import string
        alpha = list(string.ascii_lowercase)
        shuffled = alpha[:]
        rng.shuffle(shuffled)
        c2p = dict(zip(alpha, shuffled))

        n = rng.randint(20, 25)
        words = rng.sample(vocab_list, min(n, len(vocab_list)))

        map_lines = _enc_build_flat_map(c2p)
        prompt = "\n".join(map_lines) + "\n\nDecode each cipher word and match to vocabulary:\n"

        think_lines = []
        for i, word in enumerate(words[:n], 1):
            p2c = {v: k for k, v in c2p.items()}
            cipher = ''.join(p2c.get(ch, '?') for ch in word)
            decoded = ''.join(c2p.get(ch, '?') for ch in cipher)
            prompt += f"{i}. {cipher} ({len(cipher)} letters)\n"
            decode_chain = ' '.join(f"{c}={c2p.get(c,'?')}" for c in cipher)
            think_lines.append(f"{i}. {cipher}: {decode_chain} = {decoded} [vocab ✓]")

        return {"user": prompt, "think": '\n'.join(think_lines),
                "answer": f"{n} words decoded"}


@register
class TransMegaStyleDrill(MicroSkill):
    """20-25 style applications: given raw value + style, compute formatted output."""
    name = "trans_mega_style_drill"
    puzzle_type = "transformation"
    description = "20-25 style applications (raw/rev/abs/opsign) back to back"
    weight = 25.0
    max_pool = 20000

    def generate_one(self, rng, difficulty="medium"):
        n = rng.randint(20, 25)
        styles = ["raw", "rev", "abs"]
        op_chars = ['+', '-', '*', '/']

        prompt = "Apply the style to each raw value:\n"
        think_lines = []
        for i in range(1, n+1):
            raw = rng.randint(-999, 999)
            style = rng.choice(styles)
            op = rng.choice(op_chars)

            if style == "raw":
                result = str(raw)
            elif style == "rev":
                s = str(raw)
                neg = s.startswith('-')
                digits = s.lstrip('-')
                result = ('-' + digits[::-1]) if neg else digits[::-1]
            elif style == "abs":
                result = str(abs(raw))

            prompt += f"{i}. raw={raw} style={style}\n"
            think_lines.append(f"{i}. {style}({raw}) = {result}")

        return {"user": prompt, "think": '\n'.join(think_lines),
                "answer": f"{n} styles applied"}


@register
class EncMegaPatternMatchDrill(MicroSkill):
    """20 pattern→vocab match: given partial decode like _a_tle, find the vocab word."""
    name = "enc_mega_pattern_match_drill"
    puzzle_type = "encryption"
    description = "20 partial patterns → match to 77-word vocab, with same-length distractors"
    weight = 25.0
    max_pool = 20000

    def generate_one(self, rng, difficulty="medium"):
        vocab = load_vocab()
        if not vocab: return None
        vocab_list = sorted(vocab)
        by_len = {}
        for w in vocab_list:
            by_len.setdefault(len(w), []).append(w)

        n = rng.randint(18, 22)
        prompt = "Match each partial pattern to a word in Alice's 77-word vocabulary:\n"
        think_lines = []

        for i in range(1, n+1):
            # Pick a word, mask 1-3 letters
            word = rng.choice([w for w in vocab_list if len(w) >= 4])
            wlen = len(word)
            n_mask = rng.randint(1, min(3, wlen-2))
            mask_pos = rng.sample(range(wlen), n_mask)
            pattern = ''.join('_' if j in mask_pos else word[j] for j in range(wlen))

            # Get same-length candidates
            candidates = by_len.get(wlen, [word])
            # Filter to those matching known positions
            matches = [w for w in candidates
                       if all(w[j] == pattern[j] for j in range(wlen) if pattern[j] != '_')]

            prompt += f"{i}. {pattern} ({wlen} letters)\n"
            if len(matches) == 1:
                think_lines.append(f"{i}. {pattern} → only match: {matches[0]}")
            else:
                think_lines.append(f"{i}. {pattern} → candidates: {', '.join(matches[:5])} → {word}")

        return {"user": prompt, "think": '\n'.join(think_lines),
                "answer": f"{n} patterns matched"}



# ============================================================
# CIPHER-DIGIT MICRO-SKILLS (teach the cracking pipeline)
# ============================================================

CIPHER_SYMBOL_POOL = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

def _make_cipher_mapping(rng, base=10):
    syms = rng.sample(CIPHER_SYMBOL_POOL, base)
    return {s: i for i, s in enumerate(syms)}, {i: s for i, s in enumerate(syms)}


@register
class CipherDecode(MicroSkill):
    """Given a symbol→digit mapping, decode a cipher string to digits."""
    name = "cipher_decode"
    puzzle_type = "transformation"
    description = "Decode cipher symbols to digits using the mapping"
    weight = 15.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        base = rng.choice([8, 9, 10])
        mapping, inv = _make_cipher_mapping(rng, base)
        # Random cipher string (2-4 chars)
        n = rng.randint(2, 4)
        digits = [rng.randint(0, base - 1) for _ in range(n)]
        cipher = "".join(inv[d] for d in digits)
        digit_str = "".join(str(d) for d in digits)

        map_str = " ".join(f"{s}={d}" for s, d in sorted(mapping.items(), key=lambda x: x[1]))
        prompt = f"Mapping: {map_str}\n\nDecode: {cipher}"
        think = "\n".join(f"  {inv[d]}={d}" for d in digits) + f"\n= {digit_str}"
        return {"user": prompt, "think": think, "answer": digit_str}


@register
class CipherEncode(MicroSkill):
    """Given a mapping and a digit string, encode back to symbols."""
    name = "cipher_encode"
    puzzle_type = "transformation"
    description = "Encode digits back to cipher symbols using the mapping"
    weight = 15.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        base = rng.choice([8, 9, 10])
        mapping, inv = _make_cipher_mapping(rng, base)
        n = rng.randint(2, 4)
        digits = [rng.randint(0, base - 1) for _ in range(n)]
        digit_str = "".join(str(d) for d in digits)
        cipher = "".join(inv[d] for d in digits)

        map_str = " ".join(f"{s}={d}" for s, d in sorted(mapping.items(), key=lambda x: x[1]))
        prompt = f"Mapping: {map_str}\n\nEncode: {digit_str}"
        think = "\n".join(f"  {d}→{inv[d]}" for d in digits) + f"\n= {cipher}"
        return {"user": prompt, "think": think, "answer": cipher}


@register
class CipherCheckCombo(MicroSkill):
    """Given mapping + ordering + op + style, check if an example is consistent."""
    name = "cipher_check_combo"
    puzzle_type = "transformation"
    description = "Does this combo explain this cipher example? Full arithmetic check."
    weight = 20.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_transform import ALL_OPS, _apply_style
        base = rng.choice([8, 9, 10])
        mapping, inv = _make_cipher_mapping(rng, base)

        # Pick a combo
        do_rev = rng.choice([True, False])
        ord_label = "BA,DC" if do_rev else "AB,CD"
        op_name, op_fn = rng.choice(ALL_OPS)
        style = rng.choice(["plain", "rev", "abs"])

        # Generate an example that's consistent
        a0, a1, b0, b1 = [rng.randint(0, base-1) for _ in range(4)]
        a_val = a1*10+a0 if do_rev else a0*10+a1
        b_val = b1*10+b0 if do_rev else b0*10+b1
        try:
            raw = op_fn(a_val, b_val)
            result = _apply_style(raw, style, "*")
        except:
            return None
        if len(result) > 5 or len(result) == 0:
            return None

        a_sym = inv[a0] + inv[a1]
        b_sym = inv[b0] + inv[b1]
        out_sym = "".join(inv[int(c)] if c.isdigit() and int(c) < base else c for c in result)
        op_sym = rng.choice([c for c in CIPHER_SYMBOL_POOL if c not in mapping])

        # 50% show the right combo, 50% show a wrong one
        is_correct = rng.random() < 0.5
        if not is_correct:
            # Mutate: change the operation
            wrong_ops = [n for n, _ in ALL_OPS if n != op_name]
            show_op = rng.choice(wrong_ops)
        else:
            show_op = op_name

        map_str = " ".join(f"{s}={d}" for s, d in sorted(mapping.items(), key=lambda x: x[1]))
        prompt = (
            f"Mapping: {map_str}\n"
            f"Example: {a_sym}{op_sym}{b_sym} = {out_sym}\n"
            f"Combo: {ord_label} {show_op} {style}\n"
            f"Is this combo consistent with the example?"
        )

        # Show the work
        think_lines = [f"Decode: {a_sym}→{a0}{a1}, {b_sym}→{b0}{b1}"]
        if do_rev:
            think_lines.append(f"Order BA,DC: L={a1}{a0}={a_val} R={b1}{b0}={b_val}")
        else:
            think_lines.append(f"Order AB,CD: L={a0}{a1}={a_val} R={b0}{b1}={b_val}")

        try:
            check_fn = None
            for n, fn in ALL_OPS:
                if n == show_op: check_fn = fn; break
            check_raw = check_fn(a_val, b_val)
            check_result = _apply_style(check_raw, style, op_sym)
        except:
            return None

        out_decoded = "".join(str(mapping[c]) if c in mapping else c for c in out_sym)
        think_lines.append(f"{show_op}({a_val},{b_val})={check_raw}, {style}→{check_result}")
        think_lines.append(f"Expected: {out_decoded}")

        if check_result == out_decoded:
            think_lines.append("Match → YES")
            answer = "YES"
        else:
            think_lines.append(f"Mismatch: {check_result}≠{out_decoded} → NO")
            answer = "NO"

        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


@register
class CipherCrackOneSymbol(MicroSkill):
    """Given a partial mapping with 1 unknown, find the missing digit by testing against an example."""
    name = "cipher_crack_one"
    puzzle_type = "transformation"
    description = "Find the missing digit for one unknown symbol from examples"
    weight = 20.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_transform import ALL_OPS, _apply_style
        base = rng.choice([8, 9, 10])
        mapping, inv = _make_cipher_mapping(rng, base)

        # Pick a combo and generate an example
        do_rev = rng.choice([True, False])
        op_name, op_fn = rng.choice([(n,f) for n,f in ALL_OPS if n in ("add","sub","mul","absdiff","concat")])
        style = rng.choice(["plain", "rev"])

        a0, a1, b0, b1 = [rng.randint(0, base-1) for _ in range(4)]
        a_val = a1*10+a0 if do_rev else a0*10+a1
        b_val = b1*10+b0 if do_rev else b0*10+b1
        try:
            raw = op_fn(a_val, b_val)
            result_str = _apply_style(raw, style, "*")
        except:
            return None
        if len(result_str) > 5 or len(result_str) == 0:
            return None
        # All result digits must be in range
        for c in result_str:
            if c.isdigit() and int(c) >= base:
                return None

        a_sym = inv[a0] + inv[a1]
        b_sym = inv[b0] + inv[b1]
        out_sym = "".join(inv[int(c)] for c in result_str if c.isdigit())
        op_sym = rng.choice([c for c in CIPHER_SYMBOL_POOL if c not in mapping])

        # Hide one symbol
        all_syms_in_example = set(a_sym + b_sym + out_sym)
        if len(all_syms_in_example) < 2:
            return None
        hidden_sym = rng.choice(sorted(all_syms_in_example))
        hidden_digit = mapping[hidden_sym]

        # Show partial mapping (without the hidden symbol)
        partial = {s: d for s, d in mapping.items() if s != hidden_sym}
        map_str = " ".join(f"{s}={d}" for s, d in sorted(partial.items(), key=lambda x: x[1]))
        used_digits = sorted(set(partial.values()))
        available = [d for d in range(base) if d not in partial.values()]

        ord_label = "BA,DC" if do_rev else "AB,CD"
        prompt = (
            f"Base {base}. Partial mapping: {map_str}\n"
            f"Unknown symbol: {hidden_sym} (could be {available})\n"
            f"Rule: {ord_label} {op_name} {style}\n"
            f"Example: {a_sym}{op_sym}{b_sym} = {out_sym}\n"
            f"What digit does {hidden_sym} map to?"
        )

        think_lines = [f"Try each possible digit for {hidden_sym}:"]
        for d in available:
            test_map = dict(partial)
            test_map[hidden_sym] = d
            ta = test_map.get(a_sym[0], "?")
            tb = test_map.get(a_sym[1], "?")
            tc = test_map.get(b_sym[0], "?")
            td = test_map.get(b_sym[1], "?")
            if isinstance(ta, int) and isinstance(tb, int) and isinstance(tc, int) and isinstance(td, int):
                av = tb*10+ta if do_rev else ta*10+tb
                bv = td*10+tc if do_rev else tc*10+td
                try:
                    r = op_fn(av, bv)
                    styled = _apply_style(r, style, op_sym)
                    out_dec = "".join(str(test_map.get(c, "?")) for c in out_sym)
                    if styled == out_dec:
                        think_lines.append(f"  {hidden_sym}={d}: {op_name}({av},{bv})={r} {style}→{styled} = {out_dec} → match!")
                    else:
                        think_lines.append(f"  {hidden_sym}={d}: {op_name}({av},{bv})={r} {style}→{styled} ≠ {out_dec} → no")
                except:
                    think_lines.append(f"  {hidden_sym}={d}: error → no")
            else:
                think_lines.append(f"  {hidden_sym}={d}: incomplete decode → no")

        think_lines.append(f"Answer: {hidden_sym}={hidden_digit}")
        return {"user": prompt, "think": "\n".join(think_lines), "answer": str(hidden_digit)}


@register
class CipherFullPipeline(MicroSkill):
    """Full cipher mini-puzzle: decode, scan, apply, encode. 1-2 operators, 2-3 examples."""
    name = "cipher_full_pipeline"
    puzzle_type = "transformation"
    description = "Mini cipher puzzle: crack mapping → find ops → compute answer"
    weight = 25.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_transform import ALL_OPS, _apply_style, build_cipher_trace
        base = rng.choice([8, 9, 10])
        mapping, inv = _make_cipher_mapping(rng, base)

        # Pick 1-2 operators
        remaining_syms = [c for c in CIPHER_SYMBOL_POOL if c not in mapping]
        if len(remaining_syms) < 2:
            return None
        n_ops = rng.choice([1, 2])
        op_syms = rng.sample(remaining_syms, n_ops)

        # Pick combos
        COMMON_COMBOS = [
            ("AB,CD", "add", "plain"), ("AB,CD", "sub", "plain"),
            ("AB,CD", "mul", "plain"), ("BA,DC", "add", "rev"),
            ("BA,DC", "mul", "rev"), ("BA,DC", "sub", "rev"),
            ("AB,CD", "sub", "abs"), ("AB,CD", "concat", "plain"),
        ]
        op_combos = {}
        for os in op_syms:
            op_combos[os] = rng.choice(COMMON_COMBOS)

        # Generate 2-3 examples + query
        examples = []
        for os in op_syms:
            ordering, op_name, style = op_combos[os]
            do_rev = (ordering == "BA,DC")
            op_fn = None
            for n, fn in ALL_OPS:
                if n == op_name: op_fn = fn; break
            for _ in range(rng.randint(1, 2)):
                a0, a1, b0, b1 = [rng.randint(0, base-1) for _ in range(4)]
                av = a1*10+a0 if do_rev else a0*10+a1
                bv = b1*10+b0 if do_rev else b0*10+b1
                try:
                    raw = op_fn(av, bv)
                    res = _apply_style(raw, style, os)
                except:
                    continue
                for c in res:
                    if c.isdigit() and int(c) >= base:
                        break
                else:
                    a_s = inv[a0]+inv[a1]
                    b_s = inv[b0]+inv[b1]
                    out_s = "".join(inv[int(c)] if c.isdigit() and int(c)<base else c for c in res)
                    examples.append((a_s+os+b_s, out_s))

        if len(examples) < 2:
            return None

        # Query
        q_op = rng.choice(op_syms)
        qa0, qa1, qb0, qb1 = [rng.randint(0, base-1) for _ in range(4)]
        query = inv[qa0]+inv[qa1]+q_op+inv[qb0]+inv[qb1]

        ordering, op_name, style = op_combos[q_op]
        do_rev = (ordering == "BA,DC")
        op_fn = None
        for n, fn in ALL_OPS:
            if n == op_name: op_fn = fn; break
        av = qa1*10+qa0 if do_rev else qa0*10+qa1
        bv = qb1*10+qb0 if do_rev else qb0*10+qb1
        try:
            raw = op_fn(av, bv)
            res = _apply_style(raw, style, q_op)
        except:
            return None
        for c in res:
            if c.isdigit() and int(c) >= base:
                return None
        gold = "".join(inv[int(c)] if c.isdigit() and int(c)<base else c for c in res)

        # Build prompt
        prompt_lines = ["In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:"]
        for inp, out in examples:
            prompt_lines.append(f"{inp} = {out}")
        prompt_lines.append(f"Now, determine the result for: {query}")
        prompt = "\n".join(prompt_lines)

        # Build trace
        op_states = {os: (op_combos[os][1], op_combos[os][0]=="BA,DC", False) for os in op_syms}
        result = build_cipher_trace(prompt, gold, mapping, base, op_states)
        if result is None:
            return None

        from training.data import BOXED_INSTRUCTION
        reasoning, answer = result
        return {
            "user": prompt + BOXED_INSTRUCTION,
            "think": reasoning,
            "answer": gold,
        }


@register
class BitEliminationDrill(MicroSkill):
    """25 rapid Step 1/2/3 decisions: given scan stats, classify as CONST/identity/NOT/gate."""
    name = "bit_elimination_drill"
    puzzle_type = "bit_manipulation"
    description = "25 mechanical elimination decisions: CONST? identity? NOT? or gate?"
    weight = 25.0
    max_pool = 20000

    def generate_one(self, rng, difficulty="medium"):
        n = rng.randint(20, 25)
        prompt = "For each set of examples, classify: CONST, IDENTITY, NOT, or GATE:\n"
        think_lines = []
        for i in range(1, n+1):
            # Generate a random scenario
            scenario = rng.choice(["const", "identity", "not", "gate"])
            x1 = format(rng.randint(1, 254), "08b")
            x2 = format(rng.randint(1, 254), "08b")
            
            if scenario == "const":
                c = format(rng.randint(0, 255), "08b")
                o1, o2 = c, c
                answer = f"CONST ({c})"
                reason = "all outputs identical"
            elif scenario == "identity":
                o1, o2 = x1, x2
                answer = "IDENTITY"
                reason = "output = input"
            elif scenario == "not":
                o1 = "".join("1" if c=="0" else "0" for c in x1)
                o2 = "".join("1" if c=="0" else "0" for c in x2)
                answer = "NOT"
                reason = "output = NOT(input)"
            else:
                o1 = format(rng.randint(0, 255), "08b")
                o2 = format(rng.randint(0, 255), "08b")
                if o1 == x1 and o2 == x2:
                    o1 = format((int(o1, 2) ^ 1) & 0xFF, "08b")
                answer = "GATE"
                reason = "output differs from input, not constant, not NOT"
            
            prompt += f"{i}. {x1}→{o1}, {x2}→{o2}\n"
            think_lines.append(f"{i}. {reason} → {answer}")
        
        return {"user": prompt, "think": "\n".join(think_lines), "answer": f"{n} classified"}



# ============================================================
# BIT: PER-BIT COLUMN DECOMPOSITION SKILLS (new format)
# ============================================================

@register
class BitColumnRead(MicroSkill):
    """Extract bit columns from examples — the fundamental decomposition skill."""
    name = "bit_column_read"
    puzzle_type = "bit_manipulation"
    description = "Extract input/output bit columns from example pairs"
    weight = 20.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        n_ex = rng.randint(3, 6)
        inputs = [format(rng.randint(0, 255), "08b") for _ in range(n_ex)]
        outputs = [format(rng.randint(0, 255), "08b") for _ in range(n_ex)]

        # Pick a random output bit to ask about
        bit_pos = rng.randint(0, 7)

        prompt = "Examples:\n"
        for inp, out in zip(inputs, outputs):
            prompt += f"  {inp} → {out}\n"
        prompt += f"\nExtract column o[{bit_pos}] (the output bit at position {bit_pos} across all examples)."

        col = ",".join(out[bit_pos] for out in outputs)
        think = f"o[{bit_pos}]: {col}"

        return {"user": prompt, "think": think, "answer": col}


@register
class BitScanSingle(MicroSkill):
    """Practice the CONST→COPY→NOT→gate scan for one output bit.
    This is the core per-bit skill the model needs to learn."""
    name = "bit_scan_single"
    puzzle_type = "bit_manipulation"
    description = "Scan one output bit: CONST? COPY? NOT? gate? — fixed order"
    weight = 25.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        n_ex = rng.randint(4, 7)

        # Pick what the rule will be
        rule_type = rng.choices(
            ["const", "copy", "not", "gate2"],
            weights=[15, 42, 5, 38]
        )[0]

        # Generate inputs
        inputs = [format(rng.randint(0, 255), "08b") for _ in range(n_ex)]
        input_cols = ["".join(inp[b] for inp in inputs) for b in range(8)]

        bit_pos = rng.randint(0, 7)

        if rule_type == "const":
            val = rng.choice(["0", "1"])
            target_col = val * n_ex
            rule_label = f"CONST({val})"
        elif rule_type == "copy":
            src = rng.randint(0, 7)
            target_col = input_cols[src]
            rule_label = f"COPY({src})"
            # Make sure it's not also constant
            if len(set(target_col)) == 1:
                return None  # ambiguous, retry
        elif rule_type == "not":
            src = rng.randint(0, 7)
            target_col = "".join("1" if c == "0" else "0" for c in input_cols[src])
            rule_label = f"NOT({src})"
            if len(set(target_col)) == 1:
                return None
        else:  # gate2
            gate = rng.choice(["AND", "OR", "XOR", "NAND", "NOR", "XNOR"])
            a, b = rng.sample(range(8), 2)
            if a > b:
                a, b = b, a
            from generators.trace_perbit import _gate_col
            target_col = _gate_col(input_cols[a], input_cols[b], gate)
            rule_label = f"{gate}({a},{b})"
            # Make sure it's not accidentally const/copy/not
            if len(set(target_col)) == 1:
                return None
            for i in range(8):
                if input_cols[i] == target_col:
                    return None
            for i in range(8):
                inv = "".join("1" if c == "0" else "0" for c in input_cols[i])
                if inv == target_col:
                    return None

        # Build outputs with this target column at bit_pos
        outputs = []
        for ex_idx in range(n_ex):
            bits = list(format(rng.randint(0, 255), "08b"))
            bits[bit_pos] = target_col[ex_idx]
            outputs.append("".join(bits))

        target_str = ",".join(target_col)

        prompt = "Examples:\n"
        for inp, out in zip(inputs, outputs):
            prompt += f"  {inp} → {out}\n"
        prompt += f"\nScan o[{bit_pos}]: {target_str}\nWhat rule produces this column? Check CONST, then COPY, then NOT, then gates."

        # Build scan trace
        from generators.trace_perbit import _scan_bit, _col_str
        rule, scan_lines = _scan_bit(target_col, input_cols, show_rejects=2)
        think = f"o[{bit_pos}]: {target_str}\n" + "\n".join(f"  {sl}" for sl in scan_lines)

        return {"user": prompt, "think": think, "answer": rule_label}


@register
class BitApplyRules(MicroSkill):
    """Given 8 per-bit rules and a query, compute the answer bit by bit."""
    name = "bit_apply_rules"
    puzzle_type = "bit_manipulation"
    description = "Apply 8 per-bit rules to a query input — compute the answer"
    weight = 25.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        query = format(rng.randint(0, 255), "08b")

        # Generate 8 random rules (mostly simple)
        rules = []
        rule_labels = []
        answer_bits = []
        for i in range(8):
            rtype = rng.choices(["const", "copy", "not", "gate2"], weights=[15, 42, 5, 38])[0]
            if rtype == "const":
                val = rng.choice(["0", "1"])
                rules.append(("CONST", val))
                rule_labels.append(f"CONST({val})")
                answer_bits.append(val)
            elif rtype == "copy":
                src = rng.randint(0, 7)
                rules.append(("COPY", src))
                rule_labels.append(f"COPY({src})")
                answer_bits.append(query[src])
            elif rtype == "not":
                src = rng.randint(0, 7)
                rules.append(("NOT", src))
                rule_labels.append(f"NOT({src})")
                answer_bits.append("1" if query[src] == "0" else "0")
            else:
                gate = rng.choice(["AND", "OR", "XOR"])
                a, b = rng.sample(range(8), 2)
                if a > b:
                    a, b = b, a
                rules.append(("GATE2", gate, a, b))
                rule_labels.append(f"{gate}({a},{b})")
                from generators.trace_perbit import _gate
                answer_bits.append(_gate(query[a], query[b], gate))

        answer = "".join(answer_bits)

        prompt = f"Rule: {','.join(rule_labels)}\n\nQuery: {query}\nApply each rule to compute the 8-bit output."

        think_lines = []
        for i in range(8):
            r = rules[i]
            if r[0] == "CONST":
                think_lines.append(f"o[{i}]={r[1]}")
            elif r[0] == "COPY":
                think_lines.append(f"o[{i}]=i[{r[1]}]={query[r[1]]}")
            elif r[0] == "NOT":
                think_lines.append(f"o[{i}]=NOT({query[r[1]]})={answer_bits[i]}")
            else:
                _, gate, a, b = r
                think_lines.append(f"o[{i}]={gate}({query[a]},{query[b]})={answer_bits[i]}")
        think_lines.append(f"Answer: {answer}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


@register
class BitStrideDetect(MicroSkill):
    """Given per-bit COPY sources, detect if they form a shift/rotate pattern."""
    name = "bit_stride_detect"
    puzzle_type = "bit_manipulation"
    description = "Detect shift/rotate from per-bit COPY source positions"
    weight = 15.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        # 50% actual shift, 50% random permutation
        is_shift = rng.random() < 0.5

        if is_shift:
            shift_amount = rng.randint(1, 7)
            sources = [(i + shift_amount) % 8 for i in range(8)]
            label = f"rol({shift_amount})"
        else:
            sources = list(range(8))
            rng.shuffle(sources)
            # Make sure it's NOT a shift
            is_actually_shift = False
            for s in range(1, 8):
                if all(sources[i] == (i + s) % 8 for i in range(8)):
                    is_actually_shift = True
                    break
            if is_actually_shift:
                return None
            label = "no stride"

        rule_str = ",".join(f"COPY({s})" for s in sources)
        source_str = ",".join(str(s) for s in sources)

        prompt = f"Per-bit rules: {rule_str}\nSources: {source_str}\n\nDo these form a shift/rotate pattern? Check if sources[i] = (i + k) % 8 for some k."

        think_lines = [f"Sources: {source_str}"]
        if is_shift:
            think_lines.append(f"Stride: ({sources[1]} - {sources[0]}) mod 8 = {(sources[1]-sources[0])%8}")
            think_lines.append(f"Check: all match → {label}")
        else:
            stride = (sources[1] - sources[0]) % 8
            think_lines.append(f"Stride guess: ({sources[1]} - {sources[0]}) mod 8 = {stride}")
            # Find first mismatch
            for i in range(2, 8):
                expected = (sources[0] + i * stride) % 8
                if sources[i] != expected:
                    think_lines.append(f"Position {i}: expected {expected}, got {sources[i]} → no stride")
                    break
            think_lines.append(f"Result: {label}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": label}


@register
class BitPerbitMegaDrill(MicroSkill):
    """15-20 full per-bit scans: given examples + one output bit column, find the rule."""
    name = "bit_perbit_mega_drill"
    puzzle_type = "bit_manipulation"
    description = "15-20 per-bit scans back to back — the core new-format drill"
    weight = 25.0
    max_pool = 20000

    def generate_one(self, rng, difficulty="medium"):
        n_items = rng.randint(15, 20)
        n_ex = rng.randint(4, 6)

        prompt = f"For each output bit column ({n_ex} examples), find the rule.\n"
        prompt += "Scan order: CONST → COPY → NOT → gate.\n\n"

        think_lines = []
        for item in range(1, n_items + 1):
            inputs = [format(rng.randint(0, 255), "08b") for _ in range(n_ex)]
            input_cols = ["".join(inp[b] for inp in inputs) for b in range(8)]

            # Pick a rule type
            rtype = rng.choices(["const", "copy", "not", "gate2"], weights=[15, 42, 5, 38])[0]

            if rtype == "const":
                val = rng.choice(["0", "1"])
                target_col = val * n_ex
            elif rtype == "copy":
                src = rng.randint(0, 7)
                target_col = input_cols[src]
                if len(set(target_col)) == 1:
                    target_col = input_cols[(src + 1) % 8]
                    src = (src + 1) % 8
                    if len(set(target_col)) == 1:
                        rtype = "const"
                        val = target_col[0]
                        target_col = val * n_ex
            elif rtype == "not":
                src = rng.randint(0, 7)
                target_col = "".join("1" if c == "0" else "0" for c in input_cols[src])
                if len(set(target_col)) == 1:
                    rtype = "const"
                    val = target_col[0]
            else:
                gate = rng.choice(["AND", "OR", "XOR", "NAND", "NOR", "XNOR"])
                a, b = rng.sample(range(8), 2)
                if a > b:
                    a, b = b, a
                from generators.trace_perbit import _gate_col
                target_col = _gate_col(input_cols[a], input_cols[b], gate)

            target_str = ",".join(target_col)
            input_strs = " ".join(f"i[{j}]={','.join(input_cols[j])}" for j in range(8))
            prompt += f"{item}. target={target_str}\n   {input_strs}\n"

            # Find the rule
            from generators.trace_perbit import _scan_bit
            rule, scan_lines = _scan_bit(target_col, input_cols, show_rejects=1)
            from generators.trace_perbit import _rule_label
            think_lines.append(f"{item}. {target_str} → {_rule_label(rule)}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": f"{n_items} scanned"}


# ============================================================
# (end new per-bit skills)
# ============================================================


@register
class BitMegaCounterfactualDrill(MicroSkill):
    """20-25 counterfactuals: given x→y via rule Z, what if the rule were W instead? Compute diff."""
    name = "bit_mega_counterfactual_drill"
    puzzle_type = "bit_manipulation"
    description = "25 counterfactuals: if rule were different, what would the diff be?"
    weight = 25.0
    max_pool = 20000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_compact import apply_shift, apply_gate, fmt, _xor_bits
        shifts = ["shr1","shr2","shl1","shl2","rol1","rol2","ror1","ror2"]
        gates = ["xor","and","or","xnor","nand","nor"]

        n = rng.randint(20, 25)
        prompt = "For each: given the ACTUAL output under rule Z, what would the output be under rule W? Compute the diff.\n"
        think_lines = []
        for i in range(1, n+1):
            x = rng.randint(1, 254)
            x_str = format(x, "08b")
            s1, s2 = rng.sample(shifts, 2)
            real_gate = rng.choice(gates)
            wrong_gate = rng.choice([g for g in gates if g != real_gate])

            a = fmt(apply_shift(x, s1))
            b = fmt(apply_shift(x, s2))
            real_out = apply_gate(a, b, real_gate)
            wrong_out = apply_gate(a, b, wrong_gate)
            diff = _xor_bits(real_out, wrong_out)

            prompt += f"{i}. x={x_str} A={s1} B={s2} actual={real_gate}→{real_out} counterfactual={wrong_gate}→?\n"
            think_lines.append(f"{i}. {wrong_gate}({a},{b})={wrong_out} diff={diff} ({diff.count('1')} bits)")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": f"{n} counterfactuals computed"}



@register
class BitMegaGateComparisonDrill(MicroSkill):
    """15 inputs, each compute ALL 6 gates, show all outputs + diffs from expected. Full comparison."""
    name = "bit_mega_gate_comparison_drill"
    puzzle_type = "bit_manipulation"
    description = "15 inputs × 6 gates each = 90 computations. For each input, show all gate outputs side by side."
    weight = 25.0
    max_pool = 20000

    GATES = {
        "and":  lambda a, b: ''.join(str(int(x)&int(y)) for x,y in zip(a,b)),
        "or":   lambda a, b: ''.join(str(int(x)|int(y)) for x,y in zip(a,b)),
        "xor":  lambda a, b: ''.join(str(int(x)^int(y)) for x,y in zip(a,b)),
        "nand": lambda a, b: ''.join(str(1-(int(x)&int(y))) for x,y in zip(a,b)),
        "nor":  lambda a, b: ''.join(str(1-(int(x)|int(y))) for x,y in zip(a,b)),
        "xnor": lambda a, b: ''.join(str(1-(int(x)^int(y))) for x,y in zip(a,b)),
    }

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_compact import apply_shift, fmt, _xor_bits
        shifts = ["shr1","shr2","shl1","shl2","rol1","rol2","ror1","ror2"]
        gate_names = list(self.GATES.keys())

        n = rng.randint(12, 15)
        s1, s2 = rng.sample(shifts, 2)
        # Pick one "correct" gate — the expected output comes from this gate
        correct_gate = rng.choice(gate_names)

        prompt = f"Shifts: A={s1}(x), B={s2}(x)\nFor each input, compute ALL 6 gates and mark which matches expected:\n"
        think_lines = []

        for i in range(1, n+1):
            x = rng.randint(1, 254)
            x_str = format(x, "08b")
            a = fmt(apply_shift(x, s1))
            b = fmt(apply_shift(x, s2))
            expected = self.GATES[correct_gate](a, b)

            prompt += f"{i}. x={x_str} expected={expected}\n"
            think_lines.append(f"{i}. A={a} B={b}")
            for g in gate_names:
                out = self.GATES[g](a, b)
                diff = _xor_bits(out, expected)
                match = "← MATCH" if diff == "00000000" else f"diff={diff}"
                think_lines.append(f"   {g:5s}={out} {match}")

        return {"user": prompt, "think": "\n".join(think_lines), "answer": f"gate={correct_gate}"}



@register
class BitAuditDrill(MicroSkill):
    """One candidate, one witness. End with REJECT or PASS. No answer. Pure audit."""
    name = "bit_audit_drill"
    puzzle_type = "bit_manipulation"
    description = "20 single-candidate audits: compute witness, output REJECT or PASS"
    weight = 25.0
    max_pool = 20000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_compact import apply_shift, apply_gate, fmt, _xor_bits
        shifts = ["shr1","shr2","shl1","shl2","rol1","rol2","ror1","ror2"]
        gates = ["xor","and","or","xnor","nand","nor"]
        n = rng.randint(18, 22)
        
        prompt = "Audit each candidate on the witness. Output REJECT or PASS:\n"
        think_lines = []
        for i in range(1, n+1):
            x = rng.randint(1, 254)
            x_str = format(x, "08b")
            s1, s2 = rng.sample(shifts, 2)
            real_gate = rng.choice(gates)
            test_gate = rng.choice(gates)  # might be right or wrong
            
            a = fmt(apply_shift(x, s1))
            b = fmt(apply_shift(x, s2))
            expected = apply_gate(a, b, real_gate)
            computed = apply_gate(a, b, test_gate)
            diff = _xor_bits(computed, expected)
            verdict = "PASS" if diff == "00000000" else "REJECT"
            
            prompt += f"{i}. {test_gate}({a},{b}) vs expected={expected}\n"
            think_lines.append(f"{i}. computed={computed} diff={diff} → {verdict}")
        
        return {"user": prompt, "think": "\n".join(think_lines), "answer": f"{n} audited"}


@register  
class BitDuelDrill(MicroSkill):
    """Two candidates, one witness. Which survives? Pure discrimination."""
    name = "bit_duel_drill"
    puzzle_type = "bit_manipulation"
    description = "15 duels: two candidates on same witness, pick the survivor"
    weight = 25.0
    max_pool = 20000

    def generate_one(self, rng, difficulty="medium"):
        from generators.trace_compact import apply_shift, apply_gate, fmt, _xor_bits
        shifts = ["shr1","shr2","shl1","shl2","rol1","rol2","ror1","ror2"]
        gates = ["xor","and","or","xnor","nand","nor"]
        n = rng.randint(12, 15)
        
        prompt = "Two candidates per witness. Which survives?\n"
        think_lines = []
        for i in range(1, n+1):
            x = rng.randint(1, 254)
            s1, s2 = rng.sample(shifts, 2)
            real_gate = rng.choice(gates)
            wrong_gate = rng.choice([g for g in gates if g != real_gate])
            
            a = fmt(apply_shift(x, s1))
            b = fmt(apply_shift(x, s2))
            expected = apply_gate(a, b, real_gate)
            
            out_real = apply_gate(a, b, real_gate)
            out_wrong = apply_gate(a, b, wrong_gate)
            diff_real = _xor_bits(out_real, expected)
            diff_wrong = _xor_bits(out_wrong, expected)
            
            # Randomize order
            if rng.random() < 0.5:
                g1, d1, g2, d2 = real_gate, diff_real, wrong_gate, diff_wrong
            else:
                g1, d1, g2, d2 = wrong_gate, diff_wrong, real_gate, diff_real
            
            prompt += f"{i}. A={a} B={b} expected={expected} | C1={g1} C2={g2}\n"
            v1 = "PASS" if d1 == "00000000" else "REJECT"
            v2 = "PASS" if d2 == "00000000" else "REJECT"
            winner = "C1" if v1 == "PASS" else "C2" if v2 == "PASS" else "neither"
            think_lines.append(f"{i}. C1:{g1}→diff={d1}→{v1} C2:{g2}→diff={d2}→{v2} → winner={winner}")
        
        return {"user": prompt, "think": "\n".join(think_lines), "answer": f"{n} duels"}


# ============================================================
# RE-APPLY TERMINAL WEIGHTS (MUST BE LAST in file)
# ============================================================
_FINAL_WEIGHTS.update({
    "enc_table_lookup_single": 15.0,
    "enc_decode_with_rows": 15.0,
    "enc_decode_vocab_match": 12.0,
    "enc_verify_decode": 10.0,
    "enc_zero_match_recheck": 15.0,
    "enc_candidate_list_pick": 12.0,
    "enc_non_vocab_reject": 15.0,
    "bit_grid_audit": 0.0,            # DEPRECATED: GRID format gone
    "bit_terminal_reject": 0.0,       # DEPRECATED: reject state gone
    "bit_gate_trivia": 15.0,
    "bit_gate_survivor": 15.0,
    "trans_verify_lock": 10.0,
    "trans_trace_audit": 12.0,
    "trans_operand_assembly": 15.0,
    "trans_combo_survivor": 15.0,
    "trans_encode_length": 12.0,
    "enc_sentence_vocab_check": 12.0,
    "trans_order_test": 15.0,
    "trans_style_apply": 12.0,
    "bit_diff_check": 0.0,             # DEPRECATED: diff verification gone
    "bit_error_detect": 10.0,     # was 2.0 — audit traces are first-class now
    "bit_full_verify": 10.0,      # was 5.0 — execution verification is critical
    "bit_witness_select": 12.0,
    "trans_order_audit": 15.0,
    "trans_style_audit": 15.0,
    "trans_verify_op": 10.0,      # was 5.0 — auditing op choice matters
    "trans_regime_inherit": 12.0,
    "trans_mapping_audit": 10.0,
    # Per-bit column decomposition era — deprecate GRID/LOCK/diff skills
    "bit_fail_stop_drill": 0.0,        # DEPRECATED: FAIL→REJECT state machine gone
    "bit_gate_truth_table": 12.0,      # still useful: forces computation
    "bit_shift_source_backtrack": 10.0, # still useful: teaches shift identification
    "bit_gate_confusion_drill": 0.0,   # DEPRECATED: GRID-based comparison gone
    "bit_gate_from_outputs": 15.0,     # still useful: identify gate from bits
    # Rapid-fire drills — keep gate/shift, deprecate diff/pipeline
    "bit_gate_rapid_fire": 18.0,
    "bit_shift_rapid_fire": 15.0,
    "bit_diff_rapid_fire": 0.0,        # DEPRECATED: XOR diff verification gone
    "bit_full_pipeline_drill": 0.0,    # DEPRECATED: old pipeline format gone
    # MEGA drills — keep computation, deprecate GRID/LOCK/diff
    "bit_mega_gate_drill": 25.0,       # KEEP: gate computation is universal
    "bit_mega_diff_drill": 0.0,        # DEPRECATED: diff verification gone
    "bit_mega_pipeline_drill": 0.0,    # DEPRECATED: old pipeline gone
    "bit_mega_grid_drill": 0.0,        # DEPRECATED: GRID format gone
    "bit_mega_counterfactual_drill": 25.0,  # KEEP: counterfactual reasoning
    "bit_mega_gate_comparison_drill": 25.0, # KEEP: all-gates-on-same-input
    "bit_audit_drill": 0.0,            # DEPRECATED: audit format gone
    "bit_duel_drill": 0.0,             # DEPRECATED: duel format gone
    "bit_elimination_drill": 25.0,     # KEEP: CONST/identity/NOT/gate — directly teaches new format!
    "bit_decision_chain_drill": 0.0,   # DEPRECATED: STOP/PROCEED state machine gone
    "bit_full_decision_drill": 0.0,    # DEPRECATED: old decision format gone
    "bit_mega_shift_drill": 20.0,      # KEEP: shift computation
    # New per-bit column decomposition skills
    "bit_column_read": 20.0,           # fundamental decomposition skill
    "bit_scan_single": 25.0,           # core per-bit scan — highest priority
    "bit_apply_rules": 25.0,           # apply 8 rules to query — the final step
    "bit_stride_detect": 15.0,         # shift/rotate detection from COPY sources
    "bit_perbit_mega_drill": 25.0,     # bulk per-bit scans — max execution pressure
    # Cipher-digit skills
    "cipher_decode": 15.0,             # atomic: symbols → digits
    "cipher_encode": 15.0,             # atomic: digits → symbols
    "cipher_check_combo": 20.0,        # verify a combo against an example
    "cipher_crack_one": 20.0,          # find missing digit for 1 unknown symbol
    "cipher_full_pipeline": 25.0,      # full mini cipher puzzle end-to-end
    "trans_mega_op_drill": 25.0,
    "trans_mega_order_drill": 20.0,
    "enc_mega_vocab_drill": 20.0,
    "enc_mega_decode_drill": 25.0,
    "enc_mega_pattern_match_drill": 25.0,
    "trans_mega_style_drill": 25.0,
    # Downweigh old enc skills — mega drills replace them
    "enc_char_decrypt": 2.0,       # was 6
    "enc_letter_match": 2.0,       # was 5
    "enc_pattern_match": 2.0,      # was 5
    "enc_pattern_fill": 1.0,       # was 3
    "enc_can_fit": 1.0,            # was 3
    "enc_bijection": 1.0,          # was 2
    "enc_forced_mapping": 1.0,     # was 2
    "enc_not_forced": 1.0,         # was 2
    "enc_why_wrong": 1.0,          # was 2
    "enc_impossible": 1.0,         # was 2
    "enc_reverse_decrypt": 0.5,    # was 1
    "enc_repeated_letters": 0.5,   # was 1
    "str_compare": 0.5,            # was 1
    "str_count": 1.0,              # was 3
    # Reduce non-computation skills to make room
    "bit_bookend_verify": 1.0,     # was 4.0 — decorative, not computation
    "bit_pattern_count": 1.0,      # was 3.0 — pattern matching, not execution
    "bit_cascade_check": 1.0,      # was 2.0
    "bit_propagation": 1.0,        # was 1.5
    "bit_reverse": 1.0,            # was 1.0 (keep)
    "bit_ones_count": 1.0,         # was 3.0 — counting, not gate execution
    "trans_op_discriminate": 15.0,     # compute two ops, pick which matches
    "trans_cipher_decode_step": 12.0,  # cipher→digit→operand drill
    "bit_identity_detect": 12.0,       # identity detection from scan
    # Enc vocab enforcement — 56% of enc failures are OOV words
    "enc_vocab": 10.0,             # was 3.0 — model must KNOW the 77 words
    "enc_vocab_audit": 10.0,       # was 3.0 — "is this word in the vocab?"
    "enc_sentence_vocab_check": 15.0,  # was 12.0 — "check entire sentence for imposters"
    "enc_candidate_list_pick": 15.0,   # was 12.0 — "pick from valid vocab candidates"
    "enc_confusable": 10.0,        # was 6.0 — same-length word confusion is 40% of failures
    # Scale non-mega bit skills to 0.4x (target: 60% mega share)
    "bit_shift": 0.5,
    "bit_gate": 0.5,
    "bit_rule_check": 2.0,
    "bit_distinguish": 4.0,
    "bit_similarity": 0.5,
    "bit_compose2": 1.2,
    "bit_compose3": 2.0,
    "bit_edge_shift_vs_rotate": 0.5,
    "bit_edge_zeros": 0.5,
    "bit_edge_gate": 0.5,
    "bit_error_detect": 4.0,
    "bit_full_verify": 4.0,
    "general_string_diff": 0.5,
    "bit_which_op": 0.8,
    "bit_counterfactual": 0.8,
    "bit_properties": 0.5,
    "bit_step_by_step": 0.5,
    "bit_reverse_find": 0.5,
    "bit_two_step_id": 0.5,
    "bit_popcount": 0.5,
    "bit_nojump": 1.6,
    "bit_impossible": 0.5,
    "bit_spot_error": 0.5,
    "bit_rule_discriminate": 4.0,
    "bit_rule_discriminate_multi": 4.0,
    "bit_family_from_popcount": 2.0,
    "bit_eliminate_family": 4.0,
    "bit_correlate_positions": 1.6,
    "bit_narrow_sources": 2.0,
    "bit_visual_pattern": 0.6,
    "bit_gate_from_known_sources": 2.0,
    "bit_count_across_examples": 0.5,
    "bit_rank_rules": 1.6,
    "bit_second_source": 1.6,
    "bit_trace_audit": 4.0,
    "bit_confident_or_not": 1.6,
    "bit_constant_positions": 1.2,
    "bit_gate_from_properties": 1.2,
    "bit_source_consistency": 1.2,
    "bit_spot_invariant": 0.6,
    "bit_spot_invariant_open": 0.6,
    "bit_invariant_checklist": 0.6,
    "bit_compare_to_target": 0.6,
    "bit_least_error": 2.0,
    "bit_which_closer": 1.6,
    "bit_fix_one_bit": 2.0,
    "bit_predict_output": 1.6,
    "bit_where_ones": 0.5,
    "bit_and_across": 0.5,
    "bit_or_across": 0.5,
    "bit_constant_vs_variable": 0.5,
    "bit_is_rotation": 0.6,
    "bit_nibble_view": 0.5,
    "bit_complement_view": 0.5,
    "bit_grid_vs_expected": 4.8,
    "bit_honest_rule_test": 4.8,
    "bit_just_choose_rule": 4.8,
    "bit_first_fail": 3.2,
    "bit_anti_copy": 4.0,
    "bit_survivor_set": 3.2,
    "bit_reject_and_backtrack": 3.2,
    "bit_family3_execute": 2.0,
    "bit_family3_verify": 2.0,
    "bit_family3_discriminate": 2.0,
    "bit_what_produced": 2.0,
    "bit_family_permutation": 2.0,
    "bit_how_many_sources": 2.4,
    "bit_compose_identify": 2.4,
    "bit_scan_compute": 1.6,
    "bit_bookend_verify": 0.5,
    "bit_compose_two_step": 2.0,
    "bit_verify_given": 0.0,          # DEPRECATED: old verify format
    "bit_verify_given_3": 0.0,        # DEPRECATED: old 3-input verify
    "bit_ones_batch": 1.2,
    "bit_hamming_batch": 1.2,
    "bit_witness_pick": 2.4,
    "bit_grid_audit": 0.0,
    "bit_terminal_reject": 0.0,
    "bit_gate_trivia": 6.0,
    "bit_gate_survivor": 6.0,
    "bit_diff_check": 0.0,
    "bit_witness_select": 4.8,
    "bit_gate_truth_table": 4.8,
    "bit_shift_source_backtrack": 4.0,
    "bit_gate_from_outputs": 6.0,
    "bit_identity_detect": 4.8,
})

from generators.microskill_framework import REGISTRY as _FINAL_REG_2
for _name, _cls in _FINAL_REG_2.items():
    if _name in _FINAL_WEIGHTS:
        _cls.weight = _FINAL_WEIGHTS[_name]
