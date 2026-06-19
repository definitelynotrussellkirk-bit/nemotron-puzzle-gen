#!/usr/bin/env python3
"""Bit fluency micro-skills (2026-05-23 build).

Adds 13 new skills targeting in-distribution gaps in the existing bank:
  - CONST column detection (14.8% of bits, under-weighted)
  - NOT signature (5.1% of bits, under-weighted)
  - Negated-input gates OR_NOT*/AND_NOT* (26.5% of GATE2 bits, no dedicated skill)
  - Top-4 gate dominance drill (XOR/AND/XNOR/OR = 67% of GATE2 bits)
  - Whole-byte pair search (45% of solver routes)
  - Source-then-gate two-step ordering
  - Pause-then-answer (Phase B trace anchor with 32 filler tokens)
  - Compositional chains (3 skills per example)
  - Ported bit_search atoms: exec_byte, exec_bit, match_byte, support_count

Also applies weight rebalance to existing skills (CONST/NOT detection,
re-enable verify/diff atoms, boost whole-byte source skills).

See docs/BIT_FLUENCY_CURRICULUM_PLAN_20260523.md for full rationale.
"""
from generators.microskill_framework import MicroSkill, register, BYTE, SHIFT_OPS
import random


# ---------------------------------------------------------------------------
# Shared rule library — whole-byte ops + gate2 + negated gates
# ---------------------------------------------------------------------------

UNARY_SOURCES = (
    [("x", lambda x: x)]
    + [(f"shl{k}", (lambda x, k=k: (x << k) & BYTE)) for k in range(1, 8)]
    + [(f"shr{k}", (lambda x, k=k: (x >> k) & BYTE)) for k in range(1, 8)]
    + [(f"rol{k}", (lambda x, k=k: ((x << k) | (x >> (8 - k))) & BYTE)) for k in range(1, 8)]
    + [(f"ror{k}", (lambda x, k=k: ((x >> k) | (x << (8 - k))) & BYTE)) for k in range(1, 8)]
    + [("~x", lambda x: (~x) & BYTE)]
)
UNARY_BY_NAME = dict(UNARY_SOURCES)

GATE2 = {
    "AND":      lambda a, b: a & b,
    "OR":       lambda a, b: a | b,
    "XOR":      lambda a, b: a ^ b,
    "NAND":     lambda a, b: (~(a & b)) & BYTE,
    "NOR":      lambda a, b: (~(a | b)) & BYTE,
    "XNOR":     lambda a, b: (~(a ^ b)) & BYTE,
    "OR_NOTA":  lambda a, b: ((~a) & BYTE) | b,
    "OR_NOTB":  lambda a, b: a | ((~b) & BYTE),
    "AND_NOTA": lambda a, b: ((~a) & BYTE) & b,
    "AND_NOTB": lambda a, b: a & ((~b) & BYTE),
}
TOP4 = ["AND", "OR", "XOR", "XNOR"]
NEGATED = ["OR_NOTA", "OR_NOTB", "AND_NOTA", "AND_NOTB"]


def b8(x: int) -> str:
    return f"{x & BYTE:08b}"


def bit_at(byte_str: str, k: int) -> str:
    """Bit k counted left-to-right (b0 leftmost, b7 rightmost)."""
    return byte_str[k]


# ===========================================================================
# 1. bit_const_detect_drill
# ===========================================================================

@register
class BitConstDetectDrill(MicroSkill):
    name = "bit_const_detect_drill"
    puzzle_type = "bit_manipulation"
    description = "Scan output bit columns across examples, identify CONST columns and their value"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_const_detect_drill.jsonl"
    weight = 6.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        n_ex = rng.choice([5, 6, 7])
        const_mask = [rng.choice([None, "0", "1", "0", "1"]) for _ in range(8)]
        if all(c is None for c in const_mask):
            const_mask[rng.randrange(8)] = rng.choice(["0", "1"])

        examples = []
        for _ in range(n_ex):
            x = rng.randrange(256)
            xb = b8(x)
            yb_list = []
            for k in range(8):
                if const_mask[k] is not None:
                    yb_list.append(const_mask[k])
                else:
                    # Filler: copy a random input position so the bit varies
                    src = rng.randrange(8)
                    yb_list.append(xb[src])
            examples.append((xb, "".join(yb_list)))

        # Build per-column readout
        cols = []
        for k in range(8):
            col_vals = [y[k] for _, y in examples]
            if all(v == col_vals[0] for v in col_vals):
                cols.append((k, col_vals[0], "CONST"))
            else:
                cols.append((k, "".join(col_vals), "VARIES"))

        ex_block = "\n".join(f"  {x} -> {y}" for x, y in examples)
        prompt = (
            f"Look at each output bit column (b0..b7) across these {n_ex} examples.\n"
            f"For each column, decide if it is CONST (same value every example) or VARIES.\n\n"
            f"{ex_block}"
        )
        think_lines = ["Column readout (b0 leftmost .. b7 rightmost):"]
        for k, val, tag in cols:
            think_lines.append(f"  b{k}: {val} -> {tag}")
        const_bits = [(k, v) for k, v, t in cols if t == "CONST"]
        if const_bits:
            const_str = " ".join(f"b{k}={v}" for k, v in const_bits)
        else:
            const_str = "none"
        answer = f"CONST bits: {const_str}"
        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


# ===========================================================================
# 2. bit_not_signature
# ===========================================================================

@register
class BitNotSignature(MicroSkill):
    name = "bit_not_signature"
    puzzle_type = "bit_manipulation"
    description = "Decide if output bit i = input bit j or = NOT(input bit j) from examples"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_not_signature.jsonl"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        n_ex = rng.choice([4, 5, 6])
        j = rng.randrange(8)
        i = rng.randrange(8)
        is_not = rng.random() < 0.5

        examples = []
        for _ in range(n_ex):
            xb = b8(rng.randrange(256))
            src = xb[j]
            yb_list = list("00000000")
            yb_list[i] = ("1" if src == "0" else "0") if is_not else src
            # Fill other bits with random pattern (consistent doesn't matter here)
            for k in range(8):
                if k != i:
                    yb_list[k] = rng.choice("01")
            examples.append((xb, "".join(yb_list)))

        ex_block = "\n".join(f"  {x} -> {y}    (in[b{j}]={x[j]}, out[b{i}]={y[i]})" for x, y in examples)
        prompt = (
            f"For each example, compare input bit b{j} to output bit b{i}.\n"
            f"Decide: is out[b{i}] = in[b{j}], or out[b{i}] = NOT(in[b{j}])?\n\n"
            f"{ex_block}"
        )
        think_lines = []
        for xb, yb in examples:
            iv, ov = xb[j], yb[i]
            relation = "in=out" if iv == ov else "in!=out"
            think_lines.append(f"  in[b{j}]={iv} out[b{i}]={ov} -> {relation}")
        verdict = "NOT" if is_not else "COPY"
        think_lines.append(f"All examples agree -> out[b{i}] = {verdict}(in[b{j}])")
        answer = f"out[b{i}] = {verdict}(in[b{j}])"
        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


# ===========================================================================
# 3. bit_negated_gate_drill
# ===========================================================================

@register
class BitNegatedGateDrill(MicroSkill):
    name = "bit_negated_gate_drill"
    puzzle_type = "bit_manipulation"
    description = "Compute and identify OR_NOTA/OR_NOTB/AND_NOTA/AND_NOTB — 26.5% of GATE2 bits"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_negated_gate_drill.jsonl"
    weight = 10.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        gate = rng.choice(NEGATED)
        a = rng.randrange(1, 255)
        b = rng.randrange(1, 255)
        out = GATE2[gate](a, b)
        ab, bb, ob = b8(a), b8(b), b8(out)

        prompt = (
            f"Compute {gate}(A, B) where:\n"
            f"  A = {ab}\n  B = {bb}\n"
            f"Definitions:\n"
            f"  OR_NOTA(a,b)  = (NOT a) OR b\n"
            f"  OR_NOTB(a,b)  = a OR (NOT b)\n"
            f"  AND_NOTA(a,b) = (NOT a) AND b\n"
            f"  AND_NOTB(a,b) = a AND (NOT b)"
        )

        not_a = b8((~a) & BYTE)
        not_b = b8((~b) & BYTE)
        think_lines = [f"NOT A = {not_a}", f"NOT B = {not_b}"]
        if gate == "OR_NOTA":
            think_lines.append(f"OR_NOTA = (NOT A) | B")
            think_lines.append(f"        = {not_a} | {bb}")
        elif gate == "OR_NOTB":
            think_lines.append(f"OR_NOTB = A | (NOT B)")
            think_lines.append(f"        = {ab} | {not_b}")
        elif gate == "AND_NOTA":
            think_lines.append(f"AND_NOTA = (NOT A) & B")
            think_lines.append(f"         = {not_a} & {bb}")
        else:  # AND_NOTB
            think_lines.append(f"AND_NOTB = A & (NOT B)")
            think_lines.append(f"         = {ab} & {not_b}")
        think_lines.append(f"Position-by-position:")
        for k in range(8):
            ai, bi = int(ab[k]), int(bb[k])
            if gate == "OR_NOTA":
                ok = ((1 - ai) | bi) & 1
            elif gate == "OR_NOTB":
                ok = (ai | (1 - bi)) & 1
            elif gate == "AND_NOTA":
                ok = ((1 - ai) & bi) & 1
            else:
                ok = (ai & (1 - bi)) & 1
            think_lines.append(f"  b{k}: A={ai} B={bi} -> {ok}")
        think_lines.append(f"Result: {ob}")
        return {"user": prompt, "think": "\n".join(think_lines), "answer": ob}


# ===========================================================================
# 4. bit_4gate_dominance_drill
# ===========================================================================

@register
class Bit4GateDominanceDrill(MicroSkill):
    name = "bit_4gate_dominance_drill"
    puzzle_type = "bit_manipulation"
    description = "Identify which of AND/OR/XOR/XNOR produced an output — top-4 gates = 67% of GATE2"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_4gate_dominance_drill.jsonl"
    weight = 8.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        gate = rng.choice(TOP4)
        a = rng.randrange(256)
        b = rng.randrange(256)
        out = GATE2[gate](a, b)
        ab, bb, ob = b8(a), b8(b), b8(out)
        prompt = (
            f"One of AND/OR/XOR/XNOR was applied. Identify which.\n"
            f"  A = {ab}\n  B = {bb}\n  out = {ob}"
        )
        think_lines = ["Test each candidate position-by-position:"]
        for cand in TOP4:
            cand_out = b8(GATE2[cand](a, b))
            match = "match" if cand_out == ob else f"no (first diff at b{[i for i in range(8) if cand_out[i] != ob[i]][0]})"
            think_lines.append(f"  {cand:4s}: {cand_out}  {match}")
        return {"user": prompt, "think": "\n".join(think_lines), "answer": gate}


# ===========================================================================
# 5. bit_whole_byte_pair_search
# ===========================================================================

@register
class BitWholeBytePairSearch(MicroSkill):
    name = "bit_whole_byte_pair_search"
    puzzle_type = "bit_manipulation"
    description = "Joint search: find (source_A, source_B, gate) from 16-source candidate list"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_whole_byte_pair_search.jsonl"
    weight = 8.0
    max_pool = 10000

    # Narrowed candidate sources for tractable search
    SOURCES = ["x", "shl1", "shl2", "shl3", "shr1", "shr2", "shr3",
               "rol1", "rol2", "rol3", "ror1", "ror2", "ror3", "~x"]

    def generate_one(self, rng, difficulty="medium"):
        gate = rng.choice(TOP4)
        sa, sb = rng.sample(self.SOURCES, 2)
        n_ex = rng.choice([3, 4])
        examples = []
        for _ in range(n_ex):
            x = rng.randrange(256)
            va = UNARY_BY_NAME[sa](x)
            vb = UNARY_BY_NAME[sb](x)
            y = GATE2[gate](va, vb)
            examples.append((b8(x), b8(y)))

        ex_block = "\n".join(f"  {x} -> {y}" for x, y in examples)
        prompt = (
            f"Find the rule: y = gate(srcA(x), srcB(x))\n"
            f"Sources candidate set: {', '.join(self.SOURCES)}\n"
            f"Gate candidate set: {', '.join(TOP4)}\n\n"
            f"Examples:\n{ex_block}"
        )

        # Walk a short search: try a few wrong sources first, then the right one
        think_lines = []
        wrong_sa = rng.choice([s for s in self.SOURCES if s != sa])
        wrong_sb = rng.choice([s for s in self.SOURCES if s != sb])
        wrong_gate = rng.choice([g for g in TOP4 if g != gate])

        for cand in [(wrong_sa, sb, gate), (sa, wrong_sb, gate), (sa, sb, wrong_gate)]:
            csa, csb, cg = cand
            x0, y0 = examples[0]
            va = UNARY_BY_NAME[csa](int(x0, 2))
            vb = UNARY_BY_NAME[csb](int(x0, 2))
            pred = b8(GATE2[cg](va, vb))
            think_lines.append(f"  try ({csa}, {csb}, {cg}): on x={x0} -> {pred} vs gold {y0} -> no")

        x0, y0 = examples[0]
        va = UNARY_BY_NAME[sa](int(x0, 2))
        vb = UNARY_BY_NAME[sb](int(x0, 2))
        pred = b8(GATE2[gate](va, vb))
        think_lines.append(f"  try ({sa}, {sb}, {gate}): on x={x0} -> {pred} = {y0} ✓")
        for x, y in examples[1:]:
            va = UNARY_BY_NAME[sa](int(x, 2))
            vb = UNARY_BY_NAME[sb](int(x, 2))
            pred = b8(GATE2[gate](va, vb))
            think_lines.append(f"     verify on x={x} -> {pred} = {y} ✓")
        answer = f"y = {gate}({sa}(x), {sb}(x))"
        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


# ===========================================================================
# 6. bit_source_then_gate
# ===========================================================================

@register
class BitSourceThenGate(MicroSkill):
    name = "bit_source_then_gate"
    puzzle_type = "bit_manipulation"
    description = "Two-step: lock sources by COPY analysis first, then identify gate"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_source_then_gate.jsonl"
    weight = 6.0
    max_pool = 10000

    SOURCES = ["x", "shl1", "shl2", "shr1", "shr2", "rol1", "rol2", "ror1", "ror2", "~x"]

    def generate_one(self, rng, difficulty="medium"):
        gate = rng.choice(TOP4)
        sa, sb = rng.sample(self.SOURCES, 2)
        x = rng.randrange(256)
        va = UNARY_BY_NAME[sa](x)
        vb = UNARY_BY_NAME[sb](x)
        y = GATE2[gate](va, vb)
        prompt = (
            f"You have isolated this single example:\n"
            f"  x = {b8(x)}\n  y = {b8(y)}\n"
            f"It was produced by y = gate(srcA(x), srcB(x)).\n"
            f"You already know srcA = {sa} and srcB = {sb}.\n"
            f"Identify the gate."
        )
        think_lines = [
            f"Step 1 — apply sources:",
            f"  {sa}(x) = {b8(va)}",
            f"  {sb}(x) = {b8(vb)}",
            f"Step 2 — test each gate on those source values:",
        ]
        for cand in TOP4:
            pred = b8(GATE2[cand](va, vb))
            match = "match" if pred == b8(y) else "no"
            think_lines.append(f"  {cand:4s}: {pred}  {match}")
        return {"user": prompt, "think": "\n".join(think_lines), "answer": gate}


# ===========================================================================
# 7. bit_pause_then_answer
# ===========================================================================

@register
class BitPauseThenAnswer(MicroSkill):
    name = "bit_pause_then_answer"
    puzzle_type = "bit_manipulation"
    description = "Phase-B anchor: trivial bit puzzle + 32 filler pause tokens before boxed answer"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_pause_then_answer.jsonl"
    weight = 4.0
    max_pool = 10000

    PAUSE_TOKEN = "@"
    PAUSE_COUNT = 32

    def generate_one(self, rng, difficulty="medium"):
        # Trivial: single-op shift/rotate or single-gate so the model can rely on weight knowledge
        kind = rng.choice(["shift", "gate"])
        if kind == "shift":
            op = rng.choice(["shl1", "shr1", "rol1", "ror1", "shl2", "shr2"])
            x = rng.randrange(256)
            y = SHIFT_OPS[op](x)
            prompt = f"Apply {op} to x={b8(x)}."
        else:
            gate = rng.choice(TOP4)
            a, b_ = rng.randrange(256), rng.randrange(256)
            y = GATE2[gate](a, b_)
            prompt = f"Compute {gate}(A,B) where A={b8(a)} B={b8(b_)}."
        pause = " ".join([self.PAUSE_TOKEN] * self.PAUSE_COUNT)
        return {"user": prompt, "think": pause, "answer": b8(y)}


# ===========================================================================
# 8. bit_compositional_chain
# ===========================================================================

@register
class BitCompositionalChain(MicroSkill):
    name = "bit_compositional_chain"
    puzzle_type = "bit_manipulation"
    description = "Single example chains 3 skills: shift two sources, apply gate, verify diff against expected"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_compositional_chain.jsonl"
    weight = 6.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        gate = rng.choice(TOP4 + NEGATED)
        sa = rng.choice(["x", "shl1", "shl2", "shr1", "shr2", "rol1", "rol2", "ror1", "ror2"])
        sb = rng.choice(["x", "shl1", "shl2", "shr1", "shr2", "rol1", "rol2", "ror1", "ror2"])
        x = rng.randrange(256)
        va = UNARY_BY_NAME[sa](x)
        vb = UNARY_BY_NAME[sb](x)
        y = GATE2[gate](va, vb)
        # Pretend an "expected" answer differs by 0-2 bits to test the verify step
        flips = rng.choice([0, 0, 0, 1, 2])  # mostly clean, sometimes off
        expected = y
        for _ in range(flips):
            bit = rng.randrange(8)
            expected ^= (1 << (7 - bit))
        verdict = "MATCH" if expected == y else f"MISMATCH ({bin(expected ^ y).count('1')} bits)"

        prompt = (
            f"Three-step pipeline:\n"
            f"  Step 1: compute {sa}({b8(x)}) and {sb}({b8(x)})\n"
            f"  Step 2: apply {gate}\n"
            f"  Step 3: compare result against expected = {b8(expected)}"
        )
        think_lines = [
            f"Step 1:",
            f"  {sa}(x) = {b8(va)}",
            f"  {sb}(x) = {b8(vb)}",
            f"Step 2:",
            f"  {gate}({b8(va)}, {b8(vb)}) = {b8(y)}",
            f"Step 3:",
            f"  computed = {b8(y)}",
            f"  expected = {b8(expected)}",
            f"  -> {verdict}",
        ]
        return {"user": prompt, "think": "\n".join(think_lines), "answer": verdict}


# ===========================================================================
# 9. bit_micro_exec_byte — port from bit_search
# ===========================================================================

@register
class BitMicroExecByte(MicroSkill):
    name = "bit_micro_exec_byte"
    puzzle_type = "bit_manipulation"
    description = "Apply a named rule to one byte. Short, mechanical execution atom."
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_micro_exec_byte.jsonl"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        kind = rng.choice(["unary", "gate2"])
        x = rng.randrange(256)
        if kind == "unary":
            name = rng.choice(["shl1", "shl2", "shl3", "shr1", "shr2", "shr3",
                               "rol1", "rol2", "rol3", "ror1", "ror2", "ror3", "~x"])
            y = UNARY_BY_NAME[name](x)
            prompt = f"Apply rule P={name} to X={b8(x)}.\nReturn Y."
            think = f"{name}({b8(x)}) = {b8(y)}"
        else:
            gate = rng.choice(list(GATE2.keys()))
            a, b_ = rng.randrange(256), rng.randrange(256)
            y = GATE2[gate](a, b_)
            prompt = f"Apply rule P={gate}(A,B) where A={b8(a)} B={b8(b_)}.\nReturn Y."
            think = f"{gate}({b8(a)}, {b8(b_)}) = {b8(y)}"
        return {"user": prompt, "think": think, "answer": b8(y)}


# ===========================================================================
# 10. bit_micro_exec_bit — port from bit_search
# ===========================================================================

@register
class BitMicroExecBit(MicroSkill):
    name = "bit_micro_exec_bit"
    puzzle_type = "bit_manipulation"
    description = "Apply a named rule and return one specific output bit"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_micro_exec_bit.jsonl"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        kind = rng.choice(["unary", "gate2"])
        x = rng.randrange(256)
        bit = rng.randrange(8)
        if kind == "unary":
            name = rng.choice(["shl1", "shr1", "rol1", "ror1", "shl2", "shr2", "rol2", "ror2", "~x"])
            y = b8(UNARY_BY_NAME[name](x))
            prompt = f"Apply P={name} to X={b8(x)}, return bit b{bit}."
            think = f"{name}({b8(x)}) = {y}\nbit b{bit} = {y[bit]}"
        else:
            gate = rng.choice(list(GATE2.keys()))
            a, b_ = rng.randrange(256), rng.randrange(256)
            y = b8(GATE2[gate](a, b_))
            prompt = f"Apply P={gate}(A,B) where A={b8(a)} B={b8(b_)}, return bit b{bit}."
            think = f"{gate}({b8(a)}, {b8(b_)}) = {y}\nbit b{bit} = {y[bit]}"
        return {"user": prompt, "think": think, "answer": y[bit]}


# ===========================================================================
# 11. bit_micro_match_byte — port from bit_search
# ===========================================================================

@register
class BitMicroMatchByte(MicroSkill):
    name = "bit_micro_match_byte"
    puzzle_type = "bit_manipulation"
    description = "Check whether a candidate byte equals P(X) — pure verification atom"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_micro_match_byte.jsonl"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        kind = rng.choice(["unary", "gate2"])
        x = rng.randrange(256)
        positive = rng.random() < 0.5
        if kind == "unary":
            name = rng.choice(["shl1", "shr1", "rol1", "ror1", "shl2", "shr2", "~x"])
            gold = UNARY_BY_NAME[name](x)
        else:
            name = rng.choice(list(GATE2.keys()))
            b_ = rng.randrange(256)
            gold = GATE2[name](x, b_)
            kind = "gate2"
        candidate = gold if positive else (gold ^ (1 << rng.randrange(8))) & BYTE
        verdict = "yes" if candidate == gold else "no"

        if kind == "unary":
            prompt = f"P={name}\nX={b8(x)}\nY={b8(candidate)}\nDoes Y equal P(X)?"
            think = f"{name}({b8(x)}) = {b8(gold)}\nCandidate Y = {b8(candidate)}\n{b8(gold)} vs {b8(candidate)} -> {verdict}"
        else:
            prompt = f"P={name}(A,B)\nA={b8(x)} B={b8(b_)}\nY={b8(candidate)}\nDoes Y equal P(A,B)?"
            think = f"{name}({b8(x)}, {b8(b_)}) = {b8(gold)}\nCandidate Y = {b8(candidate)}\n-> {verdict}"
        return {"user": prompt, "think": think, "answer": verdict}


# ===========================================================================
# 12. bit_micro_support_count — port from bit_search
# ===========================================================================

@register
class BitMicroSupportCount(MicroSkill):
    name = "bit_micro_support_count"
    puzzle_type = "bit_manipulation"
    description = "Count how many support examples match a candidate rule"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_micro_support_count.jsonl"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        # Pick a unary-only rule for cheap exec
        name = rng.choice(["x", "shl1", "shr1", "rol1", "ror1", "shl2", "shr2", "~x"])
        n = rng.choice([3, 4, 5])
        positive = rng.random() < 0.5
        bad_count = 0 if positive else (1 if rng.random() < 0.8 else 2)
        bad_idx = set(rng.sample(range(n), bad_count)) if bad_count else set()
        pairs = []
        for i in range(n):
            x = rng.randrange(256)
            y = UNARY_BY_NAME[name](x)
            if i in bad_idx:
                y ^= (1 << rng.randrange(8))
                y &= BYTE
            pairs.append((b8(x), b8(y)))

        matches = sum(1 for (x, y) in pairs if UNARY_BY_NAME[name](int(x, 2)) == int(y, 2))
        pass_flag = "PASS" if matches == n else "FAIL"
        ex_block = "\n".join(f"  {x} -> {y}" for x, y in pairs)
        prompt = (
            f"P={name}\nCount how many of these examples match.\n\n{ex_block}"
        )
        think_lines = [f"Test P={name} on each:"]
        for x, y in pairs:
            pred = b8(UNARY_BY_NAME[name](int(x, 2)))
            ok = "✓" if pred == y else "✗"
            think_lines.append(f"  P({x}) = {pred} vs {y} {ok}")
        think_lines.append(f"Matches: {matches}/{n} -> {pass_flag}")
        answer = f"{matches}/{n} {pass_flag}"
        return {"user": prompt, "think": "\n".join(think_lines), "answer": answer}


# ===========================================================================
# 13. bit_micro_compare_rules — port from bit_search
# ===========================================================================

@register
class BitMicroCompareRules(MicroSkill):
    name = "bit_micro_compare_rules"
    puzzle_type = "bit_manipulation"
    description = "Apply two rules to the same input, report whether outputs are identical"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_micro_compare_rules.jsonl"
    weight = 4.0
    max_pool = 10000

    POOL = ["x", "shl1", "shl2", "shr1", "shr2", "rol1", "rol2", "ror1", "ror2", "~x"]

    def generate_one(self, rng, difficulty="medium"):
        a_name, b_name = rng.sample(self.POOL, 2)
        x = rng.randrange(256)
        ya = UNARY_BY_NAME[a_name](x)
        yb = UNARY_BY_NAME[b_name](x)
        same = "yes" if ya == yb else "no"
        prompt = f"A={a_name}\nB={b_name}\nX={b8(x)}\nDo A(X) and B(X) produce the same byte?"
        think = f"A({b8(x)}) = {a_name}({b8(x)}) = {b8(ya)}\nB({b8(x)}) = {b_name}({b8(x)}) = {b8(yb)}\nSAME = {same}"
        return {"user": prompt, "think": think, "answer": same}


# ===========================================================================
# Notes-driven skills (2026-05-23 batch 2)
#
# Added after reviewing experiments/bit_only/notes.txt. Key facts from notes:
#  - Whole-byte 2-input locks use ONLY AND/OR/XOR (no XNOR/NAND/NOR)
#  - 3-input locks: 4 families carry 98.4% — OR_XNOR/CH/MAJ3/GATED_XNOR_NAND
#  - Canonical 3-input shape: F(rotation, shl, shr)
#  - 12 gates never fire on training set; skip them
#  - F1-F13 elimination predicates prune candidates cheaply (popcount deltas,
#    subset relations, diff constancy, middle-bit constancy)
# ===========================================================================


# Whole-byte 3-input families (per notes: 4 of 7 firing families = 98.4%)
GATE3 = {
    "OR_XNOR":          lambda a, b, c: (c | (~(a ^ b))) & BYTE,
    "CH":               lambda a, b, c: ((a & b) | ((~a) & c)) & BYTE,
    "MAJ3":             lambda a, b, c: ((a & b) | (a & c) | (b & c)) & BYTE,
    "GATED_XNOR_NAND":  lambda a, b, c: ((c | (~(a ^ b))) & (~(a & b & c))) & BYTE,
}
TOP4_3INPUT = list(GATE3.keys())

# Whole-byte 2-input gates that ACTUALLY APPEAR per notes
WHOLE_BYTE_GATES = ["AND", "OR", "XOR"]


def popcount(x: int) -> int:
    return bin(x & BYTE).count("1")


# ===========================================================================
# 14. bit_popcount_delta_sign — F1-F4 atom
# ===========================================================================

@register
class BitPopcountDeltaSign(MicroSkill):
    name = "bit_popcount_delta_sign"
    puzzle_type = "bit_manipulation"
    description = "Compute pc(out)-pc(in) per example; classify sign pattern (eliminates gate classes)"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_popcount_delta_sign.jsonl"
    weight = 6.0
    max_pool = 10000

    PATTERNS = ["all_zero", "all_positive", "all_negative", "mixed"]

    def _make_pair(self, rng, target_sign):
        """Generate (in, out) where pc(out) - pc(in) matches target."""
        for _ in range(50):
            x = rng.randrange(256)
            pc_in = popcount(x)
            if target_sign == "zero":
                target_pc = pc_in
            elif target_sign == "positive":
                if pc_in >= 8:
                    continue
                target_pc = rng.randint(pc_in + 1, 8)
            else:  # negative
                if pc_in <= 0:
                    continue
                target_pc = rng.randint(0, pc_in - 1)
            # Build a byte with target_pc ones
            positions = rng.sample(range(8), target_pc) if target_pc > 0 else []
            y = 0
            for p in positions:
                y |= (1 << p)
            return (x, y)
        return None

    def generate_one(self, rng, difficulty="medium"):
        pattern = rng.choice(self.PATTERNS)
        n_ex = rng.choice([4, 5, 6])
        pairs = []
        if pattern == "all_zero":
            signs = ["zero"] * n_ex
        elif pattern == "all_positive":
            signs = ["positive"] * n_ex
        elif pattern == "all_negative":
            signs = ["negative"] * n_ex
        else:  # mixed
            signs = ["positive", "negative"] + [rng.choice(["positive", "negative", "zero"]) for _ in range(n_ex - 2)]
            rng.shuffle(signs)

        for s in signs:
            p = self._make_pair(rng, s)
            if p is None:
                return None
            pairs.append(p)

        ex_block = "\n".join(f"  {b8(x)} -> {b8(y)}" for x, y in pairs)
        prompt = (
            f"For each example, compute delta = popcount(out) - popcount(in).\n"
            f"Classify the sign pattern: all_zero / all_positive / all_negative / mixed.\n\n"
            f"{ex_block}"
        )
        think_lines = []
        deltas = []
        for x, y in pairs:
            pc_in, pc_out = popcount(x), popcount(y)
            d = pc_out - pc_in
            deltas.append(d)
            think_lines.append(f"  {b8(x)} pc={pc_in}, {b8(y)} pc={pc_out}, delta={d:+d}")
        # Verdict
        if all(d == 0 for d in deltas):
            verdict = "all_zero"
        elif all(d > 0 for d in deltas):
            verdict = "all_positive"
        elif all(d < 0 for d in deltas):
            verdict = "all_negative"
        else:
            verdict = "mixed"
        think_lines.append(f"deltas: {deltas} -> {verdict}")

        # Elimination commentary
        if verdict == "all_zero":
            note = "Candidates: identity, rotations, ~x (if pc(in)=4), XOR over balanced sources"
        elif verdict == "all_positive":
            note = "Eliminates: AND, pure shifts"
        elif verdict == "all_negative":
            note = "Eliminates: OR, ~x (when pc(in)<4 anywhere)"
        else:
            note = "Mixed eliminates: AND-only, OR-only, all shifts, all rotations, all NOTs"
        think_lines.append(note)
        return {"user": prompt, "think": "\n".join(think_lines), "answer": verdict}


# ===========================================================================
# 15. bit_unary_no_fit_cert — F13 combined certificate
# ===========================================================================

@register
class BitUnaryNoFitCert(MicroSkill):
    name = "bit_unary_no_fit_cert"
    puzzle_type = "bit_manipulation"
    description = "F3+F7 certificate: delta<=0 always AND middle bit constant-zero -> no unary fits (58 forms eliminated)"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_unary_no_fit_cert.jsonl"
    weight = 6.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        # Build examples where:
        #   1. delta <= 0 always (so F3 fires; rotations have delta=0 which is OK)
        #   2. some MIDDLE bit (positions 2,3,4,5) is constant ZERO across all examples
        n_ex = rng.choice([4, 5, 6])
        forced_zero = rng.choice([2, 3, 4, 5])

        pairs = []
        for _ in range(n_ex):
            x = rng.randrange(1, 256)  # avoid all-zero so pc(x) > 0
            pc_in = popcount(x)
            # Pick output popcount <= pc_in (so delta <= 0)
            target_pc = rng.randint(0, pc_in)
            # Sample positions but force position `forced_zero` to be 0
            available = [p for p in range(8) if p != (7 - forced_zero)]
            # forced_zero is b-index (left-to-right). bit position = 7 - forced_zero
            if target_pc > len(available):
                target_pc = len(available)
            positions = rng.sample(available, target_pc) if target_pc > 0 else []
            y = 0
            for p in positions:
                y |= (1 << p)
            pairs.append((x, y))

        ex_block = "\n".join(f"  {b8(x)} -> {b8(y)}" for x, y in pairs)
        prompt = (
            f"Apply the F3+F7 cheap unary certificate.\n"
            f"F3: if delta = pc(out)-pc(in) <= 0 for every example -> OR and ~x eliminated\n"
            f"F7: if some MIDDLE bit position (b2..b5) is constant-zero -> all shifts and ~shifts eliminated\n"
            f"Combined: certify 'no unary fits' or not.\n\n"
            f"{ex_block}"
        )
        think_lines = []
        deltas = []
        for x, y in pairs:
            d = popcount(y) - popcount(x)
            deltas.append(d)
            think_lines.append(f"  {b8(x)} pc={popcount(x)}, {b8(y)} pc={popcount(y)}, delta={d:+d}")
        f3 = all(d <= 0 for d in deltas)
        think_lines.append(f"F3 (delta<=0 always): {f3}")

        # Find middle constant-zero positions
        const_zero_mid = []
        for b_idx in [2, 3, 4, 5]:
            if all(b8(y)[b_idx] == "0" for _, y in pairs):
                const_zero_mid.append(b_idx)
        f7 = len(const_zero_mid) > 0
        think_lines.append(f"F7 (middle b2..b5 constant-zero): {const_zero_mid if const_zero_mid else 'none'}")

        verdict = "CERTIFIED" if (f3 and f7) else "NOT_CERTIFIED"
        think_lines.append(f"F3 AND F7 -> {verdict}")
        if verdict == "CERTIFIED":
            think_lines.append("All 58 unary forms eliminated (30 from F3, 28 from F7)")
        return {"user": prompt, "think": "\n".join(think_lines), "answer": verdict}


# ===========================================================================
# 16. bit_subset_check — F10/F11 atom
# ===========================================================================

@register
class BitSubsetCheck(MicroSkill):
    name = "bit_subset_check"
    puzzle_type = "bit_manipulation"
    description = "Check in subset_of out (OR signal) and out subset_of in (AND signal) across examples"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_subset_check.jsonl"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        mode = rng.choice(["in_subset_out", "out_subset_in", "neither"])
        n_ex = rng.choice([4, 5])
        pairs = []
        for _ in range(n_ex):
            x = rng.randrange(1, 255)
            if mode == "in_subset_out":
                # y must have ALL bits of x; add some extras
                extra = rng.randrange(256)
                y = (x | extra) & BYTE
            elif mode == "out_subset_in":
                # y bits are subset of x bits
                y = x & rng.randrange(256)
            else:  # neither
                # Force a bit in x that's not in y AND a bit in y that's not in x
                y = rng.randrange(256)
                if (x & ~y) == 0 or (y & ~x) == 0:
                    # add a bit to x not in y
                    free_x = [p for p in range(8) if not (y & (1 << p))]
                    if free_x:
                        x |= (1 << rng.choice(free_x))
                    free_y = [p for p in range(8) if not (x & (1 << p))]
                    if free_y:
                        y |= (1 << rng.choice(free_y))
            pairs.append((x, y))

        ex_block = "\n".join(f"  {b8(x)} -> {b8(y)}" for x, y in pairs)
        prompt = (
            f"For each example, check two subset relations:\n"
            f"  in subset of out: every 1-bit in input is also 1 in output (in & ~out == 0)\n"
            f"  out subset of in: every 1-bit in output is also 1 in input (out & ~in == 0)\n"
            f"Report which holds for ALL examples (or neither).\n\n"
            f"{ex_block}"
        )
        think_lines = []
        all_in_sub_out = True
        all_out_sub_in = True
        for x, y in pairs:
            in_sub = (x & (~y & BYTE)) == 0
            out_sub = (y & (~x & BYTE)) == 0
            think_lines.append(
                f"  {b8(x)} -> {b8(y)}: in&~out={b8(x & (~y & BYTE))} ({'⊆' if in_sub else '⊄'}), "
                f"out&~in={b8(y & (~x & BYTE))} ({'⊆' if out_sub else '⊄'})"
            )
            if not in_sub:
                all_in_sub_out = False
            if not out_sub:
                all_out_sub_in = False
        if all_in_sub_out:
            verdict = "in_subset_out"
            note = "Signal: y = OR(x, something)"
        elif all_out_sub_in:
            verdict = "out_subset_in"
            note = "Signal: y = AND(x, mask)"
        else:
            verdict = "neither"
            note = "Eliminates pure AND-rules and pure OR-rules"
        think_lines.append(f"-> {verdict}: {note}")
        return {"user": prompt, "think": "\n".join(think_lines), "answer": verdict}


# ===========================================================================
# 17. bit_diff_constancy — F12 atom
# ===========================================================================

@register
class BitDiffConstancy(MicroSkill):
    name = "bit_diff_constancy"
    puzzle_type = "bit_manipulation"
    description = "Check if in XOR out is constant across examples (signal for y = x XOR k)"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_diff_constancy.jsonl"
    weight = 5.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        is_constant = rng.random() < 0.5
        n_ex = rng.choice([4, 5])
        if is_constant:
            k = rng.randrange(256)
            pairs = []
            for _ in range(n_ex):
                x = rng.randrange(256)
                y = x ^ k
                pairs.append((x, y))
        else:
            pairs = [(rng.randrange(256), rng.randrange(256)) for _ in range(n_ex)]
            # ensure not accidentally constant
            diffs = [x ^ y for x, y in pairs]
            if len(set(diffs)) == 1:
                # force a different diff in the last pair
                x, y = pairs[-1]
                pairs[-1] = (x, y ^ 1)

        ex_block = "\n".join(f"  {b8(x)} -> {b8(y)}" for x, y in pairs)
        prompt = (
            f"For each example, compute diff = in XOR out.\n"
            f"If diff is the same value for every example, rule is y = x XOR k.\n\n"
            f"{ex_block}"
        )
        think_lines = []
        diffs = []
        for x, y in pairs:
            d = x ^ y
            diffs.append(d)
            think_lines.append(f"  {b8(x)} XOR {b8(y)} = {b8(d)}")
        if len(set(diffs)) == 1:
            verdict = f"constant k={b8(diffs[0])}"
            think_lines.append(f"All diffs equal -> y = x XOR {b8(diffs[0])}")
        else:
            verdict = "varies"
            think_lines.append("Diffs differ -> not a residual rule")
        return {"user": prompt, "think": "\n".join(think_lines), "answer": verdict}


# ===========================================================================
# 18. bit_3input_canonical — top-4 family execution in canonical shape
# ===========================================================================

@register
class Bit3InputCanonical(MicroSkill):
    name = "bit_3input_canonical"
    puzzle_type = "bit_manipulation"
    description = "Apply one of OR_XNOR/CH/MAJ3/GATED_XNOR_NAND in canonical (rotation, shl, shr) shape — 98.4% of 3-input locks"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_3input_canonical.jsonl"
    weight = 7.0
    max_pool = 10000

    ROTS = [f"rol{k}" for k in range(1, 5)]  # rol5-7 rarely fire per notes
    SHLS = [f"shl{k}" for k in range(1, 8)]
    SHRS = [f"shr{k}" for k in range(1, 8)]

    def generate_one(self, rng, difficulty="medium"):
        family = rng.choice(TOP4_3INPUT)
        sa = rng.choice(self.ROTS)
        sb = rng.choice(self.SHLS)
        sc = rng.choice(self.SHRS)
        x = rng.randrange(256)
        va = UNARY_BY_NAME[sa](x)
        vb = UNARY_BY_NAME[sb](x)
        vc = UNARY_BY_NAME[sc](x)
        y = GATE3[family](va, vb, vc)

        prompt = (
            f"Apply rule y = {family}(A, B, C) where:\n"
            f"  A = {sa}(x)\n  B = {sb}(x)\n  C = {sc}(x)\n"
            f"x = {b8(x)}\n\n"
            f"Family definitions:\n"
            f"  OR_XNOR(a,b,c)         = c OR NOT(a XOR b)\n"
            f"  CH(a,b,c)              = (a AND b) OR (NOT(a) AND c)\n"
            f"  MAJ3(a,b,c)            = (a AND b) OR (a AND c) OR (b AND c)\n"
            f"  GATED_XNOR_NAND(a,b,c) = (c OR NOT(a XOR b)) AND NOT(a AND b AND c)"
        )
        think_lines = [
            f"A = {sa}({b8(x)}) = {b8(va)}",
            f"B = {sb}({b8(x)}) = {b8(vb)}",
            f"C = {sc}({b8(x)}) = {b8(vc)}",
        ]
        if family == "OR_XNOR":
            xnor_ab = (~(va ^ vb)) & BYTE
            think_lines.append(f"NOT(A XOR B) = {b8(xnor_ab)}")
            think_lines.append(f"C OR NOT(A XOR B) = {b8(vc)} OR {b8(xnor_ab)} = {b8(y)}")
        elif family == "CH":
            t1 = va & vb
            t2 = ((~va) & BYTE) & vc
            think_lines.append(f"A AND B = {b8(t1)}")
            think_lines.append(f"NOT(A) AND C = {b8(t2)}")
            think_lines.append(f"OR -> {b8(y)}")
        elif family == "MAJ3":
            t1 = va & vb
            t2 = va & vc
            t3 = vb & vc
            think_lines.append(f"A&B = {b8(t1)}, A&C = {b8(t2)}, B&C = {b8(t3)}")
            think_lines.append(f"OR all -> {b8(y)}")
        else:  # GATED_XNOR_NAND
            xnor_ab = (~(va ^ vb)) & BYTE
            inner = vc | xnor_ab
            nand_abc = (~(va & vb & vc)) & BYTE
            think_lines.append(f"C OR NOT(A XOR B) = {b8(inner)}")
            think_lines.append(f"NOT(A AND B AND C) = {b8(nand_abc)}")
            think_lines.append(f"AND -> {b8(y)}")
        return {"user": prompt, "think": "\n".join(think_lines), "answer": b8(y)}


# ===========================================================================
# 19. bit_algebraic_reduce — collapse compound = unary equivalences
# ===========================================================================

@register
class BitAlgebraicReduce(MicroSkill):
    name = "bit_algebraic_reduce"
    puzzle_type = "bit_manipulation"
    description = "Recognize OR(shr_k, ror_k) = ror_k, OR(shl_k, rol_k) = rol_k, shl_k | shr_(8-k) = rol_k"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_algebraic_reduce.jsonl"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        kind = rng.choice(["shr_ror", "shl_rol", "shl_shr_complement"])
        k = rng.randint(1, 7)
        x = rng.randrange(256)
        if kind == "shr_ror":
            compound = f"OR(shr{k}, ror{k})"
            unary = f"ror{k}"
            v1 = UNARY_BY_NAME[f"shr{k}"](x)
            v2 = UNARY_BY_NAME[f"ror{k}"](x)
            compound_val = (v1 | v2) & BYTE
            unary_val = v2
        elif kind == "shl_rol":
            compound = f"OR(shl{k}, rol{k})"
            unary = f"rol{k}"
            v1 = UNARY_BY_NAME[f"shl{k}"](x)
            v2 = UNARY_BY_NAME[f"rol{k}"](x)
            compound_val = (v1 | v2) & BYTE
            unary_val = v2
        else:  # shl_shr_complement
            compound = f"OR(shl{k}, shr{8-k})"
            unary = f"rol{k}"
            v1 = UNARY_BY_NAME[f"shl{k}"](x)
            v2 = UNARY_BY_NAME[f"shr{8-k}"](x)
            compound_val = (v1 | v2) & BYTE
            unary_val = UNARY_BY_NAME[f"rol{k}"](x)

        prompt = (
            f"Show that {compound} produces the same byte as the simpler rule.\n"
            f"x = {b8(x)}\n"
            f"What is the simpler unary form?"
        )
        think_lines = [
            f"Compute compound on x={b8(x)}:",
            f"  {compound} = {b8(v1)} | {b8(v2)} = {b8(compound_val)}",
            f"Compute candidate unary {unary}({b8(x)}) = {b8(unary_val)}",
            f"Equal? {b8(compound_val)} == {b8(unary_val)}: {compound_val == unary_val}",
        ]
        if compound_val == unary_val:
            think_lines.append(f"Algebraic identity: {compound} === {unary}")
        return {"user": prompt, "think": "\n".join(think_lines), "answer": unary}


# ===========================================================================
# 20. bit_const_middle_bit_signal — F7/F8 detection
# ===========================================================================

@register
class BitConstMiddleBitSignal(MicroSkill):
    name = "bit_const_middle_bit_signal"
    puzzle_type = "bit_manipulation"
    description = "Detect middle-position constant bits (b2..b5) — fires F7/F8: eliminates all shifts and ~shifts"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_const_middle_bit_signal.jsonl"
    weight = 4.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        n_ex = rng.choice([4, 5, 6])
        # Decide whether to plant a constant-middle bit
        plant = rng.random() < 0.7
        forced_idx = None
        forced_val = None
        if plant:
            forced_idx = rng.choice([2, 3, 4, 5])
            forced_val = rng.choice(["0", "1"])

        pairs = []
        for _ in range(n_ex):
            x = rng.randrange(256)
            yb_list = list(b8(rng.randrange(256)))
            if forced_idx is not None:
                yb_list[forced_idx] = forced_val
            y = int("".join(yb_list), 2)
            pairs.append((x, y))

        ex_block = "\n".join(f"  {b8(x)} -> {b8(y)}" for x, y in pairs)
        prompt = (
            f"Scan output bit positions b2..b5 (middle bits) across these examples.\n"
            f"Report any positions where the value is constant (same in every example).\n"
            f"If a middle bit is constant, F7/F8 eliminates all shifts and ~shifts.\n\n"
            f"{ex_block}"
        )
        think_lines = ["Per-middle-position scan:"]
        const_middle = []
        for b_idx in [2, 3, 4, 5]:
            col = [b8(y)[b_idx] for _, y in pairs]
            if all(v == col[0] for v in col):
                const_middle.append((b_idx, col[0]))
                think_lines.append(f"  b{b_idx}: {''.join(col)} -> CONST {col[0]}")
            else:
                think_lines.append(f"  b{b_idx}: {''.join(col)} -> VARIES")

        if const_middle:
            pieces = " ".join(f"b{i}={v}" for i, v in const_middle)
            verdict = pieces
            think_lines.append(f"F7/F8 fires at {pieces} -> all shifts and ~shifts eliminated")
        else:
            verdict = "none"
            think_lines.append("No middle-bit constancy -> F7/F8 does not fire")
        return {"user": prompt, "think": "\n".join(think_lines), "answer": verdict}


# ===========================================================================
# 21. bit_whole_byte_top3_gate — split from 4gate; AND/OR/XOR only
# ===========================================================================

@register
class BitWholeByteTop3Gate(MicroSkill):
    name = "bit_whole_byte_top3_gate"
    puzzle_type = "bit_manipulation"
    description = "Identify whole-byte 2-input gate from {AND, OR, XOR} — covers 100% of 2-input whole-byte locks per notes"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_whole_byte_top3_gate.jsonl"
    weight = 8.0
    max_pool = 10000

    def generate_one(self, rng, difficulty="medium"):
        gate = rng.choice(WHOLE_BYTE_GATES)
        a = rng.randrange(256)
        b = rng.randrange(256)
        out = GATE2[gate](a, b)
        ab, bb, ob = b8(a), b8(b), b8(out)
        prompt = (
            f"At the whole-byte level, only AND/OR/XOR fire as 2-input gates "
            f"(verified across all 1053 training 2-input locks).\n"
            f"Identify which gate was applied.\n"
            f"  A = {ab}\n  B = {bb}\n  out = {ob}"
        )
        think_lines = ["Test each whole-byte gate position-by-position:"]
        for cand in WHOLE_BYTE_GATES:
            pred = b8(GATE2[cand](a, b))
            if pred == ob:
                think_lines.append(f"  {cand:3s}: {pred}  match")
            else:
                diff_pos = [i for i in range(8) if pred[i] != ob[i]][0]
                think_lines.append(f"  {cand:3s}: {pred}  no (first diff at b{diff_pos})")
        return {"user": prompt, "think": "\n".join(think_lines), "answer": gate}


# ===========================================================================
# ROBUSTNESS slice (2026-05-23 batch 3) — LOW WEIGHT
#
# Brief exposure to the gates / families that don't appear at whole-byte level
# in train.csv (XNOR/NAND/NOR/etc., and the 12 never-firing 3-input families).
# Goal: don't break if a private-set puzzle happens to use one. Not main focus.
# ===========================================================================

# Per-bit / rare 2-input gates (don't fire at whole-byte but appear in per-bit
# decomposition or in rarer puzzles). XNOR/NAND/NOR plus the 4 already-covered
# negated forms.
RARE_GATE2 = ["XNOR", "NAND", "NOR"]

# Rare 3-input families that GPT's notes say "never fire" — keep them at minimal
# weight so the model has *some* exposure if private set differs.
RARE_GATE3_DEFS = {
    "T1":     lambda a, b, c: (a ^ (b & c)) & BYTE,
    "PAR3":   lambda a, b, c: (a ^ b ^ c) & BYTE,
    "OX":     lambda a, b, c: ((a | b) ^ c) & BYTE,
    "XA":     lambda a, b, c: ((a ^ b) & c) & BYTE,
    "XO":     lambda a, b, c: ((a ^ b) | c) & BYTE,
    "AND_NOT": lambda a, b, c: (a & ((~b) & BYTE) & c) & BYTE,
    "OR_NOT":  lambda a, b, c: (a | ((~b) & BYTE) | c) & BYTE,
}
RARE_GATE3_NAMES = list(RARE_GATE3_DEFS.keys())


@register
class BitRareGate2ExecRobust(MicroSkill):
    name = "bit_rare_gate2_exec_robust"
    puzzle_type = "bit_manipulation"
    description = "ROBUSTNESS: execute XNOR/NAND/NOR (rare at whole-byte but possible)"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_rare_gate2_exec_robust.jsonl"
    weight = 1.5
    max_pool = 5000

    def generate_one(self, rng, difficulty="medium"):
        gate = rng.choice(RARE_GATE2)
        a, b = rng.randrange(256), rng.randrange(256)
        out = GATE2[gate](a, b)
        prompt = (
            f"[Robustness drill] Compute {gate}(A, B):\n"
            f"  A = {b8(a)}\n  B = {b8(b)}\n"
            f"Definitions:\n"
            f"  XNOR(a,b) = NOT(a XOR b)\n"
            f"  NAND(a,b) = NOT(a AND b)\n"
            f"  NOR(a,b)  = NOT(a OR b)"
        )
        # Inline computation
        if gate == "XNOR":
            xor_v = (a ^ b) & BYTE
            think = f"a XOR b = {b8(xor_v)}\nNOT = {b8(out)}"
        elif gate == "NAND":
            and_v = (a & b) & BYTE
            think = f"a AND b = {b8(and_v)}\nNOT = {b8(out)}"
        else:  # NOR
            or_v = (a | b) & BYTE
            think = f"a OR b = {b8(or_v)}\nNOT = {b8(out)}"
        return {"user": prompt, "think": think, "answer": b8(out)}


@register
class BitRareGate3Recognize(MicroSkill):
    name = "bit_rare_gate3_recognize"
    puzzle_type = "bit_manipulation"
    description = "ROBUSTNESS: brief exposure to never-firing 3-input families (T1/PAR3/OX/XA/XO/AND_NOT/OR_NOT)"
    output_dir = "data/bit_manipulation/pool/generated/ms_bit_rare_gate3_recognize.jsonl"
    weight = 1.0
    max_pool = 5000

    def generate_one(self, rng, difficulty="medium"):
        family = rng.choice(RARE_GATE3_NAMES)
        a, b, c = rng.randrange(256), rng.randrange(256), rng.randrange(256)
        out = RARE_GATE3_DEFS[family](a, b, c)
        prompt = (
            f"[Robustness drill] Apply {family}(A, B, C):\n"
            f"  A = {b8(a)}\n  B = {b8(b)}\n  C = {b8(c)}\n"
            f"Definitions (rare 3-input families):\n"
            f"  T1(a,b,c)      = a XOR (b AND c)\n"
            f"  PAR3(a,b,c)    = a XOR b XOR c\n"
            f"  OX(a,b,c)      = (a OR b) XOR c\n"
            f"  XA(a,b,c)      = (a XOR b) AND c\n"
            f"  XO(a,b,c)      = (a XOR b) OR c\n"
            f"  AND_NOT(a,b,c) = a AND NOT(b) AND c\n"
            f"  OR_NOT(a,b,c)  = a OR NOT(b) OR c"
        )
        think_lines = []
        if family == "T1":
            t = (b & c) & BYTE
            think_lines.append(f"B AND C = {b8(t)}")
            think_lines.append(f"A XOR (B AND C) = {b8(out)}")
        elif family == "PAR3":
            t = (a ^ b) & BYTE
            think_lines.append(f"A XOR B = {b8(t)}")
            think_lines.append(f"XOR C = {b8(out)}")
        elif family == "OX":
            t = (a | b) & BYTE
            think_lines.append(f"A OR B = {b8(t)}")
            think_lines.append(f"XOR C = {b8(out)}")
        elif family == "XA":
            t = (a ^ b) & BYTE
            think_lines.append(f"A XOR B = {b8(t)}")
            think_lines.append(f"AND C = {b8(out)}")
        elif family == "XO":
            t = (a ^ b) & BYTE
            think_lines.append(f"A XOR B = {b8(t)}")
            think_lines.append(f"OR C = {b8(out)}")
        elif family == "AND_NOT":
            t1 = (~b) & BYTE
            think_lines.append(f"NOT B = {b8(t1)}")
            think_lines.append(f"A AND NOT(B) AND C = {b8(out)}")
        else:  # OR_NOT
            t1 = (~b) & BYTE
            think_lines.append(f"NOT B = {b8(t1)}")
            think_lines.append(f"A OR NOT(B) OR C = {b8(out)}")
        return {"user": prompt, "think": "\n".join(think_lines), "answer": b8(out)}


# ===========================================================================
# Weight rebalance for EXISTING skills (applied at import time, after
# microskill_skills has run its _FINAL_WEIGHTS overrides).
# ===========================================================================

# Importing microskill_skills ensures its _FINAL_WEIGHTS overrides are
# applied first. Then we layer the bit-fluency rebalance on top.
from generators import microskill_skills as _ms_skills  # noqa: E402,F401
from generators.microskill_framework import REGISTRY as _REG  # noqa: E402

_BIT_FLUENCY_REBALANCE = {
    # CONST detection — 14.8% of bits, currently under-weighted
    "bit_constant_positions":   6.0,
    "bit_constant_vs_variable": 3.0,
    "bit_and_across":           2.0,
    "bit_or_across":            2.0,
    # NOT detection — 5.1% of bits + key for negated gates
    "bit_complement_view":      3.0,
    # Re-enable verify/diff atoms — needed for in-trace verification loops
    "bit_verify_given":         4.0,
    "bit_verify_given_3":       2.0,
    "bit_diff_check":           2.0,
    # Re-enable gate confusion drill — XOR↔XNOR is dominant failure
    "bit_gate_confusion_drill": 6.0,
    # Whole-byte 2-source skills — 45% of solver routes
    "bit_compose_two_step":     6.0,
    "bit_compose2":             4.0,
    "bit_narrow_sources":       5.0,
    "bit_second_source":        4.0,
    "bit_how_many_sources":     5.0,
}

for _name, _w in _BIT_FLUENCY_REBALANCE.items():
    cls = _REG.get(_name)
    if cls is not None:
        cls.weight = _w
