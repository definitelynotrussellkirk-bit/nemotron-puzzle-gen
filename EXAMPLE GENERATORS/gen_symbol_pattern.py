#!/usr/bin/env python3
"""Generate non-cipher symbolic transformation puzzles.

These are NOT cipher-digit — the symbols follow pattern/positional rules
rather than digit-arithmetic. Teaches the model that not all symbolic
transformation is cipher-digit.

Operations:
- Reverse the symbol string
- Rotate symbols by N positions
- Swap positions (e.g., swap pos 0 and 4)
- Take every other symbol
- Repeat a pattern
- First/last extraction

Usage:
    python3 -m generators.gen_symbol_pattern --n 2000
"""
import argparse
import json
import random
import time
from datetime import datetime, timezone
from training.data import BOXED_INSTRUCTION

SAFE_SYMBOLS = list('!@#$%^&*()-_=+[]:,.<>?/~`')


def _reverse(s):
    return s[::-1]

def _rotate_left(s, n):
    n = n % len(s)
    return s[n:] + s[:n]

def _swap_ends(s):
    if len(s) < 2: return s
    return s[-1] + s[1:-1] + s[0]

def _every_other(s):
    return s[::2]

def _repeat_first(s):
    return s[0] * len(s) if s else ''

def _sort_symbols(s):
    return ''.join(sorted(s))

def _delete_center(s):
    if len(s) < 3: return s
    mid = len(s) // 2
    return s[:mid] + s[mid+1:]

def _delete_last(s):
    return s[:-1] if s else ''

def _delete_first(s):
    return s[1:] if s else ''

def _duplicate(s):
    return s + s

def _keep_first_half(s):
    return s[:len(s)//2] if len(s) >= 2 else s

def _keep_last_half(s):
    return s[len(s)//2:] if len(s) >= 2 else s

def _mirror(s):
    return s + s[::-1]

def _rotate_right(s, n):
    if not s: return s
    n = n % len(s)
    return s[-n:] + s[:-n]

def _interleave_self(s):
    """Interleave first half with second half."""
    mid = len(s) // 2
    a, b = s[:mid], s[mid:]
    result = []
    for i in range(max(len(a), len(b))):
        if i < len(a): result.append(a[i])
        if i < len(b): result.append(b[i])
    return ''.join(result)

def _odd_positions(s):
    return s[1::2]


OPS = {
    "reverse": (_reverse, "Reverse the symbol string"),
    "rotate1": (lambda s: _rotate_left(s, 1), "Rotate left by 1"),
    "rotate2": (lambda s: _rotate_left(s, 2), "Rotate left by 2"),
    "rotate_right1": (lambda s: _rotate_right(s, 1), "Rotate right by 1"),
    "swap_ends": (_swap_ends, "Swap first and last symbols"),
    "every_other": (_every_other, "Take every other symbol (positions 0,2,4,...)"),
    "odd_positions": (_odd_positions, "Take odd positions (1,3,5,...)"),
    "repeat_first": (_repeat_first, "Repeat the first symbol N times"),
    "sort": (_sort_symbols, "Sort symbols alphabetically"),
    "delete_center": (_delete_center, "Delete the center symbol"),
    "delete_last": (_delete_last, "Delete the last symbol"),
    "delete_first": (_delete_first, "Delete the first symbol"),
    "duplicate": (_duplicate, "Duplicate the entire string"),
    "keep_first_half": (_keep_first_half, "Keep first half only"),
    "keep_last_half": (_keep_last_half, "Keep last half only"),
    "mirror": (_mirror, "Append reversed copy"),
    "interleave": (_interleave_self, "Interleave first and second halves"),
}


def generate_one(rng):
    """Generate one non-cipher symbolic pattern puzzle."""
    op_name = rng.choice(list(OPS.keys()))
    fn, description = OPS[op_name]

    n_symbols = rng.randint(3, 6)
    sym_pool = rng.sample(SAFE_SYMBOLS, min(12, len(SAFE_SYMBOLS)))

    # Generate 3-4 examples
    examples = []
    for _ in range(rng.randint(3, 4)):
        inp = ''.join(rng.choices(sym_pool, k=n_symbols))
        out = fn(inp)
        if not out:
            continue
        examples.append((inp, out))

    if len(examples) < 3:
        return None

    # Query
    query_inp = ''.join(rng.choices(sym_pool, k=n_symbols))
    query_out = fn(query_inp)
    if not query_out:
        return None

    # Build prompt (same Alice wrapper as other transformation)
    prompt_lines = [
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples:",
    ]
    for inp, out in examples:
        prompt_lines.append(f"{inp} = {out}")
    prompt_lines.append(f"Now, determine the result for: {query_inp}")
    prompt = "\n".join(prompt_lines)

    # Build trace — NOT cipher-digit format
    lines = [
        "Symbol pattern.",
        f"Detect: non-cipher symbols, input_len={n_symbols}, examples={len(examples)}",
        "",
        f"Rule family: {op_name}",
        "",
    ]

    # Show the pattern discovery
    inp0, out0 = examples[0]
    lines.append(f"Ex1: {inp0} → {out0}")

    if op_name == "reverse":
        lines.append(f"  Output is input reversed: {inp0} → {inp0[::-1]}")
    elif op_name in ("rotate1", "rotate2"):
        n = int(op_name[-1])
        lines.append(f"  Output is input rotated left by {n}: {inp0} → {inp0[n:]+inp0[:n]}")
    elif op_name == "rotate_right1":
        lines.append(f"  Output is input rotated right by 1: {inp0} → {inp0[-1:]+inp0[:-1]}")
    elif op_name == "swap_ends":
        lines.append(f"  First and last swapped: {inp0[0]}...{inp0[-1]} → {inp0[-1]}...{inp0[0]}")
    elif op_name == "every_other":
        lines.append(f"  Every other symbol: positions 0,2,4,... → {out0}")
    elif op_name == "odd_positions":
        lines.append(f"  Odd positions: positions 1,3,5,... → {out0}")
    elif op_name == "repeat_first":
        lines.append(f"  First symbol repeated {len(out0)} times: {inp0[0]} × {len(out0)}")
    elif op_name == "sort":
        lines.append(f"  Symbols sorted: {inp0} → {''.join(sorted(inp0))}")
    elif op_name == "delete_center":
        lines.append(f"  Delete center position {len(inp0)//2}: {inp0} → {out0}")
    elif op_name == "delete_last":
        lines.append(f"  Delete last symbol: {inp0} → {out0}")
    elif op_name == "delete_first":
        lines.append(f"  Delete first symbol: {inp0} → {out0}")
    elif op_name == "duplicate":
        lines.append(f"  Duplicate whole string: {inp0}+{inp0} → {out0}")
    elif op_name == "keep_first_half":
        lines.append(f"  Keep first half: {inp0[:len(inp0)//2]} → {out0}")
    elif op_name == "keep_last_half":
        lines.append(f"  Keep last half: {inp0[len(inp0)//2:]} → {out0}")
    elif op_name == "mirror":
        lines.append(f"  Append reversed copy: {inp0}+{inp0[::-1]} → {out0}")
    elif op_name == "interleave":
        lines.append(f"  Interleave first and second halves → {out0}")

    # Verify on Ex2
    inp1, out1 = examples[1]
    check = fn(inp1)
    lines.append(f"")
    lines.append(f"Verify Ex2: {inp1} → {check} vs {out1} → MATCH")
    lines.append(f"")

    # Query
    lines.append(f"Query: {query_inp}")
    lines.append(f"  Apply: {query_out}")

    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": f"<think>\n{chr(10).join(lines)}\n</think>\n\\boxed{{{query_out}}}"},
        ],
        "answer": query_out,
        "id": f"gen_sym_pattern_{rng.randint(0, 999999):06d}",
        "puzzle_type": "transformation",
        "mode": "symbol_pattern",
        "generator": "gen_symbol_pattern",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--output", type=str,
                        default="data/transformation/pool/generated/symbol_pattern.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    rng = random.Random(args.seed)
    t0 = time.time()
    count = 0

    with open(args.output, "w") as f:
        for _ in range(args.n * 3):
            if count >= args.n:
                break
            row = generate_one(rng)
            if row:
                f.write(json.dumps(row) + "\n")
                count += 1
                if count % 500 == 0:
                    print(f"  {count}/{args.n}", flush=True)

    dt = time.time() - t0
    print(f"Generated {count} symbol pattern traces in {dt:.1f}s → {args.output}")


if __name__ == "__main__":
    main()
