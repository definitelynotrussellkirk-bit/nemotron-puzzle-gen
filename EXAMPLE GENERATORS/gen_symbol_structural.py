#!/usr/bin/env python3
"""Generate symbol transformation training data using structural alignment traces.

Trace format designed for a 1B model that cannot do base-N arithmetic.
Instead of decoding symbols to digits and computing, the trace shows
structural patterns (operand lengths, operator identity, output lengths)
and aligns the query to the nearest matching example.

Usage:
    python3 -m generators.gen_symbol_structural [--train-csv PATH] [--output PATH]
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import re

from training.data import BOXED_INSTRUCTION, answer_needs_text_fallback, format_answer_block


def _extract_boxed(content: str) -> Optional[str]:
    """Extract last \\boxed{...} content using brace-depth counting.
    Mirrors build_manifest_training.extract_boxed exactly."""
    boxes = list(re.finditer(r'\\boxed\{', content))
    if not boxes:
        return None
    start = boxes[-1].end()
    depth = 1
    pos = start
    while pos < len(content) and depth > 0:
        if content[pos] == '{': depth += 1
        elif content[pos] == '}': depth -= 1
        pos += 1
    if depth != 0:
        return None
    return content[start:pos-1].strip()


def parse_prompt(prompt: str) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """Extract examples and query from a transformation prompt."""
    examples: List[Tuple[str, str]] = []
    query = None
    for line in prompt.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("now, determine the result for:"):
            query = line.split(":", 1)[1].strip()
        elif " = " in line and "Below are a few examples" not in line:
            lhs, rhs = line.split(" = ", 1)
            examples.append((lhs.strip(), rhs.strip()))
    return examples, query


def parse_symbol_equation(expr: str) -> Optional[Tuple[str, str, str]]:
    """Parse a 5-char symbol expression into (left, op, right).

    Symbol transformation inputs are always exactly 5 characters:
    [d1][d2][OPERATOR][d3][d4]
    The operator is at position 2.

    Returns (left_2chars, op_char, right_2chars) or None if invalid.
    """
    s = expr.replace(" ", "")
    if len(s) != 5:
        return None
    return s[:2], s[2], s[3:]


def is_symbol_row(examples: List[Tuple[str, str]], query: Optional[str]) -> bool:
    """Check if this is a symbol (non-numeric) transformation row."""
    if not examples or not query:
        return False
    # Check the first example input
    inp = examples[0][0].replace(" ", "")
    if len(inp) != 5:
        return False
    # Symbol rows have no digits in the input
    return not any(c.isdigit() for c in inp)


def build_structural_trace(prompt: str, answer: str) -> Optional[str]:
    """Build a structural alignment trace for a symbol transformation puzzle.

    Returns the trace string (without think/boxed wrapping) or None on failure.
    """
    examples, query = parse_prompt(prompt)
    if not examples or not query:
        return None

    if not is_symbol_row(examples, query):
        return None

    # Parse query
    q_parsed = parse_symbol_equation(query)
    if q_parsed is None:
        return None
    q_left, q_op, q_right = q_parsed

    # Parse all examples
    parsed_examples = []
    for inp, out in examples:
        p = parse_symbol_equation(inp)
        if p is None:
            return None
        parsed_examples.append((p[0], p[1], p[2], out))

    # Collect unique symbols across inputs and outputs
    all_symbols = set()
    for left, op, right, out in parsed_examples:
        all_symbols.update(left)
        all_symbols.update(right)
        # Output symbols (skip sign/op prefixes)
        for ch in out:
            all_symbols.add(ch)
    all_symbols.update(q_left)
    all_symbols.update(q_right)
    # Remove operator chars from symbol list
    op_chars = set()
    for _, op, _, _ in parsed_examples:
        op_chars.add(op)
    op_chars.add(q_op)
    # Digit symbols = everything that's not an operator or dash
    digit_symbols = sorted(all_symbols - op_chars - {'-'})

    # Build trace parts
    lines = []

    # [Parse]
    lines.append("[Parse]")
    lines.append(f"Left: {q_left} | Op: {q_op} | Right: {q_right}")
    lines.append("")

    # [Vocabulary]
    lines.append("[Vocabulary]")
    lines.append(f"Unique symbols in context: {','.join(digit_symbols)}")
    lines.append("")

    # [Structure Analysis]
    lines.append("[Structure Analysis]")
    for i, (left, op, right, out) in enumerate(parsed_examples, 1):
        inp_str = f"{left}{op}{right}"
        l_left = len(left)
        l_right = len(right)
        l_out = len(out)
        line = f"Ex {i}: L({l_left}) {op} L({l_right}) -> R({l_out})"
        # Find the nearest match to annotate with full structure
        if op == q_op and abs(l_left - len(q_left)) + abs(l_right - len(q_right)) == 0:
            line += f" | Structure: {inp_str} -> {out}"
        lines.append(line)
    lines.append("")

    # [Target Alignment]
    lines.append("[Target Alignment]")
    lines.append(f"Query: {q_left}{q_op}{q_right} | Structure: L({len(q_left)}) {q_op} L({len(q_right)})")

    # Find nearest matching example
    best_idx = None
    best_dist = float('inf')
    for i, (left, op, right, out) in enumerate(parsed_examples):
        if op != q_op:
            continue
        dist = abs(len(left) - len(q_left)) + abs(len(right) - len(q_right))
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    if best_idx is None:
        # No same-op example found, pick closest by length
        for i, (left, op, right, out) in enumerate(parsed_examples):
            dist = abs(len(left) - len(q_left)) + abs(len(right) - len(q_right))
            if dist < best_dist:
                best_dist = dist
                best_idx = i

    if best_idx is not None:
        ex = parsed_examples[best_idx]
        op_name = f"{ex[1]} pattern"
        lines.append(f"Nearest match: Ex {best_idx + 1} ({op_name}).")
        lines.append(f"Output length: R({len(ex[3])}) expected.")
    lines.append("")

    # [Result]
    lines.append(f"[Result] {answer}")

    return "\n".join(lines)


def generate_symbol_pool(train_csv_path: str, output_path: str) -> Dict:
    """Read train.csv, generate structural traces for symbol transformation rows.

    Returns dict with stats: total, traced, skipped, errors.
    """
    stats = {"total": 0, "traced": 0, "skipped": 0, "used_text_fallback": 0, "errors": []}
    results = []

    with open(train_csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row['prompt']
            if 'transformation rule' not in prompt.lower():
                continue

            examples, query = parse_prompt(prompt)
            if not is_symbol_row(examples, query):
                continue

            stats["total"] += 1
            answer = row['answer']
            row_id = row['id']

            trace = build_structural_trace(prompt, answer)
            if trace is None:
                stats["skipped"] += 1
                stats["errors"].append(f"{row_id}: parse failure")
                continue

            full_prompt = prompt + BOXED_INSTRUCTION
            answer_block = format_answer_block(answer)
            assistant_content = f"<think>\n{trace}\n</think>\n{answer_block}"

            # Roundtrip check
            if answer_needs_text_fallback(answer):
                expected_block = f"The final answer is: {answer}"
                if expected_block not in assistant_content:
                    stats["skipped"] += 1
                    stats["errors"].append(f"{row_id}: text fallback roundtrip failed")
                    continue
                stats["used_text_fallback"] += 1
            else:
                extracted = _extract_boxed(assistant_content)
                if extracted != answer:
                    stats["skipped"] += 1
                    stats["errors"].append(f"{row_id}: boxed roundtrip failed ({repr(extracted)} != {repr(answer)})")
                    continue

            stats["traced"] += 1

            record = {
                "messages": [
                    {"role": "user", "content": full_prompt},
                    {"role": "assistant", "content": assistant_content},
                ],
                "answer": answer,
                "id": row_id,
                "puzzle_type": "transformation",
                "mode": "symbol_structural",
                "generator": "gen_symbol_structural",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            results.append(record)

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate structural alignment traces for symbol transformation")
    parser.add_argument("--train-csv", default=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "competition", "train.csv"
    ))
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "transformation", "pool", "symbol", "symbol_structural.jsonl"
    ))
    args = parser.parse_args()

    stats = generate_symbol_pool(args.train_csv, args.output)

    print(f"Total symbol rows: {stats['total']}")
    print(f"Traced: {stats['traced']}")
    print(f"Skipped: {stats['skipped']}")
    if stats['errors']:
        print(f"\nFirst 10 errors:")
        for e in stats['errors'][:10]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
