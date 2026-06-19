#!/usr/bin/env python3
r"""Generate edit-program traces for symbol transformation puzzles.

Detects rows where the output is a simple positional rearrangement of the input
characters (delete, swap halves, extract positions). Produces short, derivable
traces that show the edit pattern across same-operator examples and apply it
to the query.

Trace format (example -- delete_center):

    mode=delete_center
    focus on examples with center symbol *

    %|*"| -> %|"|   so remove the 3rd symbol
    \(*[^ -> \([^   so remove the 3rd symbol
    |[*([ -> |[([   so remove the 3rd symbol

    Apply the same edit to the query:
    \(*[# -> \([#

This is a SEPARATE tracer from gen_symbol_structural.py. Rows that are not
edit operations return None and should fall through to structural alignment.

Usage:
    python3 -m generators.gen_symbol_editprog [--train-csv PATH] [--output PATH]
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from itertools import product
from typing import Dict, List, Optional, Tuple

from training.data import BOXED_INSTRUCTION, answer_needs_text_fallback, format_answer_block
from generators.gen_symbol_latent import (
    build_symbol_map,
    symbol_map_line,
    delex,
    delex_answer,
)


def _extract_boxed(content: str) -> Optional[str]:
    """Extract last \\boxed{...} content using brace-depth counting."""
    boxes = list(re.finditer(r"\\boxed\{", content))
    if not boxes:
        return None
    start = boxes[-1].end()
    depth = 1
    pos = start
    while pos < len(content) and depth > 0:
        if content[pos] == "{":
            depth += 1
        elif content[pos] == "}":
            depth -= 1
        pos += 1
    if depth != 0:
        return None
    return content[start : pos - 1].strip()


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
    """
    s = expr.replace(" ", "")
    if len(s) != 5:
        return None
    return s[:2], s[2], s[3:]


def is_symbol_row(examples: List[Tuple[str, str]], query: Optional[str]) -> bool:
    """Check if this is a symbol (non-numeric) transformation row."""
    if not examples or not query:
        return False
    inp = examples[0][0].replace(" ", "")
    if len(inp) != 5:
        return False
    return not any(c.isdigit() for c in inp)


def find_same_op_examples(
    examples: List[Tuple[str, str]], query_op: str
) -> List[Tuple[str, str]]:
    """Filter examples that use the same center operator character as the query.

    Returns list of (input_5chars, output_str) tuples.
    """
    result = []
    for inp, out in examples:
        inp_clean = inp.replace(" ", "")
        if len(inp_clean) == 5 and inp_clean[2] == query_op:
            result.append((inp_clean, out))
    return result


# ---------------------------------------------------------------------------
# Known edit-program types with human-readable descriptions
# ---------------------------------------------------------------------------

# Canonical permutations we recognize and can name
NAMED_PERMS = {
    (0, 1, 3, 4): ("delete_center", "remove the 3rd symbol"),
    (3, 4, 0, 1): ("swap_halves", "swap left and right halves (drop center)"),
    (4, 3, 2, 1, 0): ("reverse_all", "reverse all 5 positions"),
    (4, 3, 1, 0): ("reverse_no_center", "reverse without center"),
    (0, 3, 2, 1, 4): ("swap_inner", "swap positions 1 and 3"),
    (4, 1, 2, 3, 0): ("swap_outer", "swap positions 0 and 4"),
}


def _describe_perm(perm: Tuple[int, ...]) -> Tuple[str, str]:
    """Return (edit_type_name, rule_description) for a permutation.

    For known permutations, returns a human-friendly name.
    For unknown ones, generates a positional description.
    """
    if perm in NAMED_PERMS:
        return NAMED_PERMS[perm]

    positions = ",".join(str(p) for p in perm)
    n_out = len(perm)
    n_in = 5

    if n_out < n_in:
        return (f"select_{positions}", f"keep positions {positions}")
    elif n_out == n_in:
        return (f"rearrange_{positions}", f"rearrange to positions {positions}")
    else:
        return (f"expand_{positions}", f"map to positions {positions}")


def classify_edit_type(
    same_op_examples: List[Tuple[str, str]],
    query_input: str,
    expected_answer: str,
) -> Optional[Tuple[Tuple[int, ...], str, str]]:
    """Determine if this row is a positional edit operation.

    Checks whether the output of each same-op example can be produced by
    selecting characters from specific input positions (with possible
    repetition). The same positional mapping must hold across ALL same-op
    examples AND produce the expected answer on the query.

    Returns (perm_tuple, edit_type_name, rule_description) or None.
    """
    if not same_op_examples:
        return None

    # All same-op outputs must have the same length
    out_lens = set(len(out) for _, out in same_op_examples)
    if len(out_lens) != 1:
        return None
    out_len = out_lens.pop()

    if out_len < 1 or out_len > 5:
        return None

    # Try all position mappings (with repetition)
    # For out_len positions, each drawn from {0,1,2,3,4}
    best_perm = None
    for perm in product(range(5), repeat=out_len):
        if all(
            "".join(inp[p] for p in perm) == out
            for inp, out in same_op_examples
        ):
            expected = "".join(query_input[p] for p in perm)
            if expected == expected_answer:
                if best_perm is None:
                    best_perm = perm
                # Prefer known/named permutations
                if perm in NAMED_PERMS:
                    best_perm = perm
                    break

    if best_perm is None:
        return None

    edit_type, rule_desc = _describe_perm(best_perm)
    return best_perm, edit_type, rule_desc


def build_edit_trace(prompt: str, answer: str) -> Optional[str]:
    """Build an edit-program trace for a symbol transformation puzzle.

    Uses delexicalized canonical labels (s0, s1, ...) with spaces between
    labels so the edit operation is positionally clear to the model.

    Returns the full assistant content (with think/boxed wrapping) or None
    if this row is not an edit operation.
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
    q_inp = q_left + q_op + q_right  # full 5-char input

    # Find same-operator examples
    same_op = find_same_op_examples(examples, q_op)
    if not same_op:
        return None

    # Classify the edit type
    result = classify_edit_type(same_op, q_inp, answer)
    if result is None:
        return None

    perm, edit_type, rule_desc = result

    # --- Delexicalization ---
    smap = build_symbol_map(examples, query, answer)
    sym_line = symbol_map_line(smap)
    q_op_label = smap.get(q_op, q_op)

    # Build the trace with spaced canonical labels
    lines = []
    lines.append(f"mode={edit_type}")
    lines.append(sym_line)
    lines.append(f"focus on examples with center symbol {q_op_label}")
    lines.append("")

    for inp, out in same_op:
        inp_delex = delex_answer(inp, smap)   # spaced: "s0 s1 s2 s3 s4"
        out_delex = delex_answer(out, smap)    # spaced: "s0 s1 s3 s4"
        lines.append(f"{inp_delex} -> {out_delex}   so {rule_desc}")

    lines.append("")
    lines.append("Apply the same edit to the query:")
    q_delex = delex_answer(q_inp, smap)
    ans_delex = delex_answer(answer, smap)
    lines.append(f"{q_delex} -> {ans_delex}")

    trace = "\n".join(lines)
    return f"<think>\n{trace}\n</think>\n{format_answer_block(answer)}"


def generate_edit_pool(train_csv_path: str, output_path: str) -> Dict:
    """Read train.csv, generate edit-program traces for symbol transformation rows.

    Returns dict with stats.
    """
    stats = {
        "total_symbol": 0,
        "traced": 0,
        "skipped_no_edit": 0,
        "skipped_roundtrip_fail": 0,
        "used_text_fallback": 0,
        "edit_types": {},
    }
    results = []

    with open(train_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row["prompt"]
            if "transformation rule" not in prompt.lower():
                continue

            examples, query = parse_prompt(prompt)
            if not is_symbol_row(examples, query):
                continue

            stats["total_symbol"] += 1
            answer = row["answer"]
            row_id = row["id"]

            assistant_content = build_edit_trace(prompt, answer)
            if assistant_content is None:
                stats["skipped_no_edit"] += 1
                continue

            # Roundtrip check: brace-unsafe answers use text fallback
            if answer_needs_text_fallback(answer):
                expected_block = f"The final answer is: {answer}"
                if expected_block not in assistant_content:
                    stats["skipped_roundtrip_fail"] += 1
                    continue
                stats["used_text_fallback"] += 1
            else:
                extracted = _extract_boxed(assistant_content)
                if extracted != answer:
                    stats["skipped_roundtrip_fail"] += 1
                    continue

            # Determine edit type for stats
            # Re-parse to get the type name (cheap, already computed internally)
            q_parsed = parse_symbol_equation(query.replace(" ", "") if query else "")
            same_op = find_same_op_examples(examples, q_parsed[1] if q_parsed else "")
            result = classify_edit_type(
                same_op, q_parsed[0] + q_parsed[1] + q_parsed[2] if q_parsed else "", answer
            )
            if result:
                etype = result[1]
                stats["edit_types"][etype] = stats["edit_types"].get(etype, 0) + 1

            stats["traced"] += 1

            full_prompt = prompt + BOXED_INSTRUCTION
            record = {
                "messages": [
                    {"role": "user", "content": full_prompt},
                    {"role": "assistant", "content": assistant_content},
                ],
                "answer": answer,
                "id": row_id,
                "puzzle_type": "transformation",
                "mode": "symbol_editprog",
                "generator": "gen_symbol_editprog",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            results.append(record)

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate edit-program traces for symbol transformation"
    )
    parser.add_argument(
        "--train-csv",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "competition",
            "train.csv",
        ),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "transformation",
            "pool",
            "symbol",
            "symbol_editprog.jsonl",
        ),
    )
    args = parser.parse_args()

    stats = generate_edit_pool(args.train_csv, args.output)

    print(f"Total symbol rows: {stats['total_symbol']}")
    print(f"Traced (edit-program): {stats['traced']}")
    print(f"  (using text fallback for braces: {stats['used_text_fallback']})")
    print(f"Skipped (not edit): {stats['skipped_no_edit']}")
    print(f"Skipped (roundtrip fail): {stats['skipped_roundtrip_fail']}")
    print(f"\nEdit type breakdown:")
    for etype, count in sorted(
        stats["edit_types"].items(), key=lambda x: -x[1]
    ):
        print(f"  {etype}: {count}")

    # Print sample traces
    if os.path.exists(args.output):
        print(f"\n--- Sample traces (first 3) ---")
        with open(args.output) as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                record = json.loads(line)
                print(f"\n[{record['id']}] answer={record['answer']}")
                print(record["messages"][1]["content"])
                print("---")


if __name__ == "__main__":
    main()
