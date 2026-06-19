#!/usr/bin/env python3
r"""Generate latent-tag control traces for symbol transformation puzzles.

Short diagnostic traces that classify the puzzle rather than narrate arithmetic.
The model learns to predict classification tags (SEEN_OP, WORLD, FAMILY, LEN, SIGN)
from the prompt alone.  At inference the tags guide answer-generation policy.

Trace format (~60-100 tokens):

    <think>
    [TASK=SYMBOL] [SEEN_OP=1] [WORLD=CLOSED] [FAMILY=SHORT_ARITH] [LEN=2] [SIGN=NONE]

    Evidence:
    - Query op '*' seen in 3 examples.
    - Same-op outputs: lengths 2, 2, 3.
    - Output has new symbols -> not EDIT.
    - All output symbols in prompt vocabulary -> CLOSED.

    Answer policy:
    - use only prompt vocabulary
    - output exactly 2 symbols
    - follow same-op output style
    </think>
    \boxed{@&}

Replaces gen_symbol_structural.py.  Does NOT replace gen_symbol_editprog.py
(edit-program traces are separate; this tracer still classifies EDIT rows
in latent-tag format so both can coexist via trace-style-aware dedup).

Usage:
    python3 -m generators.gen_symbol_latent [--train-csv PATH] [--output PATH]
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from typing import Dict, List, Optional, Set, Tuple

from training.data import BOXED_INSTRUCTION, answer_needs_text_fallback, format_answer_block


# ---------------------------------------------------------------------------
# Delexicalization: map raw glyphs to canonical s0, s1, s2, ...
# ---------------------------------------------------------------------------

# Structural characters that are NOT symbols (used in prompt formatting)
_STRUCTURAL_CHARS = set("= \t\n")


def build_symbol_map(
    examples: List[Tuple[str, str]], query: Optional[str], answer: Optional[str] = None
) -> Dict[str, str]:
    """Build a mapping from raw symbol glyphs to canonical labels (s0, s1, ...).

    Scans the prompt left-to-right (examples then query) for first-appearance
    ordering.  Structural characters (=, space, tab, newline) are skipped.
    If *answer* is provided, any symbols only in the answer are appended last.
    """
    seen: Dict[str, str] = {}  # glyph -> canonical label
    counter = 0

    def _register(ch: str):
        nonlocal counter
        if ch not in seen and ch not in _STRUCTURAL_CHARS:
            seen[ch] = f"s{counter}"
            counter += 1

    # Scan examples in order
    for inp, out in examples:
        for ch in inp.replace(" ", ""):
            _register(ch)
        for ch in out:
            _register(ch)

    # Scan query
    if query:
        for ch in query.replace(" ", ""):
            _register(ch)

    # Scan answer (so answer-only symbols get labels too)
    if answer:
        for ch in answer:
            _register(ch)

    return seen


def symbol_map_line(smap: Dict[str, str]) -> str:
    """Format a compact one-line symbol legend: 'Symbols: s0=` s1=! s2=* ...'"""
    # Sort by canonical index so it reads s0 s1 s2 ...
    items = sorted(smap.items(), key=lambda kv: int(kv[1][1:]))
    pairs = " ".join(f"{label}={glyph}" for glyph, label in items)
    return f"Symbols: {pairs}"


def delex(text: str, smap: Dict[str, str], spaced: bool = False) -> str:
    """Replace every mapped glyph in *text* with its canonical label.

    If *spaced* is True, separate consecutive labels with spaces
    (useful for edit-program traces where positional clarity matters).
    """
    parts = []
    for ch in text:
        if ch in smap:
            parts.append(smap[ch])
        else:
            parts.append(ch)
    sep = " " if spaced else ""
    return sep.join(parts) if spaced else "".join(parts)


def delex_answer(answer: str, smap: Dict[str, str]) -> str:
    """Delexicalize the answer with spaces between labels."""
    return " ".join(smap.get(ch, ch) for ch in answer)


# ---------------------------------------------------------------------------
# Helpers (shared logic with gen_symbol_structural / gen_symbol_editprog)
# ---------------------------------------------------------------------------

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
    """Filter examples that use the same center operator as the query."""
    result = []
    for inp, out in examples:
        inp_clean = inp.replace(" ", "")
        if len(inp_clean) == 5 and inp_clean[2] == query_op:
            result.append((inp_clean, out))
    return result


# ---------------------------------------------------------------------------
# Edit detection (lightweight reuse from gen_symbol_editprog logic)
# ---------------------------------------------------------------------------

def _is_edit(same_op: List[Tuple[str, str]], query_input: str, answer: str) -> bool:
    """Check if the row is a positional edit operation.

    Returns True if any single positional permutation of the 5-char input
    produces the output for ALL same-op examples AND the query.
    """
    if not same_op:
        return False
    out_lens = set(len(out) for _, out in same_op)
    if len(out_lens) != 1:
        return False
    out_len = out_lens.pop()
    if out_len < 1 or out_len > 5:
        return False

    for perm in product(range(5), repeat=out_len):
        if all(
            "".join(inp[p] for p in perm) == out
            for inp, out in same_op
        ):
            expected = "".join(query_input[p] for p in perm)
            if expected == answer:
                return True
    return False


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _collect_prompt_symbols(examples: List[Tuple[str, str]], query: Optional[str]) -> Set[str]:
    """Collect all symbols visible in the prompt (examples + query)."""
    symbols: Set[str] = set()
    for inp, out in examples:
        for ch in inp.replace(" ", ""):
            symbols.add(ch)
        for ch in out:
            symbols.add(ch)
    if query:
        for ch in query.replace(" ", ""):
            symbols.add(ch)
    return symbols


def classify_row(
    examples: List[Tuple[str, str]],
    query: Optional[str],
    answer: str,
) -> Optional[Dict]:
    """Classify a symbol transformation row into diagnostic tags.

    Returns dict with keys: seen_op, world, family, length, sign,
    same_op_count, same_op_lengths, is_edit, prompt_vocab, novel_symbols.
    Or None if unparseable.
    """
    if not query:
        return None
    q_parsed = parse_symbol_equation(query)
    if q_parsed is None:
        return None
    q_left, q_op, q_right = q_parsed
    q_input = q_left + q_op + q_right

    # SEEN_OP: is the query operator present in any example's position 2?
    example_ops = set()
    for inp, out in examples:
        inp_clean = inp.replace(" ", "")
        if len(inp_clean) == 5:
            example_ops.add(inp_clean[2])
    seen_op = q_op in example_ops

    # Same-op examples
    same_op = find_same_op_examples(examples, q_op)
    same_op_count = len(same_op)
    same_op_lengths = [len(out) for _, out in same_op]

    # WORLD: does the gold answer contain symbols not in the prompt?
    prompt_symbols = _collect_prompt_symbols(examples, query)
    answer_symbols = set(answer)
    novel_symbols = answer_symbols - prompt_symbols
    world = "OPEN" if novel_symbols else "CLOSED"

    # Edit test
    is_edit = _is_edit(same_op, q_input, answer) if same_op else False

    # FAMILY
    if world == "OPEN":
        family = "OPEN_WORLD"
    elif is_edit:
        family = "EDIT"
    elif len(answer) <= 2:
        family = "SHORT_ARITH"
    else:
        family = "LONG_ARITH"

    # LEN
    length = len(answer)

    # SIGN: does answer start or end with the operator character?
    if len(answer) >= 1 and answer[0] == q_op:
        sign = "OPSIGN"
    elif len(answer) >= 1 and answer[-1] == q_op:
        sign = "TAILSIGN"
    else:
        sign = "NONE"

    return {
        "seen_op": seen_op,
        "world": world,
        "family": family,
        "length": length,
        "sign": sign,
        "same_op_count": same_op_count,
        "same_op_lengths": same_op_lengths,
        "is_edit": is_edit,
        "prompt_vocab": prompt_symbols,
        "novel_symbols": novel_symbols,
        "q_op": q_op,
    }


# ---------------------------------------------------------------------------
# Trace builder
# ---------------------------------------------------------------------------

def build_latent_trace(prompt: str, answer: str) -> Optional[str]:
    """Build a latent-tag control trace for a symbol transformation puzzle.

    Uses delexicalized canonical labels (s0, s1, ...) so the model reasons
    over clean tokens instead of raw punctuation glyphs.

    Returns the full assistant content (with think/boxed) or None on failure.
    """
    examples, query = parse_prompt(prompt)
    if not examples or not query:
        return None
    if not is_symbol_row(examples, query):
        return None

    info = classify_row(examples, query, answer)
    if info is None:
        return None

    # --- Delexicalization ---
    smap = build_symbol_map(examples, query, answer)
    sym_line = symbol_map_line(smap)

    # --- Tag line ---
    seen_val = "1" if info["seen_op"] else "0"
    tag_line = (
        f"[TASK=SYMBOL] [SEEN_OP={seen_val}] [WORLD={info['world']}] "
        f"[FAMILY={info['family']}] [LEN={info['length']}] [SIGN={info['sign']}]"
    )

    # Canonical label for query operator
    q_op = info["q_op"]
    q_op_label = smap.get(q_op, q_op)

    # --- Evidence lines ---
    evidence = []

    # 1. Query op seen/unseen (use canonical label)
    if info["seen_op"]:
        evidence.append(
            f"Query op '{q_op_label}' seen in {info['same_op_count']} examples."
        )
    else:
        evidence.append(
            f"Query op '{q_op_label}' not seen in examples."
        )

    # 2. Same-op output lengths (only if there are same-op examples)
    if info["same_op_lengths"]:
        lengths_str = ", ".join(str(l) for l in info["same_op_lengths"])
        evidence.append(f"Same-op outputs: lengths {lengths_str}.")

    # 3. Edit test
    if info["is_edit"]:
        evidence.append("Output preserves input order -> EDIT.")
    else:
        evidence.append("Output has new symbols -> not EDIT.")

    # 4. World test
    if info["novel_symbols"]:
        novel_labels = ", ".join(
            sorted(smap.get(ch, ch) for ch in info["novel_symbols"])
        )
        evidence.append(
            f"Output needs symbols outside prompt ({novel_labels}) -> OPEN."
        )
    else:
        evidence.append("All output symbols in prompt vocabulary -> CLOSED.")

    # --- Canonical answer ---
    answer_canonical = delex_answer(answer, smap)

    # --- Assemble trace ---
    lines = [tag_line, sym_line, ""]
    lines.append("Evidence:")
    for ev in evidence:
        lines.append(f"- {ev}")

    lines.append("")
    lines.append(f"Answer: {answer_canonical}")

    trace = "\n".join(lines)
    return f"<think>\n{trace}\n</think>\n{format_answer_block(answer)}"


# ---------------------------------------------------------------------------
# Pool generation
# ---------------------------------------------------------------------------

def generate_latent_pool(train_csv_path: str, output_path: str) -> Dict:
    """Read train.csv, generate latent-tag traces for all symbol transformation rows.

    Returns dict with stats.
    """
    stats = {
        "total_symbol": 0,
        "traced": 0,
        "skipped_parse_fail": 0,
        "skipped_roundtrip_fail": 0,
        "used_text_fallback": 0,
        "family_counts": Counter(),
        "seen_op_counts": Counter(),
        "world_counts": Counter(),
        "sign_counts": Counter(),
        "trace_word_counts": [],
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

            assistant_content = build_latent_trace(prompt, answer)
            if assistant_content is None:
                stats["skipped_parse_fail"] += 1
                continue

            # Roundtrip check: extraction must recover the answer.
            # Brace-unsafe answers use "The final answer is:" format,
            # so we check with the inference-matching extractor.
            if answer_needs_text_fallback(answer):
                expected_block = f"The final answer is: {answer}"
                if expected_block not in assistant_content:
                    stats["skipped_roundtrip_fail"] += 1
                    continue
            else:
                extracted = _extract_boxed(assistant_content)
                if extracted != answer:
                    stats["skipped_roundtrip_fail"] += 1
                    continue

            # Classify for stats
            info = classify_row(examples, query, answer)
            if info:
                stats["family_counts"][info["family"]] += 1
                stats["seen_op_counts"]["seen" if info["seen_op"] else "unseen"] += 1
                stats["world_counts"][info["world"]] += 1
                stats["sign_counts"][info["sign"]] += 1

            # Count words in trace (between <think> tags)
            think_match = re.search(r"<think>\n(.*?)\n</think>", assistant_content, re.DOTALL)
            if think_match:
                words = len(think_match.group(1).split())
                stats["trace_word_counts"].append(words)

            stats["traced"] += 1
            if answer_needs_text_fallback(answer):
                stats["used_text_fallback"] += 1

            full_prompt = prompt + BOXED_INSTRUCTION
            record = {
                "messages": [
                    {"role": "user", "content": full_prompt},
                    {"role": "assistant", "content": assistant_content},
                ],
                "answer": answer,
                "id": row_id,
                "puzzle_type": "transformation",
                "mode": "symbol_latent",
                "generator": "gen_symbol_latent",
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
        description="Generate latent-tag control traces for symbol transformation"
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
            "symbol_latent.jsonl",
        ),
    )
    args = parser.parse_args()

    stats = generate_latent_pool(args.train_csv, args.output)

    print(f"Total symbol rows: {stats['total_symbol']}")
    print(f"Traced: {stats['traced']}")
    print(f"  (using text fallback for braces: {stats['used_text_fallback']})")
    print(f"Skipped (parse fail): {stats['skipped_parse_fail']}")
    print(f"Skipped (roundtrip fail): {stats['skipped_roundtrip_fail']}")

    print(f"\n--- Family breakdown ---")
    for fam, count in stats["family_counts"].most_common():
        print(f"  {fam}: {count}")

    print(f"\n--- SEEN_OP breakdown ---")
    for key, count in stats["seen_op_counts"].most_common():
        print(f"  {key}: {count}")

    print(f"\n--- WORLD breakdown ---")
    for key, count in stats["world_counts"].most_common():
        print(f"  {key}: {count}")

    print(f"\n--- SIGN breakdown ---")
    for key, count in stats["sign_counts"].most_common():
        print(f"  {key}: {count}")

    if stats["trace_word_counts"]:
        avg_words = sum(stats["trace_word_counts"]) / len(stats["trace_word_counts"])
        min_words = min(stats["trace_word_counts"])
        max_words = max(stats["trace_word_counts"])
        print(f"\n--- Trace length (words) ---")
        print(f"  avg: {avg_words:.1f}, min: {min_words}, max: {max_words}")

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
