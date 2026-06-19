#!/usr/bin/env python3
"""Answer-conditioned retracer for numeric transformation rows.

For each numeric transformation row in train.csv:
1. Parse examples + query
2. Run candidate generation (CORE_OPS matching per operator)
3. Find the candidate that produces the gold answer
4. Render a trace using the gold-matching candidate

This gives correct traces even for rows where the solver currently
ranks the wrong candidate, because we use the gold answer to select.

Usage:
    python3 -m generators.retrace_numeric_explained [--limit N]
"""

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- solver imports ---
from solvers.transformation import (
    ALL_OPS,
    CORE_OPS,
    STYLES,
    _build_ops,
    _core_match_support,
    _get_modifiers,
    _is_numeric,
    _parse,
    _parse_numeric,
    _render_body,
    _split_name,
    solve_details,
)
from solvers.transformation_ops import ARITHMETIC_OPS, OP_DESCRIPTIONS, OPS_BY_NAME
from training.data import BOXED_INSTRUCTION

OUTPUT = Path("data/transformation/pool/competition/numeric_explained.jsonl")
TRAIN_CSV = Path("data/train.csv")

# Precompute style functions by name
_STYLE_DICT = dict(STYLES)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_by_op(examples):
    """Group parsed examples by operator character.
    Returns {op_char: [(a_str, b_str, out_str), ...]} and parsed query info.
    """
    by_op = {}
    for inp, out in examples:
        m = re.match(r'^(\d+)\s*(.)\s*(\d+)$', inp)
        if not m:
            return None
        a, op, b = m.group(1), m.group(2), m.group(3)
        by_op.setdefault(op, []).append((a, b, out))
    return by_op


# ---------------------------------------------------------------------------
# Answer-conditioned candidate search
# ---------------------------------------------------------------------------

def _find_gold_candidates_core(by_op, qa, qb, qop, gold_answer):
    """Find CORE_OPS candidates for each operator, conditioned on gold answer for query op.

    For each seen operator, finds all (core_name, core_fn, rev_in, rev_out, style) combos
    that match all support examples.

    For the query operator specifically, further filters to those that produce the gold answer.

    Returns:
        gold_matches: list of dicts with full candidate info for query op
        seen_op_map: {op_char: best_candidate_info} for non-query seen ops
    """
    # First: find all core candidates per operator
    all_core_cands = {}
    for op_char, entries in by_op.items():
        cands = []
        for core_name, core_fn, rev_in, rev_out in CORE_OPS:
            styles = _core_match_support(core_name, core_fn, rev_in, rev_out, entries, op_char)
            for style in styles:
                cands.append({
                    "core_name": core_name,
                    "core_fn": core_fn,
                    "rev_in": rev_in,
                    "rev_out": rev_out,
                    "style": style,
                })
        all_core_cands[op_char] = cands

    # Also find ALL_OPS flat matches per operator (fallback for concat etc)
    all_flat_cands = {}
    for op_char, entries in by_op.items():
        cands = []
        for name, fn in ALL_OPS:
            try:
                if all(fn(a, b, op_char) == out for a, b, out in entries):
                    cands.append({"name": name, "fn": fn})
            except Exception:
                pass
        all_flat_cands[op_char] = cands

    # Build modifier regime posterior from non-query ops
    mod_counts = {}
    for op_char, cands in all_flat_cands.items():
        if op_char == qop:
            continue
        for c in cands[:3]:
            n = c["name"]
            if "concat" not in n:
                mods = _get_modifiers(n)[:2]
                mod_counts[mods] = mod_counts.get(mods, 0) + 1

    # For query op: find candidates that produce the gold answer
    q_seen = qop in by_op and len(by_op.get(qop, [])) > 0

    gold_matches = []

    if q_seen:
        # Query op is in support: use core candidates
        for cand in all_core_cands.get(qop, []):
            a_val = int(qa[::-1]) if cand["rev_in"] else int(qa)
            b_val = int(qb[::-1]) if cand["rev_in"] else int(qb)
            try:
                raw = cand["core_fn"](a_val, b_val)
            except Exception:
                continue
            body, neg = _render_body(raw, cand["rev_out"])
            result = _STYLE_DICT[cand["style"]](body, neg, qop)
            if result == gold_answer:
                gold_matches.append({
                    **cand,
                    "result": result,
                    "raw": raw,
                    "a_val": a_val,
                    "b_val": b_val,
                    "body": body,
                    "neg": neg,
                    "is_unseen": False,
                })

        # Fallback: try ALL_OPS flat matches
        if not gold_matches:
            for c in all_flat_cands.get(qop, []):
                try:
                    result = c["fn"](qa, qb, qop)
                except Exception:
                    continue
                if result == gold_answer:
                    # Parse the name to extract core info
                    core, rev_in, rev_out, has_marker = _split_name(c["name"])
                    gold_matches.append({
                        "core_name": core,
                        "core_fn": OPS_BY_NAME.get(core),
                        "rev_in": rev_in,
                        "rev_out": rev_out,
                        "style": "plain",
                        "result": result,
                        "raw": None,
                        "a_val": int(qa[::-1]) if rev_in else int(qa),
                        "b_val": int(qb[::-1]) if rev_in else int(qb),
                        "body": result.lstrip("-" + qop),
                        "neg": result.startswith("-") or (result.startswith(qop) and qop not in "-0123456789"),
                        "flat_name": c["name"],
                        "flat_fn": c["fn"],
                        "is_unseen": False,
                    })
    else:
        # Query op is unseen: try ALL core ops with all styles
        for core_name, core_fn, rev_in, rev_out in CORE_OPS:
            a_val = int(qa[::-1]) if rev_in else int(qa)
            b_val = int(qb[::-1]) if rev_in else int(qb)
            try:
                raw = core_fn(a_val, b_val)
            except Exception:
                continue
            body, neg = _render_body(raw, rev_out)
            for style_name, style_fn in STYLES:
                result = style_fn(body, neg, qop)
                if result == gold_answer:
                    gold_matches.append({
                        "core_name": core_name,
                        "core_fn": core_fn,
                        "rev_in": rev_in,
                        "rev_out": rev_out,
                        "style": style_name,
                        "result": result,
                        "raw": raw,
                        "a_val": a_val,
                        "b_val": b_val,
                        "body": body,
                        "neg": neg,
                        "is_unseen": True,
                    })

        # Also try ALL_OPS flat for concat etc
        if not gold_matches:
            for name, fn in ALL_OPS:
                try:
                    result = fn(qa, qb, qop)
                except Exception:
                    continue
                if result == gold_answer:
                    core, rev_in, rev_out, has_marker = _split_name(name)
                    gold_matches.append({
                        "core_name": core,
                        "core_fn": None,
                        "rev_in": rev_in,
                        "rev_out": rev_out,
                        "style": "plain",
                        "result": result,
                        "raw": None,
                        "a_val": int(qa[::-1]) if rev_in else int(qa),
                        "b_val": int(qb[::-1]) if rev_in else int(qb),
                        "body": result,
                        "neg": False,
                        "flat_name": name,
                        "flat_fn": fn,
                        "is_unseen": True,
                    })

    # Pick best gold match: prefer regime-coherent candidate
    if gold_matches and mod_counts:
        def _regime_score(m):
            reg = (m["rev_in"], m["rev_out"])
            return mod_counts.get(reg, 0)
        gold_matches.sort(key=_regime_score, reverse=True)

    # Build seen_op_map for non-query operators
    seen_op_map = {}
    for op_char, cands in all_core_cands.items():
        if op_char == qop:
            continue
        if cands:
            seen_op_map[op_char] = cands[0]
        elif all_flat_cands.get(op_char):
            c = all_flat_cands[op_char][0]
            core, rev_in, rev_out, _ = _split_name(c["name"])
            seen_op_map[op_char] = {
                "core_name": core,
                "rev_in": rev_in,
                "rev_out": rev_out,
                "style": "plain",
                "flat_name": c["name"],
            }

    return gold_matches, seen_op_map


# ---------------------------------------------------------------------------
# Trace rendering
# ---------------------------------------------------------------------------

def _op_description(core_name, rev_in, rev_out, style):
    """Build a compact human-readable description of the operation."""
    desc = OP_DESCRIPTIONS.get(core_name, core_name)
    parts = []
    if rev_in:
        parts.append("rev_input")
    parts.append(desc)
    if rev_out:
        parts.append("rev_output")
    if style and style != "plain":
        parts.append(style)
    return ", ".join(parts)


def _compute_line(core_name, a_val, b_val, raw, rev_in, rev_out, style, qop,
                   qa_str, qb_str, gold_answer):
    """Build a compact computation line for the query."""
    # Show the arithmetic step
    desc = OP_DESCRIPTIONS.get(core_name, core_name)

    # Build arithmetic expression
    if core_name == "add":
        expr = f"{a_val}+{b_val}={raw}"
    elif core_name == "sub":
        expr = f"{a_val}-{b_val}={raw}"
    elif core_name == "bsub":
        expr = f"{b_val}-{a_val}={raw}"
    elif core_name == "mul":
        expr = f"{a_val}\u00d7{b_val}={raw}"
    elif core_name == "absdiff":
        expr = f"|{a_val}-{b_val}|={raw}"
    elif core_name == "negabsdiff":
        expr = f"-|{a_val}-{b_val}|={raw}"
    elif core_name == "add1":
        mid = a_val + b_val
        expr = f"{a_val}+{b_val}+1={mid}+1={raw}"
    elif core_name == "sub1":
        mid = a_val + b_val
        expr = f"{a_val}+{b_val}-1={mid}-1={raw}"
    elif core_name == "mul1":
        mid = a_val * b_val
        expr = f"{a_val}\u00d7{b_val}+1={mid}+1={raw}"
    elif core_name == "mulsub1":
        mid = a_val * b_val
        expr = f"{a_val}\u00d7{b_val}-1={mid}-1={raw}"
    elif core_name == "mula":
        mid = a_val * b_val
        expr = f"{a_val}\u00d7{b_val}+{a_val}={mid}+{a_val}={raw}"
    elif core_name == "mulb":
        mid = a_val * b_val
        expr = f"{a_val}\u00d7{b_val}+{b_val}={mid}+{b_val}={raw}"
    elif core_name == "mulab":
        mid = a_val * b_val
        expr = f"{a_val}\u00d7{b_val}+{a_val}+{b_val}={mid}+{a_val}+{b_val}={raw}"
    elif core_name == "mulsuba":
        mid = a_val * b_val
        expr = f"{a_val}\u00d7{b_val}-{a_val}={mid}-{a_val}={raw}"
    elif core_name == "mulsubb":
        mid = a_val * b_val
        expr = f"{a_val}\u00d7{b_val}-{b_val}={mid}-{b_val}={raw}"
    elif core_name == "mod":
        expr = f"{a_val}%{b_val}={raw}"
    elif core_name == "bmod":
        expr = f"{b_val}%{a_val}={raw}"
    elif core_name == "floordiv":
        expr = f"{a_val}\u00f7{b_val}={raw}"
    elif core_name == "bfloordiv":
        expr = f"{b_val}\u00f7{a_val}={raw}"
    elif core_name == "maxmod":
        mx, mn = max(a_val, b_val), min(a_val, b_val)
        expr = f"max({a_val},{b_val})%min({a_val},{b_val})={mx}%{mn}={raw}"
    elif core_name == "bitor":
        expr = f"{a_val}|{b_val}={raw}"
    elif core_name == "bitxor":
        expr = f"{a_val}^{b_val}={raw}"
    else:
        # Concat or unknown
        expr = f"{core_name}({a_val},{b_val})={raw}"

    return expr


def _verify_example(core_name, core_fn, rev_in, rev_out, style, a_str, b_str, out_str, op_char):
    """Build a compact verification line for one example."""
    a_val = int(a_str[::-1]) if rev_in else int(a_str)
    b_val = int(b_str[::-1]) if rev_in else int(b_str)
    try:
        raw = core_fn(a_val, b_val)
    except Exception:
        return f"{a_str}{op_char}{b_str}={out_str} \u2713"
    body, neg = _render_body(raw, rev_out)
    result = _STYLE_DICT[style](body, neg, op_char)
    if result == out_str:
        return f"{a_str}{op_char}{b_str}={out_str} \u2713"
    else:
        return f"{a_str}{op_char}{b_str}={out_str} (~{result})"


def render_trace(examples, query, gold_answer, gold_match, seen_op_map, by_op, is_unseen):
    """Render a unified-flow trace for a numeric transformation row.

    Format (from CLAUDE.md):
        Equation rules. Base 10.
        Ops:
          op1=description: verify_example check
          op2=description: verify_example check
        Query: computation
        \\boxed{answer}
    """
    qp = _parse_numeric(query)
    qa, qop, qb = qp

    lines = ["Equation rules. Base 10."]
    lines.append("Ops:")

    # Show all operators (seen ops from support + query op)
    shown_ops = set()

    # Seen operators (non-query)
    for op_char in sorted(by_op.keys()):
        if op_char == qop and not is_unseen:
            # Will be shown as part of query section
            pass
        cand = seen_op_map.get(op_char)
        if cand is None and op_char == qop:
            cand = gold_match
        if cand is None:
            continue

        desc = _op_description(
            cand["core_name"],
            cand.get("rev_in", False),
            cand.get("rev_out", False),
            cand.get("style", "plain"),
        )

        # Verify against first example for this operator
        entries = by_op.get(op_char, [])
        if entries and cand.get("core_fn") is not None:
            vline = _verify_example(
                cand["core_name"], cand["core_fn"],
                cand.get("rev_in", False), cand.get("rev_out", False),
                cand.get("style", "plain"),
                entries[0][0], entries[0][1], entries[0][2], op_char,
            )
            lines.append(f"  {op_char}={desc}: {vline}")
        else:
            # Flat name fallback
            flat = cand.get("flat_name", cand["core_name"])
            if entries:
                lines.append(f"  {op_char}={desc}: {entries[0][0]}{op_char}{entries[0][1]}={entries[0][2]} \u2713")
            else:
                lines.append(f"  {op_char}={desc}")
        shown_ops.add(op_char)

    # Query operator (if not already shown and it's a seen-op)
    if qop not in shown_ops and not is_unseen:
        desc = _op_description(
            gold_match["core_name"],
            gold_match.get("rev_in", False),
            gold_match.get("rev_out", False),
            gold_match.get("style", "plain"),
        )
        entries = by_op.get(qop, [])
        if entries and gold_match.get("core_fn") is not None:
            vline = _verify_example(
                gold_match["core_name"], gold_match["core_fn"],
                gold_match.get("rev_in", False), gold_match.get("rev_out", False),
                gold_match.get("style", "plain"),
                entries[0][0], entries[0][1], entries[0][2], qop,
            )
            lines.append(f"  {qop}={desc}: {vline}")
        else:
            if entries:
                lines.append(f"  {qop}={desc}: {entries[0][0]}{qop}{entries[0][1]}={entries[0][2]} \u2713")
            else:
                lines.append(f"  {qop}={desc}")

    # Query computation
    lines.append(f"Query: {query}")

    # Build computation detail
    gm = gold_match
    if gm.get("raw") is not None:
        # We have full core-op info
        rev_in = gm.get("rev_in", False)
        rev_out = gm.get("rev_out", False)
        style = gm.get("style", "plain")

        compute_parts = []
        if rev_in:
            compute_parts.append(f"rev({qa})={qa[::-1]}, rev({qb})={qb[::-1]}")

        expr = _compute_line(
            gm["core_name"], gm["a_val"], gm["b_val"], gm["raw"],
            rev_in, rev_out, style, qop, qa, qb, gold_answer,
        )
        compute_parts.append(expr)

        if rev_out:
            compute_parts.append(f"rev({gm['raw']})={str(abs(gm['raw']))[::-1]}")

        if style not in ("plain",) and gm.get("neg"):
            if style == "opsign":
                compute_parts.append(f"neg\u2192{qop}-prefix")
            elif style == "tailsign":
                compute_parts.append(f"neg\u2192{qop}-suffix")

        lines.append(f"  {' \u2192 '.join(compute_parts)}")
    else:
        # Flat-name fallback
        flat_name = gm.get("flat_name", gm["core_name"])
        lines.append(f"  {flat_name}({qa},{qb})={gold_answer}")

    # Note: \boxed{} is added by the caller outside <think> tags
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Answer-conditioned retracer for numeric transformation")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N rows (0=all)")
    args = parser.parse_args()

    # Load competition rows
    with open(TRAIN_CSV) as f:
        reader = csv.reader(f)
        next(reader)
        all_rows = [(r[0], r[1], r[2]) for r in reader
                     if "transformation rules" in r[1][:100]]

    # Filter to numeric only
    numeric_rows = []
    for rid, prompt, gold in all_rows:
        examples, query = _parse(prompt)
        if examples and query and _is_numeric(examples):
            numeric_rows.append((rid, prompt, gold, examples, query))

    print(f"Numeric transformation rows: {len(numeric_rows)}")
    if args.limit > 0:
        numeric_rows = numeric_rows[:args.limit]
        print(f"Processing first {args.limit} rows")

    # Run solver to check current accuracy
    results = []
    stats = {
        "total": 0,
        "gold_found": 0,
        "gold_not_found": 0,
        "solver_correct": 0,
        "solver_wrong_gold_found": 0,  # answer-conditioned wins
        "seen_op": 0,
        "unseen_op": 0,
        "seen_op_traced": 0,
        "unseen_op_traced": 0,
    }

    t0 = time.time()
    example_traces = {"seen": None, "unseen": None}

    for i, (rid, prompt, gold, examples, query) in enumerate(numeric_rows):
        stats["total"] += 1

        qp = _parse_numeric(query)
        if not qp:
            stats["gold_not_found"] += 1
            continue
        qa, qop, qb = qp

        # Group by operator
        by_op = _parse_by_op(examples)
        if by_op is None:
            stats["gold_not_found"] += 1
            continue

        q_seen = qop in by_op and len(by_op.get(qop, [])) > 0
        is_unseen = not q_seen

        if is_unseen:
            stats["unseen_op"] += 1
        else:
            stats["seen_op"] += 1

        # Check if solver gets it right
        details = solve_details(prompt)
        solver_correct = details is not None and details.get("answer") == gold

        if solver_correct:
            stats["solver_correct"] += 1

        # Find gold-matching candidates
        gold_matches, seen_op_map = _find_gold_candidates_core(
            by_op, qa, qb, qop, gold,
        )

        if not gold_matches:
            stats["gold_not_found"] += 1
            continue

        stats["gold_found"] += 1
        if not solver_correct:
            stats["solver_wrong_gold_found"] += 1

        gold_match = gold_matches[0]

        # Render trace
        trace_text = render_trace(
            examples, query, gold, gold_match, seen_op_map, by_op, is_unseen,
        )

        full_prompt = prompt + BOXED_INSTRUCTION
        mode = "answer_conditioned" if not solver_correct else "exact"

        if is_unseen:
            stats["unseen_op_traced"] += 1
        else:
            stats["seen_op_traced"] += 1

        # Store example traces for display
        trace_key = "unseen" if is_unseen else "seen"
        if example_traces[trace_key] is None:
            example_traces[trace_key] = (rid, gold, trace_text, mode)

        entry = {
            "messages": [
                {"role": "user", "content": full_prompt},
                {"role": "assistant", "content": f"<think>\n{trace_text}\n</think>\n\\boxed{{{gold}}}"},
            ],
            "id": rid,
            "answer": gold,
            "puzzle_type": "transformation",
            "mode": mode,
            "sub_type": "unseen_op" if is_unseen else "seen_op",
            "generator": "numeric_explained_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        results.append(json.dumps(entry))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(numeric_rows)}: {stats['gold_found']} traced ({elapsed:.0f}s)")

    elapsed = time.time() - t0

    # Print stats
    print(f"\n{'='*60}")
    print(f"RESULTS ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"Total numeric rows:       {stats['total']}")
    print(f"  Seen-op:                {stats['seen_op']}")
    print(f"  Unseen-op:              {stats['unseen_op']}")
    print(f"Gold found (traceable):   {stats['gold_found']}/{stats['total']} "
          f"({100*stats['gold_found']/max(stats['total'],1):.1f}%)")
    print(f"  Seen-op traced:         {stats['seen_op_traced']}")
    print(f"  Unseen-op traced:       {stats['unseen_op_traced']}")
    print(f"Gold not found:           {stats['gold_not_found']}")
    print(f"Solver already correct:   {stats['solver_correct']}")
    print(f"Answer-conditioned wins:  {stats['solver_wrong_gold_found']} "
          f"(rows solver gets wrong but gold found in candidates)")

    # Print example traces
    for label in ["seen", "unseen"]:
        ex = example_traces[label]
        if ex:
            rid, gold, trace, mode = ex
            print(f"\n{'='*60}")
            print(f"EXAMPLE TRACE ({label}-op, mode={mode}, id={rid}, gold={gold})")
            print(f"{'='*60}")
            print(trace)

    # Write output
    if results:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT, "w") as f:
            for line in results:
                f.write(line + "\n")
        print(f"\nWritten: {OUTPUT} ({len(results)} examples)")
    else:
        print("\nNo results to write.")


if __name__ == "__main__":
    main()
