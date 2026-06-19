#!/usr/bin/env python3
"""Retrace symbol competition rows using the factorized solver.

For each symbol row in train.csv where the solver gets the correct answer,
generate a high-quality reasoning trace.

Usage:
    python3 -m generators.retrace_symbol_competition
"""

import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from solvers.symbol_factorized import (
    ALL_STATES, _build_op_candidates, _infer_regime, _state_score,
    _merge_maps, _answer_from_mapping, _check, _Timeout, OPS_BY_NAME,
    solve_symbol_factorized,
)
from solvers.transformation_csp import _parse_symbol_problem, _digits_of
from training.data import BOXED_INSTRUCTION

OUTPUT = Path("data/transformation/symbol_competition.jsonl")
TRAIN_CSV = Path("data/train.csv")


def _is_symbol(prompt):
    for line in prompt.split('\n'):
        if '=' in line and 'determine' not in line.lower():
            left = line.split('=')[0].strip()
            if left and not any(c.isdigit() for c in left[:4]):
                return True
    return False


def _build_trace(prompt, gold, problem, base, op_candidates, regime_hint):
    """Build a 4-step reasoning trace for a symbol puzzle."""
    ops_list = problem['ops_list']
    digit_syms = sorted(problem['digit_syms'])
    q_l, q_op, q_r = problem['q_l'], problem['q_op'], problem['q_r']

    lines = []
    lines.append(f"Equation rules. Base {base}, {len(digit_syms)} symbols.")

    # Step 1: Identify structure
    lines.append("")
    lines.append("Step 1: Identify structure.")
    lines.append(f"Inputs: 5 chars each (2 digit-symbols, 1 operator, 2 digit-symbols)")
    lines.append(f"Unique digit symbols: {len(digit_syms)} → base-{base} encoding")
    lines.append(f"Operators: {len(ops_list)} ({', '.join(ops_list)})")

    # Output length analysis
    out_lens = []
    neg_count = 0
    has_opsign = False
    for ex in problem['parsed']:
        out = ex[3]
        if out and (out[0] == '-' or out[0] == ex[1]):
            out_lens.append(len(out) - 1)
            if out[0] != '-':
                has_opsign = True
            neg_count += 1
        else:
            out_lens.append(len(out))

    lines.append(f"Output lengths: {out_lens} (base-{base} digit counts)")
    if neg_count > 0:
        sign_type = "operator" if has_opsign else "minus"
        lines.append(f"Negative outputs: {neg_count}/{len(problem['parsed'])} (sign marker: {sign_type})")

    # Step 2: Infer modifiers
    lines.append("")
    lines.append("Step 2: Infer shared modifiers.")
    if regime_hint:
        ri, ro = regime_hint
        parts = []
        if ri:
            parts.append("rev_input")
        if ro:
            parts.append("rev_output")
        if has_opsign:
            parts.append("opsign")
        lines.append(f"Modifier regime: {' + '.join(parts) if parts else 'plain'}")
    else:
        lines.append("Modifier regime: testing plain first, then rev_input")

    # Step 3: Mapping and operators
    lines.append("")
    lines.append("Step 3: Mapping and operators.")
    lines.append(f"Symbol pool: {', '.join(digit_syms)}")
    lines.append(f"Each symbol maps to a unique digit 0-{base-1} (bijection)")

    # Show top state per operator
    for op_char in ops_list:
        if op_char in op_candidates and op_candidates[op_char]:
            best = op_candidates[op_char][0]
            name, rev_out, opsign = best['state']
            n_maps = best['n_maps']
            desc = name
            if rev_out:
                desc += ", rev_output"
            if opsign:
                desc += ", opsign"
            if n_maps == 1:
                lines.append(f"  {op_char} = {desc} (unique solution)")
            else:
                lines.append(f"  {op_char} = {desc} (best of {n_maps} consistent mappings)")

    q_seen = q_op in op_candidates
    if not q_seen:
        lines.append(f"  {q_op} = unseen in examples — infer from modifier regime")

    # Step 4: Compute query
    lines.append("")
    lines.append(f"Step 4: Compute {problem['query']}")
    lines.append(f"  Applying consistent mapping to query operands...")
    lines.append(f"  Result: {gold}")

    trace_text = "\n".join(lines)
    trace = f"<think>\n{trace_text}\n\n\\boxed{{{gold}}}\n</think>"
    return trace


def main():
    # Load competition rows
    with open(TRAIN_CSV) as f:
        reader = csv.reader(f)
        next(reader)
        rows = [(r[0], r[1], r[2]) for r in reader
                if 'transformation rules' in r[1][:120]]

    symbol_rows = [(r, p, g) for r, p, g in rows if _is_symbol(p)]
    print(f"Symbol competition rows: {len(symbol_rows)}")

    results = []
    correct = 0
    failed = 0
    t0 = time.time()

    for i, (rid, prompt, gold) in enumerate(symbol_rows):
        # Quick solve check
        ans = solve_symbol_factorized(prompt, deadline_s=3.0)
        if ans != gold:
            failed += 1
            continue

        correct += 1

        # Build detailed trace
        problem = _parse_symbol_problem(prompt)
        if problem is None:
            failed += 1
            continue

        # Find the base that worked
        deadline = time.perf_counter() + 5.0
        best_base = None
        best_op_candidates = None
        best_regime = None

        for base in problem.get('candidate_bases', [problem['base']]):
            problem['base'] = base
            try:
                op_candidates = _build_op_candidates(problem, base, deadline)
            except _Timeout:
                break
            if not op_candidates:
                continue

            regime_hint = _infer_regime(op_candidates)

            # Trim
            per_op_cap = 12
            for oc in list(op_candidates.keys()):
                cands = sorted(
                    op_candidates[oc],
                    key=lambda c: _state_score(c['state'], regime_hint) / (1.0 + math.log1p(c['n_maps'])),
                    reverse=True,
                )
                op_candidates[oc] = cands[:per_op_cap]

            best_base = base
            best_op_candidates = op_candidates
            best_regime = regime_hint
            break  # use first viable base

        if best_base is None or best_op_candidates is None:
            failed += 1
            continue

        trace = _build_trace(prompt, gold, problem, best_base,
                            best_op_candidates, best_regime)

        full_prompt = prompt + BOXED_INSTRUCTION
        results.append(json.dumps({
            "messages": [
                {"role": "user", "content": full_prompt},
                {"role": "assistant", "content": trace},
            ],
            "id": rid,
            "answer": gold,
            "puzzle_type": "transformation",
            "mode": "symbol_competition_traced",
            "generator": "retrace_symbol_factorized_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(symbol_rows)}: {correct} traced, {failed} failed ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nDone: {correct} traced, {failed} failed ({elapsed:.0f}s)")

    # Write
    with open(OUTPUT, 'w') as f:
        for line in results:
            f.write(line + '\n')
    print(f"Written: {OUTPUT} ({len(results)} examples)")


if __name__ == "__main__":
    main()
