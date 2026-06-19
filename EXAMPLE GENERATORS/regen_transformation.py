#!/usr/bin/env python3
"""Regenerate ALL competition transformation traces in scan-reject-lock format.

Replaces old structural alignment and equation traces with:
- Numeric: scan-reject-lock (visible digits)
- Cipher-digit: crack-scan-lock-encode (all symbols)

Usage:
    python3 -m generators.regen_transformation
"""

import csv
import json
import os
import signal
import time
from datetime import datetime, timezone

from generators.trace_transform import build_numeric_trace, build_cipher_trace
from generators.gen_transform_one_shot_rows import build_rows as build_one_shot_rows
from generators.gen_transform_numeric_ambiguity_rows import build_rows as build_numeric_ambiguity_rows
from generators.gen_transform_numeric_single_witness_rows import build_rows as build_numeric_single_witness_rows
from generators.gen_transform_numeric_unseen_prior_rows import build_rows as build_numeric_unseen_prior_rows
from generators.gen_transform_cipher_slot_rule_rows import build_rows as build_cipher_slot_rule_rows
from generators.gen_transform_cipher_query_local_rows import build_rows as build_cipher_query_local_rows
from generators.gen_transform_cipher_single_witness_rows import build_rows as build_cipher_single_witness_rows
from generators.gen_transform_cipher_unseen_slot_prior_rows import build_rows as build_cipher_unseen_slot_prior_rows
from generators.gen_transform_cipher_hard_way_rows import build_rows as build_cipher_hard_way_rows
from generators.gen_transform_cipher_missing_symbol_rows import build_rows as build_cipher_missing_symbol_rows
from generators.gen_transform_cipher_direct_support_prior_rows import build_rows as build_cipher_direct_support_prior_rows
from solvers.cipher_digit import solve as cipher_solve, find_op_pos
from training.data import BOXED_INSTRUCTION


class _TraceTimeout(BaseException):
    """Timeout that is not swallowed by broad `except Exception` solver code."""


def _is_transformation_prompt(prompt: str) -> bool:
    """Identify equation-transformation rows without catching bit prompts."""
    head = prompt[:240].lower()
    return (
        "secret set of transformation rules" in head
        and "determine the result for:" in prompt.lower()
    )


def regen_all(train_csv="data/competition/train.csv",
              output="data/transformation/pool/competition/competition_traced.jsonl",
              one_shot_first="data/transformation/pool/competition/one_shot_numeric_first.jsonl",
              one_shot_unique="data/transformation/pool/competition/one_shot_numeric_unique.jsonl",
              one_shot_summary="data/transformation/pool/competition/one_shot_numeric.summary.json",
              numeric_ambiguity="data/transformation/pool/competition/numeric_direct_ambiguity_prior.jsonl",
              numeric_ambiguity_summary="data/transformation/pool/competition/numeric_direct_ambiguity_prior.summary.json",
              numeric_single_witness="data/transformation/pool/competition/numeric_single_witness_prior.jsonl",
              numeric_single_witness_summary="data/transformation/pool/competition/numeric_single_witness_prior.summary.json",
              numeric_unseen_prior="data/transformation/pool/competition/numeric_unseen_operator_prior.jsonl",
              numeric_unseen_prior_summary="data/transformation/pool/competition/numeric_unseen_operator_prior.summary.json",
              cipher_slot_rule="data/transformation/pool/competition/cipher_visible_slot_rule.jsonl",
              cipher_slot_rule_summary="data/transformation/pool/competition/cipher_visible_slot_rule.summary.json",
              cipher_query_local="data/transformation/pool/competition/cipher_query_op_local_rule.jsonl",
              cipher_query_local_summary="data/transformation/pool/competition/cipher_query_op_local_rule.summary.json",
              cipher_single_witness="data/transformation/pool/competition/cipher_single_witness_local_prior.jsonl",
              cipher_single_witness_summary="data/transformation/pool/competition/cipher_single_witness_local_prior.summary.json",
              cipher_unseen_slot_prior="data/transformation/pool/competition/cipher_unseen_answer_space_prior.jsonl",
              cipher_unseen_slot_prior_summary="data/transformation/pool/competition/cipher_unseen_answer_space_prior.summary.json",
              cipher_hard_way="data/transformation/pool/competition/cipher_hard_way_prior.jsonl",
              cipher_hard_way_summary="data/transformation/pool/competition/cipher_hard_way_prior.summary.json",
              cipher_missing_symbol="data/transformation/pool/competition/cipher_missing_symbol_prior.jsonl",
              cipher_missing_symbol_summary="data/transformation/pool/competition/cipher_missing_symbol_prior.summary.json",
              cipher_direct_support_prior="data/transformation/pool/competition/cipher_direct_support_answer_space_prior.jsonl",
              cipher_direct_support_prior_summary="data/transformation/pool/competition/cipher_direct_support_answer_space_prior.summary.json",
              cipher_timeout=1.0):
    """Regenerate all stable transformation competition traces."""

    os.makedirs(os.path.dirname(output), exist_ok=True)

    results = []
    total = numeric = cipher = old_format = failed = 0
    t0 = time.time()

    with open(train_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row['prompt']
            if not _is_transformation_prompt(prompt):
                continue

            total += 1
            answer = row['answer']
            rid = row[list(row.keys())[0]]

            lines = prompt.split('\n')
            ex_lines = [l.strip() for l in lines
                       if '=' in l and not l.startswith('Now') and not l.startswith('In ')]
            if not ex_lines:
                failed += 1
                continue

            examples = [(ex.split('=')[0].strip(), ex.split('=', 1)[1].strip())
                       for ex in ex_lines]

            q_lines = [l for l in prompt.split('\n') if 'determine' in l.lower()]
            query = q_lines[0].split(':', 1)[-1].strip() if q_lines else ''

            first_lhs = examples[0][0]
            is_cipher = len(first_lhs) == 5 and not any(c.isdigit() for c in first_lhs)

            trace_text = None
            pred = None
            mode = "unknown"

            if is_cipher:
                # Visible-only cipher map. The stored answer may filter after
                # prediction, but it must not constrain the mapping search.
                def _timeout_handler(signum, frame):
                    raise _TraceTimeout()
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(10)
                try:
                    result = cipher_solve(prompt, answer=None, timeout=cipher_timeout)
                except (_TraceTimeout, Exception):
                    result = None
                signal.alarm(0)
                if result and result['answer'] == answer:
                    # Build cipher trace
                    op_pos = result.get("op_pos") or find_op_pos(examples, query, None)
                    if op_pos is not None:
                        dpos = [i for i in range(5) if i != op_pos]
                        ct = build_cipher_trace(
                            examples, query, answer,
                            result['mapping'], result.get('combos', {}),
                            result.get('maj_op', '?'), op_pos
                        )
                        if ct:
                            trace_text, pred = ct
                            mode = "cipher_digit"
                            cipher += 1
            else:
                # Try numeric trace (full scan with examples)
                import random
                result = build_numeric_trace(examples, query, answer, random.Random(hash(rid)))
                if result:
                    trace_text, pred = result
                    if pred == answer:
                        mode = "numeric_scan"
                        numeric += 1
                    else:
                        trace_text = None

                # If full scan failed, try gold-only: just find combo for query
                if trace_text is None and answer:
                    from generators.trace_transform import (
                        _make_operands, _calc, _fmt, SCAN_ORDER, COMBO_DISPLAY
                    )
                    q_op_pos = None
                    for i, c in enumerate(query):
                        if not c.isdigit() and c not in ' ':
                            q_op_pos = i; break
                    if q_op_pos is not None:
                        try:
                            q_a = int(query[:q_op_pos])
                            q_b = int(query[q_op_pos+1:])
                            q_op = query[q_op_pos]
                            targets = [answer]
                            if len(answer) >= 2 and not answer[0].isdigit() and answer[0] != '-':
                                targets.append(answer[1:])
                            # Skip gold-only traces — they teach execution-after-oracle,
                            # not derivable search. Model can't learn to find the combo.
                            pass
                        except (ValueError, IndexError):
                            pass

            if trace_text is None:
                # Skip — fallback traces teach nothing and waste training slots
                old_format += 1
                if total % 200 == 0:
                    dt = time.time() - t0
                    print(f"  {total}: numeric={numeric} cipher={cipher} fallback={old_format} ({dt:.0f}s)",
                          flush=True)
                continue

            if "???" in trace_text or "???" in answer:
                old_format += 1
                continue

            results.append({
                "messages": [
                    {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": f"<think>\n{trace_text}\n</think>\n" + (f"The final answer is: {answer}" if ('}' in answer or '{' in answer) else f"\\boxed{{{answer}}}")},
                ],
                "answer": answer,
                "id": rid,
                "puzzle_type": "transformation",
                "mode": mode,
                "generator": "regen_transformation",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })

            if total % 200 == 0:
                dt = time.time() - t0
                print(f"  {total}: numeric={numeric} cipher={cipher} fallback={old_format} ({dt:.0f}s)",
                      flush=True)

    # Save
    with open(output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    dt = time.time() - t0
    print(f"\nDone in {dt:.0f}s → {output}")
    print(f"  Total: {total}")
    print(f"  Numeric (scan-reject-lock): {numeric}")
    print(f"  Cipher-digit (crack-scan-lock): {cipher}")
    print(f"  Fallback (old/minimal): {old_format}")
    print(f"  Failed: {failed}")

    one_shot = build_one_shot_rows(
        train_csv=train_csv,
        out_first=one_shot_first,
        out_unique=one_shot_unique,
        summary_out=one_shot_summary,
    )
    one_shot_counts = one_shot.get("counters", {})
    print("\nNumeric one-shot stable subset:")
    print(f"  First-order rows: {one_shot_counts.get('first_order_rows', 0)}")
    print(f"  Unique-answer rows: {one_shot_counts.get('unique_rows', 0)}")
    print(f"  Summary: {one_shot_summary}")

    numeric_ambiguity_result = build_numeric_ambiguity_rows(
        train_csv=train_csv,
        out=numeric_ambiguity,
        summary_out=numeric_ambiguity_summary,
    )
    numeric_ambiguity_counts = numeric_ambiguity_result.get("counters", {})
    print("\nNumeric direct ambiguity prior subset:")
    print(f"  Rows: {numeric_ambiguity_counts.get('rows', 0)}")
    print(f"  Summary: {numeric_ambiguity_summary}")

    numeric_single_witness_result = build_numeric_single_witness_rows(
        train_csv=train_csv,
        out=numeric_single_witness,
        summary_out=numeric_single_witness_summary,
    )
    numeric_single_witness_counts = numeric_single_witness_result.get("counters", {})
    print("\nNumeric single-witness prior subset:")
    print(f"  Rows: {numeric_single_witness_counts.get('rows', 0)}")
    print(f"  Summary: {numeric_single_witness_summary}")

    numeric_unseen_result = build_numeric_unseen_prior_rows(
        train_csv=train_csv,
        out=numeric_unseen_prior,
        summary_out=numeric_unseen_prior_summary,
    )
    numeric_unseen_counts = numeric_unseen_result.get("counters", {})
    print("\nNumeric unseen-operator prior subset:")
    print(f"  Rows: {numeric_unseen_counts.get('rows', 0)}")
    print(f"  Summary: {numeric_unseen_prior_summary}")

    cipher_slot_result = build_cipher_slot_rule_rows(
        train_csv=train_csv,
        out=cipher_slot_rule,
        summary_out=cipher_slot_rule_summary,
    )
    cipher_slot_counts = cipher_slot_result.get("counters", {})
    print("\nCipher visible-slot rule subset:")
    print(f"  Rows: {cipher_slot_counts.get('rows', 0)}")
    print(f"  Summary: {cipher_slot_rule_summary}")

    cipher_local_result = build_cipher_query_local_rows(
        train_csv=train_csv,
        out=cipher_query_local,
        summary_out=cipher_query_local_summary,
        timeout=1.0,
    )
    cipher_local_counts = cipher_local_result.get("counters", {})
    print("\nCipher query-op local rule subset:")
    print(f"  Rows: {cipher_local_counts.get('rows', 0)}")
    print(f"  Summary: {cipher_query_local_summary}")

    cipher_single_result = build_cipher_single_witness_rows(
        train_csv=train_csv,
        out=cipher_single_witness,
        summary_out=cipher_single_witness_summary,
        timeout=0.2,
    )
    cipher_single_counts = cipher_single_result.get("counters", {})
    print("\nCipher single-witness local-prior subset:")
    print(f"  Rows: {cipher_single_counts.get('rows', 0)}")
    print(f"  Summary: {cipher_single_witness_summary}")

    cipher_unseen_slot = build_cipher_unseen_slot_prior_rows(
        train_csv=train_csv,
        out=cipher_unseen_slot_prior,
        summary_out=cipher_unseen_slot_prior_summary,
    )
    cipher_unseen_slot_counts = cipher_unseen_slot.get("counters", {})
    print("\nCipher unseen query-slot prior subset:")
    print(f"  Rows: {cipher_unseen_slot_counts.get('rows', 0)}")
    print(f"  Summary: {cipher_unseen_slot_prior_summary}")

    cipher_hard = build_cipher_hard_way_rows(
        train_csv=train_csv,
        out=cipher_hard_way,
        summary_out=cipher_hard_way_summary,
        timeout=cipher_timeout,
    )
    cipher_hard_counts = cipher_hard.get("counters", {})
    print("\nCipher best-fit fallback subset:")
    print(f"  Rows: {cipher_hard_counts.get('rows', 0)}")
    print(f"  Summary: {cipher_hard_way_summary}")

    cipher_missing = build_cipher_missing_symbol_rows(
        train_csv=train_csv,
        out=cipher_missing_symbol,
        summary_out=cipher_missing_symbol_summary,
        timeout=cipher_timeout,
    )
    cipher_missing_counts = cipher_missing.get("counters", {})
    print("\nCipher missing-symbol prior subset:")
    print(f"  Rows: {cipher_missing_counts.get('rows', 0)}")
    print(f"  Summary: {cipher_missing_symbol_summary}")

    cipher_direct_support = build_cipher_direct_support_prior_rows(
        train_csv=train_csv,
        out=cipher_direct_support_prior,
        summary_out=cipher_direct_support_prior_summary,
    )
    cipher_direct_support_counts = cipher_direct_support.get("counters", {})
    print("\nCipher direct-support answer-space prior subset:")
    print(f"  Rows: {cipher_direct_support_counts.get('rows', 0)}")
    print(f"  Summary: {cipher_direct_support_prior_summary}")


def perturb_and_retrace(train_csv="data/competition/train.csv",
                        output="data/transformation/pool/generated/perturbed_traces.jsonl",
                        n_per_row=2, max_rows=500):
    """Generate augmented training rows by perturbing solved competition rows.

    Takes a solved row, swaps operand digits while keeping the same combo/mapping,
    retraces with the active trace builder. Produces on-manifold diversity.
    """
    import random as _rng_mod

    os.makedirs(os.path.dirname(output), exist_ok=True)
    results = []
    rng = _rng_mod.Random(42)
    t0 = time.time()

    with open(train_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(results) >= max_rows:
                break
            prompt = row['prompt']
            if not _is_transformation_prompt(prompt):
                continue
            answer = row['answer']

            lines = prompt.split('\n')
            ex_lines = [l.strip() for l in lines
                       if '=' in l and not l.startswith('Now') and not l.startswith('In ')]
            if not ex_lines:
                continue
            examples = [(ex.split('=')[0].strip(), ex.split('=', 1)[1].strip())
                       for ex in ex_lines]
            q_lines = [l for l in prompt.split('\n') if 'determine' in l.lower()]
            query = q_lines[0].split(':', 1)[-1].strip() if q_lines else ''
            if not query:
                continue

            first_lhs = examples[0][0]
            is_cipher = len(first_lhs) == 5 and not any(c.isdigit() for c in first_lhs)

            if is_cipher:
                continue  # cipher perturbation is harder, skip for now

            # Numeric: perturb by swapping operand digits
            for _ in range(n_per_row):
                new_examples = []
                for lhs, rhs_orig in examples:
                    # Find operator position
                    op_pos_found = None
                    for i, c in enumerate(lhs):
                        if not c.isdigit() and c not in ' ':
                            op_pos_found = i
                            break
                    if op_pos_found is None:
                        break
                    op_c = lhs[op_pos_found]
                    # Perturb: swap one digit randomly
                    new_lhs = list(lhs)
                    digit_positions = [i for i in range(len(lhs)) if i != op_pos_found and lhs[i].isdigit()]
                    if digit_positions:
                        pos = rng.choice(digit_positions)
                        new_lhs[pos] = str(rng.randint(1, 9))
                    new_lhs_str = ''.join(new_lhs)
                    new_examples.append((new_lhs_str, ''))  # rhs will be computed by trace builder

                if len(new_examples) != len(examples):
                    continue

                # Perturb query too
                new_query_list = list(query)
                q_digit_pos = [i for i in range(len(query)) if query[i].isdigit()]
                if q_digit_pos:
                    pos = rng.choice(q_digit_pos)
                    new_query_list[pos] = str(rng.randint(1, 9))
                new_query = ''.join(new_query_list)

                # Try to retrace with the perturbed examples
                # We need the trace builder to find the same combo
                result = build_numeric_trace(new_examples, new_query, None, rng)
                if result is None:
                    continue
                trace_text, pred = result
                if pred is None:
                    continue

                # Build perturbed prompt
                prompt_lines = [
                    "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
                    "Below are a few examples:",
                ]
                for lhs_new, _ in new_examples:
                    # Need to compute the actual RHS from the trace
                    pass

                # Actually, build_numeric_trace needs (lhs, rhs) pairs where rhs is known.
                # We can't perturb without knowing the combo. Skip rows where trace builder fails.
                # The trace builder WILL find the combo from perturbed examples if the same combo works.
                # But we passed rhs='' so it can't match. We need a different approach.
                # Instead: find the combo from the ORIGINAL examples, then apply to perturbed.
                break  # this approach needs rework

    # Simpler approach: use the ORIGINAL combo, generate new random operands
    results = []
    with open(train_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(results) >= max_rows:
                break
            prompt = row['prompt']
            if not _is_transformation_prompt(prompt):
                continue
            answer = row['answer']

            lines = prompt.split('\n')
            ex_lines = [l.strip() for l in lines
                       if '=' in l and not l.startswith('Now') and not l.startswith('In ')]
            if not ex_lines:
                continue
            examples = [(ex.split('=')[0].strip(), ex.split('=', 1)[1].strip())
                       for ex in ex_lines]
            q_lines = [l for l in prompt.split('\n') if 'determine' in l.lower()]
            query = q_lines[0].split(':', 1)[-1].strip() if q_lines else ''
            if not query:
                continue

            first_lhs = examples[0][0]
            is_cipher = len(first_lhs) == 5 and not any(c.isdigit() for c in first_lhs)
            if is_cipher:
                continue

            # First: solve the ORIGINAL to get the combo
            orig_result = build_numeric_trace(examples, query, answer, rng)
            if orig_result is None:
                continue

            # Extract operator chars and combo from the trace
            # Parse the original examples to get operator info
            parsed_ops = set()
            for lhs, rhs in examples:
                for c in lhs:
                    if not c.isdigit() and c not in ' ':
                        parsed_ops.add(c)
                        break

            # Now generate perturbed versions with new random operands
            for _ in range(n_per_row):
                new_examples = []
                for lhs, rhs in examples:
                    op_pos_found = None
                    for i, c in enumerate(lhs):
                        if not c.isdigit() and c not in ' ':
                            op_pos_found = i
                            break
                    if op_pos_found is None:
                        break
                    op_c = lhs[op_pos_found]
                    # New random operands, same operator
                    a_new = rng.randint(10, 99)
                    b_new = rng.randint(10, 99)
                    new_lhs = f"{a_new}{op_c}{b_new}"
                    new_examples.append((new_lhs, ''))  # rhs unknown for now

                if len(new_examples) != len(examples):
                    continue

                # New query with same operator
                q_op = None
                for c in query:
                    if not c.isdigit() and c not in ' ':
                        q_op = c
                        break
                if q_op is None:
                    continue
                qa_new = rng.randint(10, 99)
                qb_new = rng.randint(10, 99)
                new_query = f"{qa_new}{q_op}{qb_new}"

                # Try to trace — the trace builder will scan and find the same combo
                # if the original combo still works on the new operands
                # We pass answer=None so it uses the first match
                trace_result = build_numeric_trace(
                    [(lhs, '') for lhs, _ in new_examples],
                    new_query, None, rng)

                # That won't work because rhs is empty. We need to compute rhs.
                # Let me use a different approach: directly compute with the found combo.
                # Actually the trace builder needs rhs to match against.
                # The real fix: compute rhs from the combo we already know.

                # Parse the original trace to get the locked combo
                orig_trace = orig_result[0]
                import re
                lock_match = re.search(r'Lock\[.\]: ([A-Z,]+)\|(\w+)\|(\w+)', orig_trace)
                if not lock_match:
                    break
                order_map = {"BA,DC": "BA_DC", "AB,CD": "AB_CD", "AB,DC": "AB_DC", "BA,CD": "BA_CD"}
                locked_order = order_map.get(lock_match.group(1), lock_match.group(1))
                locked_op = lock_match.group(2)
                locked_fmt = lock_match.group(3)

                # Compute RHS for each new example
                from generators.trace_transform import _make_operands, _calc, _fmt
                new_examples_with_rhs = []
                valid = True
                for lhs_new, _ in new_examples:
                    op_idx = None
                    for i, c in enumerate(lhs_new):
                        if not c.isdigit():
                            op_idx = i
                            break
                    if op_idx is None:
                        valid = False
                        break
                    a_str = lhs_new[:op_idx]
                    b_str = lhs_new[op_idx+1:]
                    op_c = lhs_new[op_idx]
                    try:
                        a_val = int(a_str)
                        b_val = int(b_str)
                    except ValueError:
                        valid = False
                        break
                    L, R = _make_operands(a_val//10, a_val%10, b_val//10, b_val%10, locked_order)
                    val = _calc(L, R, locked_op)
                    if val is None:
                        valid = False
                        break
                    fval = _fmt(val, locked_fmt, op_char=op_c)
                    if fval is None:
                        valid = False
                        break
                    new_examples_with_rhs.append((lhs_new, str(fval)))

                if not valid or len(new_examples_with_rhs) != len(examples):
                    continue

                # Compute query answer
                qa_val, qb_val = qa_new, qb_new
                qL, qR = _make_operands(qa_val//10, qa_val%10, qb_val//10, qb_val%10, locked_order)
                qval = _calc(qL, qR, locked_op)
                if qval is None:
                    continue
                q_answer = _fmt(qval, locked_fmt, op_char=q_op)
                if q_answer is None:
                    continue

                # Now build trace with known correct examples + answer
                trace_result = build_numeric_trace(
                    new_examples_with_rhs, new_query, str(q_answer), rng)
                if trace_result is None:
                    continue
                trace_text, pred = trace_result
                if pred != str(q_answer):
                    continue

                # Build prompt
                prompt_lines = [
                    "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
                    "Below are a few examples:",
                ]
                for lhs_new, rhs_new in new_examples_with_rhs:
                    prompt_lines.append(f"{lhs_new} = {rhs_new}")
                prompt_lines.append(f"Now, determine the result for: {new_query}")
                new_prompt = "\n".join(prompt_lines)

                results.append({
                    "messages": [
                        {"role": "user", "content": new_prompt + BOXED_INSTRUCTION},
                        {"role": "assistant", "content": f"<think>\n{trace_text}\n</think>\n\\boxed{{{q_answer}}}"},
                    ],
                    "answer": str(q_answer),
                    "id": f"perturb_{row[list(row.keys())[0]]}_{rng.randint(0,9999):04d}",
                    "puzzle_type": "transformation",
                    "mode": "perturbed_numeric",
                    "source_id": row[list(row.keys())[0]],
                    "generator": "perturb_and_retrace",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                })

    with open(output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    dt = time.time() - t0
    print(f"Perturbed {len(results)} rows in {dt:.0f}s → {output}")


if __name__ == "__main__":
    regen_all()
