#!/usr/bin/env python3
"""Generate symbol transformation training data (v3 — interpreter format).

Every trace is a tiny base-b interpreter:
  Use: → Check: (1 diagnostic support) → Query:

No meta-analysis. No family names as macros. Reversal shown on digit strings.
Executable operator formulas. Compact digit map (only symbols used).

Usage:
    python3 -m generators.gen_symbol_transformation --n 5000 --output data/transformation/pool/symbol/symbol.jsonl
"""

import argparse
import json
import random
from training.data import BOXED_INSTRUCTION, format_answer_block
from solvers.transformation_ops import ARITHMETIC_OPS, OP_DESCRIPTIONS
from solvers.transformation_csp import OPS_BY_NAME


# Full symbol pool including {} — brace-unsafe answers use text fallback
SYMBOL_POOL = list("!#$%&'()*+,-/:;<=>?@[\\]^`|~{}")
OP_POOL = list("!@#$%^&*()-+=[]|;:<>?/\\~`")


def _digits_of(val, base):
    if val == 0:
        return [0]
    ds = []
    v = abs(val)
    while v > 0:
        ds.append(v % base)
        v //= base
    ds.reverse()
    return ds


def _rev2(v, base):
    """Reverse a 2-digit base-b number: swap high and low digits."""
    return (v % base) * base + (v // base)


def _digit_str(hi, lo):
    """Format two digits as a string like '74'."""
    return f"{hi}{lo}"


def _is_diagnostic_symbol(use_rev_input, use_rev_output, use_opsign,
                          a_val, b_val, compute_a, compute_b, result, base):
    """Check if a support row exposes the rule for a 1B model."""
    # For rev_input: reversal should actually change the value
    if use_rev_input and a_val == compute_a and b_val == compute_b:
        return False
    # For opsign: prefer a negative result to show the sign rule
    if use_opsign and result >= 0:
        return False  # not ideal, but don't require it
    # For rev_output: result should have >1 digit and not be palindromic
    if use_rev_output:
        digits = _digits_of(abs(result), base)
        if len(digits) > 1 and digits == list(reversed(digits)):
            return False
    # Output shouldn't be trivially zero
    if result == 0:
        return False
    return True


def generate_symbol_puzzle(rng, base=None, mode=None):
    """Generate one symbol transformation puzzle with interpreter-style trace."""
    if base is None:
        base = rng.choices([8, 9, 10, 11], weights=[6, 25, 56, 13])[0]

    if base > len(SYMBOL_POOL):
        return None

    symbols = rng.sample(SYMBOL_POOL, base)
    sym_to_digit = {s: i for i, s in enumerate(symbols)}
    digit_to_sym = {i: s for i, s in enumerate(symbols)}

    available_ops = [c for c in OP_POOL if c not in symbols]
    if mode == 'sparse':
        n_ops = 1
    elif mode == 'unseen_op':
        n_ops = rng.choice([2, 3])
    else:
        n_ops = rng.choices([1, 2, 3], weights=[5, 49, 46])[0]
    op_syms = rng.sample(available_ops, min(n_ops, len(available_ops)))

    all_ops = [(n, f) for n, f, w in ARITHMETIC_OPS]
    for extra_name in ['concat', 'bconcat', 'interleave', 'rev_interleave', 'cross_swap', 'perm0321']:
        fn = OPS_BY_NAME.get(extra_name)
        if fn:
            all_ops.append((extra_name, fn))

    regime = rng.choices(
        ['plain', 'rev_both', 'rev_input', 'rev_output'],
        weights=[52, 35, 9, 4]
    )[0]
    use_rev_input = regime in ('rev_both', 'rev_input')
    use_rev_output = regime in ('rev_both', 'rev_output')
    use_opsign = rng.random() < 0.13

    if mode == 'negative':
        use_opsign = True
        neg_ops = [(n, f) for n, f, w in ARITHMETIC_OPS if n in ('sub', 'bsub', 'mulsub1', 'mulsuba', 'mulsubb')]
        if not neg_ops:
            neg_ops = all_ops

    op_assignments = {}
    for op_sym in op_syms:
        if mode == 'negative':
            op_name, op_fn = rng.choice(neg_ops)
        else:
            op_name, op_fn = rng.choice(all_ops)
        op_assignments[op_sym] = (op_name, op_fn)

    # Generate examples
    examples = []
    n_examples = rng.choices([3, 4, 5], weights=[236, 301, 286])[0]
    for _ in range(n_examples * 3):
        if len(examples) >= n_examples:
            break

        if mode == 'unseen_op':
            usable_ops = op_syms[:-1] if len(op_syms) > 1 else op_syms
        else:
            usable_ops = op_syms

        op_sym = rng.choice(usable_ops)
        op_name, op_fn = op_assignments[op_sym]

        a_hi, a_lo = rng.randrange(base), rng.randrange(base)
        b_hi, b_lo = rng.randrange(base), rng.randrange(base)
        a_val = a_hi * base + a_lo
        b_val = b_hi * base + b_lo

        compute_a = _rev2(a_val, base) if use_rev_input else a_val
        compute_b = _rev2(b_val, base) if use_rev_input else b_val

        try:
            result = op_fn(compute_a, compute_b)
        except:
            continue

        neg = result < 0
        if neg and not use_opsign:
            sign_char = '-'
        elif neg and use_opsign:
            sign_char = op_sym
        else:
            sign_char = None

        abs_result = abs(result)
        digits = _digits_of(abs_result, base)

        if use_rev_output:
            digits = list(reversed(digits))

        if any(d >= base for d in digits):
            continue

        inp = digit_to_sym[a_hi] + digit_to_sym[a_lo] + op_sym + digit_to_sym[b_hi] + digit_to_sym[b_lo]
        out_chars = ''.join(digit_to_sym[d] for d in digits)
        if sign_char:
            out_chars = sign_char + out_chars

        examples.append({
            'inp': inp, 'out': out_chars, 'op_sym': op_sym, 'op_name': op_name,
            'a_hi': a_hi, 'a_lo': a_lo, 'b_hi': b_hi, 'b_lo': b_lo,
            'a_val': a_val, 'b_val': b_val,
            'compute_a': compute_a, 'compute_b': compute_b,
            'result': result,
        })

    if len(examples) < 2:
        return None

    # Generate query
    if mode == 'unseen_op' and len(op_syms) > 1:
        q_op_sym = op_syms[-1]
    else:
        q_op_sym = rng.choice(op_syms)

    q_op_name, q_op_fn = op_assignments[q_op_sym]
    qa_hi, qa_lo = rng.randrange(base), rng.randrange(base)
    qb_hi, qb_lo = rng.randrange(base), rng.randrange(base)
    qa_val = qa_hi * base + qa_lo
    qb_val = qb_hi * base + qb_lo
    compute_qa = _rev2(qa_val, base) if use_rev_input else qa_val
    compute_qb = _rev2(qb_val, base) if use_rev_input else qb_val

    try:
        q_result = q_op_fn(compute_qa, compute_qb)
    except:
        return None

    q_neg = q_result < 0
    q_abs = abs(q_result)
    q_digits = _digits_of(q_abs, base)
    if use_rev_output:
        q_digits_out = list(reversed(q_digits))
    else:
        q_digits_out = list(q_digits)
    if any(d >= base for d in q_digits_out):
        return None

    query = digit_to_sym[qa_hi] + digit_to_sym[qa_lo] + q_op_sym + digit_to_sym[qb_hi] + digit_to_sym[qb_lo]
    answer_chars = ''.join(digit_to_sym[d] for d in q_digits_out)
    if q_neg:
        sign_char = q_op_sym if use_opsign else '-'
        answer_chars = sign_char + answer_chars

    # Build prompt
    rng.shuffle(examples)
    prompt_lines = [f"{e['inp']} = {e['out']}" for e in examples]
    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules "
        "is applied to equations. Below are a few examples:\n"
        + "\n".join(prompt_lines)
        + f"\nNow, determine the result for: {query}"
    )

    # === Build interpreter-style trace ===
    q_desc = OP_DESCRIPTIONS.get(q_op_name, q_op_name)

    # Collect all symbols used in check + query for compact digit map
    def _used_syms(hi, lo, digits_out):
        used = {hi, lo}
        for d in digits_out:
            used.add(d)
        return used

    # Find diagnostic support — MUST use the same operator as the query
    # so Use: compute and Check: computation agree.
    best_ex = None
    # First: try examples with the same operator
    for ex in examples:
        if ex['op_sym'] == q_op_sym and _is_diagnostic_symbol(
                use_rev_input, use_rev_output, use_opsign,
                ex['a_val'], ex['b_val'],
                ex['compute_a'], ex['compute_b'],
                ex['result'], base):
            best_ex = ex
            break
    # If no diagnostic match with same op, take any with same op
    if best_ex is None:
        for ex in examples:
            if ex['op_sym'] == q_op_sym:
                best_ex = ex
                break
    # If query op is unseen (not in any example), use first diagnostic example
    # and label the check with its own operator
    if best_ex is None:
        for ex in examples:
            if _is_diagnostic_symbol(use_rev_input, use_rev_output, use_opsign,
                                      ex['a_val'], ex['b_val'],
                                      ex['compute_a'], ex['compute_b'],
                                      ex['result'], base):
                best_ex = ex
                break
    if best_ex is None:
        best_ex = examples[0]

    # Determine if check uses same op as query
    check_uses_same_op = (best_ex['op_sym'] == q_op_sym)

    # Gather digits used in check + query for compact map
    used_digits = set()
    for d in [best_ex['a_hi'], best_ex['a_lo'], best_ex['b_hi'], best_ex['b_lo']]:
        used_digits.add(d)
    s_result = best_ex['result']
    s_digits = _digits_of(abs(s_result), base)
    if use_rev_output:
        s_digits_out = list(reversed(s_digits))
    else:
        s_digits_out = list(s_digits)
    for d in s_digits_out:
        used_digits.add(d)
    for d in [qa_hi, qa_lo, qb_hi, qb_lo]:
        used_digits.add(d)
    for d in q_digits_out:
        used_digits.add(d)

    # Build compact digit map (only used symbols)
    map_parts = ", ".join(f'"{digit_to_sym[d]}"={d}' for d in sorted(used_digits))

    # Input/output transform descriptions
    input_desc = "reverse base-{} digit strings".format(base) if use_rev_input else "plain"
    output_desc = "reverse output digit string" if use_rev_output else "plain"
    sign_desc = f"prefix operator on negative" if use_opsign else "none"

    trace_lines = ["Use:"]
    trace_lines.append(f"  base = {base}")
    trace_lines.append(f"  digits: {map_parts}")
    trace_lines.append(f"  input = {input_desc}")
    trace_lines.append(f"  compute = {q_desc}")
    if use_opsign or any(e['result'] < 0 for e in examples):
        trace_lines.append(f"  sign = {sign_desc}")
    if use_rev_output:
        trace_lines.append(f"  output = {output_desc}")

    # === Helper to build one row (check or query) ===
    def _build_row(a_hi, a_lo, b_hi, b_lo, a_val, b_val, compute_a, compute_b,
                   result, op_desc, is_check=False):
        row = []
        left_sym = digit_to_sym[a_hi] + digit_to_sym[a_lo]
        right_sym = digit_to_sym[b_hi] + digit_to_sym[b_lo]
        left_digits = _digit_str(a_hi, a_lo)
        right_digits = _digit_str(b_hi, b_lo)

        row.append(f"  left = \"{left_sym}\" → \"{left_digits}\"_{base} = {a_val}")
        row.append(f"  right = \"{right_sym}\" → \"{right_digits}\"_{base} = {b_val}")

        if use_rev_input:
            rev_left = _digit_str(a_lo, a_hi)
            rev_right = _digit_str(b_lo, b_hi)
            row.append(f"  rev input: \"{left_digits}\" → \"{rev_left}\"_{base} = {compute_a}")
            row.append(f"  rev input: \"{right_digits}\" → \"{rev_right}\"_{base} = {compute_b}")

        row.append(f"  {op_desc}: {compute_a}, {compute_b} = {result}")

        neg = result < 0
        magnitude = abs(result)
        mag_digits = _digits_of(magnitude, base)

        if use_rev_output and len(mag_digits) > 0:
            mag_digits_str = ','.join(str(d) for d in mag_digits)
            rev_digits = list(reversed(mag_digits))
            rev_digits_str = ','.join(str(d) for d in rev_digits)
            row.append(f"  magnitude digits = [{mag_digits_str}]_{base}")
            row.append(f"  rev output = [{rev_digits_str}]_{base}")
            encode_digits = rev_digits
        else:
            digits_str = ','.join(str(d) for d in mag_digits)
            row.append(f"  magnitude digits = [{digits_str}]_{base}")
            encode_digits = mag_digits

        encoded = ''.join(digit_to_sym[d] for d in encode_digits)
        if neg:
            if use_opsign:
                row.append(f"  negative → prefix operator → {q_op_sym}{encoded}")
                encoded = q_op_sym + encoded
            else:
                row.append(f"  negative → -{encoded}")
                encoded = '-' + encoded
        else:
            row.append(f"  encode → {encoded}")

        return row, encoded

    # Check: one diagnostic support
    trace_lines.append("")
    ex = best_ex
    ex_op_desc = OP_DESCRIPTIONS.get(ex['op_name'], ex['op_name'])
    if check_uses_same_op:
        trace_lines.append("Check:")
    else:
        # Label the check with its own operator so there's no contradiction
        trace_lines.append(f"Check (operator {ex['op_sym']} = {ex_op_desc}):")
    check_lines, check_encoded = _build_row(
        ex['a_hi'], ex['a_lo'], ex['b_hi'], ex['b_lo'],
        ex['a_val'], ex['b_val'], ex['compute_a'], ex['compute_b'],
        ex['result'], ex_op_desc, is_check=True
    )
    trace_lines.extend(check_lines)
    trace_lines[-1] += " → MATCH"

    # Query: same shape
    trace_lines.append("")
    trace_lines.append("Query:")
    q_lines, q_encoded = _build_row(
        qa_hi, qa_lo, qb_hi, qb_lo,
        qa_val, qb_val, compute_qa, compute_qb,
        q_result, q_desc
    )
    trace_lines.extend(q_lines)

    reasoning = "\n".join(trace_lines)
    answer_block = format_answer_block(answer_chars)
    trace = f"<think>\n{reasoning}\n</think>\n{answer_block}"

    base_mode = f"symbol_{mode or 'regular'}"
    tagged_mode = f"{base_mode}_answer"

    from datetime import datetime, timezone
    return {
        "messages": [
            {"role": "user", "content": prompt + BOXED_INSTRUCTION},
            {"role": "assistant", "content": trace},
        ],
        "answer": answer_chars,
        "id": f"gen_symbol_trans_{rng.randint(0,999999):06d}",
        "puzzle_type": "transformation",
        "mode": tagged_mode,
        "generator": "gen_symbol_transformation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--output", type=str, default="data/transformation/pool/symbol/symbol.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    mode_weights = {
        None: 50,
        'unseen_op': 20,
        'negative': 20,
        'sparse': 10,
    }
    modes = list(mode_weights.keys())
    weights = list(mode_weights.values())

    count = 0
    with open(args.output, "w") as out:
        for _ in range(args.n * 3):
            if count >= args.n:
                break
            mode = rng.choices(modes, weights=weights)[0]
            result = generate_symbol_puzzle(rng, mode=mode)
            if result is None:
                continue
            out.write(json.dumps(result) + "\n")
            count += 1
            if count % 1000 == 0:
                print(f"  {count}/{args.n}")

    print(f"Done: {count} symbol transformation examples → {args.output}")


if __name__ == "__main__":
    main()
