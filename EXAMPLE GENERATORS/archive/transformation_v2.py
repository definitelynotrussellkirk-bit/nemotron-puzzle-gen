"""Coverage-oriented symbolic transformation generator.

This generator builds examples from templates, not random operands. It can
enumerate the full latent program space:
  - base
  - operator combo
  - per-operator output reversal
  - per-operator operator-as-sign behavior

Input reversal is already encoded inside the operation menu (`R:*`).
"""

import random
from itertools import product as iprod

from solvers.transformation_csp import K, OPS, solve_symbol_details


# Exclude '-' because it is also the negative sign when opsign=False.
SYMBOL_POOL = list("!@#$%^&*()_+=[]{}|;':\",./<>?\\`~")
_NEGATIVE_CAPABLE = {}
_REVERSAL_CAPABLE = {}


def enumerate_modifier_assignments(n_operators):
    """Yield every `(rev_outs, opsigns)` assignment for `n_operators`."""
    for rev_outs in iprod([False, True], repeat=n_operators):
        for opsigns in iprod([False, True], repeat=n_operators):
            yield tuple(rev_outs), tuple(opsigns)


def _encode_val(val, val_to_sym, base, op_char, rev_out, opsign):
    neg = val < 0
    val = abs(val)
    if val == 0:
        return val_to_sym[0]

    digits = []
    while val > 0:
        digits.append(val_to_sym[val % base])
        val //= base
    digits.reverse()
    if rev_out:
        digits = list(reversed(digits))
    s = "".join(digits)
    if neg:
        s = (op_char if opsign else "-") + s
    return s


def _encode_2digit(val, val_to_sym, base):
    high = val // base
    low = val % base
    return val_to_sym[high] + val_to_sym[low]


def _ordered_values(base):
    max_val = base * base - 1
    preferred = [
        0, 1, 2, 3,
        base - 1, base, base + 1, base + 2,
        2 * base - 1, 2 * base, 2 * base + 1,
        max(0, base * base // 2 - 1), base * base // 2, min(max_val, base * base // 2 + 1),
        max(0, max_val - base - 1), max(0, max_val - base), max(0, max_val - 1), max_val,
    ]
    seen = set()
    for val in preferred:
        if 0 <= val <= max_val and val not in seen:
            seen.add(val)
            yield val
    for val in range(max_val + 1):
        if val not in seen:
            yield val


def _negative_capable(base, op_idx):
    key = (base, op_idx)
    if key in _NEGATIVE_CAPABLE:
        return _NEGATIVE_CAPABLE[key]
    _, op_fn = OPS[op_idx]
    capable = False
    for left_val in range(base * base):
        for right_val in range(base * base):
            try:
                if op_fn(left_val, right_val, base) < 0:
                    capable = True
                    break
            except Exception:
                continue
        if capable:
            break
    _NEGATIVE_CAPABLE[key] = capable
    return capable


def _rev_observable_capable(base, op_idx):
    key = (base, op_idx)
    if key in _REVERSAL_CAPABLE:
        return _REVERSAL_CAPABLE[key]
    _, op_fn = OPS[op_idx]
    capable = False
    for left_val in range(base * base):
        for right_val in range(base * base):
            try:
                result_val = op_fn(left_val, right_val, base)
            except Exception:
                continue
            if _nonpal_multidigit(result_val, base):
                capable = True
                break
        if capable:
            break
    _REVERSAL_CAPABLE[key] = capable
    return capable


def _latent_config_is_satisfiable(base, op_combo, rev_outs, opsigns):
    for op_idx, rev_out, opsign in zip(op_combo, rev_outs, opsigns):
        if opsign and not _negative_capable(base, op_idx):
            return False
        if rev_out and not _rev_observable_capable(base, op_idx):
            return False
    return True


def _make_example(left_val, right_val, op_fn, val_to_sym, base, op_char, rev_out, opsign):
    try:
        result_val = op_fn(left_val, right_val, base)
    except Exception:
        return None
    return (
        _encode_2digit(left_val, val_to_sym, base),
        _encode_2digit(right_val, val_to_sym, base),
        _encode_val(result_val, val_to_sym, base, op_char, rev_out, opsign),
        result_val,
    )


def _strip_sign(result_str, op_char):
    if result_str and result_str[0] in {op_char, "-"} and len(result_str) > 1:
        return result_str[1:]
    return result_str


def _nonpal_multidigit(result_val, base):
    if result_val < 0 or result_val < base:
        return False
    digits = []
    value = result_val
    while value > 0:
        digits.append(value % base)
        value //= base
    digits.reverse()
    return digits != list(reversed(digits))


def _add_rendered_example(examples, seen, op_char, rendered):
    if rendered is None:
        return False
    left_str, right_str, result_str, _ = rendered
    key = (left_str, op_char, right_str, result_str)
    if key in seen:
        return False
    seen.add(key)
    examples.append(key)
    return True


def _find_pair(op_fn, base, predicate, val_to_sym, op_char, rev_out, opsign):
    for left_val in _ordered_values(base):
        for right_val in _ordered_values(base):
            rendered = _make_example(
                left_val,
                right_val,
                op_fn,
                val_to_sym,
                base,
                op_char,
                rev_out,
                opsign,
            )
            if rendered is None:
                continue
            if predicate(left_val, right_val, rendered[3]):
                return rendered
    return None


def _find_swap_pair(op_fn, base, val_to_sym, op_char, rev_out, opsign):
    fallback = None
    for left_val in _ordered_values(base):
        for right_val in _ordered_values(base):
            if left_val == right_val:
                continue
            forward = _make_example(
                left_val, right_val, op_fn, val_to_sym, base, op_char, rev_out, opsign
            )
            backward = _make_example(
                right_val, left_val, op_fn, val_to_sym, base, op_char, rev_out, opsign
            )
            if forward is None or backward is None:
                continue
            if fallback is None:
                fallback = (forward, backward)
            if forward[3] != backward[3]:
                return forward, backward
    return fallback


def _build_operator_examples(op_fn, base, val_to_sym, op_char, rev_out, opsign):
    examples = []
    seen = set()

    swap_pair = _find_swap_pair(op_fn, base, val_to_sym, op_char, rev_out, opsign)
    if swap_pair is None:
        return None
    _add_rendered_example(examples, seen, op_char, swap_pair[0])
    _add_rendered_example(examples, seen, op_char, swap_pair[1])

    diagonal = _find_pair(
        op_fn,
        base,
        lambda left_val, right_val, result_val: left_val == right_val and left_val > 0,
        val_to_sym,
        op_char,
        rev_out,
        opsign,
    )
    if diagonal is None:
        return None
    _add_rendered_example(examples, seen, op_char, diagonal)

    zero_right = _find_pair(
        op_fn,
        base,
        lambda left_val, right_val, result_val: right_val == 0 and left_val != 0,
        val_to_sym,
        op_char,
        rev_out,
        opsign,
    )
    zero_left = _find_pair(
        op_fn,
        base,
        lambda left_val, right_val, result_val: left_val == 0 and right_val != 0,
        val_to_sym,
        op_char,
        rev_out,
        opsign,
    )
    if zero_right is None or zero_left is None:
        return None
    _add_rendered_example(examples, seen, op_char, zero_right)
    _add_rendered_example(examples, seen, op_char, zero_left)

    one_right = _find_pair(
        op_fn,
        base,
        lambda left_val, right_val, result_val: right_val == 1 and left_val not in {0, 1},
        val_to_sym,
        op_char,
        rev_out,
        opsign,
    )
    one_left = _find_pair(
        op_fn,
        base,
        lambda left_val, right_val, result_val: left_val == 1 and right_val not in {0, 1},
        val_to_sym,
        op_char,
        rev_out,
        opsign,
    )
    if one_right is None or one_left is None:
        return None
    _add_rendered_example(examples, seen, op_char, one_right)
    _add_rendered_example(examples, seen, op_char, one_left)

    if opsign:
        negative = _find_pair(
            op_fn,
            base,
            lambda left_val, right_val, result_val: result_val < 0,
            val_to_sym,
            op_char,
            rev_out,
            opsign,
        )
        if negative is None:
            return None
        _add_rendered_example(examples, seen, op_char, negative)

    if rev_out:
        reversed_witness = _find_pair(
            op_fn,
            base,
            lambda left_val, right_val, result_val: _nonpal_multidigit(result_val, base),
            val_to_sym,
            op_char,
            rev_out,
            opsign,
        )
        if reversed_witness is None:
            return None
        _add_rendered_example(examples, seen, op_char, reversed_witness)

    return examples


def _collect_seen_digit_symbols(examples, op_symbols):
    seen = set()
    for left_str, op_char, right_str, result_str in examples:
        seen.update(left_str)
        seen.update(right_str)
        seen.update(_strip_sign(result_str, op_char))
    return seen - set(op_symbols)


def _cover_missing_digits(examples, op_combo, op_symbols, rev_outs, opsigns, base, val_to_sym, sym_to_val):
    seen = _collect_seen_digit_symbols(examples, op_symbols)
    digit_symbols = set(sym_to_val)
    missing = digit_symbols - seen
    if not missing:
        return examples

    seen_examples = set(examples)
    op_char = op_symbols[0]
    op_idx = op_combo[0]
    _, op_fn = OPS[op_idx]
    rev_out = rev_outs[0]
    opsign = opsigns[0]

    for miss_sym in sorted(missing):
        miss_val = sym_to_val[miss_sym]
        added = False
        candidate_inputs = []
        for other_digit in _ordered_values(base):
            if other_digit >= base:
                break
            candidate_inputs.append(miss_val * base + other_digit)
            candidate_inputs.append(other_digit * base + miss_val)

        for left_val in candidate_inputs:
            if left_val < 0 or left_val >= base * base:
                continue
            for right_val in _ordered_values(base):
                rendered = _make_example(
                    left_val,
                    right_val,
                    op_fn,
                    val_to_sym,
                    base,
                    op_char,
                    rev_out,
                    opsign,
                )
                if _add_rendered_example(examples, seen_examples, op_char, rendered):
                    added = True
                    break
            if added:
                break
        if not added:
            return None

    final_seen = _collect_seen_digit_symbols(examples, op_symbols)
    if digit_symbols - final_seen:
        return None
    return examples


def _check_observability(examples, op_symbols, rev_outs, opsigns):
    for op_i, op_char in enumerate(op_symbols):
        op_examples = [example for example in examples if example[1] == op_char]
        if not op_examples:
            return False

        if opsigns[op_i]:
            if not any(result_str.startswith(op_char) for _, _, _, result_str in op_examples):
                return False

        if rev_outs[op_i]:
            found = False
            for _, _, _, result_str in op_examples:
                clean = _strip_sign(result_str, op_char)
                if len(clean) >= 2 and clean != clean[::-1]:
                    found = True
                    break
            if not found:
                return False
    return True


def _build_prompt(examples, query_str):
    example_lines = [f"{left}{op}{right} = {result}" for left, op, right, result in examples]
    return (
        "In Alice's Wonderland, a secret set of transformation rules "
        "is applied to equations. Below are a few examples:\n"
        + "\n".join(example_lines)
        + f"\nNow, determine the result for: {query_str}"
    )


def generate_puzzle_for_config(
    base,
    op_combo,
    n_operators,
    rev_outs,
    opsigns,
    seed=42,
    verify_mode="answer",
    max_retries=4,
):
    """Generate one symbolic puzzle for a specific latent config."""
    if not _latent_config_is_satisfiable(base, op_combo, rev_outs, opsigns):
        return None

    rng = random.Random(seed)
    n_total = base + n_operators
    if n_total > len(SYMBOL_POOL):
        return None

    for _ in range(max_retries):
        chosen = rng.sample(SYMBOL_POOL, n_total)
        digit_syms = chosen[:base]
        op_syms = chosen[base : base + n_operators]

        perm = list(range(base))
        rng.shuffle(perm)
        sym_to_val = {digit_syms[i]: perm[i] for i in range(base)}
        val_to_sym = {perm[i]: digit_syms[i] for i in range(base)}

        examples = []
        ok = True
        for op_i in range(n_operators):
            op_char = op_syms[op_i]
            _, op_fn = OPS[op_combo[op_i]]
            built = _build_operator_examples(
                op_fn,
                base,
                val_to_sym,
                op_char,
                rev_outs[op_i],
                opsigns[op_i],
            )
            if built is None:
                ok = False
                break
            examples.extend(built)

        if not ok:
            continue

        examples = _cover_missing_digits(
            examples,
            op_combo,
            op_syms,
            rev_outs,
            opsigns,
            base,
            val_to_sym,
            sym_to_val,
        )
        if examples is None:
            continue
        if not _check_observability(examples, op_syms, rev_outs, opsigns):
            continue

        seen_inputs = {(left, op, right) for left, op, right, _ in examples}
        for op_i in range(n_operators):
            op_char = op_syms[op_i]
            _, op_fn = OPS[op_combo[op_i]]
            for left_val in _ordered_values(base):
                for right_val in _ordered_values(base):
                    rendered = _make_example(
                        left_val,
                        right_val,
                        op_fn,
                        val_to_sym,
                        base,
                        op_char,
                        rev_outs[op_i],
                        opsigns[op_i],
                    )
                    if rendered is None:
                        continue
                    left_str, right_str, answer, _ = rendered
                    query_key = (left_str, op_char, right_str)
                    if query_key in seen_inputs:
                        continue

                    prompt = _build_prompt(examples, f"{left_str}{op_char}{right_str}")
                    if verify_mode is not None:
                        solved = solve_symbol_details(prompt, mode=verify_mode, known_answer=answer)
                        if solved is None:
                            continue

                    return {
                        "prompt": prompt,
                        "answer": answer,
                        "query": f"{left_str}{op_char}{right_str}",
                        "examples": list(examples),
                        "n_examples": len(examples),
                        "config": {
                            "base": base,
                            "n_operators": n_operators,
                            "op_combo": op_combo,
                            "rev_outs": rev_outs,
                            "opsigns": opsigns,
                            "op_names": [OPS[i][0] for i in op_combo],
                            "op_symbols": list(op_syms),
                            "sym_to_val": sym_to_val,
                        },
                    }
        continue

    return None


def generate_full_coverage(
    bases=None,
    n_operators_list=None,
    seed=42,
    max_per_base=None,
    enumerate_modifiers=True,
    verify_mode="answer",
):
    """Yield one puzzle per latent configuration.

    When `enumerate_modifiers=True`, the latent config includes both `rev_out`
    and `opsign` flags. Generation raises on the first uncovered config so that
    gaps are surfaced instead of silently skipped.
    """
    if bases is None:
        bases = [8, 9, 10, 11]
    if n_operators_list is None:
        n_operators_list = [2, 3]

    rng = random.Random(seed)

    for base in bases:
        for n_ops in n_operators_list:
            count = 0
            modifier_assignments = (
                list(enumerate_modifier_assignments(n_ops))
                if enumerate_modifiers
                else [
                    (
                        tuple(rng.random() < 0.5 for _ in range(n_ops)),
                        tuple(rng.random() < 0.5 for _ in range(n_ops)),
                    )
                ]
            )
            for op_combo in iprod(range(K), repeat=n_ops):
                for rev_outs, opsigns in modifier_assignments:
                    if not _latent_config_is_satisfiable(base, op_combo, rev_outs, opsigns):
                        continue
                    puzzle = generate_puzzle_for_config(
                        base=base,
                        op_combo=op_combo,
                        n_operators=n_ops,
                        rev_outs=rev_outs,
                        opsigns=opsigns,
                        seed=rng.randint(0, 2**31 - 1),
                        verify_mode=verify_mode,
                    )
                    if puzzle is None:
                        raise RuntimeError(
                            "uncovered latent config: "
                            f"base={base} n_ops={n_ops} "
                            f"ops={[OPS[i][0] for i in op_combo]} "
                            f"rev={rev_outs} opsign={opsigns}"
                        )
                    yield puzzle
                    count += 1
                    if max_per_base and count >= max_per_base:
                        break
                if max_per_base and count >= max_per_base:
                    break


if __name__ == "__main__":
    generated = 0
    for puzzle in generate_full_coverage(
        bases=[10],
        n_operators_list=[2],
        seed=7,
        max_per_base=8,
    ):
        generated += 1
        print(
            f"{generated}: base={puzzle['config']['base']} "
            f"ops={puzzle['config']['op_names']} "
            f"rev={puzzle['config']['rev_outs']} "
            f"opsign={puzzle['config']['opsigns']} "
            f"query={puzzle['query']} answer={puzzle['answer']}"
        )
