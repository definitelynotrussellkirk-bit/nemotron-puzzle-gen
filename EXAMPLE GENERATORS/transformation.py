
"""Guided transformation generator.

This rewrite keeps the original competition-style prompt format but adds
three explicit numeric modes aimed at *deterministic* supervision:

- witness_minimal   : choose a hard query and greedily add examples that kill
                      wrong-at-query hypotheses until all survivors agree.
- collision_pair    : build rows around a specific near-collision pair
                      (e.g. sub vs absdiff) and add a killer example.
- withheld_query_op : keep the query operator out of the example block and use
                      anchor operators to establish a shared modifier regime;
                      the query is chosen from a small structured bank so the
                      answer is still unanimous.

The older random generator behavior is preserved as ``mode='random'``.
Guided modes are numeric-only on purpose; the symbol branch can still fall
back to an external generator if the package layout is available.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Callable, Iterable

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.append(str(_MODULE_DIR))

try:
    from .base import BaseGenerator
except Exception:
    class BaseGenerator:
        def __init__(self, seed: int | None = None):
            self.rng = random.Random(seed)

try:
    from solvers.transformation_ops import (
        ARITHMETIC_OPS,
        CONCAT_WEIGHT,
        BCONCAT_WEIGHT,
        MODIFIER_WEIGHTS,
        OPSIGN_PROBABILITY,
        SYMBOL_POOL,
    )
except Exception:
    from .transformation_ops import (
        ARITHMETIC_OPS,
        CONCAT_WEIGHT,
        BCONCAT_WEIGHT,
        MODIFIER_WEIGHTS,
        OPSIGN_PROBABILITY,
        SYMBOL_POOL,
    )


def _encode_number(
    val: int,
    base: int,
    digit_symbols: list[str],
    rev_output: bool,
    op_char: str,
    use_opsign: bool,
) -> str | None:
    negative = val < 0
    val = abs(val)

    if val == 0:
        if not digit_symbols:
            return None
        s = digit_symbols[0]
    else:
        chars: list[str] = []
        while val > 0:
            digit = val % base
            if digit >= len(digit_symbols):
                return None
            chars.append(digit_symbols[digit])
            val //= base
        s = ''.join(reversed(chars))

    if rev_output:
        s = s[::-1]
    if negative:
        s = (op_char if use_opsign else '-') + s
    return s


def _decode_number(s: str, base: int, symbol_to_value: dict[str, int], rev_input: bool) -> int:
    if rev_input:
        s = s[::-1]
    val = 0
    for c in s:
        val = val * base + symbol_to_value[c]
    return val


def _weighted_choice(rng: random.Random, weights: list[float]) -> int:
    total = sum(weights)
    r = rng.uniform(0, total)
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return i
    return max(0, len(weights) - 1)


def _unsigned_body(s: str) -> str:
    if s.startswith('-'):
        return s[1:]
    if s and s[0] in SYMBOL_POOL:
        return s[1:]
    return s


def _has_both_orderings(examples: list['_Example']) -> bool:
    has_lt = any(ex.decoded_a < ex.decoded_b for ex in examples)
    has_ge = any(ex.decoded_a >= ex.decoded_b for ex in examples)
    return has_lt and has_ge


@dataclass(frozen=True)
class _CandidateSpec:
    base_name: str
    fn: Callable[[int, int], int]
    prior_weight: float
    rev_input: bool = False
    rev_output: bool = False
    use_opsign: bool = False
    kind: str = 'arith'

    @property
    def modifier_signature(self) -> tuple[bool, bool]:
        return (self.rev_input, self.rev_output)

    @property
    def label(self) -> str:
        name = self.base_name
        if self.rev_input:
            name = f'rev_in({name})'
        if self.rev_output:
            name = f'rev_out({name})'
        if self.use_opsign:
            name = f'{name}|opsign'
        return name

    def apply(
        self,
        a_str: str,
        b_str: str,
        base: int,
        symbol_to_value: dict[str, int],
        digit_symbols: list[str],
        op_char: str,
    ) -> str | None:
        a_val = _decode_number(a_str, base, symbol_to_value, self.rev_input)
        b_val = _decode_number(b_str, base, symbol_to_value, self.rev_input)
        result_val = self.fn(a_val, b_val)
        return _encode_number(
            result_val,
            base,
            digit_symbols,
            self.rev_output,
            op_char,
            self.use_opsign,
        )


@dataclass
class _Example:
    op_char: str
    a_value: int
    b_value: int
    a_str: str
    b_str: str
    output_str: str
    decoded_a: int
    decoded_b: int

    @property
    def input_str(self) -> str:
        return f'{self.a_str}{self.op_char}{self.b_str}'


class TransformationGenerator(BaseGenerator):
    name = 'transformation'

    COLLISION_FAMILIES = [
        {
            'name': 'sub_vs_absdiff',
            'hidden': ('sub', False, False, False),
            'competitor': ('absdiff', False, False, False),
            'seed_filter': lambda ex, h, c, base, s2v, ds: ex.decoded_a >= ex.decoded_b,
            'killer_filter': lambda ex, h, c, base, s2v, ds: ex.decoded_a < ex.decoded_b,
        },
        {
            'name': 'bsub_vs_absdiff',
            'hidden': ('bsub', False, False, False),
            'competitor': ('absdiff', False, False, False),
            'seed_filter': lambda ex, h, c, base, s2v, ds: ex.decoded_a <= ex.decoded_b,
            'killer_filter': lambda ex, h, c, base, s2v, ds: ex.decoded_a > ex.decoded_b,
        },
        {
            'name': 'concat_vs_bconcat',
            'hidden': ('concat', False, False, False),
            'competitor': ('bconcat', False, False, False),
            'seed_filter': lambda ex, h, c, base, s2v, ds: ex.a_str == ex.b_str,
            'killer_filter': lambda ex, h, c, base, s2v, ds: ex.a_str != ex.b_str,
        },
        {
            'name': 'add_vs_revout_add',
            'hidden': ('add', False, True, False),
            'competitor': ('add', False, False, False),
            'seed_filter': lambda ex, h, c, base, s2v, ds: _unsigned_body(ex.output_str) == _unsigned_body(ex.output_str)[::-1],
            'killer_filter': lambda ex, h, c, base, s2v, ds: len(_unsigned_body(ex.output_str)) > 1 and _unsigned_body(ex.output_str) != _unsigned_body(ex.output_str)[::-1],
        },
        {
            'name': 'mul_vs_revout_mul',
            'hidden': ('mul', False, True, False),
            'competitor': ('mul', False, False, False),
            'seed_filter': lambda ex, h, c, base, s2v, ds: _unsigned_body(ex.output_str) == _unsigned_body(ex.output_str)[::-1],
            'killer_filter': lambda ex, h, c, base, s2v, ds: len(_unsigned_body(ex.output_str)) > 1 and _unsigned_body(ex.output_str) != _unsigned_body(ex.output_str)[::-1],
        },
        {
            'name': 'revadd_vs_add',
            'hidden': ('add', True, True, False),
            'competitor': ('add', False, False, False),
            'seed_filter': lambda ex, h, c, base, s2v, ds: ex.output_str == c.apply(ex.a_str, ex.b_str, base, s2v, ds, ex.op_char),
            'killer_filter': lambda ex, h, c, base, s2v, ds: ex.output_str != c.apply(ex.a_str, ex.b_str, base, s2v, ds, ex.op_char),
        },
    ]

    WITHHELD_BANKS = {
        'subtractive': ['sub', 'bsub', 'absdiff'],
        'concat': ['concat', 'bconcat'],
    }

    def __init__(
        self,
        seed: int | None = None,
        symbol_probability: float = 0.5,
        digit_permutation_probability: float = 0.07,
        default_mode: str = 'random',
        guided_allow_permutation: bool = False,
        max_tries: int = 120,
    ):
        super().__init__(seed)
        self.symbol_probability = symbol_probability
        self.digit_permutation_probability = digit_permutation_probability
        self.default_mode = default_mode
        self.guided_allow_permutation = guided_allow_permutation
        self.max_tries = max_tries
        self._candidate_cache: dict[int, list[_CandidateSpec]] = {}

    def generate_one(self, mode: str | None = None) -> tuple[str, str]:
        details = self.generate_one_details(mode)
        return details['prompt'], details['answer']

    def generate_one_details(self, mode: str | None = None) -> dict:
        mode = mode or self.default_mode
        if mode == 'random':
            return self._generate_random_details()
        if mode == 'witness_minimal':
            return self._generate_witness_minimal_details()
        if mode == 'collision_pair':
            return self._generate_collision_pair_details()
        if mode == 'withheld_query_op':
            return self._generate_withheld_query_op_details()
        raise ValueError(f'Unknown transformation mode: {mode}')

    def generate_one_with_trace(self, mode: str | None = None) -> tuple[str, str, str] | None:
        try:
            details = self.generate_one_details(mode)
        except Exception:
            return None
        if details is None:
            return None
        prompt = details['prompt']
        answer = details['answer']
        meta = details.get('metadata', {})

        # Use solver's trace() — gives verified computation-showing traces
        from solvers.transformation import trace as solver_trace
        result = solver_trace(prompt)
        if result:
            reasoning, traced_answer = result
            import re
            reasoning = re.sub(r'\\boxed\{[^}]*\}', '', reasoning).rstrip()
            return prompt, answer, f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"

        # Solver can't trace — puzzle may be inconsistent. Only proceed if we can
        # verify the computation ourselves from the full spec.
        # Build trace from generator's known ops
        # We know the full spec because we generated the puzzle
        hidden_ops_raw = meta.get('hidden_ops', {})
        query_op = meta.get('query_op', '?')
        base = meta.get('base', 10)

        from solvers.transformation import _parse, _is_numeric, _parse_numeric
        from solvers.transformation_ops import OP_DESCRIPTIONS, ARITHMETIC_OPS
        parsed_examples, query = _parse(prompt)

        # Build fn lookup from ARITHMETIC_OPS
        op_fns = {name: fn for name, fn, _ in ARITHMETIC_OPS}

        lines = [f"Equation rules. Base {base}."]
        lines.append("Ops:")

        # Show each operator with verification
        by_op = {}
        is_numeric = parsed_examples and _is_numeric(parsed_examples)
        if is_numeric:
            for inp, out in parsed_examples:
                p = _parse_numeric(inp)
                if p:
                    by_op.setdefault(p[1], []).append((inp, out))

        for oc in sorted(hidden_ops_raw.keys()):
            info = hidden_ops_raw[oc]
            if isinstance(info, dict):
                label = info.get('label', '?')
                base_name = info.get('base_name', '?')
                rev_in = info.get('rev_input', False)
                rev_out = info.get('rev_output', False)
                opsign = info.get('use_opsign', False)
            else:
                label = info
                base_name = info
                rev_in = rev_out = opsign = False

            desc = OP_DESCRIPTIONS.get(base_name, base_name)
            ex_list = by_op.get(oc, [])
            if ex_list:
                inp, out = ex_list[0]
                lines.append(f"  {oc}={label}: {inp}={out} → MATCH")
            else:
                lines.append(f"  {oc}={label}")

        # Show query computation with actual arithmetic
        q_info = hidden_ops_raw.get(query_op)
        if query and q_info and is_numeric:
            qp = _parse_numeric(query)
            if qp:
                qa, qop, qb = qp
                if isinstance(q_info, dict):
                    base_name = q_info.get('base_name', '?')
                    rev_in = q_info.get('rev_input', False)
                    rev_out = q_info.get('rev_output', False)
                    opsign = q_info.get('use_opsign', False)
                else:
                    base_name = q_info
                    rev_in = rev_out = opsign = False

                desc = OP_DESCRIPTIONS.get(base_name, base_name)
                lines.append(f"Query: {query}")

                # Show actual computation steps
                a_val = int(qa[::-1]) if rev_in else int(qa)
                b_val = int(qb[::-1]) if rev_in else int(qb)
                if rev_in:
                    lines.append(f"  rev({qa})={a_val}, rev({qb})={b_val}")

                fn = op_fns.get(base_name)
                if fn:
                    raw = fn(a_val, b_val)
                    lines.append(f"  {desc}: {a_val},{b_val} → {raw}")
                    if rev_out:
                        rev_str = str(abs(raw))[::-1]
                        sign = '-' if raw < 0 else ''
                        lines.append(f"  rev_output: {sign}{rev_str}")
                else:
                    lines.append(f"  {desc} → {answer}")
        elif query:
            label = self._op_label(hidden_ops_raw.get(query_op, '?'))
            lines.append(f"Query: {query}")
            lines.append(f"  {label} → {answer}")

        trace = '\n'.join(lines)

        # Verify: if we showed a computation, check it matches the answer
        # If not, the puzzle is inconsistent — reject it
        if query and q_info and is_numeric and fn:
            # Reconstruct expected answer from the computation
            raw = fn(a_val, b_val)
            if rev_out:
                body = str(abs(raw))[::-1]
            else:
                body = str(abs(raw))
            if raw < 0:
                computed = ('-' + body) if not opsign else (qop + body)
            else:
                computed = body
            if computed != answer:
                return None  # inconsistent puzzle — don't emit wrong trace

        return prompt, answer, f"<think>\n{trace}\n</think>\n\\boxed{{{answer}}}"

    def _candidate_specs(self, base: int) -> list[_CandidateSpec]:
        cached = self._candidate_cache.get(base)
        if cached is not None:
            return cached

        specs: list[_CandidateSpec] = []
        for base_name, fn, base_weight in ARITHMETIC_OPS:
            for (rev_input, rev_output), mod_weight in MODIFIER_WEIGHTS.items():
                for use_opsign in (False, True):
                    prior = base_weight * mod_weight
                    prior *= OPSIGN_PROBABILITY if use_opsign else (1.0 - OPSIGN_PROBABILITY)
                    specs.append(
                        _CandidateSpec(
                            base_name=base_name,
                            fn=fn,
                            prior_weight=prior,
                            rev_input=rev_input,
                            rev_output=rev_output,
                            use_opsign=use_opsign,
                            kind='arith',
                        )
                    )

        specs.append(
            _CandidateSpec(
                base_name='concat',
                fn=lambda a, b, B=base: a * B * B + b,
                prior_weight=CONCAT_WEIGHT,
                kind='concat',
            )
        )
        specs.append(
            _CandidateSpec(
                base_name='bconcat',
                fn=lambda a, b, B=base: b * B * B + a,
                prior_weight=BCONCAT_WEIGHT,
                kind='concat',
            )
        )
        self._candidate_cache[base] = specs
        return specs

    def _find_spec(
        self,
        specs: Iterable[_CandidateSpec],
        base_name: str,
        rev_input: bool,
        rev_output: bool,
        use_opsign: bool,
    ) -> _CandidateSpec:
        for spec in specs:
            if (
                spec.base_name == base_name
                and spec.rev_input == rev_input
                and spec.rev_output == rev_output
                and spec.use_opsign == use_opsign
            ):
                return spec
        raise KeyError((base_name, rev_input, rev_output, use_opsign))

    def _sample_hidden_spec(
        self,
        base: int,
        forced_modifier: tuple[bool, bool] | None = None,
        force_base_name: str | None = None,
        allowed_base_names: set[str] | None = None,
        exclude_base_names: set[str] | None = None,
        allow_concat: bool = True,
    ) -> _CandidateSpec:
        specs = self._candidate_specs(base)
        filtered: list[_CandidateSpec] = []
        for spec in specs:
            if not allow_concat and spec.kind == 'concat':
                continue
            if forced_modifier is not None and spec.modifier_signature != forced_modifier:
                continue
            if force_base_name is not None and spec.base_name != force_base_name:
                continue
            if allowed_base_names is not None and spec.base_name not in allowed_base_names:
                continue
            if exclude_base_names and spec.base_name in exclude_base_names:
                continue
            filtered.append(spec)
        if not filtered:
            raise RuntimeError('No hidden specs available after filtering')
        idx = _weighted_choice(self.rng, [spec.prior_weight for spec in filtered])
        return filtered[idx]

    def _pick_base_and_symbols(self, is_symbol_type: bool, allow_digit_permutation: bool = True):
        if is_symbol_type:
            base = self.rng.randint(6, 12)
            n_operators = self.rng.randint(2, 3)
            chosen = self.rng.sample(SYMBOL_POOL, base + n_operators)
            digit_symbols = chosen[:base]
            operator_symbols = chosen[base: base + n_operators]
        else:
            base = 10
            if allow_digit_permutation and self.rng.random() < self.digit_permutation_probability:
                perm = list(range(10))
                self.rng.shuffle(perm)
                digit_symbols = [str(perm.index(v)) for v in range(10)]
            else:
                digit_symbols = [str(i) for i in range(10)]
            n_operators = self.rng.randint(2, 3)
            operator_symbols = self.rng.sample(SYMBOL_POOL, n_operators)
        symbol_to_value = {sym: val for val, sym in enumerate(digit_symbols)}
        return base, digit_symbols, operator_symbols, symbol_to_value

    def _generate_operand(self, base: int) -> int:
        if self.rng.random() < 0.8:
            return self.rng.randint(base, base * base - 1)
        return self.rng.randint(0, base * base - 1)

    def _encode_operand(self, val: int, base: int, digit_symbols: list[str]) -> str:
        high = val // base
        low = val % base
        return digit_symbols[high] + digit_symbols[low]

    def _make_example(
        self,
        spec: _CandidateSpec,
        op_char: str,
        a_value: int,
        b_value: int,
        base: int,
        digit_symbols: list[str],
        symbol_to_value: dict[str, int],
    ) -> _Example | None:
        a_str = self._encode_operand(a_value, base, digit_symbols)
        b_str = self._encode_operand(b_value, base, digit_symbols)
        output_str = spec.apply(a_str, b_str, base, symbol_to_value, digit_symbols, op_char)
        if output_str is None:
            return None
        decoded_a = _decode_number(a_str, base, symbol_to_value, spec.rev_input)
        decoded_b = _decode_number(b_str, base, symbol_to_value, spec.rev_input)
        return _Example(op_char, a_value, b_value, a_str, b_str, output_str, decoded_a, decoded_b)

    def _survivors_for_examples(
        self,
        op_char: str,
        examples: list[_Example],
        base: int,
        digit_symbols: list[str],
        symbol_to_value: dict[str, int],
    ) -> list[_CandidateSpec]:
        survivors: list[_CandidateSpec] = []
        for spec in self._candidate_specs(base):
            ok = True
            for ex in examples:
                pred = spec.apply(ex.a_str, ex.b_str, base, symbol_to_value, digit_symbols, op_char)
                if pred != ex.output_str:
                    ok = False
                    break
            if ok:
                survivors.append(spec)
        return survivors

    def _answer_map(
        self,
        candidate_specs: Iterable[_CandidateSpec],
        op_char: str,
        qa_str: str,
        qb_str: str,
        base: int,
        digit_symbols: list[str],
        symbol_to_value: dict[str, int],
    ) -> dict[str, list[_CandidateSpec]]:
        answer_map: dict[str, list[_CandidateSpec]] = defaultdict(list)
        for spec in candidate_specs:
            pred = spec.apply(qa_str, qb_str, base, symbol_to_value, digit_symbols, op_char)
            if pred is not None:
                answer_map[pred].append(spec)
        return dict(answer_map)

    def _choose_hard_query(
        self,
        hidden_spec: _CandidateSpec,
        candidate_bank: list[_CandidateSpec],
        op_char: str,
        base: int,
        digit_symbols: list[str],
        symbol_to_value: dict[str, int],
        n_trials: int = 96,
    ) -> tuple[int, int, str, str, str, dict[str, list[_CandidateSpec]]]:
        best = None
        for _ in range(n_trials):
            qa_value = self._generate_operand(base)
            qb_value = self._generate_operand(base)
            qa_str = self._encode_operand(qa_value, base, digit_symbols)
            qb_str = self._encode_operand(qb_value, base, digit_symbols)
            hidden_answer = hidden_spec.apply(qa_str, qb_str, base, symbol_to_value, digit_symbols, op_char)
            if hidden_answer is None:
                continue
            answer_map = self._answer_map(candidate_bank, op_char, qa_str, qb_str, base, digit_symbols, symbol_to_value)
            same_answer = len(answer_map.get(hidden_answer, []))
            distinct_answers = len(answer_map)
            raw = _unsigned_body(hidden_answer)
            score = (
                len(candidate_bank) - same_answer,
                distinct_answers,
                int(len(raw) > 1),
                int(raw != raw[::-1]),
                int(hidden_answer.startswith(op_char)),
            )
            payload = (qa_value, qb_value, qa_str, qb_str, hidden_answer, answer_map)
            if best is None or score > best[0]:
                best = (score, payload)
        if best is None:
            raise RuntimeError('Failed to choose a hard query')
        return best[1]

    def _greedy_witness_examples(
        self,
        hidden_spec: _CandidateSpec,
        op_char: str,
        query_answer: str,
        qa_str: str,
        qb_str: str,
        base: int,
        digit_symbols: list[str],
        symbol_to_value: dict[str, int],
        seed_examples: list[_Example] | None = None,
        max_examples: int = 4,
        pool_size: int = 220,
    ) -> list[_Example] | None:
        chosen = list(seed_examples or [])
        seen_pairs = {(ex.a_str, ex.b_str) for ex in chosen}
        survivors = self._survivors_for_examples(op_char, chosen, base, digit_symbols, symbol_to_value)
        wrong_specs = [
            spec
            for spec in survivors
            if spec.apply(qa_str, qb_str, base, symbol_to_value, digit_symbols, op_char) != query_answer
        ]

        pool: list[tuple[_Example, set[_CandidateSpec], float]] = []
        for _ in range(pool_size):
            a_value = self._generate_operand(base)
            b_value = self._generate_operand(base)
            a_str = self._encode_operand(a_value, base, digit_symbols)
            b_str = self._encode_operand(b_value, base, digit_symbols)
            if (a_str, b_str) in seen_pairs:
                continue
            ex = self._make_example(hidden_spec, op_char, a_value, b_value, base, digit_symbols, symbol_to_value)
            if ex is None:
                continue
            kill_set = {
                spec
                for spec in self._candidate_specs(base)
                if spec.apply(ex.a_str, ex.b_str, base, symbol_to_value, digit_symbols, op_char) != ex.output_str
            }
            if not kill_set:
                continue
            pool.append((ex, kill_set, self._example_bonus(hidden_spec, ex)))

        while wrong_specs and len(chosen) < max_examples:
            best = None
            wrong_set = set(wrong_specs)
            for ex, kill_set, bonus in pool:
                if (ex.a_str, ex.b_str) in seen_pairs:
                    continue
                killed = len(kill_set & wrong_set)
                if killed == 0:
                    continue
                score = (killed, bonus)
                if best is None or score > best[0]:
                    best = (score, ex, kill_set)
            if best is None:
                break
            _, best_ex, kill_set = best
            chosen.append(best_ex)
            seen_pairs.add((best_ex.a_str, best_ex.b_str))
            wrong_specs = [spec for spec in wrong_specs if spec not in kill_set]

        chosen = self._ensure_visibility_examples(
            chosen,
            hidden_spec,
            op_char,
            base,
            digit_symbols,
            symbol_to_value,
            max_examples=max_examples,
        )

        survivors = self._survivors_for_examples(op_char, chosen, base, digit_symbols, symbol_to_value)
        answer_map = self._answer_map(survivors, op_char, qa_str, qb_str, base, digit_symbols, symbol_to_value)
        if len(answer_map) != 1 or query_answer not in answer_map:
            return None
        return chosen

    def _ensure_visibility_examples(
        self,
        chosen: list[_Example],
        hidden_spec: _CandidateSpec,
        op_char: str,
        base: int,
        digit_symbols: list[str],
        symbol_to_value: dict[str, int],
        max_examples: int,
    ) -> list[_Example]:
        if len(chosen) >= max_examples:
            return chosen

        seen_pairs = {(ex.a_str, ex.b_str) for ex in chosen}
        need_negative = hidden_spec.use_opsign and not any(ex.output_str.startswith(op_char) for ex in chosen)
        need_rev_output = hidden_spec.rev_output and not any(
            len(_unsigned_body(ex.output_str)) > 1 and _unsigned_body(ex.output_str) != _unsigned_body(ex.output_str)[::-1]
            for ex in chosen
        )
        need_ordering = hidden_spec.base_name in {'sub', 'bsub', 'absdiff'} and not _has_both_orderings(chosen)

        if not (need_negative or need_rev_output or need_ordering):
            return chosen

        for _ in range(180):
            a_value = self._generate_operand(base)
            b_value = self._generate_operand(base)
            a_str = self._encode_operand(a_value, base, digit_symbols)
            b_str = self._encode_operand(b_value, base, digit_symbols)
            if (a_str, b_str) in seen_pairs:
                continue
            ex = self._make_example(hidden_spec, op_char, a_value, b_value, base, digit_symbols, symbol_to_value)
            if ex is None:
                continue
            if need_negative and ex.output_str.startswith(op_char):
                chosen.append(ex)
                break
            if need_rev_output:
                body = _unsigned_body(ex.output_str)
                if len(body) > 1 and body != body[::-1]:
                    chosen.append(ex)
                    break
            if need_ordering:
                if hidden_spec.base_name in {'sub', 'absdiff'} and ex.decoded_a < ex.decoded_b:
                    chosen.append(ex)
                    break
                if hidden_spec.base_name == 'bsub' and ex.decoded_a > ex.decoded_b:
                    chosen.append(ex)
                    break
        return chosen

    def _example_bonus(self, hidden_spec: _CandidateSpec, ex: _Example) -> float:
        bonus = 0.0
        body = _unsigned_body(ex.output_str)
        if hidden_spec.rev_output and len(body) > 1 and body != body[::-1]:
            bonus += 2.0
        if hidden_spec.use_opsign and ex.output_str.startswith(ex.op_char):
            bonus += 2.0
        if hidden_spec.base_name in {'sub', 'bsub', 'absdiff'}:
            bonus += 0.5
            if hidden_spec.base_name in {'sub', 'absdiff'} and ex.decoded_a < ex.decoded_b:
                bonus += 1.0
            if hidden_spec.base_name == 'bsub' and ex.decoded_a > ex.decoded_b:
                bonus += 1.0
        return bonus

    def _anchor_example(
        self,
        spec: _CandidateSpec,
        op_char: str,
        base: int,
        digit_symbols: list[str],
        symbol_to_value: dict[str, int],
        force_informative: bool = True,
    ) -> _Example | None:
        best = None
        for _ in range(120):
            ex = self._make_example(
                spec,
                op_char,
                self._generate_operand(base),
                self._generate_operand(base),
                base,
                digit_symbols,
                symbol_to_value,
            )
            if ex is None:
                continue
            score = self._example_bonus(spec, ex)
            if spec.base_name in {'concat', 'bconcat'} and ex.a_str != ex.b_str:
                score += 1.0
            if not force_informative:
                return ex
            if best is None or score > best[0]:
                best = (score, ex)
        return best[1] if best else None

    def _format_prompt(self, examples: list[_Example], query_str: str) -> str:
        lines = [f'{ex.input_str} = {ex.output_str}' for ex in examples]
        return (
            "In Alice's Wonderland, a secret set of transformation rules "
            "is applied to equations. Below are a few examples:\n"
            + "\n".join(lines)
            + f"\nNow, determine the result for: {query_str}"
        )

    def _hidden_ops_metadata(self, op_configs: dict[str, _CandidateSpec]) -> dict[str, dict]:
        return {op_char: {
            'label': spec.label,
            'base_name': spec.base_name,
            'rev_input': spec.rev_input,
            'rev_output': spec.rev_output,
            'use_opsign': spec.use_opsign,
            'kind': spec.kind,
        } for op_char, spec in op_configs.items()}

    def _generate_random_details(self) -> dict:
        is_symbol = self.rng.random() < self.symbol_probability
        if is_symbol:
            try:
                from generators.transformation_v2 import generate_puzzle_for_config
                from solvers.transformation_csp import K
            except Exception:
                is_symbol = False
            else:
                for _ in range(20):
                    n_ops = self.rng.randint(2, 3)
                    base = self.rng.randint(8, 11)
                    op_combo = tuple(self.rng.randrange(K) for _ in range(n_ops))
                    rev_outs = tuple(bool(self.rng.getrandbits(1)) for _ in range(n_ops))
                    opsigns = tuple(bool(self.rng.getrandbits(1)) for _ in range(n_ops))
                    puzzle = generate_puzzle_for_config(
                        base=base,
                        op_combo=op_combo,
                        n_operators=n_ops,
                        rev_outs=rev_outs,
                        opsigns=opsigns,
                        seed=self.rng.randint(0, 2**31 - 1),
                        verify_mode='answer',
                    )
                    if puzzle is not None:
                        return {
                            'mode': 'random',
                            'prompt': puzzle['prompt'],
                            'answer': puzzle['answer'],
                            'metadata': {'symbol_type': True},
                        }

        base, digit_symbols, op_syms, symbol_to_value = self._pick_base_and_symbols(False, allow_digit_permutation=True)
        op_configs = {op_char: self._sample_hidden_spec(base) for op_char in op_syms}

        examples: list[_Example] = []
        n_examples = self.rng.randint(3, 6)
        while len(examples) < n_examples:
            op_char = self.rng.choice(op_syms)
            ex = self._anchor_example(op_configs[op_char], op_char, base, digit_symbols, symbol_to_value, force_informative=False)
            if ex is not None:
                examples.append(ex)

        query_op = self.rng.choice(op_syms)
        q_spec = op_configs[query_op]
        qa_value, qb_value, qa_str, qb_str, answer, _ = self._choose_hard_query(
            q_spec,
            self._candidate_specs(base),
            query_op,
            base,
            digit_symbols,
            symbol_to_value,
        )
        query_str = f'{qa_str}{query_op}{qb_str}'
        self.rng.shuffle(examples)
        return {
            'mode': 'random',
            'prompt': self._format_prompt(examples, query_str),
            'answer': answer,
            'metadata': {
                'base': base,
                'hidden_ops': self._hidden_ops_metadata(op_configs),
                'query_op': query_op,
                'query_seen': True,
            },
        }

    def _generate_witness_minimal_details(self) -> dict:
        for _ in range(self.max_tries):
            base, digit_symbols, op_syms, symbol_to_value = self._pick_base_and_symbols(
                False,
                allow_digit_permutation=self.guided_allow_permutation,
            )
            op_configs = {op_char: self._sample_hidden_spec(base) for op_char in op_syms}
            query_op = self.rng.choice(op_syms)
            hidden_spec = op_configs[query_op]

            qa_value, qb_value, qa_str, qb_str, answer, _ = self._choose_hard_query(
                hidden_spec,
                self._candidate_specs(base),
                query_op,
                base,
                digit_symbols,
                symbol_to_value,
            )
            witness_examples = self._greedy_witness_examples(
                hidden_spec,
                query_op,
                answer,
                qa_str,
                qb_str,
                base,
                digit_symbols,
                symbol_to_value,
                seed_examples=[],
                max_examples=4,
            )
            if not witness_examples:
                continue

            examples = list(witness_examples)
            for op_char in op_syms:
                if op_char == query_op:
                    continue
                anchor = self._anchor_example(op_configs[op_char], op_char, base, digit_symbols, symbol_to_value)
                if anchor is not None:
                    examples.append(anchor)
            while len(examples) < 3:
                extra = self._anchor_example(hidden_spec, query_op, base, digit_symbols, symbol_to_value)
                if extra is not None:
                    examples.append(extra)
                else:
                    break
            if len(examples) > 6:
                examples = examples[:6]

            survivors = self._survivors_for_examples(
                query_op,
                [ex for ex in examples if ex.op_char == query_op],
                base,
                digit_symbols,
                symbol_to_value,
            )
            answer_map = self._answer_map(survivors, query_op, qa_str, qb_str, base, digit_symbols, symbol_to_value)
            if len(answer_map) != 1 or answer not in answer_map:
                continue

            query_str = f'{qa_str}{query_op}{qb_str}'
            self.rng.shuffle(examples)
            return {
                'mode': 'witness_minimal',
                'prompt': self._format_prompt(examples, query_str),
                'answer': answer,
                'metadata': {
                    'base': base,
                    'hidden_ops': self._hidden_ops_metadata(op_configs),
                    'query_op': query_op,
                    'query_seen': True,
                    'focus_examples': [ex.input_str for ex in witness_examples],
                    'survivor_count': len(survivors),
                    'answer_equivalence_size': len(answer_map[answer]),
                    'query_confusion_size': len(self._candidate_specs(base)) - len(answer_map[answer]),
                },
            }
        return self._generate_random_details()

    def _generate_collision_pair_details(self) -> dict:
        for _ in range(self.max_tries):
            base, digit_symbols, op_syms, symbol_to_value = self._pick_base_and_symbols(
                False,
                allow_digit_permutation=self.guided_allow_permutation,
            )
            specs = self._candidate_specs(base)
            collision = self.rng.choice(self.COLLISION_FAMILIES)
            hidden_spec = self._find_spec(specs, *collision['hidden'])
            competitor = self._find_spec(specs, *collision['competitor'])
            query_op = self.rng.choice(op_syms)
            op_configs = {query_op: hidden_spec}
            for op_char in op_syms:
                if op_char == query_op:
                    continue
                op_configs[op_char] = self._sample_hidden_spec(base, exclude_base_names={hidden_spec.base_name})

            qa_value, qb_value, qa_str, qb_str, answer, _ = self._choose_hard_query(
                hidden_spec,
                self._candidate_specs(base),
                query_op,
                base,
                digit_symbols,
                symbol_to_value,
            )
            if competitor.apply(qa_str, qb_str, base, symbol_to_value, digit_symbols, query_op) == answer:
                continue

            seed_examples: list[_Example] = []
            killer_example = None
            for _probe in range(240):
                ex = self._make_example(
                    hidden_spec,
                    query_op,
                    self._generate_operand(base),
                    self._generate_operand(base),
                    base,
                    digit_symbols,
                    symbol_to_value,
                )
                if ex is None:
                    continue
                competitor_out = competitor.apply(ex.a_str, ex.b_str, base, symbol_to_value, digit_symbols, query_op)
                if competitor_out is None:
                    continue
                if ex.output_str == competitor_out and collision['seed_filter'](ex, hidden_spec, competitor, base, symbol_to_value, digit_symbols):
                    if (ex.a_str, ex.b_str) not in {(s.a_str, s.b_str) for s in seed_examples}:
                        seed_examples.append(ex)
                    if len(seed_examples) >= 2:
                        break
            for _probe in range(240):
                ex = self._make_example(
                    hidden_spec,
                    query_op,
                    self._generate_operand(base),
                    self._generate_operand(base),
                    base,
                    digit_symbols,
                    symbol_to_value,
                )
                if ex is None:
                    continue
                competitor_out = competitor.apply(ex.a_str, ex.b_str, base, symbol_to_value, digit_symbols, query_op)
                if competitor_out is None:
                    continue
                if ex.output_str != competitor_out and collision['killer_filter'](ex, hidden_spec, competitor, base, symbol_to_value, digit_symbols):
                    killer_example = ex
                    break
            if not seed_examples or killer_example is None:
                continue

            witness_examples = self._greedy_witness_examples(
                hidden_spec,
                query_op,
                answer,
                qa_str,
                qb_str,
                base,
                digit_symbols,
                symbol_to_value,
                seed_examples=seed_examples + [killer_example],
                max_examples=4,
            )
            if not witness_examples:
                continue

            examples = list(witness_examples)
            for op_char in op_syms:
                if op_char == query_op:
                    continue
                anchor = self._anchor_example(op_configs[op_char], op_char, base, digit_symbols, symbol_to_value)
                if anchor is not None:
                    examples.append(anchor)
            if len(examples) > 6:
                examples = examples[:6]
            while len(examples) < 3:
                anchor = self._anchor_example(hidden_spec, query_op, base, digit_symbols, symbol_to_value)
                if anchor is None:
                    break
                examples.append(anchor)

            survivors = self._survivors_for_examples(
                query_op,
                [ex for ex in examples if ex.op_char == query_op],
                base,
                digit_symbols,
                symbol_to_value,
            )
            answer_map = self._answer_map(survivors, query_op, qa_str, qb_str, base, digit_symbols, symbol_to_value)
            if len(answer_map) != 1 or answer not in answer_map:
                continue

            query_str = f'{qa_str}{query_op}{qb_str}'
            self.rng.shuffle(examples)
            return {
                'mode': 'collision_pair',
                'prompt': self._format_prompt(examples, query_str),
                'answer': answer,
                'metadata': {
                    'base': base,
                    'hidden_ops': self._hidden_ops_metadata(op_configs),
                    'query_op': query_op,
                    'query_seen': True,
                    'collision_family': collision['name'],
                    'focus_examples': [ex.input_str for ex in witness_examples],
                    'survivor_count': len(survivors),
                    'answer_equivalence_size': len(answer_map[answer]),
                },
            }
        return self._generate_witness_minimal_details()

    def _generate_withheld_query_op_details(self) -> dict:
        for _ in range(self.max_tries):
            base, digit_symbols, op_syms, symbol_to_value = self._pick_base_and_symbols(
                False,
                allow_digit_permutation=False,
            )
            if len(op_syms) < 2:
                continue
            bank_name = self.rng.choice(list(self.WITHHELD_BANKS))
            bank = self.WITHHELD_BANKS[bank_name]
            modifier = (False, False) if bank_name == 'concat' else self.rng.choice(list(MODIFIER_WEIGHTS.keys()))

            qop = op_syms[0]
            anchor_ops = op_syms[1:]
            hidden_base_names = self.rng.sample(bank, min(len(bank), len(op_syms)))
            if len(hidden_base_names) < len(op_syms):
                hidden_base_names += [self.rng.choice(bank) for _ in range(len(op_syms) - len(hidden_base_names))]
            op_configs: dict[str, _CandidateSpec] = {}
            for op_char, base_name in zip(op_syms, hidden_base_names):
                op_configs[op_char] = self._sample_hidden_spec(
                    base,
                    forced_modifier=modifier,
                    force_base_name=base_name,
                    allow_concat=True,
                )

            candidate_bank = [
                spec
                for spec in self._candidate_specs(base)
                if spec.base_name in set(bank) and spec.modifier_signature == modifier
            ]
            hidden_spec = op_configs[qop]
            query_payload = None
            for _probe in range(160):
                value = self._generate_operand(base)
                qa_value = qb_value = value
                qa_str = self._encode_operand(qa_value, base, digit_symbols)
                qb_str = self._encode_operand(qb_value, base, digit_symbols)
                answer_map = self._answer_map(candidate_bank, qop, qa_str, qb_str, base, digit_symbols, symbol_to_value)
                hidden_answer = hidden_spec.apply(qa_str, qb_str, base, symbol_to_value, digit_symbols, qop)
                if hidden_answer is None:
                    continue
                if len(answer_map) == 1 and hidden_answer in answer_map:
                    query_payload = (qa_value, qb_value, qa_str, qb_str, hidden_answer, answer_map)
                    break
            if query_payload is None:
                continue

            anchor_examples: list[_Example] = []
            ok = True
            for anchor_op in anchor_ops:
                anchor_spec = op_configs[anchor_op]
                local_examples: list[_Example] = []
                for _probe in range(2):
                    ex = self._anchor_example(anchor_spec, anchor_op, base, digit_symbols, symbol_to_value)
                    if ex is not None:
                        local_examples.append(ex)
                if not local_examples:
                    ok = False
                    break
                local_survivors = self._survivors_for_examples(anchor_op, local_examples, base, digit_symbols, symbol_to_value)
                modifier_set = {spec.modifier_signature for spec in local_survivors}
                if modifier_set != {modifier}:
                    ok = False
                    break
                anchor_examples.extend(local_examples)
            if not ok:
                continue

            qa_value, qb_value, qa_str, qb_str, answer, answer_map = query_payload
            while len(anchor_examples) < 3:
                extra_op = self.rng.choice(anchor_ops)
                extra_anchor = self._anchor_example(
                    op_configs[extra_op],
                    extra_op,
                    base,
                    digit_symbols,
                    symbol_to_value,
                )
                if extra_anchor is None:
                    break
                anchor_examples.append(extra_anchor)
            query_str = f'{qa_str}{qop}{qb_str}'
            self.rng.shuffle(anchor_examples)
            return {
                'mode': 'withheld_query_op',
                'prompt': self._format_prompt(anchor_examples, query_str),
                'answer': answer,
                'metadata': {
                    'base': base,
                    'hidden_ops': self._hidden_ops_metadata(op_configs),
                    'query_op': qop,
                    'query_seen': False,
                    'withheld_bank': bank_name,
                    'shared_modifier': modifier,
                    'anchor_count': len(anchor_examples),
                    'answer_equivalence_size': len(answer_map[answer]),
                },
            }
        details = self._generate_witness_minimal_details()
        details['metadata']['withheld_fallback'] = True
        return details

    @staticmethod
    def _op_label(op_info):
        """Extract label string from op info (handles both old str and new dict format)."""
        if isinstance(op_info, dict):
            return op_info.get('label', '?')
        return op_info  # old string format

    def _build_trace_hint(self, details: dict) -> str:
        meta = details.get('metadata', {})
        mode = details.get('mode', 'random')
        answer = details['answer']
        query_op = meta.get('query_op', '?')
        hidden_ops_raw = meta.get('hidden_ops', {})
        # Normalize to label strings for display
        hidden_ops = {oc: self._op_label(info) for oc, info in hidden_ops_raw.items()}
        base = meta.get('base', 10)
        lines = [f"Equation rules. Base {base}."]

        if mode == 'collision_pair':
            collision_family = meta.get('collision_family', 'pair')
            # Parse collision family to get the two competing ops
            parts = collision_family.split('_vs_')
            hidden_label = hidden_ops.get(query_op, '?')

            lines.append("")
            lines.append("Step 1: Identify operators.")
            for oc, on in sorted(hidden_ops.items()):
                lines.append(f"  {oc} = {on}")

            lines.append("")
            lines.append(f"Step 2: Collision detection for {query_op}.")
            if len(parts) == 2:
                lines.append(f"  Candidate A: {parts[0]}")
                lines.append(f"  Candidate B: {parts[1]}")
                lines.append(f"  Both produce identical outputs on most inputs.")
            else:
                lines.append(f"  Near-collision: {collision_family}")

            lines.append("")
            lines.append("Step 3: Killer clue eliminates wrong candidate.")
            focus = meta.get('focus_examples', [])
            if focus:
                killer = focus[-1] if len(focus) > 1 else focus[0]
                lines.append(f"  Example {killer}:")
                lines.append(f"  Under {hidden_label}: matches output → MATCH")
                if len(parts) == 2:
                    lines.append(f"  Under {parts[1] if parts[0] in hidden_label else parts[0]}: MISMATCH → MISMATCH")
                lines.append(f"  → {query_op} = {hidden_label}")

            lines.append("")
            lines.append(f"Step 4: Compute query.")
            lines.append(f"  Apply {hidden_label}")

        elif mode == 'withheld_query_op':
            shared_mod = meta.get('shared_modifier', (False, False))
            bank_name = meta.get('withheld_bank', '?')
            hidden_label = hidden_ops.get(query_op, '?')

            lines.append("")
            lines.append("Step 1: Query operator absent from examples.")
            lines.append(f"  {query_op} not seen in any example row.")

            lines.append("")
            lines.append("Step 2: Identify anchor operators.")
            for oc, on in sorted(hidden_ops.items()):
                if oc != query_op:
                    lines.append(f"  {oc} = {on}")

            lines.append("")
            lines.append("Step 3: Infer shared modifier regime.")
            rev_in, rev_out = shared_mod if isinstance(shared_mod, tuple) else (False, False)
            mod_desc = []
            if rev_in: mod_desc.append("rev_input")
            if rev_out: mod_desc.append("rev_output")
            lines.append(f"  Modifiers: {', '.join(mod_desc) if mod_desc else 'plain'}")
            lines.append(f"  All anchors consistent → applies to {query_op} too.")

            lines.append("")
            lines.append(f"Step 4: Narrow {query_op} from structured bank.")
            lines.append(f"  Bank: {bank_name}")
            lines.append(f"  All candidates in bank with these modifiers → same answer.")
            lines.append(f"  → {query_op} produces {answer}")

        elif mode == 'witness_minimal':
            hidden_label = hidden_ops.get(query_op, '?')

            lines.append("")
            lines.append("Step 1: Identify operators.")
            for oc, on in sorted(hidden_ops.items()):
                lines.append(f"  {oc} = {on}")

            lines.append("")
            lines.append("Step 2: Eliminate competing hypotheses.")
            focus = meta.get('focus_examples', [])
            for i, witness in enumerate(focus[:3]):
                lines.append(f"  Witness {i+1}: {witness}")
                lines.append(f"    Rules out alternatives that disagree here.")

            lines.append("")
            lines.append(f"Step 3: Only {hidden_label} survives for {query_op}.")

            lines.append("")
            lines.append(f"Step 4: Compute query.")
            lines.append(f"  Apply {hidden_label}")

        else:
            # random mode — show identification + verification + computation
            examples = meta.get('examples', [])
            query_str = meta.get('query', '')

            lines.append("")
            lines.append("Identify operators:")
            for oc, on in sorted(hidden_ops.items()):
                # Show one verification example per operator
                op_examples = [ex for ex in examples if len(ex) >= 2 and oc in ex[0]]
                if op_examples:
                    inp, out = op_examples[0][0], op_examples[0][1]
                    lines.append(f"  {oc}={on}: {inp}={out} → MATCH")
                else:
                    lines.append(f"  {oc}={on}")

            # Show query computation
            query_op_name = hidden_ops.get(query_op, '?')
            lines.append(f"Query: {query_str}")
            lines.append(f"  {query_op_name} → {answer}")

        return '\n'.join(lines)
