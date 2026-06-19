"""Generator for transformation rule reasoning problems.

=== HOW THE COMPETITION GENERATES THESE (REVERSE-ENGINEERED) ===

Every transformation problem is secretly BASE-N ARITHMETIC with an encoded
symbol system. The generation algorithm is:

1. CHOOSE A SYMBOL SET
   - "Numeric" sub-type: uses digits 0-9 directly (identity mapping)
   - "Pure symbol" sub-type: uses arbitrary symbols (!@#$%^&*...) that
     each map to a hidden digit value (0 to base-1)

2. CHOOSE A BASE (typically 6-12; 10 is most common for numeric type)

3. CHOOSE 2-3 OPERATOR SYMBOLS and assign each a COMPOUND OPERATION
   from this menu:

   Base operations:
     add      : a + b
     sub      : a - b
     bsub     : b - a
     mul      : a * b
     absdiff  : |a - b|
     add1     : a + b + 1
     sub1     : a + b - 1
     mul1     : a * b + 1
     mulsub1  : a * b - 1
     concat   : a * base^2 + b  (i.e., concatenate digit strings)
     bconcat  : b * base^2 + a  (reverse concatenation)

   Each operation can ALSO have these independently combinable MODIFIERS:
     - Input reversal:  reverse the digit-string of a and b before operating
     - Output reversal: reverse the digit-string of the result after operating
     - Both:            reverse inputs AND reverse output

4. INPUT FORMAT (always 5 characters):
     [digit1][digit2][OPERATOR][digit3][digit4]
   Positions 0,1 = left operand (2-digit number in base N)
   Position 2    = operator symbol
   Positions 3,4 = right operand (2-digit number in base N)

5. OUTPUT ENCODING:
   - Positive results: encode as digit string in the same base/symbol system
   - Negative results: the OPERATOR SYMBOL replaces the minus sign
     e.g., if op='#' and result is -35, output is "#35" not "-35"
   - Output digit reversal is applied BEFORE adding the sign prefix

6. Generate 3-6 examples + 1 query, format as competition prompt.

=== OBSERVED FREQUENCIES (from 1728 operator instances, 92.7% cracked) ===

Top operations:
  rev(rev(a)*rev(b))    174   — reverse inputs, multiply, reverse output
  rev(rev(a)+rev(b))    159   — reverse inputs, add, reverse output
  concat(b,a)           117   — concatenate b before a
  a-b                   112   — normal subtraction
  concat(a,b)           105   — concatenate a before b
  rev(rev(a)-rev(b))     90   — reverse inputs, subtract, reverse output
  a+b                    80   — normal addition
  a*b                    69   — normal multiplication
  rev(rev(a)*rev(b)+1)   69   — reversed multiply + 1
  rev(rev(a)*rev(b)-1)   60   — reversed multiply - 1
  a+b+1                  54   — add + 1
  a-b (opsign)           52   — subtract, op char as negative sign
  ... and many more combinations
"""

import random
import string
from .base import BaseGenerator


# ---------------------------------------------------------------------------
# The operation menu
# ---------------------------------------------------------------------------
# Each entry: (name, function, weight)
# The function signature is always fn(a: int, b: int) -> int
# Weights approximate observed frequencies across all operator instances.

# Import from shared spec to prevent drift
from solvers.transformation_ops import ARITHMETIC_OPS, SYMBOL_POOL, MODIFIER_WEIGHTS, OPSIGN_PROBABILITY, CONCAT_WEIGHT, BCONCAT_WEIGHT

# Concat, modifier, opsign, and symbol pool constants imported from shared spec above.


# ---------------------------------------------------------------------------
# Encoding / decoding helpers
# ---------------------------------------------------------------------------

def _encode_number(val: int, base: int, digit_symbols: list[str],
                   rev_output: bool, op_char: str, use_opsign: bool) -> str | None:
    """Encode an integer as a digit-string in the given base and symbol set.

    Args:
        val:           The integer to encode.
        base:          The number base (e.g., 10).
        digit_symbols: List where digit_symbols[i] is the symbol for value i.
        rev_output:    If True, reverse the digit characters before returning.
        op_char:       The operator character (used as negative sign if use_opsign).
        use_opsign:    If True, use op_char instead of '-' for negative numbers.

    Returns:
        The encoded string, or None if a digit value has no symbol.
    """
    # Handle sign
    negative = val < 0
    val = abs(val)

    # Special case: zero
    if val == 0:
        if 0 >= len(digit_symbols):
            return None
        s = digit_symbols[0]
    else:
        # Convert to base-N digit string
        chars = []
        while val > 0:
            digit = val % base
            if digit >= len(digit_symbols):
                return None  # digit value not representable
            chars.append(digit_symbols[digit])
            val //= base
        # chars is in reverse order (least significant first), flip it
        s = "".join(reversed(chars))

    # Apply output reversal modifier
    if rev_output:
        s = s[::-1]

    # Prepend sign for negative numbers
    if negative:
        sign = op_char if use_opsign else "-"
        s = sign + s

    return s


def _decode_number(s: str, base: int, symbol_to_value: dict[str, int],
                   rev_input: bool) -> int:
    """Decode a digit-string from the given base and symbol mapping.

    Args:
        s:               The digit string (e.g., "3A" or "/`").
        base:            The number base.
        symbol_to_value: Maps each symbol character to its digit value.
        rev_input:       If True, reverse the string before decoding.

    Returns:
        The decoded integer value.
    """
    if rev_input:
        s = s[::-1]
    val = 0
    for c in s:
        val = val * base + symbol_to_value[c]
    return val


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------

class TransformationGenerator(BaseGenerator):
    """Generates transformation rule puzzles matching the competition format.

    Supports two sub-types:
    - Numeric: digits 0-9 are used directly, operators are non-digit symbols
    - Pure symbol: all characters (digits and operators) are arbitrary symbols

    The mix is controlled by `symbol_probability` (default 0.5 = equal mix).
    """

    name = "transformation"

    def __init__(self, seed: int | None = None, symbol_probability: float = 0.5,
                 digit_permutation_probability: float = 0.07):
        """
        Args:
            seed:               Random seed for reproducibility.
            symbol_probability: Probability of generating pure-symbol (vs numeric) type.
                                Set to 0.0 for all numeric, 1.0 for all symbol.
            digit_permutation_probability: For numeric type, probability of permuting
                                digit values (so '3' doesn't mean value 3). This covers
                                the ~7.3% of numeric problems that use hidden digit mappings.
        """
        super().__init__(seed)
        self.symbol_probability = symbol_probability
        self.digit_permutation_probability = digit_permutation_probability

    def _pick_base_and_symbols(self, is_symbol_type: bool):
        """Choose the base, digit symbols, and operator symbols.

        For numeric type:
            base=10, digit_symbols=['0','1',...,'9'], operators from symbol pool.
        For pure symbol type:
            base=6-12, all symbols randomly assigned.

        Returns:
            (base, digit_symbols, operator_symbols, symbol_to_value)
        """
        if is_symbol_type:
            # Pure symbol type: random base, random symbol assignments
            base = self.rng.randint(6, 12)

            # Pick enough symbols for digits + 2-3 operators
            n_operators = self.rng.randint(2, 3)
            n_total = base + n_operators

            # Sample from the symbol pool (no repeats)
            chosen = self.rng.sample(SYMBOL_POOL, min(n_total, len(SYMBOL_POOL)))
            digit_symbols = chosen[:base]        # symbols representing values 0..base-1
            operator_symbols = chosen[base:base + n_operators]
        else:
            # Numeric type: base 10, digits are '0'-'9'
            base = 10

            # ~7.3% of real numeric problems use a PERMUTED digit mapping:
            # the digit characters are still '0'-'9', but their VALUES are shuffled.
            # e.g., '3' might represent value 7, '7' might represent value 3.
            # This makes the problems much harder — the model must figure out
            # the hidden digit→value mapping from the examples.
            if self.rng.random() < self.digit_permutation_probability:
                # Permuted: digit_symbols[value] = the character for that value
                # Create a random permutation of 0-9
                perm = list(range(10))
                self.rng.shuffle(perm)
                # digit_symbols[v] = character whose VALUE is v
                # perm[i] = the value assigned to digit character str(i)
                # So the character for value v is str(perm.index(v))
                digit_symbols = [str(perm.index(v)) for v in range(10)]
            else:
                # Identity mapping: '0'=0, '1'=1, ..., '9'=9
                digit_symbols = [str(i) for i in range(10)]

            n_operators = self.rng.randint(2, 3)
            # Operators: pick non-digit symbols
            operator_symbols = self.rng.sample(SYMBOL_POOL, n_operators)

        # Build reverse mapping: symbol character -> digit value
        # digit_symbols[value] = character, so character -> value
        symbol_to_value = {sym: val for val, sym in enumerate(digit_symbols)}

        return base, digit_symbols, operator_symbols, symbol_to_value

    def _pick_operation(self, base: int):
        """Choose a compound operation for one operator symbol.

        Returns:
            (op_name, op_fn, rev_input, rev_output, use_opsign)
        """
        # Decide: arithmetic op or concatenation?
        total_arith_weight = sum(w for _, _, w in ARITHMETIC_OPS)
        total_concat_weight = CONCAT_WEIGHT + BCONCAT_WEIGHT
        total = total_arith_weight + total_concat_weight

        r = self.rng.uniform(0, total)
        if r < total_arith_weight:
            # Pick an arithmetic operation (weighted)
            weights = [w for _, _, w in ARITHMETIC_OPS]
            idx = _weighted_choice(self.rng, weights)
            op_name, op_fn, _ = ARITHMETIC_OPS[idx]
        else:
            # Pick concatenation
            if self.rng.random() < CONCAT_WEIGHT / total_concat_weight:
                op_name = "concat"
                op_fn = lambda a, b, B=base: a * B * B + b
            else:
                op_name = "bconcat"
                op_fn = lambda a, b, B=base: b * B * B + a

        # Pick modifiers (rev_input, rev_output)
        mod_items = list(MODIFIER_WEIGHTS.items())
        mod_weights = [w for _, w in mod_items]
        mod_idx = _weighted_choice(self.rng, mod_weights)
        rev_input, rev_output = mod_items[mod_idx][0]

        # Concat operations typically don't use reversal — override
        if op_name in ("concat", "bconcat"):
            rev_input = False
            rev_output = False

        # Decide: use operator-as-negative-sign?
        use_opsign = self.rng.random() < OPSIGN_PROBABILITY

        return op_name, op_fn, rev_input, rev_output, use_opsign

    def _generate_operand(self, base: int) -> int:
        """Generate a random 2-digit operand value in the given base.

        The value must be representable as exactly 2 digits, so it's
        in range [0, base^2 - 1]. We bias toward values that need 2 digits
        (>= base) so the leading digit isn't always zero.
        """
        # 80% chance of 2-digit number (>= base), 20% chance of any value
        if self.rng.random() < 0.8:
            return self.rng.randint(base, base * base - 1)
        else:
            return self.rng.randint(0, base * base - 1)

    def _encode_operand(self, val: int, base: int, digit_symbols: list[str]) -> str:
        """Encode a 2-digit operand as a 2-character string.

        Always produces exactly 2 characters (zero-padded on the left).
        """
        high = val // base
        low = val % base
        return digit_symbols[high] + digit_symbols[low]

    def generate_one(self) -> tuple[str, str]:
        """Generate one transformation rule puzzle (prompt, answer) pair.

        Steps:
        1. Decide numeric vs pure-symbol sub-type
        2. Choose base, symbols, operators
        3. Assign each operator a compound operation
        4. Generate 3-6 example pairs + 1 query
        5. Format as competition-style prompt
        """
        # Step 1: numeric or symbol?
        is_symbol = self.rng.random() < self.symbol_probability

        if is_symbol:
            from generators.transformation_v2 import generate_puzzle_for_config
            from solvers.transformation_csp import K

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
                    verify_mode="answer",
                )
                if puzzle is not None:
                    return puzzle["prompt"], puzzle["answer"]

        # Step 2: set up the number system
        base, digit_symbols, op_syms, sym_to_val = self._pick_base_and_symbols(is_symbol)

        # Step 3: assign operations to operators
        op_configs = {}  # op_char -> (name, fn, rev_in, rev_out, opsign)
        for op_char in op_syms:
            op_name, op_fn, rev_in, rev_out, opsign = self._pick_operation(base)
            op_configs[op_char] = (op_name, op_fn, rev_in, rev_out, opsign)

        # Step 4: generate examples with witness-quality constraints
        # A.1: Rev_output examples must have visible (non-palindromic, multi-digit) outputs
        # A.2: Sub-family ops must have both a>b and a<b examples
        # A.3: Opsign ops must have at least one negative output
        # A.4: Concat ops must have asymmetric operands
        n_examples = self.rng.randint(3, 6)
        lines = []
        example_meta = []  # track (op_char, a_val, b_val, result_val) for validation

        for _ in range(n_examples * 3):  # oversample to allow filtering
            if len(lines) >= n_examples:
                break

            op_char = self.rng.choice(op_syms)
            op_name, op_fn, rev_in, rev_out, opsign = op_configs[op_char]

            a_val = self._generate_operand(base)
            b_val = self._generate_operand(base)

            a_str = self._encode_operand(a_val, base, digit_symbols)
            b_str = self._encode_operand(b_val, base, digit_symbols)

            a_decode = _decode_number(a_str, base, sym_to_val, rev_in)
            b_decode = _decode_number(b_str, base, sym_to_val, rev_in)

            result_val = op_fn(a_decode, b_decode)

            result_str = _encode_number(
                result_val, base, digit_symbols, rev_out, op_char, opsign
            )
            if result_str is None:
                continue

            # A.4: Concat — reject symmetric operands
            if op_name in ('concat', 'bconcat') and a_str == b_str:
                continue

            input_str = a_str + op_char + b_str
            lines.append(f"{input_str} = {result_str}")
            example_meta.append((op_char, a_decode, b_decode, result_val, result_str))

        # Post-validation per operator
        for op_char in op_syms:
            op_name, op_fn, rev_in, rev_out, opsign = op_configs[op_char]
            op_examples = [(a, b, r, rs) for oc, a, b, r, rs in example_meta if oc == op_char]

            if not op_examples:
                continue

            # A.1: Check reversal visibility — at least one non-palindromic multi-digit output
            if rev_out and all(
                len(rs.lstrip('-').lstrip(op_char)) <= 1 or
                rs.lstrip('-').lstrip(op_char) == rs.lstrip('-').lstrip(op_char)[::-1]
                for _, _, _, rs in op_examples
            ):
                # Add a witness example with visible reversal
                for _ in range(20):
                    a_val = self.rng.randint(base, base * base - 1)
                    b_val = self.rng.randint(base, base * base - 1)
                    a_str = self._encode_operand(a_val, base, digit_symbols)
                    b_str = self._encode_operand(b_val, base, digit_symbols)
                    a_d = _decode_number(a_str, base, sym_to_val, rev_in)
                    b_d = _decode_number(b_str, base, sym_to_val, rev_in)
                    r = op_fn(a_d, b_d)
                    rs = _encode_number(r, base, digit_symbols, rev_out, op_char, opsign)
                    if rs and len(rs.lstrip('-').lstrip(op_char)) >= 2:
                        clean = rs.lstrip('-').lstrip(op_char)
                        if clean != clean[::-1]:
                            lines.append(f"{a_str}{op_char}{b_str} = {rs}")
                            example_meta.append((op_char, a_d, b_d, r, rs))
                            break

            # A.2: Sub-family — need both orderings
            if op_name in ('sub', 'bsub', 'absdiff') and len(op_examples) >= 2:
                has_pos = any(r >= 0 for _, _, r, _ in op_examples)
                has_neg = any(r < 0 for _, _, r, _ in op_examples)
                if not (has_pos and has_neg) and op_name != 'absdiff':
                    # Add a witness with opposite sign
                    for _ in range(20):
                        a_val = self.rng.randint(base, base * base - 1)
                        b_val = self.rng.randint(base, base * base - 1)
                        if has_pos and not has_neg:
                            a_val, b_val = min(a_val, b_val), max(a_val, b_val)
                        else:
                            a_val, b_val = max(a_val, b_val), min(a_val, b_val)
                        a_str = self._encode_operand(a_val, base, digit_symbols)
                        b_str = self._encode_operand(b_val, base, digit_symbols)
                        a_d = _decode_number(a_str, base, sym_to_val, rev_in)
                        b_d = _decode_number(b_str, base, sym_to_val, rev_in)
                        r = op_fn(a_d, b_d)
                        rs = _encode_number(r, base, digit_symbols, rev_out, op_char, opsign)
                        if rs:
                            lines.append(f"{a_str}{op_char}{b_str} = {rs}")
                            example_meta.append((op_char, a_d, b_d, r, rs))
                            break

            # A.3: Opsign — need at least one negative
            if opsign and all(r >= 0 for _, _, r, _ in op_examples):
                for _ in range(20):
                    a_val = self.rng.randint(0, base - 1)
                    b_val = self.rng.randint(base, base * base - 1)
                    a_str = self._encode_operand(a_val, base, digit_symbols)
                    b_str = self._encode_operand(b_val, base, digit_symbols)
                    a_d = _decode_number(a_str, base, sym_to_val, rev_in)
                    b_d = _decode_number(b_str, base, sym_to_val, rev_in)
                    r = op_fn(a_d, b_d)
                    if r < 0:
                        rs = _encode_number(r, base, digit_symbols, rev_out, op_char, opsign)
                        if rs:
                            lines.append(f"{a_str}{op_char}{b_str} = {rs}")
                            example_meta.append((op_char, a_d, b_d, r, rs))
                            break

        # Step 5: generate the query
        query_op = self.rng.choice(op_syms)
        qop_name, qop_fn, qrev_in, qrev_out, qopsign = op_configs[query_op]

        qa_val = self._generate_operand(base)
        qb_val = self._generate_operand(base)
        qa_str = self._encode_operand(qa_val, base, digit_symbols)
        qb_str = self._encode_operand(qb_val, base, digit_symbols)

        qa_decode = _decode_number(qa_str, base, sym_to_val, qrev_in)
        qb_decode = _decode_number(qb_str, base, sym_to_val, qrev_in)
        query_result = qop_fn(qa_decode, qb_decode)
        answer = _encode_number(
            query_result, base, digit_symbols, qrev_out, query_op, qopsign
        )

        # Safety fallback
        if answer is None:
            answer = "0"

        query_str = qa_str + query_op + qb_str

        # Step 6: format the prompt
        prompt = (
            "In Alice's Wonderland, a secret set of transformation rules "
            "is applied to equations. Below are a few examples:\n"
            + "\n".join(lines)
            + f"\nNow, determine the result for: {query_str}"
        )

        return prompt, answer


    def _format_op_name(self, op_name, rev_in, rev_out, opsign=False):
        """Format a human-readable operation description."""
        from solvers.transformation_ops import OP_DESCRIPTIONS
        desc = OP_DESCRIPTIONS.get(op_name, op_name)
        parts = []
        if rev_in:
            parts.append("rev inputs")
        parts.append(desc)
        if rev_out:
            parts.append("rev output")
        if opsign:
            parts.append("opsign")
        return ", ".join(parts)

    def generate_one_with_trace(self) -> tuple[str, str, str] | None:
        """Generate a transformation puzzle WITH a reasoning trace.

        Returns (prompt, answer, trace_text) or None if generation fails.
        The trace is built directly from the known latent parameters.
        """
        is_symbol = self.rng.random() < self.symbol_probability

        if is_symbol:
            # Symbol type — use transformation_v2 + CSP trace
            from generators.transformation_v2 import generate_puzzle_for_config
            from solvers.transformation_csp import K, OPS_BY_NAME as CSP_OPS

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
                    verify_mode="answer",
                )
                if puzzle is not None:
                    # Build trace from known latent params
                    from solvers.transformation import trace as solver_trace
                    result = solver_trace(puzzle["prompt"])
                    if result:
                        return puzzle["prompt"], puzzle["answer"], result[0]
                    # Fallback: bare boxed
                    return puzzle["prompt"], puzzle["answer"], f"\\boxed{{{puzzle['answer']}}}"
            return None

        # Numeric type — we know everything
        base, digit_symbols, op_syms, sym_to_val = self._pick_base_and_symbols(is_symbol)

        op_configs = {}
        for op_char in op_syms:
            op_name, op_fn, rev_in, rev_out, opsign = self._pick_operation(base)
            op_configs[op_char] = (op_name, op_fn, rev_in, rev_out, opsign)

        n_examples = self.rng.randint(3, 6)
        lines = []
        example_data = []  # (op_char, a_decode, b_decode, result_val, result_str)

        for _ in range(n_examples):
            op_char = self.rng.choice(op_syms)
            op_name, op_fn, rev_in, rev_out, opsign = op_configs[op_char]

            a_val = self._generate_operand(base)
            b_val = self._generate_operand(base)
            a_str = self._encode_operand(a_val, base, digit_symbols)
            b_str = self._encode_operand(b_val, base, digit_symbols)

            a_decode = _decode_number(a_str, base, sym_to_val, rev_in)
            b_decode = _decode_number(b_str, base, sym_to_val, rev_in)
            result_val = op_fn(a_decode, b_decode)

            result_str = _encode_number(
                result_val, base, digit_symbols, rev_out, op_char, opsign
            )
            if result_str is None:
                continue

            input_str = a_str + op_char + b_str
            lines.append(f"{input_str} = {result_str}")
            example_data.append((op_char, a_decode, b_decode, result_val, input_str, result_str))

        if not lines:
            return None

        # Query
        query_op = self.rng.choice(op_syms)
        qop_name, qop_fn, qrev_in, qrev_out, qopsign = op_configs[query_op]

        qa_val = self._generate_operand(base)
        qb_val = self._generate_operand(base)
        qa_str = self._encode_operand(qa_val, base, digit_symbols)
        qb_str = self._encode_operand(qb_val, base, digit_symbols)

        qa_decode = _decode_number(qa_str, base, sym_to_val, qrev_in)
        qb_decode = _decode_number(qb_str, base, sym_to_val, qrev_in)
        query_result = qop_fn(qa_decode, qb_decode)
        answer = _encode_number(
            query_result, base, digit_symbols, qrev_out, query_op, qopsign
        )
        if answer is None:
            return None

        query_str = qa_str + query_op + qb_str

        prompt = (
            "In Alice's Wonderland, a secret set of transformation rules "
            "is applied to equations. Below are a few examples:\n"
            + "\n".join(lines)
            + f"\nNow, determine the result for: {query_str}"
        )

        # Build trace from known latent params
        trace_lines = [f"Equation rules. Base {base}."]

        # Show ops with verification from examples
        trace_lines.append("Ops:")
        seen_ops = {}
        for op_char, a_dec, b_dec, res_val, inp_str, res_str in example_data:
            if op_char not in seen_ops:
                op_name, _, rev_in, rev_out, opsign = op_configs[op_char]
                display = self._format_op_name(op_name, rev_in, rev_out, opsign)
                trace_lines.append(f"{op_char}={display}: {inp_str}→{res_str} ✓")
                seen_ops[op_char] = True

        # Query computation — show step-by-step work
        trace_lines.append(f"Query: {query_str}")
        steps = []
        if qrev_in:
            qa_str_r = self._encode_operand(qa_val, base, digit_symbols)[::-1]
            qb_str_r = self._encode_operand(qb_val, base, digit_symbols)[::-1]
            steps.append(f"rev inputs: {qa_decode},{qb_decode}")
        from solvers.transformation_ops import OP_DESCRIPTIONS
        op_desc = OP_DESCRIPTIONS.get(qop_name, qop_name)
        steps.append(f"{qa_decode}{op_desc.replace('a','').replace('b','')}{qb_decode}={query_result}")
        if qrev_out:
            steps.append(f"rev({query_result})→{answer}")
        elif qopsign and query_result < 0:
            steps.append(f"opsign→{answer}")
        trace_lines.append(f"compute: {' → '.join(steps)}")

        trace_text = "\n".join(trace_lines) + f"\n\n\\boxed{{{answer}}}"

        # Metadata for downstream filtering/emitting
        self._last_metadata = {
            "trace_mode": "deterministic",  # we know all latent params
            "query_op_seen": True,  # generator always shows all ops
            "digit_permuted": hasattr(self, '_last_digit_permuted') and self._last_digit_permuted,
            "is_symbol": is_symbol,
            "n_operators": len(op_configs) if not is_symbol else 0,
        }

        return prompt, answer, trace_text


def _weighted_choice(rng: random.Random, weights: list[float]) -> int:
    """Weighted random selection. Returns the index of the chosen item."""
    total = sum(weights)
    r = rng.uniform(0, total)
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return i
    return len(weights) - 1
