"""Shared latent-circuit trace builder for bit generators.

When the solver gets the answer wrong, we can still build a truthful trace
from the generator's known circuit. This avoids both:
- Contradictory traces (wrong reasoning + right boxed answer)
- Minimal fallback (no reasoning at all)
"""

from generators.bit_manipulation import _get_bit, _bits_to_str, _apply_bit_spec
from solvers.bit_manipulation import _eval_function


def build_latent_trace(circuit, examples, query, answer):
    """Build a truthful trace from the known circuit structure.

    Uses human-readable format matching the solver's new trace style.
    All function evaluations use _eval_function for correctness.
    """
    query_bits = [_get_bit(query, pos) for pos in range(8)]
    n = len(examples)

    lines = [
        f"Bit manipulation rule. {n} examples, query={_bits_to_str(query)}.",
        f"Each output bit is an independent function of input bits.",
        f"",
    ]

    for bp in range(8):
        spec = circuit[bp]
        family = spec.get('family', '?')
        inputs = spec.get('inputs', ())
        result_bit = int(answer[bp])

        # Compute actual function result for correctness
        fn_result = _eval_function(family, inputs, query_bits)
        if fn_result is None:
            fn_result = result_bit

        # Human-readable description
        if family in ('CONST_0', 'CONST_1'):
            lines.append(f"  b{bp}: always {fn_result}.")
        elif family == 'COPY' and len(inputs) == 1:
            lines.append(f"  b{bp}: copies input bit {inputs[0]}. Query: {fn_result}.")
        elif family == 'NOT' and len(inputs) == 1:
            lines.append(f"  b{bp}: inverts input bit {inputs[0]}. Query: {fn_result}.")
        elif len(inputs) == 2:
            fn_name = family.split("(")[0] if "(" in family else family
            lines.append(f"  b{bp}: {fn_name} of input bits {inputs[0]} and {inputs[1]}. "
                        f"Matches all {n} examples. Query: {fn_result}.")
        elif len(inputs) == 3:
            fn_name = family.split("(")[0] if "(" in family else family
            if "TT3" in fn_name: fn_name = "f"
            lines.append(f"  b{bp}: {fn_name} of input bits {inputs[0]}, {inputs[1]}, {inputs[2]}. "
                        f"Matches all {n} examples. Query: {fn_result}.")
        else:
            lines.append(f"  b{bp}: {family}. Query: {fn_result}.")

    lines.append(f"")
    lines.append(f"Result: {answer}")
    return "\n".join(lines) + f"\n\n\\boxed{{{answer}}}"
