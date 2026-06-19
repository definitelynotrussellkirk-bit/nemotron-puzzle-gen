"""Generator for number conversion reasoning problems.

=== HOW THE COMPETITION GENERATES THESE ===

Every number conversion problem is DECIMAL TO ROMAN NUMERAL conversion.
  - 3-5 examples of "decimal -> Roman numeral" are shown.
  - The model is asked to convert one more decimal number.
  - All numbers are in the range 1-100 (mean ~49.3).
  - Standard Roman numeral rules apply (I, IV, V, IX, X, XL, L, XC, C).

This is the EASIEST category — models get ~100% zero-shot accuracy.
The answer is always a valid Roman numeral string, 1-8 characters.
"""

import random
from .base import BaseGenerator

# ---------------------------------------------------------------------------
# Roman numeral conversion table
# ---------------------------------------------------------------------------
# Standard subtractive-notation rules: values in descending order.
# Each (value, numeral) pair is used greedily from largest to smallest.

_ROMAN_TABLE = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]


def to_roman(n: int) -> str:
    """Convert a positive integer to its Roman numeral representation.

    Uses standard subtractive notation (e.g., 4=IV, 9=IX, 40=XL, 90=XC).

    >>> to_roman(42)
    'XLII'
    >>> to_roman(99)
    'XCIX'
    """
    if n <= 0:
        raise ValueError(f"Roman numerals require positive integers, got {n}")
    parts = []
    for value, numeral in _ROMAN_TABLE:
        while n >= value:
            parts.append(numeral)
            n -= value
    return "".join(parts)


class NumberConversionGenerator(BaseGenerator):
    """Generates number conversion puzzles matching the competition format.

    Each puzzle:
    1. Selects 3-5 random numbers in range [1, 100] as examples
    2. Selects 1 random number as the query
    3. Shows "decimal -> Roman" examples
    4. Asks the model to convert the query number

    The answer is always the Roman numeral string.
    """

    name = "number_conversion"

    def generate_one(self) -> tuple[str, str]:
        """Generate one number conversion puzzle (prompt, answer) pair."""
        # Step 1: pick example numbers and query (all unique)
        n_examples = self.rng.randint(3, 5)
        pool = self.rng.sample(range(1, 101), n_examples + 1)
        examples = pool[:n_examples]
        query = pool[n_examples]

        # Step 2: format example lines
        lines = [f"{n} -> {to_roman(n)}" for n in examples]

        # Step 3: compute the answer
        answer = to_roman(query)

        # Step 4: format the prompt to match competition style
        prompt = (
            "In Alice's Wonderland, numbers are secretly converted into a "
            "different numeral system. Some examples are given below:\n"
            + "\n".join(lines)
            + f"\nNow, write the number {query} in the Wonderland numeral system."
        )

        return prompt, answer
