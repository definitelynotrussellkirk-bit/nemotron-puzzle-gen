"""Generator for unit conversion reasoning problems.

=== HOW THE COMPETITION GENERATES THESE ===

Every unit conversion problem uses a SECRET LINEAR SCALING FACTOR:
  - A random factor in range [0.50, 2.00] is chosen.
  - 3-5 examples show "X.XX m becomes Y.YY" where Y = X * factor.
  - The model is asked to convert one more measurement.

The model must:
  1. Infer the scaling factor from the examples (divide output by input).
  2. Multiply the query value by that factor.

The factor is perfectly consistent within each problem (±0.0003).
Answers are decimal numbers with 2 decimal places, typically 4-5 characters.

Zero-shot accuracy is ~33% (the model sometimes gets the ratio right).
"""

import math
import random
from .base import BaseGenerator


class UnitConversionGenerator(BaseGenerator):
    """Generates unit conversion puzzles matching the competition format.

    Each puzzle:
    1. Picks a random linear scaling factor in [0.50, 2.00]
    2. Generates 3-5 example measurements and their converted values
    3. Generates 1 query measurement
    4. Answer = query * factor, rounded to 2 decimal places

    The scaling factor is consistent across all examples in a puzzle.
    """

    name = "unit_conversion"

    def __init__(self, seed: int | None = None, boundary_mode: bool = False):
        """Initialize the generator.

        Parameters
        ----------
        boundary_mode : bool
            If True, generate puzzles where the query answer is at a rounding
            boundary (2*remainder ~ divisor), i.e., the intermediate result is
            near X.XX5. These are the cases where the model is most likely to
            produce off-by-one-hundredth errors.
        """
        super().__init__(seed)
        self.boundary_mode = boundary_mode

    def _is_near_boundary(self, ref_out_int: int, ref_v_int: int, q_int: int,
                          tolerance: float = 0.02) -> bool:
        """Check if a query value produces a rounding-boundary answer.

        The rounding decision is: d_int = quot + 1 if 2*rem >= ref_v_int else quot.
        A boundary case is when 2*rem is very close to ref_v_int, i.e.,
        abs(2*rem - ref_v_int) / ref_v_int < tolerance.

        This corresponds to the intermediate (exact) result being near X.XX5,
        where the model must round correctly to get the right hundredth.
        """
        num = ref_out_int * q_int
        _quot, rem = divmod(num, ref_v_int)
        # How close is 2*rem to ref_v_int? (0 = exactly at boundary)
        closeness = abs(2 * rem - ref_v_int) / ref_v_int
        return closeness < tolerance

    def _find_boundary_query(self, ref_out_int: int, ref_v_int: int,
                             exclude: set, max_attempts: int = 500) -> float | None:
        """Search for a query value that lands near a rounding boundary.

        Tries random values in [5.00, 49.99] and returns the first one where
        the intermediate result is near X.XX5. Returns None if no boundary
        value is found within max_attempts.
        """
        for _ in range(max_attempts):
            q = round(self.rng.uniform(5.0, 49.99), 2)
            if q in exclude:
                continue
            q_int = int(round(q * 100))
            if self._is_near_boundary(ref_out_int, ref_v_int, q_int, tolerance=0.02):
                return q
        return None

    def generate_one(self) -> tuple[str, str]:
        """Generate one unit conversion puzzle (prompt, answer) pair."""
        # Step 1: pick a random scaling factor
        # 6dp precision matches competition data (4dp is too coarse)
        factor = round(self.rng.uniform(0.50, 2.00), 6)

        # Step 2: generate example measurements
        n_examples = self.rng.randint(3, 5)
        values = [round(self.rng.uniform(5.0, 49.99), 2) for _ in range(n_examples + 1)]
        while len(set(values)) < len(values):
            values = [round(self.rng.uniform(5.0, 49.99), 2) for _ in range(n_examples + 1)]
        examples = values[:n_examples]
        query = values[n_examples]

        # Step 3: pick reference (largest input) and compute ALL outputs
        # via integer ratio from reference, so traces are internally consistent
        ref_v = max(examples)
        ref_out = round(ref_v * factor, 2)
        ref_v_int = int(round(ref_v * 100))
        ref_out_int = int(round(ref_out * 100))

        # Step 3b (boundary_mode): replace the query with one that lands
        # near a rounding boundary (X.XX5), if possible. These are the
        # hardest cases for the model — off-by-one-hundredth errors.
        if self.boundary_mode:
            exclude = set(examples)
            boundary_q = self._find_boundary_query(
                ref_out_int, ref_v_int, exclude, max_attempts=500
            )
            if boundary_q is not None:
                query = boundary_q

        lines = []
        for v in examples:
            v_int = int(round(v * 100))
            num = ref_out_int * v_int
            quot, rem = divmod(num, ref_v_int)
            d_int = quot + 1 if 2 * rem >= ref_v_int else quot
            converted = d_int / 100
            lines.append(f"{v} m becomes {converted:.2f}")

        # Step 4: compute the answer via same ratio
        q_int = int(round(query * 100))
        q_num = ref_out_int * q_int
        q_quot, q_rem = divmod(q_num, ref_v_int)
        q_d_int = q_quot + 1 if 2 * q_rem >= ref_v_int else q_quot
        answer = f"{q_d_int / 100:.2f}"

        # Step 5: format the prompt to match competition style
        prompt = (
            "In Alice's Wonderland, a secret unit conversion is applied to "
            "measurements. For example:\n"
            + "\n".join(lines)
            + f"\nNow, convert the following measurement: {query} m"
        )

        return prompt, answer
