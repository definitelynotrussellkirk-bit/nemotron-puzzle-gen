"""Validation script: generate samples from each generator and verify correctness.

For each generator, this script:
1. Generates N sample puzzles
2. Attempts to solve each puzzle deterministically (round-trip test)
3. Compares the computed answer to the generator's answer
4. Reports success/failure rates

This ensures our generators produce solvable, self-consistent puzzles
that match the competition format.
"""

import re
from generators import GENERATORS
from generators.number_conversion import to_roman


def _solve_number_conversion(prompt: str) -> str | None:
    """Solve a number conversion puzzle by parsing the query number."""
    m = re.search(r"write the number (\d+)", prompt)
    if m:
        return to_roman(int(m.group(1)))
    return None


def _solve_unit_conversion(prompt: str) -> str | None:
    """Solve a unit conversion puzzle using the full solver."""
    from solvers.unit_conversion import solve
    return solve(prompt)


def _solve_gravitational(prompt: str) -> str | None:
    """Solve a gravitational puzzle using the full solver."""
    from solvers.gravitational import solve
    return solve(prompt)


def _solve_encryption(prompt: str) -> str | None:
    """Solve an encryption puzzle using the full solver (cross-word + Zipf)."""
    from solvers.encryption import solve
    return solve(prompt)


def _solve_bit_manipulation(prompt: str) -> str | None:
    """Solve a bit manipulation puzzle with the shared deterministic solver."""
    from solvers.bit_manipulation import solve
    return solve(prompt)


def _solve_transformation(prompt: str) -> str | None:
    """Solve a transformation puzzle with the shared deterministic solver."""
    from solvers.transformation import solve
    return solve(prompt)


def validate_all(n_samples: int = 50, seed: int = 123):
    """Generate and validate samples from all generators."""
    results = {}

    for name, cls in GENERATORS.items():
        gen = cls(seed=seed)
        correct = 0
        solvable = 0
        total = n_samples

        solver = {
            "bit_manipulation": _solve_bit_manipulation,
            "encryption": _solve_encryption,
            "number_conversion": _solve_number_conversion,
            "unit_conversion": _solve_unit_conversion,
            "gravitational": _solve_gravitational,
            "transformation": _solve_transformation,
        }.get(name)

        for i in range(total):
            prompt, answer = gen.generate_one()

            computed = solver(prompt)
            if computed is not None:
                solvable += 1
                if computed == answer:
                    correct += 1
                elif i < 3:  # show first few mismatches
                    print(f"  MISMATCH [{name}]: expected={answer!r}, got={computed!r}")

        results[name] = (correct, solvable, total)
        status = "PASS" if correct == solvable else "PARTIAL"
        print(f"{name:25s}: {correct}/{solvable} correct "
              f"({solvable}/{total} solvable) [{status}]")

    return results


if __name__ == "__main__":
    validate_all()
