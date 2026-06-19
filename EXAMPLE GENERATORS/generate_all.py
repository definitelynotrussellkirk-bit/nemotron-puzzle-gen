"""Generate synthetic training data from all generators.

Usage:
    python -m generators.generate_all                        # defaults: 1000 per type
    python -m generators.generate_all --n-per-type 5000      # more data
    python -m generators.generate_all --types bit_manipulation encryption  # specific types
    python -m generators.generate_all --validate             # also run validation

Output goes to data/synthetic/ as one CSV per problem type, plus a combined file.
Each CSV has columns: id, prompt, answer (matching the competition format).
"""

import argparse
import csv
from pathlib import Path

from generators import GENERATORS


def generate(n_per_type: int = 1000, seed: int = 42,
             output_dir: str = "data/synthetic",
             types: list[str] | None = None) -> dict[str, Path]:
    """Generate synthetic training data.

    Args:
        n_per_type:  Number of samples per problem type.
        seed:        Random seed for reproducibility.
        output_dir:  Directory to write CSVs to.
        types:       List of type names to generate (None = all).

    Returns:
        Dict mapping type name to output file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {}
    all_rows = []

    for name, gen_cls in GENERATORS.items():
        if types and name not in types:
            continue

        # Derive per-type seed to reduce accidental correlation between types
        type_seed = seed + hash(name) % 10000
        gen = gen_cls(seed=type_seed)
        path = out / f"{name}.csv"

        rows = []
        for i in range(n_per_type):
            prompt, answer = gen.generate_one()
            row_id = f"syn_{name}_{i:06d}"
            rows.append((row_id, prompt, answer))
            all_rows.append((row_id, prompt, answer))

        # Write per-type CSV
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "prompt", "answer"])
            writer.writerows(rows)

        paths[name] = path
        print(f"Generated {n_per_type:,d} {name} samples -> {path}")

    # Write combined CSV
    combined_path = out / "combined.csv"
    with open(combined_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prompt", "answer"])
        writer.writerows(all_rows)

    print(f"\nCombined: {len(all_rows):,d} total samples -> {combined_path}")
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for all puzzle types"
    )
    parser.add_argument(
        "--n-per-type", type=int, default=1000,
        help="Number of samples per problem type (default: 1000)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=str, default="data/synthetic",
        help="Output directory (default: data/synthetic)"
    )
    parser.add_argument(
        "--types", nargs="+", default=None,
        help="Specific types to generate (default: all)"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run validation after generating"
    )
    args = parser.parse_args()

    generate(
        n_per_type=args.n_per_type,
        seed=args.seed,
        output_dir=args.output_dir,
        types=args.types,
    )

    if args.validate:
        print("\n--- Running validation ---\n")
        from generators.validate import validate_all
        validate_all(n_samples=50, seed=args.seed + 1)


if __name__ == "__main__":
    main()
