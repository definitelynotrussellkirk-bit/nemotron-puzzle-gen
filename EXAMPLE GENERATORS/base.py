"""Base class for synthetic data generators."""

import csv
import random
from abc import ABC, abstractmethod
from pathlib import Path


class BaseGenerator(ABC):
    """Base class that all problem-type generators inherit from.

    Each generator produces (prompt, answer) pairs that mimic the
    competition's format: a short narrative with worked examples
    followed by a question, plus the ground-truth answer.
    """

    name: str = "base"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    @abstractmethod
    def generate_one(self) -> tuple[str, str]:
        """Return a single (prompt, answer) pair."""
        ...

    def generate(self, n: int) -> list[tuple[str, str]]:
        """Generate *n* (prompt, answer) pairs."""
        return [self.generate_one() for _ in range(n)]

    def to_csv(self, path: str | Path, n: int) -> Path:
        """Generate *n* pairs and write them to a CSV with columns id, prompt, answer."""
        path = Path(path)
        rows = self.generate(n)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "prompt", "answer"])
            for i, (prompt, answer) in enumerate(rows):
                writer.writerow([f"syn_{self.name}_{i:06d}", prompt, answer])
        return path
