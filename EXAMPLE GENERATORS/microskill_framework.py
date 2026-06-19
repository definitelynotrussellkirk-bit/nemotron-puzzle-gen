#!/usr/bin/env python3
"""Micro-skill generator framework.

Every micro-skill is a class that implements generate_one(rng, difficulty).
The framework handles tagging, formatting, validation, and output.

Usage:
    # Generate a specific skill
    python3 -m generators.microskill_framework --skill bit_shift --n 2000

    # Generate all skills
    python3 -m generators.microskill_framework --all --n 2000

    # List available skills
    python3 -m generators.microskill_framework --list
"""
import argparse
import json
import random
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path


BYTE = 0xFF
TAG = "[Alice's Training House] "
DRILL = "[TRAINING DRILL]"
BOXED_INSTRUCTION = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'


class MicroSkill(ABC):
    """Base class for all micro-skills."""

    name: str = ""           # unique skill ID
    puzzle_type: str = ""    # bit_manipulation, encryption, transformation, etc.
    description: str = ""    # human-readable description
    output_dir: str = ""     # relative path for output file
    weight: float = 1.0      # sampling weight (higher = picked more often)
    max_pool: int = 10000    # max rows to generate (limited by sample space)

    @abstractmethod
    def generate_one(self, rng: random.Random, difficulty: str = "medium") -> dict | None:
        """Generate one training example.

        Args:
            rng: seeded random instance
            difficulty: "easy", "medium", or "hard"
                easy: obvious answers, simple inputs
                medium: standard difficulty
                hard: near-misses, edge cases, adversarial distractors

        Returns:
            dict with keys: user, think, answer
            or None to skip (e.g., degenerate case)
        """
        pass

    def format_row(self, example: dict, idx: int) -> dict:
        """Format a generate_one result into a standard training row."""
        user = TAG + example["user"] + BOXED_INSTRUCTION
        think = f"{DRILL}\n{example['think']}"
        answer = example["answer"]
        assistant = f"<think>\n{think}\n</think>\n\\boxed{{{answer}}}"

        return {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "answer": answer,
            "id": f"ms_{self.name}_{idx:05d}",
            "puzzle_type": self.puzzle_type,
            "mode": f"microskill_{self.name}",
            "skill_name": self.name,
            "generator": f"ms_{self.name}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def generate(self, n: int = 2000, seed: int = 42,
                 difficulty: str = "medium",
                 difficulty_mix: dict | None = None) -> list[dict]:
        """Generate n examples.

        Args:
            n: number of examples to generate
            seed: random seed
            difficulty: fixed difficulty level
            difficulty_mix: override with {"easy": 0.2, "medium": 0.5, "hard": 0.3}
        """
        rng = random.Random(seed)
        results = []
        attempts = 0
        max_attempts = n * 3

        while len(results) < n and attempts < max_attempts:
            attempts += 1

            # Pick difficulty
            if difficulty_mix:
                d = rng.choices(
                    list(difficulty_mix.keys()),
                    weights=list(difficulty_mix.values())
                )[0]
            else:
                d = difficulty

            example = self.generate_one(rng, d)
            if example is None:
                continue

            row = self.format_row(example, len(results))
            results.append(row)

        return results

    def save(self, results: list[dict], path: str | None = None):
        """Save results to JSONL."""
        if path is None:
            path = self.output_dir
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"  {self.name}: {len(results)} -> {path}")


# ============================================================
# Shared utilities
# ============================================================

def rol(x, k):
    return ((x << k) | (x >> (8 - k))) & BYTE

def ror(x, k):
    return ((x >> k) | (x << (8 - k))) & BYTE

SHIFT_OPS = {}
for _k in range(1, 8):
    SHIFT_OPS[f"shl{_k}"] = lambda x, k=_k: (x << k) & BYTE
    SHIFT_OPS[f"shr{_k}"] = lambda x, k=_k: (x >> k) & BYTE
    SHIFT_OPS[f"rol{_k}"] = lambda x, k=_k: rol(x, k)
    SHIFT_OPS[f"ror{_k}"] = lambda x, k=_k: ror(x, k)

GATE_OPS = {
    "A ^ B":  lambda a, b: (a ^ b) & BYTE,
    "A & B":  lambda a, b: (a & b) & BYTE,
    "A | B":  lambda a, b: (a | b) & BYTE,
    "A & ~B": lambda a, b: (a & (~b & BYTE)) & BYTE,
    "~A & B": lambda a, b: ((~a & BYTE) & b) & BYTE,
}

def shift_str(name, bits):
    """Describe a shift as a string operation."""
    import re
    m = re.match(r'(shl|shr|rol|ror)(\d+)', name)
    if not m:
        return f"{name}({bits})"
    op, k = m.group(1), int(m.group(2))
    if op == "shr":
        result = "0" * k + bits[:-k]
        return f"shr{k}: prepend {k} zeros, drop last {k} -> {result}"
    elif op == "shl":
        result = bits[k:] + "0" * k
        return f"shl{k}: drop first {k}, append {k} zeros -> {result}"
    elif op == "rol":
        result = bits[k:] + bits[:k]
        return f"rol{k}: move first {k} to end -> {result}"
    elif op == "ror":
        result = bits[-k:] + bits[:-k]
        return f"ror{k}: move last {k} to front -> {result}"

def gate_position_by_position(a_bits, b_bits, gate_name):
    """Compute gate and return (result_bits, display_lines)."""
    lines = []
    lines.append(f"  {' '.join(a_bits)}")
    lines.append(f"  {' '.join(b_bits)}")
    if "^" in gate_name and "~" not in gate_name:
        result = "".join("1" if a_bits[i] != b_bits[i] else "0" for i in range(8))
        lines.append(f"  {' '.join(result)}  (diff->1 same->0)")
    elif "& ~" in gate_name:
        result = "".join("1" if a_bits[i] == "1" and b_bits[i] == "0" else "0" for i in range(8))
        lines.append(f"  {' '.join(result)}  (A=1 B=0 -> 1)")
    elif "~" in gate_name and "&" in gate_name:
        result = "".join("1" if a_bits[i] == "0" and b_bits[i] == "1" else "0" for i in range(8))
        lines.append(f"  {' '.join(result)}  (A=0 B=1 -> 1)")
    elif "&" in gate_name:
        result = "".join("1" if a_bits[i] == "1" and b_bits[i] == "1" else "0" for i in range(8))
        lines.append(f"  {' '.join(result)}  (both 1->1)")
    elif "|" in gate_name:
        result = "".join("1" if a_bits[i] == "1" or b_bits[i] == "1" else "0" for i in range(8))
        lines.append(f"  {' '.join(result)}  (either 1->1)")
    else:
        result = "0" * 8
    return result, lines


def load_vocab():
    """Load competition vocabulary."""
    for path in [
        "solvers/competition_vocabulary.txt",
        Path(__file__).parent.parent / "solvers" / "competition_vocabulary.txt",
    ]:
        try:
            with open(path) as f:
                return [l.strip().lower() for l in f if l.strip()]
        except FileNotFoundError:
            continue
    return []


# ============================================================
# Registry
# ============================================================

REGISTRY: dict[str, type[MicroSkill]] = {}

def register(cls):
    """Decorator to register a micro-skill class."""
    REGISTRY[cls.name] = cls
    return cls


# ============================================================
# CLI
# ============================================================

def _stable_hash(name: str) -> int:
    """Deterministic hash from skill name (stable across Python versions)."""
    import hashlib
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)


# Domain quotas: what fraction of total micro-skills each domain gets.
# Prevents bit skills (72 of 107) from crowding out other types.
# Within each domain, skills are sampled proportional to weight.
DOMAIN_QUOTAS = {
    "bit_manipulation":  0.58,   # 62% — primary failure mode
    "encryption":        0.18,   # 20% — second priority
    "transformation":    0.18,   # 12% — third priority
    "_other":            0.06,   # 6% — grav/unit/numconv at ~2% each (already 99-100%)
}


def sample_micro_skills(n: int, seed: int = 42, difficulty: str = "mixed") -> list[dict]:
    """Sample n micro-skill examples with two-level allocation.

    Level 1: allocate n across domains using DOMAIN_QUOTAS.
    Level 2: within each domain, allocate by skill weight.

    This prevents any single domain from crowding out others as
    skills are added, while still respecting relative weights within
    each domain.
    """
    from generators import microskill_skills  # noqa

    rng = random.Random(seed)
    difficulty_mix = {"easy": 0.2, "medium": 0.5, "hard": 0.3} if difficulty == "mixed" else None

    # Group skills by domain
    domains = {}
    for name, cls in REGISTRY.items():
        domain = cls.puzzle_type if cls.puzzle_type in DOMAIN_QUOTAS else "_other"
        domains.setdefault(domain, {})[name] = cls

    # Level 1: allocate counts per domain
    domain_counts = {}
    allocated = 0
    for domain in sorted(DOMAIN_QUOTAS):
        if domain not in domains:
            continue
        count = int(n * DOMAIN_QUOTAS[domain])
        domain_counts[domain] = count
        allocated += count
    # Distribute remainder to largest domain
    remainder = n - allocated
    if remainder > 0:
        largest = max(domain_counts, key=domain_counts.get)
        domain_counts[largest] += remainder

    # Level 2: within each domain, allocate by weight
    skill_counts = {}
    for domain, domain_n in domain_counts.items():
        skills = domains.get(domain, {})
        if not skills:
            continue
        total_w = sum(cls.weight for cls in skills.values())
        if total_w <= 0:
            continue

        remaining = domain_n
        for name in sorted(skills):
            count = int(domain_n * skills[name].weight / total_w)
            skill_counts[name] = count
            remaining -= count
        # Distribute remainder to highest-weight skills in this domain
        for name in sorted(skills, key=lambda k: -skills[k].weight):
            if remaining <= 0:
                break
            skill_counts[name] = skill_counts.get(name, 0) + 1
            remaining -= 1

    # Generate from each skill
    all_rows = []
    for name in sorted(skill_counts):
        count = skill_counts[name]
        if count <= 0:
            continue
        skill = REGISTRY[name]()
        results = skill.generate(
            n=count, seed=seed + _stable_hash(name),
            difficulty=difficulty if difficulty != "mixed" else "medium",
            difficulty_mix=difficulty_mix,
        )
        all_rows.extend(results)

    rng.shuffle(all_rows)
    return all_rows[:n]


def generate_all_pools(seed: int = 42, difficulty: str = "mixed"):
    """Generate max-size pools for ALL skills. Run once to populate pools."""
    from generators import microskill_skills  # noqa

    difficulty_mix = {"easy": 0.2, "medium": 0.5, "hard": 0.3}
    total = 0
    for name, cls in sorted(REGISTRY.items()):
        skill = cls()
        n = skill.max_pool
        results = skill.generate(n=n, seed=seed, difficulty="medium", difficulty_mix=difficulty_mix)
        skill.save(results)
        total += len(results)
    print(f"\nTotal: {total} rows across {len(REGISTRY)} skills")


def main():
    # Import all skill modules to populate REGISTRY
    from generators import microskill_skills  # noqa

    parser = argparse.ArgumentParser(description="Micro-skill generator")
    parser.add_argument("--skill", type=str, help="Skill name to generate")
    parser.add_argument("--all", action="store_true", help="Generate all skill pools (max size)")
    parser.add_argument("--sample", type=int, help="Sample N weighted micro-skill rows (for DNA)")
    parser.add_argument("--list", action="store_true", help="List available skills with weights")
    parser.add_argument("--n", type=int, default=2000, help="Pool size (if --skill)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=str, default="mixed",
                        choices=["easy", "medium", "hard", "mixed"])
    args = parser.parse_args()

    if args.list:
        print(f"{'Skill':35s} {'Type':20s} {'Weight':>6s} {'Pool':>6s} Description")
        print("-" * 110)
        for name, cls in sorted(REGISTRY.items()):
            print(f"  {name:33s} {cls.puzzle_type:20s} {cls.weight:6.1f} {cls.max_pool:6d} {cls.description}")
        print(f"\nTotal: {len(REGISTRY)} skills, total weight: {sum(c.weight for c in REGISTRY.values()):.1f}")
        return

    if args.sample:
        rows = sample_micro_skills(args.sample, args.seed, args.difficulty)
        print(f"Sampled {len(rows)} micro-skill rows")
        return

    difficulty_mix = {"easy": 0.2, "medium": 0.5, "hard": 0.3} if args.difficulty == "mixed" else None
    difficulty = args.difficulty if args.difficulty != "mixed" else "medium"

    if args.all:
        generate_all_pools(args.seed, args.difficulty)
    elif args.skill:
        if args.skill not in REGISTRY:
            print(f"Unknown skill: {args.skill}. Use --list to see available.")
            return
        skill = REGISTRY[args.skill]()
        results = skill.generate(n=args.n, seed=args.seed, difficulty=difficulty, difficulty_mix=difficulty_mix)
        skill.save(results)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
