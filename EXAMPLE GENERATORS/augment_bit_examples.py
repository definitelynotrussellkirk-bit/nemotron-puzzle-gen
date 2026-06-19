#!/usr/bin/env python3
"""Augment bit competition traces by varying which examples are shown.

For each competition puzzle, the rule is known (from the solver).
We can generate ANY valid input→output pair for that rule.
This script creates augmented versions with different example subsets,
teaching the model the ALGORITHM not the specific examples.

Usage:
    python3 -m generators.augment_bit_examples --n 5000 --output data/bit_manipulation/pool/generated/augmented_competition.jsonl
"""
import argparse
import json
import random
import re
import time
from datetime import datetime, timezone

from training.data import BOXED_INSTRUCTION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--output", type=str,
                        default="data/bit_manipulation/pool/generated/augmented_competition.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", type=str,
                        default="data/bit_manipulation/pool/competition/competition_traced.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load competition traced rows — they have the rule embedded in the trace
    source_rows = []
    with open(args.source) as f:
        for line in f:
            r = json.loads(line)
            source_rows.append(r)

    print(f"Loaded {len(source_rows)} source rows")

    # For each source row, extract the rule from the trace,
    # then generate new examples using that rule
    count = 0
    skipped = 0
    t0 = time.time()

    with open(args.output, 'w') as out:
        while count < args.n:
            # Pick a random source row
            src = rng.choice(source_rows)
            user = src['messages'][0]['content']
            asst = src['messages'][1]['content']
            answer = src.get('answer', '')

            # Extract the original examples and query from the prompt
            orig_examples = re.findall(r'([01]{8})\s*->\s*([01]{8})', user)
            query_m = re.search(r'(?:output for|determine the output).*?([01]{8})', user)
            if not orig_examples or not query_m:
                skipped += 1
                continue

            query_str = query_m.group(1)

            # We know the rule produces these outputs for these inputs.
            # Build a lookup: for each input, compute output
            rule_map = {inp: out for inp, out in orig_examples}
            rule_map[query_str] = answer

            # Try to infer the rule by testing random inputs against known examples
            # Simple approach: just use the known examples but SHUFFLE and SUBSAMPLE
            n_new = rng.randint(7, 10)

            # Generate new random inputs and compute outputs using the ORIGINAL examples
            # as a truth oracle. Since we can't easily extract the boolean function,
            # we'll resample from a larger set.

            # Actually, we CAN generate new examples if we know the transformation.
            # But extracting the rule from the trace is complex.
            # Simpler approach: just resample WHICH of the original examples to show.

            if len(orig_examples) < 7:
                skipped += 1
                continue

            # Resample: pick n_new examples from the original set (different subset each time)
            available = list(orig_examples)
            rng.shuffle(available)
            new_examples = available[:n_new]

            # Build new prompt with same format but different example subset/order
            prompt_lines = [
                "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. "
                "The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, "
                "and possibly majority or choice functions.",
                "",
                "Here are some examples of input -> output:",
            ]
            for inp, out_val in new_examples:
                prompt_lines.append(f"{inp} -> {out_val}")
            prompt_lines.append(f"\nNow, determine the output for: {query_str}")
            new_prompt = "\n".join(prompt_lines)

            # REGENERATE the trace from the new examples — DO NOT reuse old trace
            # The old trace references old examples in Scan, which creates prompt/trace mismatch
            # Extract rule info from original trace and rebuild
            import re as _re
            # Try to extract sources and gate from original trace
            src_matches = _re.findall(r'([A-C])\s*=\s*(?:~)?(\w+)\(x\)', asst)
            gate_match = _re.search(r'output\s*=\s*(\w+)\(', asst)

            if src_matches and gate_match:
                from generators.trace_compact import build_compact_trace
                src_names = [m[1] for m in src_matches]
                complements = ['~' in asst.split(m[1])[0].split('\n')[-1] for m in src_matches]
                gate = gate_match.group(1)

                # Build fresh trace from extracted rule + new examples
                try:
                    trace_result = build_compact_trace(
                        [(chr(65+i), src_names[i], complements[i]) for i in range(len(src_names))],
                        gate if len(src_names) <= 2 else {"family": gate, "inputs": [chr(65+i) for i in range(len(src_names))]},
                        new_examples,
                        query_str,
                        seed=rng.randint(0, 999999),
                    )
                except Exception:
                    trace_result = None

                if trace_result is None:
                    skipped += 1
                    continue

                if isinstance(trace_result, tuple) and len(trace_result) >= 2:
                    new_trace, trace_answer = trace_result[0], trace_result[1]
                    if trace_answer != answer:
                        skipped += 1
                        continue
                    new_asst = f"<think>\n{new_trace}\n</think>\n\\boxed{{{answer}}}"
                else:
                    skipped += 1
                    continue
            else:
                # Can't extract rule — skip this row
                skipped += 1
                continue

            augmented = {
                "messages": [
                    {"role": "user", "content": new_prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": new_asst},
                ],
                "answer": answer,
                "id": f"aug_bit_{src.get('id', '')}_{count:06d}",
                "puzzle_type": "bit_manipulation",
                "mode": "augmented_competition",
                "source_id": src.get('id', ''),
                "generator": "augment_bit_examples",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            out.write(json.dumps(augmented) + '\n')
            count += 1

            if count % 1000 == 0:
                print(f"  {count}/{args.n}")

    dt = time.time() - t0
    print(f"Generated {count} augmented rows in {dt:.1f}s → {args.output}")
    print(f"Skipped: {skipped}")


if __name__ == '__main__':
    main()
