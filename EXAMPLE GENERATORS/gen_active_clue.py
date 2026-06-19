#!/usr/bin/env python3
"""Active clue generation framework.

Shared pattern across bit, symbol, and encryption puzzles:
1. Start with a latent program (circuit, mapping, or cipher)
2. Generate a query
3. Greedily select examples that maximize information gain:
   - Each example eliminates the most wrong-at-query candidates
   - Stop when the query answer is uniquely determined, or
   - Stop when a target number of ambiguous bits/positions remain

This produces "teaching set" puzzles where every example is informative,
rather than random examples that might be redundant.

Usage:
    python3 -m generators.gen_active_clue --type bit --n 500
    python3 -m generators.gen_active_clue --type transformation --n 500
"""

import argparse
import json
import random
import time
from pathlib import Path


def generate_active_bit(rng, target_ambiguous=0):
    """Generate a bit puzzle with actively selected examples.

    Uses the existing bit generator infrastructure but replaces
    random example selection with greedy information-maximizing selection.
    """
    from generators.bit_manipulation import BitManipulationGenerator, _bits_to_str, _get_bit
    from solvers.bit_manipulation import _enumerate_candidates, trace as bm_trace
    from training.data import BOXED_INSTRUCTION

    gen = BitManipulationGenerator.__new__(BitManipulationGenerator)
    gen.rng = rng

    for attempt in range(30):
        circuit = gen._build_circuit()
        query = rng.randrange(256)
        answer = _bits_to_str(gen._apply_circuit(circuit, query))

        all_inputs = [x for x in range(256) if x != query]
        rng.shuffle(all_inputs)

        # Start with 2 random examples
        examples = [(x, gen._apply_circuit(circuit, x)) for x in all_inputs[:2]]

        # Greedily add examples that eliminate the most wrong candidates
        for _ in range(12):
            # Count current ambiguous bits
            query_bits = [_get_bit(query, pos) for pos in range(8)]
            input_bit_matrix = [[_get_bit(inp, pos) for inp, _ in examples] for pos in range(8)]

            n_ambiguous = 0
            for bp in range(8):
                true_bit = int(answer[bp])
                target_bits = [_get_bit(out, bp) for _, out in examples]
                cands = _enumerate_candidates(input_bit_matrix, target_bits, query_bits, bp)
                query_vals = set(qbit for _, _, qbit, _ in cands)
                if len(query_vals) > 1:
                    n_ambiguous += 1

            if n_ambiguous <= target_ambiguous:
                break

            # Find best next example
            best_input = None
            best_reduction = -1
            for x in rng.sample(all_inputs, min(40, len(all_inputs))):
                if any(x == inp for inp, _ in examples):
                    continue
                trial = examples + [(x, gen._apply_circuit(circuit, x))]
                trial_matrix = [[_get_bit(inp, pos) for inp, _ in trial] for pos in range(8)]

                trial_amb = 0
                for bp in range(8):
                    true_bit = int(answer[bp])
                    target_bits = [_get_bit(out, bp) for _, out in trial]
                    cands = _enumerate_candidates(trial_matrix, target_bits, query_bits, bp)
                    query_vals = set(qbit for _, _, qbit, _ in cands)
                    if len(query_vals) > 1:
                        trial_amb += 1

                reduction = n_ambiguous - trial_amb
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_input = x

            if best_input is not None:
                examples.append((best_input, gen._apply_circuit(circuit, best_input)))
            else:
                break

        # Check final ambiguity
        final_amb = 0
        input_bit_matrix = [[_get_bit(inp, pos) for inp, _ in examples] for pos in range(8)]
        query_bits = [_get_bit(query, pos) for pos in range(8)]
        for bp in range(8):
            target_bits = [_get_bit(out, bp) for _, out in examples]
            cands = _enumerate_candidates(input_bit_matrix, target_bits, query_bits, bp)
            query_vals = set(qbit for _, _, qbit, _ in cands)
            if len(query_vals) > 1:
                final_amb += 1

        if final_amb > target_ambiguous:
            continue

        rng.shuffle(examples)
        prompt = gen._format_prompt(examples, query)
        result = bm_trace(prompt)
        if result is None:
            continue

        reasoning, pred = result
        if pred != answer:
            # Solver disagrees with latent circuit — use ambiguity trace
            from solvers.bit_manipulation import trace_with_gold
            amb_result = trace_with_gold(prompt, answer)
            if amb_result:
                reasoning, pred = amb_result
            else:
                continue
        return {
            "messages": [
                {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"},
            ],
            "answer": answer,
            "id": f"gen_active_bit_{rng.randint(0,999999):06d}",
            "puzzle_type": "bit_manipulation",
            "mode": f"active_clue_amb{final_amb}",
            "n_examples": len(examples),
            "n_ambiguous_bits": final_amb,
            "generator": "gen_active_clue",
        }

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["bit"], default="bit")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--ambiguous", type=int, default=0, help="Target ambiguous bits (0=fully determined)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/bit_manipulation/active_clue.jsonl"

    rng = random.Random(int(time.time()))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    t0 = time.time()
    with open(output, "w") as out:
        for _ in range(args.n * 3):
            if count >= args.n:
                break
            result = generate_active_bit(rng, target_ambiguous=args.ambiguous)
            if result:
                out.write(json.dumps(result) + "\n")
                count += 1
                if count % 50 == 0:
                    print(f"  {count}/{args.n} ({time.time()-t0:.0f}s)", flush=True)

    print(f"Done: {count} active-clue examples in {time.time()-t0:.0f}s → {output}")


if __name__ == "__main__":
    main()
