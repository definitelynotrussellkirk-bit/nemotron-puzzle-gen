#!/usr/bin/env python3
"""Parallel bit manipulation data generator.

Uses multiprocessing to generate puzzles across all CPU cores.
Each worker generates puzzles independently with different seeds.
"""

import json
import multiprocessing as mp
import time
import sys
from pathlib import Path

# Must be importable at module level for multiprocessing
from generators.bit_manipulation import BitManipulationGenerator
from solvers.bit_manipulation import trace as bm_trace
from training.data import BOXED_INSTRUCTION


def _generate_batch(args):
    """Worker function: generate a batch of puzzles."""
    from datetime import datetime, timezone
    seed, batch_size, worker_id = args
    gen = BitManipulationGenerator(seed=seed)
    now = datetime.now(timezone.utc).isoformat()
    results = []

    for i in range(batch_size * 3):  # oversample
        if len(results) >= batch_size:
            break
        prompt, answer = gen.generate_one()
        trace_result = bm_trace(prompt)
        if trace_result and trace_result[1] == answer:
            reasoning, _ = trace_result
            msg = {
                'messages': [
                    {'role': 'user', 'content': prompt + BOXED_INSTRUCTION},
                    {'role': 'assistant', 'content': f'<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}'},
                ],
                'answer': answer,
                'id': f'gen_bit_manipulation_w{worker_id}_{i:06d}',
                'puzzle_type': 'bit_manipulation',
                'mode': 'regular',
                'generator': 'gen_bit_manipulation_parallel',
                'generated_at': now,
            }
            results.append(msg)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=5000)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--output', type=str,
                        default='data/bit_manipulation/pool/generated/parallel.jsonl')
    args = parser.parse_args()

    n_workers = args.workers or mp.cpu_count()
    target = args.target
    per_worker = (target + n_workers - 1) // n_workers

    print(f"Generating {target} bit manipulation examples")
    print(f"  Workers: {n_workers}, {per_worker} per worker")

    tasks = [(42 + i * 1000, per_worker, i) for i in range(n_workers)]

    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        batches = pool.map(_generate_batch, tasks)

    all_results = []
    for batch in batches:
        all_results.extend(batch)

    # Trim to target
    all_results = all_results[:target]

    # Re-number IDs
    for i, msg in enumerate(all_results):
        msg['id'] = f'gen_bit_manipulation_{i:06d}'

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        for msg in all_results:
            f.write(json.dumps(msg) + '\n')

    elapsed = time.time() - t0
    print(f"Done: {len(all_results)} examples in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Rate: {len(all_results)/elapsed:.1f} examples/s")
    print(f"  Output: {output}")


if __name__ == '__main__':
    main()
