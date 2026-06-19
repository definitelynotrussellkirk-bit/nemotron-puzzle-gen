#!/usr/bin/env python3
"""Generate witness-set bit manipulation training data.

Instead of maximizing posterior margin (current approach), this generator
chooses examples that ELIMINATE wrong-at-query candidates. Each example
is chosen to kill the maximum number of surviving wrong hypotheses.

This produces "hard but deterministic" puzzles — the minimal proof
that the query answer must be what it is.

Usage:
    python3 -m generators.gen_bit_witness --n 500
"""

from datetime import datetime, timezone
import argparse
import json
import random
import time
from pathlib import Path

from generators.bit_manipulation import BitManipulationGenerator, _bits_to_str, _get_bit
from solvers.bit_manipulation import _enumerate_candidates, trace as bm_trace
from training.data import BOXED_INSTRUCTION


def _count_wrong_survivors(examples, query, circuit, gen):
    """Count candidates that are consistent with examples but wrong on query."""
    query_output = gen._apply_circuit(circuit, query)
    input_bit_matrix = [[_get_bit(inp, pos) for inp, _ in examples] for pos in range(8)]
    query_bits = [_get_bit(query, pos) for pos in range(8)]

    total_wrong = 0
    for bp in range(8):
        true_bit = _get_bit(query_output, bp)
        target_bits = [_get_bit(out, bp) for _, out in examples]
        cands = _enumerate_candidates(input_bit_matrix, target_bits, query_bits, bp)

        for fam, inputs, qbit, mass in cands:
            if qbit != true_bit:
                total_wrong += 1

    return total_wrong


def _build_witness_trace(examples, query, circuit, gen, answer):
    """Build a trace showing which examples kill which wrong candidates.

    For each output bit, identifies the true function and shows which
    examples eliminate the strongest competing hypotheses.
    """
    query_output = gen._apply_circuit(circuit, query)
    input_bit_matrix = [[_get_bit(inp, pos) for inp, _ in examples] for pos in range(8)]
    query_bits = [_get_bit(query, pos) for pos in range(8)]
    n = len(examples)

    lines = [f"Bit rule. Witness-minimal proof. Ex[{n}] Q={_bits_to_str(query)}"]

    for bp in range(8):
        true_bit = _get_bit(query_output, bp)
        target_bits = [_get_bit(out, bp) for _, out in examples]
        spec = circuit[bp]
        true_fam = spec.get('family', '?')
        true_inputs = spec.get('inputs', ())

        # Format true function
        if true_fam in ('CONST_0', 'CONST_1'):
            true_app = f"={true_bit}"
        elif true_fam == 'COPY' and len(true_inputs) == 1:
            true_app = f"=({true_inputs[0]})\u2192{query_bits[true_inputs[0]]}"
        elif true_fam == 'NOT' and len(true_inputs) == 1:
            true_app = f"!({true_inputs[0]})\u2192{1 - query_bits[true_inputs[0]]}"
        elif len(true_inputs) == 2:
            i, j = true_inputs
            true_app = f"{true_fam}({i},{j})\u2192{true_bit}"
        elif len(true_inputs) == 3:
            i, j, k = true_inputs
            true_app = f"{true_fam}({i},{j},{k})\u2192{true_bit}"
        else:
            true_app = f"{true_fam}\u2192{true_bit}"

        # Find which examples were critical for this bit
        # An example is "critical" if removing it would allow a wrong candidate to survive
        cands = _enumerate_candidates(input_bit_matrix, target_bits, query_bits, bp)
        wrong_cands = [(fam, inp, qbit, mass) for fam, inp, qbit, mass in cands
                       if qbit != true_bit]

        if not wrong_cands:
            # No wrong candidates — bit is easy
            lines.append(f"  b{bp}: {true_app}")
        elif len(wrong_cands) <= 2:
            # Show which examples kill the wrong candidates
            kill_parts = []
            for wfam, winp, wqbit, wmass in wrong_cands[:2]:
                # Find which example(s) contradict this wrong candidate
                for ex_idx, (inp, out) in enumerate(examples):
                    inp_bits = [_get_bit(inp, p) for p in range(8)]
                    out_bit = _get_bit(out, bp)
                    # Check if this wrong candidate would predict differently
                    if wfam in ('CONST_0', 'CONST_1'):
                        wrong_pred = 0 if wfam == 'CONST_0' else 1
                    elif wfam == 'COPY' and len(winp) == 1:
                        wrong_pred = inp_bits[winp[0]]
                    elif wfam == 'NOT' and len(winp) == 1:
                        wrong_pred = 1 - inp_bits[winp[0]]
                    else:
                        continue  # complex — skip detail
                    if wrong_pred != out_bit:
                        kill_parts.append(f"{wfam} killed by ex {_bits_to_str(inp)}")
                        break

            if kill_parts:
                lines.append(f"  b{bp}: {true_app} ({'; '.join(kill_parts)})")
            else:
                lines.append(f"  b{bp}: {true_app} ({len(wrong_cands)} wrong ruled out)")
        else:
            lines.append(f"  b{bp}: {true_app} ({len(wrong_cands)} wrong ruled out)")

    return "\n".join(lines) + f"\n\n\\boxed{{{answer}}}"


def _bits_unanimous(examples, query, circuit, gen):
    """Check if all consistent candidates agree on every query bit."""
    query_output = gen._apply_circuit(circuit, query)
    input_bit_matrix = [[_get_bit(inp, pos) for inp, _ in examples] for pos in range(8)]
    query_bits = [_get_bit(query, pos) for pos in range(8)]

    for bp in range(8):
        true_bit = _get_bit(query_output, bp)
        target_bits = [_get_bit(out, bp) for _, out in examples]
        cands = _enumerate_candidates(input_bit_matrix, target_bits, query_bits, bp)

        if any(qbit != true_bit for _, _, qbit, _ in cands):
            return False
    return True


def generate_witness_puzzle(rng):
    """Generate a puzzle with witness-minimal examples."""
    gen = BitManipulationGenerator.__new__(BitManipulationGenerator)
    gen.rng = rng

    for circuit_attempt in range(20):
        circuit = gen._build_circuit()
        query = rng.randrange(256)

        # Start with a small seed set (3-4 diverse examples)
        all_inputs = [x for x in range(256) if x != query]
        rng.shuffle(all_inputs)
        examples = [(x, gen._apply_circuit(circuit, x)) for x in all_inputs[:3]]

        # Greedily add examples that kill the most wrong-at-query candidates
        for _ in range(12):  # max 15 total examples
            if _bits_unanimous(examples, query, circuit, gen):
                break

            # Try candidate inputs, pick the one that kills the most wrong survivors
            best_input = None
            best_killed = -1

            candidates = rng.sample(all_inputs, min(30, len(all_inputs)))
            current_wrong = _count_wrong_survivors(examples, query, circuit, gen)

            for x in candidates:
                if any(x == inp for inp, _ in examples):
                    continue
                trial = examples + [(x, gen._apply_circuit(circuit, x))]
                trial_wrong = _count_wrong_survivors(trial, query, circuit, gen)
                killed = current_wrong - trial_wrong
                if killed > best_killed:
                    best_killed = killed
                    best_input = x

            if best_input is None or best_killed <= 0:
                # Add a random one
                for x in all_inputs:
                    if not any(x == inp for inp, _ in examples):
                        best_input = x
                        break

            if best_input is not None:
                examples.append((best_input, gen._apply_circuit(circuit, best_input)))

        # Check if we achieved unanimity
        if not _bits_unanimous(examples, query, circuit, gen):
            continue

        rng.shuffle(examples)
        prompt = gen._format_prompt(examples, query)
        answer = _bits_to_str(gen._apply_circuit(circuit, query))

        # Get trace — only keep if solver agrees (compact format)
        result = bm_trace(prompt)
        if result is None:
            continue

        reasoning, pred = result
        if pred != answer:
            continue

        return {
            "messages": [
                {"role": "user", "content": prompt + BOXED_INSTRUCTION},
                {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"},
            ],
            "answer": answer,
            "id": f"gen_bit_witness_{rng.randint(0,999999):06d}",
            "puzzle_type": "bit_manipulation",
            "mode": "witness_minimal",
            "n_examples": len(examples),
            "generator": "gen_bit_witness",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--output", type=str,
                        default="data/bit_manipulation/pool/generated/witness.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    t0 = time.time()
    with open(output, "w") as out:
        for i in range(args.n * 3):
            if count >= args.n:
                break
            result = generate_witness_puzzle(rng)
            if result:
                out.write(json.dumps(result) + "\n")
                count += 1
                if count % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  {count}/{args.n} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"Done: {count} witness examples in {elapsed:.0f}s → {output}")


if __name__ == "__main__":
    main()
