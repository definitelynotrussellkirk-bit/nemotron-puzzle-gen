#!/usr/bin/env python3
"""Augment encryption puzzles by re-encrypting with different ciphers.

Same plaintext words, different cipher mappings. Teaches the model
the ALGORITHM (build table → decode → match vocab) not specific letter patterns.

Usage:
    python3 -m generators.augment_enc_examples --n 50000
"""
import argparse
import json
import random
import string
import time
from datetime import datetime, timezone

from solvers.encryption import trace as enc_trace
from training.data import BOXED_INSTRUCTION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--output", type=str,
                        default="data/encryption/pool/generated/augmented_enc.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", type=str,
                        default="data/encryption/pool/competition/competition_traced.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    source_rows = [json.loads(l) for l in open(args.source)]
    print(f"Loaded {len(source_rows)} source rows")

    count = 0
    skipped = 0
    t0 = time.time()

    with open(args.output, 'w') as out:
        while count < args.n:
            src = rng.choice(source_rows)
            user_content = src['messages'][0]['content']

            # Extract the plaintext from the original trace
            # The original answer IS the plaintext query answer
            original_answer = src.get('answer', '')
            if not original_answer:
                skipped += 1
                continue

            # Create a new random cipher
            letters = list(string.ascii_lowercase)
            shuffled = letters[:]
            rng.shuffle(shuffled)
            new_cipher = dict(zip(letters, shuffled))
            new_decipher = {v: k for k, v in new_cipher.items()}

            # Re-encrypt the original prompt with the new cipher
            # Parse original prompt to extract examples and query
            lines = user_content.split('\n')
            new_lines = []
            for line in lines:
                if '->' in line and 'example' not in line.lower() and 'encrypt' not in line.lower():
                    # This is an example line: "ciphertext -> plaintext"
                    parts = line.split('->')
                    if len(parts) == 2:
                        # Re-encrypt the ciphertext with new cipher
                        # The plaintext stays the same
                        plaintext = parts[1].strip()
                        new_ciphertext = ''.join(
                            new_cipher.get(c, c) for c in plaintext
                        )
                        new_lines.append(f"{new_ciphertext} -> {plaintext}")
                    else:
                        new_lines.append(line)
                elif 'decrypt:' in line.lower() or 'determine' in line.lower():
                    # This is the query line — re-encrypt the cipher words
                    # Find cipher words in the original
                    # The query has cipher text that decodes to the answer
                    answer_words = original_answer.split()
                    new_query_words = []
                    for word in answer_words:
                        new_cipher_word = ''.join(new_cipher.get(c, c) for c in word)
                        new_query_words.append(new_cipher_word)
                    new_lines.append(f"Decrypt: {' '.join(new_query_words)}")
                else:
                    new_lines.append(line)

            new_prompt = '\n'.join(new_lines)

            # Generate trace for the new puzzle
            result = enc_trace(new_prompt)
            if result is None:
                skipped += 1
                continue

            reasoning, traced_answer = result
            if traced_answer.lower() != original_answer.lower():
                skipped += 1
                continue

            content = f"<think>\n{reasoning}\n</think>\n\\boxed{{{traced_answer}}}"

            row = {
                "messages": [
                    {"role": "user", "content": new_prompt + BOXED_INSTRUCTION},
                    {"role": "assistant", "content": content},
                ],
                "answer": traced_answer,
                "id": f"aug_enc_{src.get('id', '')}_{count:06d}",
                "puzzle_type": "encryption",
                "mode": "augmented_competition",
                "source_id": src.get('id', ''),
                "generator": "augment_enc_examples",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            out.write(json.dumps(row) + '\n')
            count += 1

            if count % 5000 == 0:
                print(f"  {count}/{args.n} ({time.time()-t0:.0f}s, {skipped} skipped)")

    dt = time.time() - t0
    print(f"Generated {count} augmented enc rows in {dt:.0f}s → {args.output}")
    print(f"Skipped: {skipped}")


if __name__ == '__main__':
    main()
