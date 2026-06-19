#!/usr/bin/env python3
"""Micro-skill: Does this rule fit ALL examples?

The model sees 3-5 (input, output) pairs and a proposed rule.
It must check each example and say yes/no.
This is THE core skill — the model currently fakes this check.

2K rows. Tagged [Alice's Training House].
"""
import json, random, re
from datetime import datetime, timezone

BYTE = 0xFF
TAG = "[Alice's Training House] "
DRILL = "[TRAINING DRILL]"

def rol(x, k): return ((x << k) | (x >> (8 - k))) & BYTE
def ror(x, k): return ((x >> k) | (x << (8 - k))) & BYTE

TRANSFORMS = {}
for k in range(1, 8):
    TRANSFORMS[f"shl{k}"] = lambda x, k=k: (x << k) & BYTE
    TRANSFORMS[f"shr{k}"] = lambda x, k=k: (x >> k) & BYTE
    TRANSFORMS[f"rol{k}"] = lambda x, k=k: rol(x, k)
    TRANSFORMS[f"ror{k}"] = lambda x, k=k: ror(x, k)
TRANSFORMS["x"] = lambda x: x

GATES = {
    "A ^ B":  lambda a, b: (a ^ b) & BYTE,
    "A & B":  lambda a, b: (a & b) & BYTE,
    "A | B":  lambda a, b: (a | b) & BYTE,
    "A & ~B": lambda a, b: (a & (~b & BYTE)) & BYTE,
    "~A & B": lambda a, b: ((~a & BYTE) & b) & BYTE,
}

def shift_str(name, bits):
    """Show shift as string operation."""
    k = int(re.search(r'\d+', name).group()) if re.search(r'\d+', name) else 0
    if name.startswith("shr"):
        return f"shr{k}: prepend {k} zeros, drop last {k} -> {'0'*k + bits[:-k]}"
    elif name.startswith("shl"):
        return f"shl{k}: drop first {k}, append {k} zeros -> {bits[k:] + '0'*k}"
    elif name.startswith("rol"):
        return f"rol{k}: move first {k} to end -> {bits[k:] + bits[:k]}"
    elif name.startswith("ror"):
        return f"ror{k}: move last {k} to front -> {bits[-k:] + bits[:-k]}"
    elif name == "x":
        return f"x = {bits}"
    return f"{name}({bits})"

def gate_str(gate_name, a_bits, b_bits):
    """Show gate position-by-position."""
    lines = []
    lines.append(f"  {' '.join(a_bits)}")
    lines.append(f"  {' '.join(b_bits)}")
    if "^" in gate_name:
        result = "".join("1" if a_bits[i] != b_bits[i] else "0" for i in range(8))
        lines.append(f"  {' '.join(result)}  (diff->1 same->0)")
    elif "& ~" in gate_name:
        result = "".join("1" if a_bits[i] == "1" and b_bits[i] == "0" else "0" for i in range(8))
        lines.append(f"  {' '.join(result)}  (A=1 and B=0 -> 1)")
    elif "~A &" in gate_name:
        result = "".join("1" if a_bits[i] == "0" and b_bits[i] == "1" else "0" for i in range(8))
        lines.append(f"  {' '.join(result)}  (A=0 and B=1 -> 1)")
    elif "&" in gate_name:
        result = "".join("1" if a_bits[i] == "1" and b_bits[i] == "1" else "0" for i in range(8))
        lines.append(f"  {' '.join(result)}  (both 1->1)")
    elif "|" in gate_name:
        result = "".join("1" if a_bits[i] == "1" or b_bits[i] == "1" else "0" for i in range(8))
        lines.append(f"  {' '.join(result)}  (either 1->1)")
    return result, lines


def generate(n=2000, seed=42):
    rng = random.Random(seed)
    results = []
    
    src_names = list(TRANSFORMS.keys())
    gate_names = list(GATES.keys())
    
    for i in range(n):
        # Pick a real rule
        src_a_name = rng.choice(src_names)
        src_b_name = rng.choice([s for s in src_names if s != src_a_name])
        gate_name = rng.choice(gate_names)
        
        src_a_fn = TRANSFORMS[src_a_name]
        src_b_fn = TRANSFORMS[src_b_name]
        gate_fn = GATES[gate_name]
        
        def compute(x):
            return gate_fn(src_a_fn(x), src_b_fn(x))
        
        # Generate 4 examples
        inputs = rng.sample(range(256), 4)
        examples = [(format(x, "08b"), format(compute(x), "08b")) for x in inputs]
        
        # 50% correct rule, 50% wrong rule (different gate or source)
        is_correct = rng.random() < 0.5
        
        if is_correct:
            proposed_a, proposed_b, proposed_gate = src_a_name, src_b_name, gate_name
        else:
            # Change one thing
            change = rng.choice(["gate", "src_a", "src_b"])
            if change == "gate":
                proposed_gate = rng.choice([g for g in gate_names if g != gate_name])
                proposed_a, proposed_b = src_a_name, src_b_name
            elif change == "src_a":
                proposed_a = rng.choice([s for s in src_names if s != src_a_name])
                proposed_b, proposed_gate = src_b_name, gate_name
            else:
                proposed_b = rng.choice([s for s in src_names if s != src_b_name])
                proposed_a, proposed_gate = src_a_name, gate_name
        
        proposed_a_fn = TRANSFORMS[proposed_a]
        proposed_b_fn = TRANSFORMS[proposed_b]
        proposed_gate_fn = GATES[proposed_gate]
        
        # Build the check trace — verify proposed rule against each example
        ex_str = "\n".join(f"  {inp} -> {out}" for inp, out in examples)
        rule_str = f"A = {proposed_a}(x), B = {proposed_b}(x), output = {proposed_gate}"
        
        think_lines = []
        all_pass = True
        first_fail = None
        
        for j, (inp, expected_out) in enumerate(examples):
            x = int(inp, 2)
            a = proposed_a_fn(x)
            b = proposed_b_fn(x)
            computed = proposed_gate_fn(a, b)
            computed_str = format(computed, "08b")
            
            a_str = format(a, "08b")
            b_str = format(b, "08b")
            
            if computed_str == expected_out:
                think_lines.append(f"Ex {j+1}: x={inp}")
                think_lines.append(f"  A = {shift_str(proposed_a, inp).split(' -> ')[-1] if '->' in shift_str(proposed_a, inp) else a_str}")
                think_lines.append(f"  B = {shift_str(proposed_b, inp).split(' -> ')[-1] if '->' in shift_str(proposed_b, inp) else b_str}")
                think_lines.append(f"  {proposed_gate} = {computed_str} = {expected_out} ✓")
            else:
                think_lines.append(f"Ex {j+1}: x={inp}")
                think_lines.append(f"  A = {shift_str(proposed_a, inp).split(' -> ')[-1] if '->' in shift_str(proposed_a, inp) else a_str}")
                think_lines.append(f"  B = {shift_str(proposed_b, inp).split(' -> ')[-1] if '->' in shift_str(proposed_b, inp) else b_str}")
                think_lines.append(f"  {proposed_gate} = {computed_str} != {expected_out} ✗ FAIL")
                all_pass = False
                if first_fail is None:
                    first_fail = j + 1
                break  # Stop at first failure
        
        if all_pass:
            think_lines.append(f"All {len(examples)} examples match -> rule fits")
            answer = "Yes"
        else:
            think_lines.append(f"Failed at example {first_fail} -> rule does NOT fit")
            answer = "No"
        
        user = f"Does this rule fit all examples?\nRule: {rule_str}\nExamples:\n{ex_str}"
        think = "\n".join(think_lines)
        
        r = {
            "messages": [
                {"role": "user", "content": TAG + user + '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'},
                {"role": "assistant", "content": f"<think>\n{DRILL}\n{think}\n</think>\n\\boxed{{{answer}}}"},
            ],
            "id": f"ms_bit_rule_check_{i:04d}",
            "puzzle_type": "bit_manipulation",
            "mode": "microskill_rule_check",
            "generator": "gen_ms_bit_rule_check",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        results.append(r)
    
    return results


if __name__ == "__main__":
    results = generate(2000, 42)
    outpath = "data/bit_manipulation/pool/generated/ms_rule_check.jsonl"
    with open(outpath, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    yes = sum(1 for r in results if "Yes" in r["messages"][1]["content"].split("boxed{")[1])
    print(f"Generated {len(results)} rule-check drills ({yes} yes, {len(results)-yes} no)")
    print(f"Written to {outpath}")
