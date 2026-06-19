"""Generator for bit manipulation reasoning problems.

Each puzzle hides an 8-bit boolean circuit where every output bit depends on
at most 3 input bits. Unlike the original random-example version, this
generator now chooses examples using the hidden circuit so the query answer is
usually identifiable in answer mode rather than left to heuristic guesswork.
"""

import random

from .base import BaseGenerator


def _get_bit(val, pos):
    return (val >> (7 - pos)) & 1


def _bits_to_str(val):
    return format(val & 0xFF, "08b")


def _weighted_choice(rng, weights):
    total = sum(weights)
    pick = rng.uniform(0, total)
    cumulative = 0.0
    for idx, weight in enumerate(weights):
        cumulative += weight
        if pick <= cumulative:
            return idx
    return len(weights) - 1


def _eval_tt(tt, pattern):
    return (tt >> pattern) & 1


def _apply_bit_spec(spec, input_val):
    family = spec["family"]
    inputs = spec["inputs"]
    if family == "CONST_0":
        return 0
    if family == "CONST_1":
        return 1
    if family == "COPY":
        return _get_bit(input_val, inputs[0])
    if family == "NOT":
        return 1 - _get_bit(input_val, inputs[0])
    if "tt" in spec:
        pattern = 0
        for pos in inputs:
            pattern = (pattern << 1) | _get_bit(input_val, pos)
        return _eval_tt(spec["tt"], pattern)
    raise ValueError(f"unknown spec family: {family}")


BIT_FUNCTION_MENU = [
    ("CONST_0", 1439),
    ("CONST_1", 457),
    ("COPY", 5280),
    ("NOT", 722),
    ("AND", 1149),
    ("XOR", 1094),
    ("XNOR", 858),
    ("OR", 630),
    ("NAND", 285),
    ("NOR", 249),
    ("a_OR_NOTb", 111),
    ("NOTa_AND_b", 69),
    ("NOTa_OR_b", 64),
    ("a_AND_NOTb", 46),
    ("MAJ3", 80),
    ("CH", 63),
    ("NOT_CH", 50),
    ("NOT_MAJ3", 24),
    ("TT3_random", 146),
]

_TT2_BY_FAMILY = {
    "AND": 0b0001,
    "XOR": 0b0110,
    "XNOR": 0b1001,
    "OR": 0b0111,
    "NAND": 0b1110,
    "NOR": 0b1000,
    "a_OR_NOTb": 0b1101,
    "NOTa_AND_b": 0b0100,
    "NOTa_OR_b": 0b1011,
    "a_AND_NOTb": 0b0010,
}

_TT3_BY_FAMILY = {
    "MAJ3": 0b11101000,
    "CH": 0b11000101,
    "NOT_CH": 0b00111010,
    "NOT_MAJ3": 0b00010111,
}


class BitManipulationGenerator(BaseGenerator):
    """Generates solver-friendly bit manipulation puzzles."""

    name = "bit_manipulation"

    def _make_spec(self, family):
        if family == "CONST_0":
            return {"family": family, "inputs": (), "tt": 0}
        if family == "CONST_1":
            return {"family": family, "inputs": (), "tt": 1}
        if family in {"COPY", "NOT"}:
            return {"family": family, "inputs": (self.rng.randint(0, 7),)}
        if family in _TT2_BY_FAMILY:
            left, right = self.rng.sample(range(8), 2)
            return {
                "family": family,
                "inputs": (left, right),
                "tt": _TT2_BY_FAMILY[family],
            }
        if family in _TT3_BY_FAMILY:
            first, second, third = self.rng.sample(range(8), 3)
            return {
                "family": family,
                "inputs": (first, second, third),
                "tt": _TT3_BY_FAMILY[family],
            }
        if family == "TT3_random":
            first, second, third = self.rng.sample(range(8), 3)
            return {
                "family": family,
                "inputs": (first, second, third),
                "tt": self.rng.randint(0, 255),
            }
        raise ValueError(f"unknown family: {family}")

    def _build_circuit(self):
        weights = [weight for _, weight in BIT_FUNCTION_MENU]
        families = [name for name, _ in BIT_FUNCTION_MENU]
        return [self._make_spec(families[_weighted_choice(self.rng, weights)]) for _ in range(8)]

    def _apply_circuit(self, circuit, input_val):
        output = 0
        for pos, spec in enumerate(circuit):
            output |= (_apply_bit_spec(spec, input_val) << (7 - pos))
        return output & 0xFF

    def _candidate_inputs(self, circuit, query_input):
        query_bits = list(_bits_to_str(query_input))
        candidates = set()

        for spec in circuit:
            support = spec["inputs"]
            if not support:
                continue
            for pattern in range(1 << len(support)):
                bits = list(query_bits)
                for idx, pos in enumerate(support):
                    bits[pos] = "1" if ((pattern >> (len(support) - idx - 1)) & 1) else "0"
                value = int("".join(bits), 2)
                if value != query_input:
                    candidates.add(value)

        while len(candidates) < 48:
            value = self.rng.randrange(256)
            if value != query_input:
                candidates.add(value)

        return list(candidates)

    def _seed_examples(self, circuit, query_input, candidate_inputs):
        covered = set()
        chosen = []
        remaining = list(candidate_inputs)

        while len(chosen) < 4 and remaining:
            best_input = None
            best_gain = -1
            for value in remaining:
                gain = 0
                for out_pos, spec in enumerate(circuit):
                    support = spec["inputs"]
                    if not support:
                        continue
                    pattern = tuple(_get_bit(value, pos) for pos in support)
                    key = (out_pos, support, pattern)
                    if key not in covered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_input = value
            if best_input is None:
                break
            chosen.append(best_input)
            remaining.remove(best_input)
            for out_pos, spec in enumerate(circuit):
                support = spec["inputs"]
                if not support:
                    continue
                pattern = tuple(_get_bit(best_input, pos) for pos in support)
                covered.add((out_pos, support, pattern))

        return chosen

    def _format_prompt(self, examples, query_input):
        lines = [f"{_bits_to_str(inp)} -> {_bits_to_str(out)}" for inp, out in examples]
        return (
            "In Alice's Wonderland, a secret bit manipulation rule transforms "
            "8-bit binary numbers. The transformation involves operations like "
            "bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority "
            "or choice functions.\n\n"
            "Here are some examples of input -> output:\n"
            + "\n".join(lines)
            + f"\n\nNow, determine the output for: {_bits_to_str(query_input)}"
        )

    def _design_examples(self, circuit, query_input, max_examples):
        from solvers.bit_manipulation import solve_details

        query_output = _bits_to_str(self._apply_circuit(circuit, query_input))
        query_bits = [int(bit) for bit in query_output]

        candidate_inputs = self._candidate_inputs(circuit, query_input)
        outputs = {value: self._apply_circuit(circuit, value) for value in candidate_inputs}

        chosen_inputs = self._seed_examples(circuit, query_input, candidate_inputs)
        chosen_set = set(chosen_inputs)

        while len(chosen_inputs) <= max_examples:
            if len(chosen_inputs) >= 4:
                examples = [(value, outputs[value]) for value in chosen_inputs]
                prompt = self._format_prompt(examples, query_input)
                details = solve_details(prompt)
                if details is not None and all(bit["mode"] == "answer" for bit in details["bit_details"]):
                    return examples

            if len(chosen_inputs) == max_examples:
                break

            best_input = None
            best_score = None
            for value in candidate_inputs:
                if value in chosen_set:
                    continue
                trial_inputs = chosen_inputs + [value]
                if len(trial_inputs) < 4:
                    continue
                examples = [(inp, outputs[inp]) for inp in trial_inputs]
                prompt = self._format_prompt(examples, query_input)
                details = solve_details(prompt)
                if details is None:
                    continue
                answer_bits = sum(1 for bit in details["bit_details"] if bit["mode"] == "answer")
                hidden_margin = 0.0
                for out_pos, bit_detail in enumerate(details["bit_details"]):
                    masses = bit_detail["masses"]
                    hidden_margin += masses[query_bits[out_pos]] - masses[1 - query_bits[out_pos]]
                score = (answer_bits, hidden_margin)
                if best_score is None or score > best_score:
                    best_score = score
                    best_input = value

            if best_input is None:
                break
            chosen_inputs.append(best_input)
            chosen_set.add(best_input)

        return None

    def generate_one(self):
        for _ in range(20):
            circuit = self._build_circuit()
            query_input = self.rng.randrange(256)
            examples = self._design_examples(circuit, query_input, max_examples=14)
            if examples is None:
                continue

            self.rng.shuffle(examples)
            prompt = self._format_prompt(examples, query_input)
            answer = _bits_to_str(self._apply_circuit(circuit, query_input))
            return prompt, answer

        # Fallback: emit a random circuit with random examples.
        circuit = self._build_circuit()
        inputs = self.rng.sample(range(256), 9)
        examples = [(value, self._apply_circuit(circuit, value)) for value in inputs[:8]]
        query_input = inputs[8]
        return self._format_prompt(examples, query_input), _bits_to_str(self._apply_circuit(circuit, query_input))
