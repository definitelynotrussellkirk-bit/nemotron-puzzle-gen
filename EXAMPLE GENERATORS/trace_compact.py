#!/usr/bin/env python3
"""Compact bit trace format — high gradient density.

Format:
    Scan:
    Ex1: IIIIIIII→OOOOOOOO ones:N→N diff=N same=N set=N clr=N
    ...

    Rule:
      A = src1(x)
      B = src2(x)
      output = A gate B

    Check 1: x=IIIIIIII
      A=src1(IIIIIIII)=RRRRRRRR
      B=src2(IIIIIIII)=RRRRRRRR
      GRID(A,B,gate):
      R R R R R R R R
      R R R R R R R R
      R R R R R R R R
      =OOOOOOOO ones=N → MATCH

    Query: x=IIIIIIII ones=N
      A=src1(IIIIIIII)=RRRRRRRR
      B=src2(IIIIIIII)=RRRRRRRR
      GRID(A,B,gate):
      R R R R R R R R
      R R R R R R R R
      R R R R R R R R
      =OOOOOOOO ones=N delta=±N match=N/8
"""

import re
import random

BYTE = 0xFF

# ── Primitives ──

def shl(x, k): return ((x << k) & BYTE)
def shr(x, k): return ((x >> k) & BYTE)
def rol(x, k): return ((x << k) | (x >> (8 - k))) & BYTE
def ror(x, k): return ((x >> k) | (x << (8 - k))) & BYTE

def apply_shift(x_int, src_name):
    m = re.match(r'(shl|shr|rol|ror)(\d+)', src_name)
    if not m: return x_int
    op, k = m.group(1), int(m.group(2))
    return {'shl': shl, 'shr': shr, 'rol': rol, 'ror': ror}[op](x_int, k)

def apply_gate(a_str, b_str, gate):
    r = []
    for i in range(8):
        a, b = int(a_str[i]), int(b_str[i])
        if gate == 'xor':       r.append(str(a ^ b))
        elif gate == 'xnor':    r.append(str(1 - (a ^ b)))
        elif gate == 'and':     r.append(str(a & b))
        elif gate == 'or':      r.append(str(a | b))
        elif gate == 'nand':    r.append(str(1 - (a & b)))
        elif gate == 'nor':     r.append(str(1 - (a | b)))
        elif gate == 'and_not': r.append(str(a & (1 - b)))
        elif gate == 'or_not':  r.append(str(a | (1 - b)))
        else: r.append('0')
    return ''.join(r)

def fmt(x): return format(x, '08b')
def spaced(s): return ' '.join(s)
def ones(s): return sum(1 for c in s if c == '1')


# ── Scan preamble ──

def scan_line(idx, inp, out):
    """One scan line: Ex1: IIII→OOOO ones:N→N diff=N same=N set=N clr=N"""
    in_ones = ones(inp)
    out_ones = ones(out)
    diff_n = sum(1 for a, b in zip(inp, out) if a != b)
    set_n = sum(1 for a, b in zip(inp, out) if a == '0' and b == '1')
    clr_n = sum(1 for a, b in zip(inp, out) if a == '1' and b == '0')
    return f"Ex{idx}: {inp}→{out} ones:{in_ones}→{out_ones} diff={diff_n} set={set_n} clr={clr_n}"

def build_scan(examples, max_scan=3):
    """Build Scan: block from examples list. Limit to max_scan for token budget."""
    lines = ["Scan:"]
    scan_examples = examples[:max_scan]
    for i, (inp, out) in enumerate(scan_examples):
        lines.append(scan_line(i + 1, inp, out))
    return lines


# ── Shift lines ──

def shift_line(label, src_name, x_str):
    """A=shr3(10110011)=00010110"""
    result = fmt(apply_shift(int(x_str, 2), src_name))
    return f"{label}={src_name}({x_str})={result}", result

def not_line(label, src_str):
    """~A=not(00010110)=11101001"""
    result = ''.join('1' if c == '0' else '0' for c in src_str)
    return f"{label}=not({src_str})={result}", result


# ── GRID blocks ──

def grid_2(a_str, b_str, gate, a_label="A", b_label="B"):
    """GRID block for 2-input gate. Compact: no spaces between bits."""
    result = apply_gate(a_str, b_str, gate)
    lines = [
        f"GRID({a_label},{b_label},{gate}):",
        a_str,
        b_str,
        result,
        f"={result}",
    ]
    return lines, result


def apply_family_3(a_str, b_str, c_str, family):
    """Apply a 3-input family function position-by-position."""
    result = []
    for i in range(8):
        a, b, c = int(a_str[i]), int(b_str[i]), int(c_str[i])
        if family == 'or_xnor':
            r = c | (1 - (a ^ b))  # C | XNOR(A,B)
        elif family == 'gated_xnor_nand':
            r = (c | (1 - (a ^ b))) & (1 - (a & b & c))
        elif family == 'ch':
            r = (a & b) | ((1 - a) & c)
        elif family == 'maj':
            r = (a & b) | (a & c) | (b & c)
        elif family == 'tt121':
            r = ((1 - a) & (1 - (b ^ c))) | (a & (1 - (b & c)))
        elif family == 't1':
            r = (1 - (a ^ b ^ c)) | ((1 - a) & (1 - b) & c)
        else:
            r = 0
        result.append(str(r & 1))
    return ''.join(result)


def grid_3(a_str, b_str, c_str, family, a_label="A", b_label="B", c_label="C"):
    """GRID block for 3-input family. Compact: no spaces between bits."""
    result = apply_family_3(a_str, b_str, c_str, family)
    lines = [
        f"GRID({a_label},{b_label},{c_label},{family}):",
        a_str,
        b_str,
        c_str,
        result,
        f"={result}",
    ]
    return lines, result


# ── Formula computation ──

def compute_sources(sources, x_str):
    """Compute all source values. Returns dict label→result_str and trace lines."""
    vals = {}
    lines = []
    for label, src_name, is_comp in sources:
        if src_name == 'x':
            val = x_str
            lines.append(f"{label}=x={x_str}")
        else:
            sl, val = shift_line(label, src_name, x_str)
            lines.append(sl)
        if is_comp:
            nl, val = not_line(f"~{label}", val)
            lines.append(nl)
            vals[label] = val  # store complemented version under original label
        else:
            vals[label] = val
    return vals, lines

def compute_formula(formula, vals, lines):
    """Compute gate formula, append GRID blocks. Returns output string.

    formula types:
      {"gate": "xor", "inputs": ["A", "B"]}                    — 2-input
      {"family": "or_xnor", "inputs": ["A", "B", "C"]}         — 3-input family
      {"chain": [{"gate": "xnor", "inputs": ["A","B"], "out": "P"},
                 {"gate": "or", "inputs": ["C","P"]}]}          — chained 2-input
      {"not_of": {"gate": "or", "inputs": ["A","B"]}}           — NOT of 2-input
    """
    if "family" in formula:
        # 3-input family — single GRID with 4 rows (A,B,C,result)
        fam = formula["family"]
        a_label, b_label, c_label = formula["inputs"]
        gl, result = grid_3(vals[a_label], vals[b_label], vals[c_label],
                           fam, a_label, b_label, c_label)
        lines.extend(gl)
        return result
    elif "chain" in formula:
        for step in formula["chain"]:
            g = step["gate"]
            a_label, b_label = step["inputs"]
            out_label = step.get("out", "output")
            gl, result = grid_2(vals[a_label], vals[b_label], g, a_label, b_label)
            lines.extend(gl)
            vals[out_label] = result
        return result
    elif "not_of" in formula:
        inner = formula["not_of"]
        if "family" in inner:
            fam = inner["family"]
            labels = inner["inputs"]
            gl, result = grid_3(vals[labels[0]], vals[labels[1]], vals[labels[2]],
                               fam, *labels)
            lines.extend(gl)
        else:
            g = inner["gate"]
            a_label, b_label = inner["inputs"]
            gl, result = grid_2(vals[a_label], vals[b_label], g, a_label, b_label)
            lines.extend(gl)
        nr = ''.join('1' if c == '0' else '0' for c in result)
        lines.append(f"not({result})={nr}")
        return nr
    else:
        g = formula["gate"]
        a_label, b_label = formula["inputs"]
        gl, result = grid_2(vals[a_label], vals[b_label], g, a_label, b_label)
        lines.extend(gl)
        return result


# ── Check / Query / Full trace ──

def _xor_bits(a: str, b: str) -> str:
    """XOR two 8-bit binary strings."""
    return ''.join('0' if x == y else '1' for x, y in zip(a, b))


def compact_check(sources, formula, x_str, expected, check_num, example_label=None):
    """Build a compact witness block with two-pass verify.

    Two-pass: model produces output FIRST (committed), then sees expected,
    then must compute XOR diff. Prevents fake-verify by forcing sequential
    commitment: compute → compare → decide.
    """
    label = f" ({example_label})" if example_label else ""
    lines = [f"Witness {check_num}{label}: x={x_str}"]
    vals, src_lines = compute_sources(sources, x_str)
    lines.extend(src_lines)
    output = compute_formula(formula, vals, lines)
    # Three-pass verify: output → expected → per-bit XOR → verdict
    # Model must compute each XOR bit, can't fabricate diff=00000000
    lines.append(f"  output={output}")
    lines.append(f"  expected={expected}")
    # Show per-bit XOR so model can't fake the diff
    xor_parts = ' '.join(f"{o}⊕{e}={int(o)^int(e)}" for o, e in zip(output, expected))
    diff = _xor_bits(output, expected)
    lines.append(f"  XOR: {xor_parts}")
    lines.append(f"  diff={diff} → {'PASS' if diff == '00000000' else 'FAIL'}")
    return lines

def compact_query(sources, formula, x_str):
    """Build a compact query block — no decorative stats.

    Dropped ones=/delta=/match= from query (R13: creates room for
    heuristic storytelling, not load-bearing for verification).
    """
    lines = [f"Query: x={x_str}"]
    vals, src_lines = compute_sources(sources, x_str)
    lines.extend(src_lines)
    output = compute_formula(formula, vals, lines)
    return lines, output

# ── Witness scoring: rank examples by how many rivals they kill ──

def _chain_to_family(formula):
    """Convert a chain formula to its equivalent 3-input family (if known).

    Chain formulas like P=xnor(A,B)→or(C,P) are semantically identical to
    the or_xnor family. This lets EXCLUDE work on chain formulas by treating
    them as families for rival selection and output computation.
    """
    if "chain" not in formula:
        return None
    steps = formula["chain"]
    if len(steps) != 2:
        return None
    inner_gate = steps[0].get("gate", "")
    outer_gate = steps[1].get("gate", "")
    # Map known chain patterns to families
    chain_key = f"{outer_gate}({inner_gate})"
    CHAIN_TO_FAMILY = {
        "or(xnor)": "or_xnor",
        "and(xnor)": "gated_xnor_nand",  # close enough for rival purposes
        "or(and)": "maj",  # not exact but useful for EXCLUDE
    }
    return CHAIN_TO_FAMILY.get(chain_key)


def _get_rivals(formula):
    """Get rival gates/families for a formula."""
    if "family" in formula:
        return _RIVAL_FAMILY.get(formula["family"], [])
    elif "gate" in formula:
        return _RIVAL_2INPUT.get(formula["gate"], [])
    elif "chain" in formula:
        fam = _chain_to_family(formula)
        if fam:
            return _RIVAL_FAMILY.get(fam, [])
    return []


def _compute_output(sources, formula_override, x_str):
    """Compute output for a given formula on input x_str."""
    vals = {}
    for label, src_name, is_comp in sources:
        v = fmt(apply_shift(int(x_str, 2), src_name))
        if is_comp:
            v = ''.join('1' if c == '0' else '0' for c in v)
        vals[label] = v

    if "gate" in formula_override:
        return apply_gate(vals.get('A','00000000'), vals.get('B','00000000'), formula_override["gate"])
    elif "family" in formula_override:
        return apply_family_3(
            vals.get('A','00000000'), vals.get('B','00000000'),
            vals.get('C','00000000'), formula_override["family"])
    elif "chain" in formula_override:
        # Compute chain step by step
        for step in formula_override["chain"]:
            g = step["gate"]
            a_l, b_l = step["inputs"]
            out_l = step.get("out", "output")
            vals[out_l] = apply_gate(vals.get(a_l,'00000000'), vals.get(b_l,'00000000'), g)
        return vals.get("output", vals.get(out_l))
    return None


def _score_witness_power(sources, formula, examples):
    """Score each example by how many rival rules it kills.

    Returns list of (idx, kill_count, best_killed_rival) sorted by kill_count desc.
    """
    rivals = _get_rivals(formula)
    if not rivals:
        return [(i, 0, None) for i in range(len(examples))]

    scores = []
    for i, (inp, expected) in enumerate(examples):
        kills = 0
        best_rival = None
        for rival in rivals:
            if "family" in formula:
                rival_formula = {"family": rival, "inputs": formula.get("inputs", ["A","B","C"])}
            else:
                rival_formula = {"gate": rival, "inputs": formula.get("inputs", ["A","B"])}
            rival_out = _compute_output(sources, rival_formula, inp)
            if rival_out is not None and rival_out != expected:
                kills += 1
                best_rival = rival
        scores.append((i, kills, best_rival))

    scores.sort(key=lambda x: -x[1])
    return scores


def _most_confusable_rival(sources, formula, examples):
    """Find the rival that survives the MOST examples (hardest to distinguish).

    Returns (rival_name, list_of_examples_where_it_survives).
    """
    rivals = _get_rivals(formula)
    if not rivals:
        return None, []

    best_rival = None
    best_survive_count = -1

    for rival in rivals:
        if "family" in formula:
            rival_formula = {"family": rival, "inputs": formula.get("inputs", ["A","B","C"])}
        else:
            rival_formula = {"gate": rival, "inputs": formula.get("inputs", ["A","B"])}

        survives = 0
        for inp, expected in examples:
            rival_out = _compute_output(sources, rival_formula, inp)
            if rival_out == expected:
                survives += 1

        if survives > best_survive_count:
            best_survive_count = survives
            best_rival = rival

    return best_rival, best_survive_count


# ── EXCLUDE block: show a wrong gate failing on a witness example ──

_RIVAL_2INPUT = {
    'xor': ['and', 'or'],
    'and': ['xor', 'or'],
    'or': ['xor', 'and'],
    'xnor': ['and', 'or'],
    'nand': ['xor', 'or'],
    'nor': ['xor', 'and'],
    'and_not': ['xor', 'and'],
    'or_not': ['xor', 'or'],
}

_RIVAL_FAMILY = {
    'or_xnor': ['ch', 'maj'],
    'gated_xnor_nand': ['or_xnor', 'ch'],
    'ch': ['maj', 'or_xnor'],
    'maj': ['ch', 'or_xnor'],
    'tt121': ['or_xnor', 'ch'],
    't1': ['or_xnor', 'maj'],
}


def _build_source_exclude(sources, formula, inp, expected, rng):
    """Exclude a wrong SOURCE: swap one source for a different shift, show it fails."""
    import re as _re
    all_shifts = [f"shl{k}" for k in range(1,8)] + [f"shr{k}" for k in range(1,8)] + \
                 [f"rol{k}" for k in range(1,8)] + [f"ror{k}" for k in range(1,8)]

    # Pick which source to swap
    src_idx = rng.randrange(len(sources))
    label, real_src, is_comp = sources[src_idx]
    candidates = [s for s in all_shifts if s != real_src]
    if not candidates:
        return []
    wrong_src = rng.choice(candidates)

    # Compute with wrong source
    wrong_sources = list(sources)
    wrong_sources[src_idx] = (label, wrong_src, is_comp)

    vals = {}
    for lbl, sn, ic in wrong_sources:
        v = fmt(apply_shift(int(inp, 2), sn))
        if ic:
            v = ''.join('1' if c == '0' else '0' for c in v)
        vals[lbl] = v

    if "gate" in formula:
        wrong_out = apply_gate(vals['A'], vals['B'], formula['gate'])
    elif "family" in formula:
        wrong_out = _apply_family_str(vals['A'], vals['B'], vals.get('C','00000000'), formula['family'])
    else:
        return []

    if wrong_out == expected:
        return []

    return [f"Try {label}={wrong_src}: {wrong_out} vs {expected} → EXCLUDE (rejected)"]


def _build_exclude(sources, formula, examples, rng):
    """Build an EXCLUDE line using witness-powered selection.

    Picks the MOST CONFUSABLE rival and the BEST WITNESS example that kills it.
    ~70% of traces get an exclude block.
    """
    if rng.random() > 0.7:
        return []

    # 20% source-exclude for variety
    if rng.random() < 0.2 and len(sources) >= 2:
        idx = rng.randrange(len(examples))
        return _build_source_exclude(sources, formula, examples[idx][0], examples[idx][1], rng)

    # Chain formulas: convert to equivalent family for rival selection
    effective_formula = formula
    if "chain" in formula:
        fam = _chain_to_family(formula)
        if fam is None or len(sources) < 3:
            return []
        effective_formula = {"family": fam, "inputs": ["A", "B", "C"]}

    # Find most confusable rival (survives most examples)
    confusable_rival, _ = _most_confusable_rival(sources, effective_formula, examples)
    if confusable_rival is None:
        return []

    # Find best witness example that kills this rival
    if "family" in effective_formula:
        rival_formula = {"family": confusable_rival, "inputs": effective_formula.get("inputs", ["A","B","C"])}
    else:
        rival_formula = {"gate": confusable_rival, "inputs": effective_formula.get("inputs", ["A","B"])}

    best_idx = None
    for i, (inp, expected) in enumerate(examples):
        rival_out = _compute_output(sources, rival_formula, inp)
        if rival_out is not None and rival_out != expected:
            best_idx = i
            break  # first killer is fine

    if best_idx is None:
        # Confusable rival survives ALL examples — can't exclude it
        # Fall back to any rival on any example
        rivals = _get_rivals(effective_formula)
        rng.shuffle(rivals)
        for wrong_gate in rivals:
            idx = rng.randrange(len(examples))
            inp, expected = examples[idx]
            if "family" in effective_formula:
                rf = {"family": wrong_gate, "inputs": effective_formula.get("inputs",["A","B","C"])}
            else:
                rf = {"gate": wrong_gate, "inputs": effective_formula.get("inputs",["A","B"])}
            rival_out = _compute_output(sources, rf, inp)
            if rival_out is not None and rival_out != expected:
                return [f"Try {wrong_gate}: {rival_out} vs {expected} → EXCLUDE (rejected)"]
        return []

    inp, expected = examples[best_idx]
    wrong_gate = confusable_rival

    vals = {}
    for label, src_name, is_comp in sources:
        v = fmt(apply_shift(int(inp, 2), src_name))
        if is_comp:
            v = ''.join('1' if c == '0' else '0' for c in v)
        vals[label] = v

    if len(sources) == 2:
        wrong_out = apply_gate(vals['A'], vals['B'], wrong_gate)
    elif len(sources) == 3:
        wrong_out = _apply_family_str(vals['A'], vals['B'], vals['C'], wrong_gate)
    else:
        return []

    if wrong_out == expected:
        return []

    lines = [f"Try {wrong_gate}: {wrong_out} vs {expected} → EXCLUDE (rejected)"]
    return lines


def _apply_family_str(a, b, c, family):
    """Apply a 3-input family to 8-bit strings. Uses apply_family_3 for correctness."""
    return apply_family_3(a, b, c, family)


def build_compact_trace(sources, formula, examples, query_str, seed=None):
    """Build a complete compact trace with Scan preamble.

    Returns (trace_text, answer_str) or (trace_text, answer_str, metadata) if metadata requested.
    Metadata includes witness_strength for stratification.
    """
    rng = random.Random(seed) if seed is not None else random.Random(hash(query_str))
    lines = []

    # Type preamble — helps model commit to the right template
    lines.append("Bit rule.")
    lines.append("")

    # Scan preamble
    scan = build_scan(examples)
    lines.extend(scan)
    lines.append("")

    # Mechanical elimination preamble — like grav's "extract rate" but for bit.
    # Each step is zero-search, pure extraction. Covers 62% of competition mechanically.
    outputs = [out for _, out in examples]
    inputs = [inp for inp, _ in examples]

    # Step 1: CONST?
    if len(set(outputs)) == 1:
        lines.append(f"Step 1: all outputs same? Yes → {outputs[0]}")
    else:
        lines.append(f"Step 1: all outputs same? No ({len(set(outputs))} distinct)")

    # Step 2: Identity?
    identity_matches = sum(1 for i, o in zip(inputs, outputs) if i == o)
    if identity_matches == len(examples):
        lines.append(f"Step 2: output=input? Yes (all {len(examples)} match)")
    else:
        # Show first mismatch
        for inp, out in examples:
            if inp != out:
                n_diff = sum(1 for a, b in zip(inp, out) if a != b)
                lines.append(f"Step 2: output=input? No (Ex1 differs at {n_diff} positions)")
                break

    # Step 3: NOT?
    not_matches = sum(1 for i, o in zip(inputs, outputs)
                      if o == ''.join('1' if c == '0' else '0' for c in i))
    if not_matches == len(examples):
        lines.append(f"Step 3: output=NOT(input)? Yes")
    else:
        lines.append(f"Step 3: output=NOT(input)? No")

    lines.append("")

    # EXCLUDE block: randomly pick a wrong gate, show it fails on a random example.
    # This teaches the model that rejection comes BEFORE acceptance.
    exclude_lines = _build_exclude(sources, formula, examples, rng)
    if exclude_lines:
        lines.extend(exclude_lines)
        lines.append("")

    # MANDATORY BACKTRACKING: 100% of traces show at least one failed candidate.
    # R14 feedback: the model NEVER sees FAIL→Query in training, only FAIL→STOP→new candidate.
    # This is THE fix for the #1 failure mode (93% of wrong bit rows proceed after FAIL).
    candidate_num = 0
    effective_formula = formula
    if "chain" in formula:
        fam = _chain_to_family(formula)
        if fam and len(sources) >= 3:
            effective_formula = {"family": fam, "inputs": ["A", "B", "C"]}

    # Always show at least 1 failed candidate, sometimes 2
    rivals_to_show = []
    rival1, _ = _most_confusable_rival(sources, effective_formula, examples)
    if rival1:
        rivals_to_show.append(rival1)
        # 30% show a second failed candidate
        all_gates = ["xor", "and", "or", "xnor", "nand", "nor"]
        correct_gate = formula.get("gate", formula.get("family", ""))
        other_gates = [g for g in all_gates if g != correct_gate and g != rival1]
        if rng.random() < 0.30 and other_gates:
            rivals_to_show.append(rng.choice(other_gates))

    for rival in rivals_to_show:
        candidate_num += 1
        if "family" in effective_formula:
            rival_formula = {"family": rival, "inputs": effective_formula.get("inputs", ["A","B","C"])}
        else:
            rival_formula = {"gate": rival, "inputs": effective_formula.get("inputs", ["A","B"])}

        # Find a witness that kills this rival
        for wi, (winp, wexp) in enumerate(examples):
            rival_out = _compute_output(sources, rival_formula, winp)
            if rival_out is not None and rival_out != wexp:
                # Full rule specification — sources + gate (B1: never just gate name)
                src_parts = []
                for label, src_name, is_comp in sources:
                    prefix = "~" if is_comp else ""
                    src_parts.append(f"{label}={prefix}{src_name}(x)")
                lines.append(f"Try[{candidate_num}]: {', '.join(src_parts)}, gate={rival}")
                lines.append(f"  Witness (Ex{wi+1}): x={winp}")
                wvals = {}
                for label, src_name, is_comp in sources:
                    v = fmt(apply_shift(int(winp, 2), src_name))
                    if is_comp:
                        v = ''.join('1' if c == '0' else '0' for c in v)
                    wvals[label] = v
                if len(sources) == 2:
                    lines.append(f"  {rival}({wvals['A']},{wvals['B']})={rival_out}")
                elif len(sources) == 3:
                    lines.append(f"  {rival}({wvals['A']},{wvals['B']},{wvals['C']})={rival_out}")
                diff_bt = _xor_bits(rival_out, wexp)
                lines.append(f"  expected={wexp} diff={diff_bt} → FAIL")
                lines.append(f"  Decision[{candidate_num}]: REJECT")
                lines.append("")
                break
    candidate_num += 1  # correct candidate gets next number

    # Rule declaration
    rule_parts = []
    for label, src_name, is_comp in sources:
        prefix = "~" if is_comp else ""
        if src_name == 'x':
            rule_parts.append(f"{label} = {prefix}x")
        else:
            rule_parts.append(f"{label} = {prefix}{src_name}(x)")
    # Format formula for display
    if "chain" in formula:
        steps = formula["chain"]
        formula_str = " → ".join(f"{s.get('out','output')}={s['gate']}({','.join(s['inputs'])})" for s in steps)
    elif "family" in formula:
        formula_str = f"{formula['family']}({','.join(formula['inputs'])})"
    elif "not_of" in formula:
        inner = formula["not_of"]
        if "family" in inner:
            formula_str = f"not({inner['family']}({','.join(inner['inputs'])}))"
        else:
            formula_str = f"not({inner['gate']}({','.join(inner['inputs'])}))"
    else:
        formula_str = f"{formula['gate']}({','.join(formula['inputs'])})"

    lines.append(f"Try[{candidate_num}]:")
    for rp in rule_parts:
        lines.append(f"  {rp}")
    lines.append(f"  output = {formula_str}")
    lines.append("")

    # Witness-powered checks: Check 1 = best witness, Check 2 = diversity
    # Also use effective_formula for chain formulas
    effective_formula = formula
    if "chain" in formula:
        fam = _chain_to_family(formula)
        if fam and len(sources) >= 3:
            effective_formula = {"family": fam, "inputs": ["A", "B", "C"]}
    witness_scores = _score_witness_power(sources, effective_formula, examples)
    max_witness_kills = witness_scores[0][1] if witness_scores else 0
    n_checks = 3  # 3 checks — 2 witnesses let 6.6% of wrong gates pass, 3 cuts it to 2%

    if witness_scores and witness_scores[0][1] > 0:
        check1_idx = witness_scores[0][0]
        check2_idx = witness_scores[1][0] if len(witness_scores) > 1 and witness_scores[1][1] > 0 else None
        if check2_idx is None:
            remaining = [i for i in range(len(examples)) if i != check1_idx]
            check2_idx = rng.choice(remaining) if remaining else check1_idx
        # Check 3: different from 1 and 2
        remaining3 = [i for i in range(len(examples)) if i != check1_idx and i != check2_idx]
        check3_idx = rng.choice(remaining3) if remaining3 else check2_idx
        check_indices = [check1_idx, check2_idx, check3_idx][:n_checks]
    else:
        check_indices = rng.sample(range(len(examples)), min(n_checks, len(examples)))

    for ci, idx in enumerate(check_indices):
        inp, expected = examples[idx]
        check_lines = compact_check(sources, formula, inp, expected, ci + 1,
                                    example_label=f"Ex{idx+1}")
        lines.extend(check_lines)
        lines.append("")

    # Elimination table: 30% of traces show rivals dying on witnesses
    # Uses extra token budget to make LOCK more principled (Idea #5)
    if rng.random() < 0.30 and "gate" in formula:
        correct_gate = formula["gate"]
        rivals = _get_rivals(formula)
        if rivals:
            killed_lines = []
            check_inp = examples[check_indices[0]][0]  # use first witness input
            check_exp = examples[check_indices[0]][1]
            for rival in rivals[:3]:  # max 3 rivals shown
                if rival == correct_gate:
                    continue
                rival_formula = {"gate": rival, "inputs": formula.get("inputs", ["A", "B"])}
                rival_out = _compute_output(sources, rival_formula, check_inp)
                if rival_out and rival_out != check_exp:
                    diff_r = _xor_bits(rival_out, check_exp)
                    killed_lines.append(f"  {rival}: {rival_out} diff={diff_r} → dead")
            if killed_lines:
                lines.append("Rivals on W1:")
                lines.extend(killed_lines)
                lines.append(f"  {correct_gate} survives both witnesses.")
                lines.append("")

    # Decision + explicit LOCK object (B2: LOCK exists as named object before Query)
    lines.append(f"Decision[{candidate_num}]: LOCK")
    lines.append(f"LOCK[{candidate_num}]:")
    for rp in rule_parts:
        lines.append(f"  {rp}")
    lines.append(f"  output = {formula_str}")
    lines.append("")

    # Query (only allowed after LOCK — must reference LOCK by number)
    lines.append(f"Query (use LOCK[{candidate_num}]):")
    _, answer = compact_query(sources, formula, query_str)
    # compact_query returns (lines, answer) — we already added the header
    q_lines, _ = compact_query(sources, formula, query_str)
    lines.extend(q_lines[1:])  # skip the "Query: x=..." line since we replaced it

    # Witness strength metadata for stratification
    # w0 = no rivals killed, w1 = weak, w2 = moderate, w3+ = strong
    if max_witness_kills == 0:
        witness_strength = "w0"
    elif max_witness_kills == 1:
        witness_strength = "w1"
    elif max_witness_kills == 2:
        witness_strength = "w2"
    else:
        witness_strength = f"w{max_witness_kills}"

    return '\n'.join(lines), answer, {"witness_strength": witness_strength,
                                       "max_witness_kills": max_witness_kills,
                                       "has_exclude": "EXCLUDE" in '\n'.join(lines)}


# ── Convenience: build from solver-style args ──

def build_trace_from_solver(src_names, src_complements, gate_or_formula, examples, query_str, seed=None):
    """Build compact trace from solver-style arguments.

    src_names: ['shr3', 'rol2'] or ['shr3', 'rol2', 'shl1']
    src_complements: [False, False] or [True, False, False]
    gate_or_formula: 'xor' or a formula dict for 3-input
    examples: [(inp_str, out_str), ...]
    query_str: 8-bit string
    """
    labels = [chr(65 + i) for i in range(len(src_names))]  # A, B, C
    sources = list(zip(labels, src_names, src_complements))

    if isinstance(gate_or_formula, str):
        formula = {"gate": gate_or_formula, "inputs": labels[:2]}
    else:
        formula = gate_or_formula

    trace, answer, metadata = build_compact_trace(sources, formula, examples, query_str, seed=seed)
    return trace, answer


def build_trace_from_solver_with_meta(src_names, src_complements, gate_or_formula, examples, query_str, seed=None):
    """Same as build_trace_from_solver but also returns witness metadata."""
    labels = [chr(65 + i) for i in range(len(src_names))]
    sources = list(zip(labels, src_names, src_complements))
    if isinstance(gate_or_formula, str):
        formula = {"gate": gate_or_formula, "inputs": labels[:2]}
    else:
        formula = gate_or_formula
    return build_compact_trace(sources, formula, examples, query_str, seed=seed)


# ── Test ──

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('unsloth/Nemotron-3-Nano-30B-A3B', trust_remote_code=True)

    # 2-input
    examples = [("11100000", ""), ("11001010", ""), ("10110011", ""),
                ("01010101", ""), ("00001111", "")]
    for i, (inp, _) in enumerate(examples):
        a = fmt(apply_shift(int(inp, 2), "shr3"))
        b = fmt(apply_shift(int(inp, 2), "rol2"))
        examples[i] = (inp, apply_gate(a, b, "xor"))

    trace, answer = build_trace_from_solver(
        ["shr3", "rol2"], [False, False], "xor",
        examples, "10110011", seed=42)
    print("=== 2-INPUT ===")
    print(trace)
    print(f"\nAnswer: {answer}")
    print(f"Exact tokens: {len(tok.encode(trace))}")

    # 3-input
    print("\n=== 3-INPUT ===")
    examples3 = [("10110011", ""), ("11001010", ""), ("11100000", "")]
    formula3 = {"chain": [
        {"gate": "xnor", "inputs": ["A", "B"], "out": "P"},
        {"gate": "or", "inputs": ["C", "P"]},
    ]}
    for i, (inp, _) in enumerate(examples3):
        a = fmt(apply_shift(int(inp, 2), "shr3"))
        b = fmt(apply_shift(int(inp, 2), "rol2"))
        c = fmt(apply_shift(int(inp, 2), "shl1"))
        p = apply_gate(a, b, "xnor")
        examples3[i] = (inp, apply_gate(c, p, "or"))

    trace3, answer3 = build_trace_from_solver(
        ["shr3", "rol2", "shl1"], [False, False, False], formula3,
        examples3, "01010101", seed=42)
    print(trace3)
    print(f"\nAnswer: {answer3}")
    print(f"Exact tokens: {len(tok.encode(trace3))}")

    # 2-input with complement
    print("\n=== 2-INPUT WITH COMPLEMENT ===")
    examples_c = [("11100000", ""), ("11001010", "")]
    for i, (inp, _) in enumerate(examples_c):
        a = fmt(apply_shift(int(inp, 2), "shr3"))
        b_raw = fmt(apply_shift(int(inp, 2), "rol2"))
        b = ''.join('1' if c == '0' else '0' for c in b_raw)
        examples_c[i] = (inp, apply_gate(a, b, "and"))

    trace_c, answer_c = build_trace_from_solver(
        ["shr3", "rol2"], [False, True],
        {"gate": "and", "inputs": ["A", "B"]},
        examples_c, "10110011", seed=42)
    print(trace_c)
    print(f"\nAnswer: {answer_c}")
    print(f"Exact tokens: {len(tok.encode(trace_c))}")
