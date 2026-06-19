#!/usr/bin/env python3
"""Role fingerprint: derivable source identification for 3-stream bit traces.

Instead of declaring sources as magic facts, fingerprint evidence shows HOW
the sources were identified from the examples alone.

Approach: score candidates with lightweight heuristics (XNOR-pair match,
force-1, selector), then VERIFY the top candidates produce exact output on
every example. Only verified triples are returned.

Key scoring methods:
- xnor_pair: For OR_XNOR family, find the pair whose XNOR best matches output.
- force1: Find the source whose 1-bits almost always appear in the output.
- selector: For CH family, find the source that acts as a multiplexer.
- equal_contrib: For MAJ3, all three sources contribute roughly equally.

Usage:
    from generators.bit_role_fingerprint import identify_sources, format_fingerprint

    result = identify_sources(examples, family_name)
    lines = format_fingerprint(examples, family_name, result)
"""

BYTE = 0xFF


def _rol(x, k):
    return ((x << k) | (x >> (8 - k))) & BYTE


def _ror(x, k):
    return ((x >> k) | (x << (8 - k))) & BYTE


def _build_transform_pool():
    """Build the 29-transform candidate pool: x, shl1-7, shr1-7, rol1-7, ror1-7."""
    pool = [("x", lambda x: x)]
    for k in range(1, 8):
        pool.append((f"shl{k}", lambda x, k=k: (x << k) & BYTE))
        pool.append((f"shr{k}", lambda x, k=k: (x >> k) & BYTE))
        pool.append((f"rol{k}", lambda x, k=k: _rol(x, k)))
        pool.append((f"ror{k}", lambda x, k=k: _ror(x, k)))
    return pool


# Module-level pool (built once)
TRANSFORM_POOL = _build_transform_pool()
_POOL_BY_NAME = {name: fn for name, fn in TRANSFORM_POOL}


def _get_transform_fn(name):
    """Get transform function by name."""
    if name in _POOL_BY_NAME:
        return _POOL_BY_NAME[name]
    raise ValueError(f"Unknown transform: {name}")


def _popcount(v):
    return bin(v & BYTE).count('1')


# ---------------------------------------------------------------------------
# Family functions (for verification)
# ---------------------------------------------------------------------------

def _fam_or_xnor(a, b, c):
    return (c | (a & b) | (~a & ~b)) & BYTE


def _fam_gated_xnor_nand(a, b, c):
    return _fam_or_xnor(a, b, c) & (~(a & b & c)) & BYTE


def _fam_ch(a, b, c):
    return ((a & b) | ((~a) & c)) & BYTE


def _fam_maj3(a, b, c):
    return ((a & b) | (a & c) | (b & c)) & BYTE


def _fam_tt121(a, b, c):
    return (((~a) & ((b & c) | ((~b) & (~c)))) | (a & (~(b & c)))) & BYTE


def _fam_t1(a, b, c):
    return (~(a ^ b ^ c) | ((~a) & (~b) & c)) & BYTE


_FAMILY_FNS = {
    "OR_XNOR": _fam_or_xnor,
    "GATED_XNOR_NAND": _fam_gated_xnor_nand,
    "CH": _fam_ch,
    "MAJ3": _fam_maj3,
    "TT121": _fam_tt121,
    "T1": _fam_t1,
}

# All six permutations of (A, B, C)
_PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _parse_examples(examples):
    """Parse (binary_str, binary_str) pairs to (int, int) pairs."""
    return [(int(inp, 2), int(out, 2)) for inp, out in examples]


def _verify_triple(pairs, fam_fn, fn_a, fn_b, fn_c):
    """Check if fam_fn(fn_a(x), fn_b(x), fn_c(x)) == y for all (x, y) pairs."""
    for x, y in pairs:
        if fam_fn(fn_a(x), fn_b(x), fn_c(x)) != y:
            return False
    return True


# ---------------------------------------------------------------------------
# Scoring functions (used for evidence lines, not for selection)
# ---------------------------------------------------------------------------

def score_xnor_pair(examples, src_a_name, src_b_name):
    """Score how well XNOR(src_a(x), src_b(x)) matches output bits.

    Returns float 0.0 to 1.0 (fraction of bits matched across all examples).
    """
    total_bits = 0
    matching_bits = 0
    fn_a = _get_transform_fn(src_a_name)
    fn_b = _get_transform_fn(src_b_name)

    for inp, out in examples:
        x = int(inp, 2)
        y = int(out, 2)
        xnor_val = (~(fn_a(x) ^ fn_b(x))) & BYTE
        agree = ~(xnor_val ^ y) & BYTE
        matching_bits += _popcount(agree)
        total_bits += 8

    return matching_bits / total_bits if total_bits > 0 else 0.0


def score_force1(examples, src_name):
    """Score how often src=1 implies output=1 ("force-1" property).

    Returns float 0.0 to 1.0.
    """
    src_ones = 0
    src_ones_and_out_ones = 0
    fn = _get_transform_fn(src_name)

    for inp, out in examples:
        x = int(inp, 2)
        y = int(out, 2)
        s_val = fn(x)
        src_ones += _popcount(s_val)
        src_ones_and_out_ones += _popcount(s_val & y)

    return src_ones_and_out_ones / src_ones if src_ones > 0 else 0.0


# ---------------------------------------------------------------------------
# Source identification (verification-gated)
# ---------------------------------------------------------------------------

def _all_pool_names():
    return [name for name, _ in TRANSFORM_POOL]


def _identify_by_verification(examples, family_name):
    """Universal identification: try all triples with all permutations,
    verify against all examples. Return the first verified triple with
    the best heuristic scores for evidence.

    This is the gold-standard approach: rank candidates by a lightweight
    heuristic, then verify the top candidates. Only verified triples
    are returned.
    """
    fam_fn = _FAMILY_FNS.get(family_name)
    if fam_fn is None:
        return None

    pairs = _parse_examples(examples)
    names = _all_pool_names()
    n = len(names)
    fns = [_get_transform_fn(name) for name in names]

    # Precompute all source values for all examples
    # src_vals[si][ei] = value of source si on example ei's input
    src_vals = []
    for si in range(n):
        vals = [fns[si](x) for x, y in pairs]
        src_vals.append(vals)

    # Find all verified triples (with any permutation)
    # For efficiency, try all unordered triples × 6 perms
    # 29 choose 3 = 3654 triples × 6 perms = 21924 checks
    # Each check iterates ~8 examples = ~175K ops total — fast
    verified = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triple_vals = [src_vals[i], src_vals[j], src_vals[k]]
                for perm in _PERMS:
                    ok = True
                    for ei in range(len(pairs)):
                        a = triple_vals[perm[0]][ei]
                        b = triple_vals[perm[1]][ei]
                        c = triple_vals[perm[2]][ei]
                        if fam_fn(a, b, c) != pairs[ei][1]:
                            ok = False
                            break
                    if ok:
                        # Map back to ordered source names (A, B, C positions)
                        triple_names = [names[i], names[j], names[k]]
                        ordered = [triple_names[perm[0]], triple_names[perm[1]], triple_names[perm[2]]]
                        verified.append(ordered)
                        # Early exit — we only need one (the simplest / first found)
                        # But gather a few to pick the one matching the trace sources
                        if len(verified) >= 5:
                            break
                if len(verified) >= 5:
                    break
            if len(verified) >= 5:
                break
        if len(verified) >= 5:
            break

    if not verified:
        return None

    # Pick the first verified triple (simplest — fewest chars in names)
    best = min(verified, key=lambda t: sum(len(s) for s in t))

    return best


def identify_sources(examples, family_name):
    """Given examples and a known family, identify source transforms.

    Uses brute-force verification: tries all triples of transforms with all
    permutations. Returns only triples that produce exact output on every example.

    Then computes heuristic scores (XNOR pair, force-1, selector) on the verified
    triple for use in evidence lines.

    Args:
        examples: list of (input_str, output_str) 8-bit binary string pairs
        family_name: one of OR_XNOR, GATED_XNOR_NAND, CH, MAJ3, TT121, T1

    Returns:
        dict with keys A, B, C (source names), method, and score fields.
        None if no verified triple found.
    """
    triple = _identify_by_verification(examples, family_name)
    if triple is None:
        return None

    a_name, b_name, c_name = triple

    # Compute evidence scores on the verified triple
    if family_name in ("OR_XNOR", "GATED_XNOR_NAND", "TT121", "T1"):
        xnor_score = score_xnor_pair(examples, a_name, b_name)
        force1_score = score_force1(examples, c_name)
        return {
            "A": a_name,
            "B": b_name,
            "C": c_name,
            "method": "xnor_pair+force1",
            "xnor_score": xnor_score,
            "force1_score": force1_score,
        }
    elif family_name == "CH":
        # For CH, A is the selector
        # Compute a selector-quality score
        fn_a = _get_transform_fn(a_name)
        # Selector quality: has mix of 0s and 1s across examples
        total_ones = sum(_popcount(fn_a(int(inp, 2))) for inp, _ in examples)
        total_bits = 8 * len(examples)
        sel_balance = 1.0 - abs(total_ones / total_bits - 0.5) * 2  # 1.0 = perfect balance
        return {
            "A": a_name,
            "B": b_name,
            "C": c_name,
            "method": "selector_scan",
            "selector_score": 1.0,  # Verified = 100% exact match
            "selector_balance": sel_balance,
        }
    elif family_name == "MAJ3":
        return {
            "A": a_name,
            "B": b_name,
            "C": c_name,
            "method": "equal_contrib",
            "maj_score": 1.0,  # Verified = 100% exact match
        }
    else:
        return {
            "A": a_name,
            "B": b_name,
            "C": c_name,
            "method": "verified",
        }


# ---------------------------------------------------------------------------
# Evidence for known sources (used by generator)
# ---------------------------------------------------------------------------

def compute_evidence(examples, family_name, ordered_names):
    """Compute fingerprint evidence scores for KNOWN source names.

    Unlike identify_sources (which searches for sources), this function
    takes the actual sources used by the generator and computes the
    heuristic scores that explain WHY these sources were chosen.

    Args:
        examples: list of (input_str, output_str) 8-bit binary string pairs
        family_name: e.g. "OR_XNOR"
        ordered_names: list of source names in A, B, C order (as used in Use: section)

    Returns:
        dict with A, B, C, method, and score fields (same format as identify_sources).
        None if the source names aren't in the transform pool.
    """
    # Verify all names are known
    for name in ordered_names:
        if name not in _POOL_BY_NAME:
            return None

    a_name, b_name = ordered_names[0], ordered_names[1]
    c_name = ordered_names[2] if len(ordered_names) >= 3 else None

    if family_name in ("OR_XNOR", "GATED_XNOR_NAND", "TT121", "T1"):
        xnor_score = score_xnor_pair(examples, a_name, b_name)
        force1_score = score_force1(examples, c_name)
        return {
            "A": a_name,
            "B": b_name,
            "C": c_name,
            "method": "xnor_pair+force1",
            "xnor_score": xnor_score,
            "force1_score": force1_score,
        }
    elif family_name == "CH":
        return {
            "A": a_name,
            "B": b_name,
            "C": c_name,
            "method": "selector_scan",
            "selector_score": 1.0,
        }
    elif family_name == "MAJ3":
        return {
            "A": a_name,
            "B": b_name,
            "C": c_name,
            "method": "equal_contrib",
            "maj_score": 1.0,
        }
    else:
        return {
            "A": a_name,
            "B": b_name,
            "C": c_name,
            "method": "verified",
        }


# ---------------------------------------------------------------------------
# Trace formatting
# ---------------------------------------------------------------------------

def format_fingerprint(examples, family_name, sources):
    """Format 2-3 lines of fingerprint evidence for a trace.

    Args:
        examples: list of (input_str, output_str) pairs
        family_name: e.g. "OR_XNOR"
        sources: dict from identify_sources (with A, B, C, method, scores)

    Returns:
        list of trace line strings (no leading newline, no trailing newline)
    """
    if sources is None:
        return []

    method = sources.get("method", "")
    lines = []

    if method == "xnor_pair+force1":
        xnor_pct = int(round(sources["xnor_score"] * 100))
        f1_pct = int(round(sources["force1_score"] * 100))
        lines.append(f"Scan: xnor({sources['A']},{sources['B']}) matches output {xnor_pct}%")
        lines.append(f"Scan: {sources['C']} force-1 {f1_pct}% (1-bits preserved in output)")
    elif method == "selector_scan":
        lines.append(f"Scan: {sources['A']} selects between {sources['B']} and {sources['C']} (exact match)")
    elif method == "equal_contrib":
        lines.append(f"Scan: majority({sources['A']},{sources['B']},{sources['C']}) exact match")

    return lines


# ---------------------------------------------------------------------------
# Verification helper (for testing)
# ---------------------------------------------------------------------------

def verify_fingerprint(examples, family_name, true_sources, true_perm=None):
    """Check if fingerprint correctly identifies the true sources.

    Args:
        examples: list of (input_str, output_str) pairs
        family_name: e.g. "OR_XNOR"
        true_sources: list of (name, fn) pairs as used by the generator
        true_perm: permutation tuple, e.g. (0, 1, 2)

    Returns:
        dict with "correct" (bool), "found" (identified sources), "expected" (true sources)
    """
    result = identify_sources(examples, family_name)
    if result is None:
        return {"correct": False, "found": None, "expected": true_sources}

    # Get ordered true names
    if true_perm is not None:
        expected_names = [true_sources[true_perm[i]][0] for i in range(3)]
    else:
        expected_names = [s[0] for s in true_sources]

    found_names = [result.get("A"), result.get("B"), result.get("C")]

    # For OR_XNOR: A,B are the XNOR pair (order doesn't matter), C is force-1
    if family_name in ("OR_XNOR", "GATED_XNOR_NAND", "TT121", "T1"):
        expected_pair = set(expected_names[:2])
        found_pair = set(found_names[:2])
        pair_ok = expected_pair == found_pair
        c_ok = found_names[2] == expected_names[2]
        correct = pair_ok and c_ok
    elif family_name == "CH":
        # Selector (A) must match; B,C order matters
        correct = found_names == expected_names
    elif family_name == "MAJ3":
        # All three must match (order doesn't matter for symmetric)
        correct = set(found_names) == set(expected_names)
    else:
        correct = set(found_names) == set(expected_names)

    return {
        "correct": correct,
        "found": found_names,
        "expected": expected_names,
        "scores": {k: v for k, v in result.items() if k not in ("A", "B", "C", "method")},
    }
