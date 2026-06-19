#!/usr/bin/env python3
"""
measure_bit_space.py — recompute every headline BIT statistic on the ACTUAL rule
space that generators/bit.py samples (58 named streams, 11 gates), using the
generator's own GATE_WEIGHTS sampling prior.

Two parts:
  * EXACT enumeration (deterministic): stream aliasing, syntactic-rule count,
    distinct induced functions, unreachable (x->y) pairs. Asserted against the
    published headline numbers below.
  * MONTE-CARLO simulation (sampled): survivor / uniqueness / answer-determinacy
    curves under the generator's prior. Reported as estimates with N and seed.

Provenance: NO raw competition rows are read. GATE_WEIGHTS are aggregate-derived
priors (approximate family frequencies over competition bit rows), baked into
generators/bit.py — not raw data.

Dependency note: this analysis uses NumPy (the six clean generators are
standard-library-only; only this script needs NumPy).

Usage:
  python3 analysis/measure_bit_space.py                 # fast: 2000 samples
  python3 analysis/measure_bit_space.py --samples 20000 # heavier
  python3 analysis/measure_bit_space.py --full          # 20000 (paper setting)
  python3 analysis/measure_bit_space.py --seed 7
"""
import sys, os, argparse, random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generators.bit import STREAMS, GATES, GATE_WEIGHTS, _TWO_INPUT  # noqa

# published headline numbers (exact enumeration) — asserted at runtime
EXP_NAMED, EXP_STREAM_FNS = 58, 44
EXP_SYNTACTIC, EXP_FUNCS, EXP_UNREACHABLE = 1_379_240, 89_086, 2_266

ALL = np.arange(256, dtype=np.uint8)
names = list(STREAMS)

def gate_out(gate, f, g, h):
    return np.asarray(GATES[gate](f.astype(np.int64), g.astype(np.int64),
                                  h.astype(np.int64)) & 0xFF, dtype=np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--full", action="store_true", help="use 20000 samples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    N = 20000 if args.full else args.samples

    # ---- EXACT: streams + aliasing ----
    stab = {nm: np.array([STREAMS[nm](int(x)) & 0xFF for x in ALL], dtype=np.uint8)
            for nm in names}
    distinct_streams = len({stab[nm].tobytes() for nm in names})
    print(f"[exact] named streams            : {len(names)}")
    print(f"[exact] distinct stream functions: {distinct_streams} "
          f"(alias factor {len(names)/distinct_streams:.3f}x)")

    # ---- EXACT: full hypothesis space + distinct induced functions ----
    syntactic = 0
    funcs = {}
    for gate in GATES:
        two = gate in _TWO_INPUT
        gh = names if not two else [None]
        for f in names:
            for g in names:
                for h in (gh):
                    syntactic += 1
                    hh = stab[f] if two else stab[h]
                    tt = gate_out(gate, stab[f], stab[g], hh).tobytes()
                    funcs.setdefault(tt, len(funcs))
    D = len(funcs)
    print(f"[exact] syntactic hypotheses     : {syntactic:,}")
    print(f"[exact] distinct induced functions: {D:,}")
    Dtab = np.frombuffer(b"".join(funcs.keys()), dtype=np.uint8).reshape(D, 256)

    # ---- EXACT: reachable (x->y) pairs ----
    reach = set()
    for r in range(D):
        reach.update(zip(ALL.tolist(), Dtab[r].tolist()))
    unreachable = 256*256 - len(reach)
    print(f"[exact] unreachable (x->y) pairs : {unreachable:,} / 65,536")

    # ---- assertions on the published headline numbers ----
    assert len(names) == EXP_NAMED, (len(names), EXP_NAMED)
    assert distinct_streams == EXP_STREAM_FNS, (distinct_streams, EXP_STREAM_FNS)
    assert syntactic == EXP_SYNTACTIC, (syntactic, EXP_SYNTACTIC)
    assert D == EXP_FUNCS, (D, EXP_FUNCS)
    assert unreachable == EXP_UNREACHABLE, (unreachable, EXP_UNREACHABLE)
    print("[exact] assertions passed: matches the numbers quoted in the write-up.\n")

    # ---- MONTE-CARLO: survivor / uniqueness / answer-determinacy ----
    rng = random.Random(args.seed)
    gate_list, gate_w = list(GATE_WEIGHTS), list(GATE_WEIGHTS.values())
    KMAX = 10
    surv = {k: [] for k in range(1, KMAX+1)}
    uniq = {k: 0 for k in range(1, KMAX+1)}
    qdet = {k: 0 for k in range(1, KMAX+1)}
    racc = {k: [] for k in range(1, KMAX+1)}   # random-over-survivors query accuracy
    for _ in range(N):
        gate = rng.choices(gate_list, weights=gate_w)[0]
        two = gate in _TWO_INPUT
        f = rng.choice(names); g = rng.choice(names)
        h = f if two else rng.choice(names)
        rule_vec = gate_out(gate, stab[f], stab[g], stab[h])
        xs = rng.sample(range(256), KMAX+1); q = xs[-1]; ex = xs[:-1]
        mask = np.ones(D, dtype=bool)
        for k in range(1, KMAX+1):
            xi = ex[k-1]
            mask &= (Dtab[:, xi] == rule_vec[xi])
            n = int(mask.sum()); surv[k].append(n)
            if n == 1: uniq[k] += 1
            qa = Dtab[mask, q]
            if qa.size and np.all(qa == qa[0]): qdet[k] += 1
            # expected accuracy of a RANDOM surviving function on the query byte
            racc[k].append(float(np.mean(qa == rule_vec[q])) if qa.size else 0.0)
    print(f"[monte-carlo] N={N} samples, seed={args.seed} (estimates):")
    print("  k | median surv | mean surv | function unique | query determined | random-survivor query-acc")
    for k in range(1, KMAX+1):
        a = np.array(surv[k])
        print(f"  {k:2d} | {int(np.median(a)):11d} | {a.mean():9.1f} | "
              f"{100*uniq[k]/N:14.1f}% | {100*qdet[k]/N:15.1f}% | {100*np.mean(racc[k]):.1f}%")
    print("  ('random-survivor query-acc' = a random function CONSISTENT with the k examples,")
    print("   right on the query. Baseline: random over ALL functions ~ 1/256 = 0.39%.")
    print("   These use RANDOM example inputs; measured on the real train.csv examples the")
    print("   numbers are about the SAME (function-det ~58%, query-det ~83%, random ~93%).")
    print("   The big lever is the simplicity PRIOR: Occam-on-arity ~99% -> see occam_solver.py.)")
    med1 = int(np.median(surv[1]))
    print(f"\n  total function class: log2({D}) = {np.log2(D):.1f} bits")
    print(f"  first-example reduction: median {med1} survivors  "
          f"=> ~{np.log2(D/med1):.1f} bits gained "
          f"(mean elimination {100*np.mean([1-x/D for x in surv[1]]):.2f}%; "
          f"uniform-8-bit ceiling {100*(1-1/256):.2f}%)")
    print("  NOTE: percentages above are Monte-Carlo estimates (vary with --seed/--samples).")

if __name__ == "__main__":
    main()
