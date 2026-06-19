#!/usr/bin/env python3
"""
occam_solver.py — the "Autotracer": a compact whole-byte BIT solver that derives a
chain-of-thought from the examples, using OCCAM'S RAZOR ON ARITY.

Given some (input -> output) byte examples and a query, it finds the SIMPLEST
rule of the form  out = GATE(f(x), g(x), h(x))  consistent with every example
(simplest = lowest gate-arity + simplest streams), then applies it to the query.
"Simplest consistent rule" is the prior that actually solves these puzzles.

Why this matters (all reproducible here, no competition data):
  * The examples do NOT logically force the answer. Within this class, the rule is
    uniquely determined only ~58% of the time and the query answer is forced only
    ~83% of the time (`--stats`).
  * But the generator picks SIMPLE rules, so the simplest consistent rule is almost
    always the true one: Occam-on-arity recovers the exact answer ~99% of the time.
  * Distinguish three things, they are different numbers:
        FUNCTION exactly determined   (one consistent rule)            ~58%
        QUERY  exactly determined     (all consistent rules agree)     ~83%
        ANSWER via Occam prior        (pick the simplest consistent)   ~99%
    Baseline, random over ALL functions ignoring examples: 1/256 = 0.39%.

Usage:
  python3 analysis/occam_solver.py --demo
  python3 analysis/occam_solver.py --stats --samples 3000
  echo "01010001->11011101 ... 00110100" | python3 analysis/occam_solver.py
"""
import sys, os, argparse, random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generators"))
import bit as B  # noqa

NAMES = list(B.STREAMS)
ALL = np.arange(256, dtype=np.uint8)
STAB = {nm: np.array([B.STREAMS[nm](int(x)) & 0xFF for x in ALL], dtype=np.uint8) for nm in NAMES}

def _gout(g, f, gg, h):
    return np.asarray(B.GATES[g](f.astype(np.int64), gg.astype(np.int64), h.astype(np.int64)) & 0xFF, dtype=np.uint8)

def _scost(nm):
    if nm == "x": return 0
    if nm == "~x": return 1
    inv = nm.startswith("~"); base = nm[1:] if inv else nm
    return {"shl": 2, "shr": 2, "rol": 4, "ror": 4}[base[:3]] + int(base[3]) + (1 if inv else 0)

_GCOST = {g: (10 if g in B._TWO_INPUT else 20) for g in B.GATES}

def build():
    """Enumerate distinct whole-byte functions; keep each one's simplest spelling."""
    funcs = {}; spell = {}; comp = {}
    for g in B.GATES:
        two = g in B._TWO_INPUT
        for f in NAMES:
            for gg in NAMES:
                for h in (NAMES if not two else [f]):
                    tt = _gout(g, STAB[f], STAB[gg], STAB[f] if two else STAB[h]).tobytes()
                    c = _GCOST[g] + _scost(f) + _scost(gg) + (0 if two else _scost(h))
                    i = funcs.get(tt)
                    if i is None:
                        i = funcs[tt] = len(funcs); comp[i] = c; spell[i] = (g, f, gg, None if two else h)
                    elif c < comp[i]:
                        comp[i] = c; spell[i] = (g, f, gg, None if two else h)
    D = len(funcs)
    Dtab = np.frombuffer(b"".join(funcs.keys()), dtype=np.uint8).reshape(D, 256)
    return Dtab, np.array([comp[i] for i in range(D)]), [spell[i] for i in range(D)]

def solve(Dtab, COMP, SPELL, examples, query):
    mask = np.ones(Dtab.shape[0], dtype=bool)
    for xi, yo in examples:
        mask &= (Dtab[:, xi] == yo)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    qa = Dtab[idx, query]
    best = idx[np.argmin(COMP[idx])]
    return {
        "answer": int(Dtab[best, query]),
        "rule": SPELL[best],
        "n_consistent": int(idx.size),
        "function_determined": int(idx.size) == 1,
        "query_determined": len(set(qa.tolist())) == 1,
    }

def trace(res, examples, query):
    g, f, gg, h = res["rule"]
    rule = f"{g}({f}, {gg}" + (f", {h})" if h else ")")
    L = []
    L.append(f"{res['n_consistent']} whole-byte rule(s) fit all {len(examples)} examples; "
             f"taking the SIMPLEST (Occam on arity).")
    L.append(f"Rule: out = {rule}")
    L.append(f"  (query answer forced by examples? {res['query_determined']}; "
             f"rule unique? {res['function_determined']})")
    L.append(f"Apply to {query:08b}: -> {res['answer']:08b}")
    return "\n".join(L)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    Dtab, COMP, SPELL = build()

    if args.stats:
        rng = random.Random(args.seed); gl, gw = list(B.GATE_WEIGHTS), list(B.GATE_WEIGHTS.values())
        print("Occam-on-arity vs determinacy (random examples; estimates):")
        print("  k | function-determined | query-determined | random-consistent | OCCAM (simplest)")
        for K in (7, 8, 9, 10):
            occ = qd = fd = 0; rand = []
            for _ in range(args.samples):
                g = rng.choices(gl, weights=gw)[0]; two = g in B._TWO_INPUT
                f = rng.choice(NAMES); gg = rng.choice(NAMES); h = f if two else rng.choice(NAMES)
                rv = _gout(g, STAB[f], STAB[gg], STAB[f] if two else STAB[h])
                xs = rng.sample(range(256), K + 1); q = xs[-1]
                ex = [(xi, int(rv[xi])) for xi in xs[:-1]]
                r = solve(Dtab, COMP, SPELL, ex, q)
                fd += r["function_determined"]; qd += r["query_determined"]
                m = np.ones(Dtab.shape[0], bool)
                for xi, yo in ex: m &= (Dtab[:, xi] == yo)
                rand.append(float(np.mean(Dtab[m, q] == int(rv[q]))))
                occ += (r["answer"] == int(rv[q]))
            n = args.samples
            print(f"  {K:2d} | {100*fd/n:18.0f}% | {100*qd/n:15.0f}% | {100*np.mean(rand):16.0f}% | {100*occ/n:.1f}%")
        print("  (random over ALL functions, ignoring examples: 1/256 = 0.39%)")
        return

    if args.demo:
        ex = [(0b01010001, 0b11011101), (0b00001001, 0b01101101), (0b00010101, 0b01010101),
              (0b11111111, 0b10000001), (0b10011101, 0b01000101), (0b00111011, 0b00001001),
              (0b10111101, 0b00000101), (0b00100110, 0b10110011)]
        q = 0b00110100
    else:
        toks = sys.stdin.read().split()
        ex = []; q = None
        for t in toks:
            if "->" in t:
                a, b = t.split("->"); ex.append((int(a, 2), int(b, 2)))
            elif set(t) <= {"0", "1"} and len(t) == 8:
                q = int(t, 2)
        if q is None: print("need a query byte"); return
    res = solve(Dtab, COMP, SPELL, ex, q)
    if res is None:
        print("no whole-byte rule fits (likely a per-bit puzzle outside this class)")
    else:
        print(trace(res, ex, q))

if __name__ == "__main__":
    main()
