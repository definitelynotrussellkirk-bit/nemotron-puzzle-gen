#!/usr/bin/env python3
"""
Build the Kaggle Open-Contribution notebook from the repo files.

Output: nemotron_data_method.ipynb  (category: Best Data/Synthetic Data Method)

The notebook is self-contained: it recreates the generators with %%writefile,
runs them live (the data method demonstrated, sanitization-proof), links the two
interactive HTML viewers (with a copy-paste IFrame snippet), and carries the
methodology write-up. Re-run after
editing any source file so the notebook always matches the repo.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def read(p):
    with open(os.path.join(HERE, p), encoding="utf-8") as f:
        return f.read()


_CELL_N = [0]
def _cid():
    _CELL_N[0] += 1
    return f"cell-{_CELL_N[0]:03d}"


def md(src):
    return {"cell_type": "markdown", "id": _cid(), "metadata": {},
            "source": src.splitlines(keepends=True)}


def code(src, outputs=None):
    return {"cell_type": "code", "id": _cid(), "metadata": {}, "execution_count": None,
            "outputs": outputs or [], "source": src.splitlines(keepends=True)}


def run_capture(src):
    """Execute a snippet against the repo's generators, capture stdout as a saved
    notebook stream output so the published (un-run) view shows real results."""
    import io, contextlib, runpy, sys
    sys.path.insert(0, os.path.join(HERE, "generators"))
    buf = io.StringIO()
    g = {}
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src, "<demo>", "exec"), g)
    except Exception as e:  # keep building even if a demo errors
        buf.write(f"\n[demo capture error: {e}]\n")
    return [{"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(keepends=True)}]


GENERATORS = ["bit", "encryption", "transformation",
              "gravitational", "unit_conversion", "number_conversion"]

REPO_URL = "https://github.com/definitelynotrussellkirk-bit/nemotron-puzzle-gen"
PAGES_URL = "https://definitelynotrussellkirk-bit.github.io/nemotron-puzzle-gen/"

cells = []

# ---- 1. title -------------------------------------------------------------
cells.append(md(f"""# Understanding the BIT puzzle as GATE(f, g, h) — generators + an honest search write-up

**Open Contribution Award — Best Data / Synthetic Data Method.** Team: 147 / 4,354 (top 3.4%).

We set out to teach an LLM to **search** these reasoning puzzles — enumerate
candidate rules, test them against the examples, prune, lock. We didn't fully get
there. This notebook is **what we learned along the way** and the **base generators**
we built to explore it, shared as honest work-in-progress.

Every generated row has a known latent rule and an exact gold answer (we sampled the
rule), so in our working system each forward generator was paired with a solver/verifier
that acted as a free oracle — which let us cheaply mint data in many custom
chain-of-thought (CoT) formats and keep only the derivable ones. (A known rule is not
the same as a rule *uniquely identifiable* from the shown examples — see the BIT section.
The solvers/validators were internal; the six generators written below are the *forward
samplers* only.) The piece we're happiest with: a clean way to **show** and **generate**
the hardest type, BIT, as `output = GATE(f(x), g(x), h(x))`, with an interactive viewer
to help others *visualize* the problem and watch the search space collapse.

- 🔗 **Repo (generators + interactive viewers):** {REPO_URL}
- 🌐 **Live interactive pages:** {PAGES_URL}

> **Tone & scope.** Nothing here is a confident claim or a leaderboard result — it's
> a log of what we tried, what we observed, and what we now speculate. We hoped to
> transfer a search ability into the 30B; BIT results varied a lot (better runs ~50–60%,
> worst ~12% — a deterministic search where every step was locally solvable but the model
> couldn't chain them). Whether that was a failure on our part or something more
> fundamental, we honestly don't know. We've included lots of training traces for others.
> The generators below run live and contain **no raw competition rows**; a few sampling
> priors (e.g. the BIT gate-family weights) were estimated from *aggregate statistics* over
> competition data. More in *Budget & hindsight*.
"""))

# ---- 2. the method --------------------------------------------------------
cells.append(md("""## Why a generator platform

Four properties made the generators a fast CoT-experimentation loop:

1. **A free oracle.** Because we sampled the latent rule, an internal solver could
   verify *any* intermediate state for free, so we kept only derivable traces. (Those
   solvers were internal and are not part of this public release — the cells below
   write the forward samplers only.)
2. **Derivability as a hard gate.** A trace is allowed only if each step follows
   mechanically from the previous one (no naming a rule without the evidence, no
   decimal division a small model can't carry). A lint enforces it.
3. **A representation that's easy to show.** Reading BIT as one gate over three
   whole-byte streams, `GATE(f(x), g(x), h(x))`, instead of eight independent
   per-bit problems, is the cleanest thing to look at — and to generate.
4. **The data is over-determined — but *softly*.** On the whole-byte rule space (89,086
   functions, which covers 100% of real BIT puzzles), the exact *rule* is pinned only
   ~58% of the time and the *answer* is logically forced ~83%; yet picking the SIMPLEST
   consistent rule (Occam on arity) gets the exact answer ~99%. The answer is recovered by
   a simplicity prior, not by logic. See the BIT section below and `analysis/occam_solver.py`.

Because of (1)+(2), a $150 budget bought a lot of experimentation: no human labels,
no learned verifier to pay for. (Budget breakdown at the end.)
"""))

# ---- 3. recreate generators ----------------------------------------------
cells.append(md("## The generators\n\nRecreate the six samplers, then run them."))
cells.append(code("import os\nos.makedirs('generators', exist_ok=True)"))
for g in GENERATORS:
    body = read(f"generators/{g}.py")
    cells.append(code(f"%%writefile generators/{g}.py\n{body}"))

# ---- 4. live demo (outputs captured at build time, so the published view isn't blank)
cells.append(md("### Live: create puzzles\n\nEach call picks a hidden rule, samples inputs, and emits `examples + query + answer + the exact rule used`. The outputs below are real, captured when this notebook was built."))
DEMO1 = (
    "import random, sys\n"
    "sys.path.insert(0, 'generators')\n"
    "import bit\n"
    "rng = random.Random(0)\n"
    "p = bit.sample(rng)\n"
    "print('BIT  rule =', p['rule'])\n"
    "for e in p['examples'][:4]:\n"
    "    print(' ', e['input'], '->', e['output'])\n"
    "print('  QUERY', p['query'], '->', p['answer'])"
)
cells.append(code(DEMO1, outputs=run_capture(DEMO1)))
DEMO2 = (
    "import random, sys\n"
    "sys.path.insert(0, 'generators')\n"
    "import encryption, transformation, gravitational, unit_conversion, number_conversion\n"
    "rng = random.Random(1)\n"
    "for mod in (encryption, transformation, gravitational, unit_conversion, number_conversion):\n"
    "    q = mod.sample(rng)\n"
    "    print(f\"\\n{q['type']}  rule={q['rule']}\")\n"
    "    print('  example:', q['examples'][0])\n"
    "    print('  query  :', q['query'], '->', q['answer'])"
)
cells.append(code(DEMO2, outputs=run_capture(DEMO2)))

# ---- 5. interactive viewers (linked, not embedded) ------------------------
cells.append(md(f"""## Interactive viewers (best viewed live)

Three browser tools ship with the repo. Kaggle's *published* view strips interactive
JS, so they are best opened live on the project site:

- **BIT viewer** — pick three byte-streams and a gate, watch the output byte computed
  bit-by-bit, with the real per-bit data distribution → {PAGES_URL}
- **Over-determination explorer** — sample K examples from a rule and watch the
  candidate hypotheses shrink one example at a time, with a uniqueness verdict
  (exact within its catalog) → {PAGES_URL}over-determination.html
- **Mini-skill gallery** — shuffle a small random sample of the atomic drills, filter
  by type, pull another generation of any skill → {PAGES_URL}miniskills.html

We hope the BIT viewer helps others *visualize* the bit problems and how the search
space collapses — using an LLM to build quick data visualizations, which many across
past competitions have found valuable.

*To embed one live in a fork:* `from IPython.display import IFrame` then
`IFrame("{PAGES_URL}", width="100%", height=720)`.
"""))

# ---- 6. insights ----------------------------------------------------------
cells.append(md("""## Per-type insights

**BIT — `GATE(f,g,h)`.** A stream is the input through one cheap whole-byte op
(`shl/shr/rol/ror`, complement, identity). Output = one gate across three streams;
dominant families are `OR_XNOR` and a `XNOR/NAND` selector. For an 8-bit byte
`rol_k == ror_(8-k)`, so 58 named streams are only **44 distinct functions** —
score the function, not the label.

**BIT: a search for FUNCTIONS vs a search for ANSWERS.** The whole-byte `GATE(f,g,h)`
class has **89,086** distinct functions, and it reproduces **100%** of the real BIT
puzzles (every gold answer is a gate over streams).
Measured on the real train.csv puzzles (median 9 examples), success depends entirely on
*which question you ask* (reproducible: `analysis/occam_solver.py`):

| Question | How often it works |
|---|---|
| pin the exact RULE (function search) | ~58% |
| all consistent rules agree on the ANSWER | ~83% |
| pick a *random* consistent rule | ~93% |
| **pick the SIMPLEST consistent rule (Occam on arity)** | **~99%** |
| ignore the examples (random byte) | 0.39% |

So the answer is **not logically forced** — it's recovered by a **prior**: the generator
picks simple rules, so the simplest consistent rule is almost always the true one, and
Occam-on-arity lands the exact answer ~99% on random examples (our internal solver scored
~99.3% on the real competition rows, measured on our machine; train.csv isn't shipped here,
so that figure isn't reproducible from this notebook). That's
*soft*, prior-recovered over-determination, not hard logic. In hindsight our effort went
into the harder framing (**function search**, underdetermined ~42% of the time); the easy
target was **answer search** — and Tong's per-column / frequency-prior method reads like
exactly that. Our `occam_solver.py` "Autotracer" is the answer-search form: paste examples
+ a query, it derives the simplest consistent rule and the CoT.

**Encryption — closed vocabulary.** Every plaintext word is in a fixed list, so
unknown cipher-words are finished by exhaustive same-length candidate testing +
bijection rejection. That closure is the whole reason the type is exact.

**Transformation — the one piece of ours that actually shipped.** operation ×
ordering × style, layered as an encoding on the raw arithmetic. We left the *numeric*
lane unexplored, but the **cipher (cryptarithm) transform rows** from this generator —
about **2,500 verified rows** (a CSP solve for the symbol→digit map + honest priors) —
were the one part of our own work that made it into our best scoring blend, on top of
a strong public CoT base. Honest caveat: we couldn't cleanly attribute the lift, and
cipher-type accuracy stayed ~7.5%. The numeric lane is the most open next step.

**Numeric rates — clear the decimals first, like a human.** For `d = ½·g·t²` and
`out = in·factor`, cancel the hidden decimal constant by **ratio** instead of
recovering it: `d_q = d_e·(t_q/t_e)²`. The unknown divides out and the trace
collapses to one step. Honest caveat: the shown values are *rounded*, so this is a
**tolerance-compatible shortcut**, not exact cancellation — in a quick simulation it
hit the exact 2-decimal answer ~74% (gravitational) / ~68% (unit) from one example,
with nearly all predictions inside the task's 1% tolerance.

**Number conversion — no hidden parameter.** int↔Roman is a fixed, table-driven
additive/subtractive notation (not positional) — a deterministic greedy walk of the
value table; the examples only signal direction.
"""))

# ---- 6b. search approaches + papers --------------------------------------
cells.append(md("""## The CoT formats were search traces — approaches & papers

Most custom formats are **search traces** (enumerate candidate rules → test → prune
→ lock). Our initial framing was **recursive bisection**: to get from a state `A`
to the goal `Z`, if it fails in one shot, introduce a waypoint `N`, solve `A→N` and
`N→Z` separately, and keep bisecting until each hop is trivial. That divide-and-
conquer instinct led us to the search-supervision literature, whose training shapes
are the same idea made trainable. We *hoped* one thing would be our edge: our solver
is an exact, free verifier of every intermediate state, so the weak/learned-verifier
bottleneck the literature describes wouldn't apply to us. It didn't translate into a
solving win for us (see the dissociation note and budget below).

- **Teach-long enumerate→test→prune SFT** — every step a solver-checkable
  state→action→observation→prune transition; rejected branches shown in-trace.
- **STaR / expert-iteration trim** — *inspired by* Zelikman et al., *STaR* (2022)
  and Lehnert et al., *Searchformer / "Beyond A\\*"* (2024). Those papers filter
  sampled rationales / learn A\\* dynamics and bootstrap shorter traces; "keep the
  shortest **verified** trace under a declining length budget, else the canonical
  trace" is *our* adaptation, not their method.
- **GRPO process / progress reward** — GRPO is from Shao et al., *DeepSeekMath*
  (2024, arXiv:2402.03300); the Hamming-distance / "gold still reachable" rewards
  are *our* proposed shaping, not from that paper.
- **Step-level process supervision** — Lightman et al., *Let's Verify Step by Step*
  (2023). Our solver emits the earliest-wrong-step label for free.
- **Off the table at submission:** tree-of-thought / MCTS / self-consistency —
  competition inference is greedy single-pass, so these stayed training-data tools.

**Compared to Tong Huikang's 0.85 (speculation).** We used *first divergence* — the
first example/position where candidate rules disagree — to build our atomic A→B
drills, much like Tong's per-column method (freq priors + stride extrapolation;
[github.com/tonghuikang/nemotron](https://github.com/tonghuikang/nemotron)).
Supportable facts: his repo presents a Progress-Prize submission and the linked
notebook is titled "End-to-end finetuning for LB 0.85". *Speculating* from the public
repo (we did not reverse-engineer it): we wonder whether his traces baked the bit
computation, the search priors, *and* the answer into **one coherent trace** — and
that "all in one trace" was the important move. That's a testable hypothesis, not a
verified comparison. We instead fragmented the same
material across a library of micro-skills and many CoT formats; the pieces were
there, but the model may never have seen one trace teaching search + the bit op +
the answer together. Consolidating into a single unified trace is near the top of
our "if we ran it again" list.
"""))

# ---- 6b1. the dissociation finding ---------------------------------------
cells.append(md("""## A divergence we *think* we saw: rank *given* candidates vs *generate* them

A BIT answer is one byte — only 256 possibilities — so we tried turning solving into
**ranking**: hand the model a shortlist (64 → 32 → 16 → 8 candidates) and have it
pick the most likely. Given the shortlist, it *seemed* to pick reliably. When it had to
**produce** the shortlist itself — even from a deterministic recipe — accuracy seemed to
fall to ~78–80%. (Earlier slate formats had parsing failures; the K=8 format we report
parsed 64/64, so the loss looked like candidate *recall*, not formatting.)

On a small (64-row) eval we tried to factorize that condition
(end-to-end ≈ parse × candidate-set × selection-given-set):

| Stage (generate-then-pick) | What we saw |
|---|---|
| **Parse** — clean, parseable 8-candidate set | 64/64 (always exactly 8) |
| **Candidate set correct** — its 8 contained/identified gold | ~78–80% (50–51/64) |
| **Selection given set correct** — picked gold given its set was right | 50/50, 51/51 |

So the errors we saw *seemed* to land in **candidate-set construction**: parsing held, and
on the 50–51 rows whose set contained gold we happened to see no selection errors. **Big
caveat:** 50–51 rows is a small sample with real variance — "no errors" is suggestive, not
proof that selection is perfect. The fuller picture:

| What the model had to do | Accuracy we saw (small evals) |
|---|---|
| Execute a *given* function (scaffolded) | up to ~100% (64/64, 512/512) |
| Generate 8 candidates, then pick | ~78–80% (50–51/64) |
| Solve raw end-to-end (no scaffold) | ~11–22% (7/64–14/64) |

As scaffolding was removed, accuracy went down. This is the same local-vs-global pattern
as the alphabet point in the budget section: the model seemed able to do each step alone
but not chain them.

For reference, "random" depends on what you average over (see the BIT section): ignoring
the examples is 0.39% (1/256); a guess that stays *consistent* with the examples is right
about 93%; the simplest consistent rule (Occam) about 99%. So the ~78–80% on the
generate-then-pick condition is below a random-consistent guess. Within-slate baselines:
12.5% (uniform over 8), 3.1% (a random 8-slate contains gold).
*(Source: `bit_alice_cache_slate_mask_qset_k8` / `qset_only_k8`, 64 rows, greedy decode;
LoRA-SFT BIT search lineage on Nemotron-3-Nano-30B — internal evals, not a leaderboard
number.)*
"""))

# ---- 6b2. mini-skills -----------------------------------------------------
cells.append(md(f"""## Mini-skills — from holistic inspection of mistakes

Beneath the full traces sits a library of **239 atomic "mini-skill" drills**, each
designed from **holistic inspection of the model's mistakes**: we read *how* it
failed and distilled the missing micro-competence into a small, self-contained
drill (a single gate evaluation, a shift, a bijection check, a popcount-delta
sign, …). The base generators mint these on demand, so the library grew out of the
failure analysis rather than a fixed syllabus.

- **Browse:** `miniskills.html` — gallery covers **233 of 239 registered** skills
  (up to 2 random generations each)
- **Full random dump:** `miniskills_sample.jsonl` — 5,891 weighted-random rows,
  **221** of 239 skills (a random sample, so not all 239 appear)
- Live gallery: {PAGES_URL}miniskills.html

**The recipe (so you can build your own).** The model could often do each step of a
procedure but not chain them. So we generated one drill per step. It's task-decomposition
turned into data — not magic:

```
full task:   A ───────────────▶ Z
                  decompose into the steps a solver does
             A ─▶ B ─▶ C ─▶ … ─▶ Z

per step:  a sampler emits random instances of "given A, produce B"  (one generator each)
           plus one "chain A ─▶ … ─▶ Z" drill
gate each row on the solver  ─▶  keep only derivable ones  =  the dataset
```

List your solver's mechanical steps, write a sampler per step (and one for the whole
chain), and check each row against the solver. That transfers to any decomposable task.
"""))

# ---- 6c. budget / hindsight ----------------------------------------------
cells.append(md("""## Budget & hindsight

We earmarked about **$150** at the start — roughly **$100 for Tinker** (LoRA SFT /
RL) and **~$50 / ~600 credits for Colab Pro+**. We kept each **experiment** cheap —
only about **$2 of Tinker compute apiece** — by running lots of small steps. To be
precise about the bookkeeping: an "experiment" means one format/idea, which usually
bundled several short training *runs* plus evals; the ~$100 spread over a few dozen
experiments, which is why the per-experiment figure and the larger run count below
aren't in tension. Exact, free oracles meant every generated trace was usable
without human labels or a learned verifier; and **having CODEX (an agent) manage the
Tinker runs end-to-end — launch a small step, evaluate, hand off, repeat, no idle GPU
or manual babysitting — saved a lot of money** and made the volume affordable.

**Trace length was our other cost lever.** Shorter chains-of-thought are cheaper to
train on and to run at inference (output is capped at 7,800 tokens), so for most of the
competition we kept traces tight — aiming under **~900 tokens**, then relaxing to
**~1.1k**, then **~1.3k** as harder cases needed room. Only near the end did we spend the
*full* budget, letting the enumerate-test-prune search traces run long to try to teach
BIT — where cost climbed. On training, we had good luck with **QLoRA (4-bit LoRA) on an
H100** (via Colab Pro+, alongside Tinker for the LoRA SFT/RL), especially at the *lower*
token counts where short traces packed efficiently and runs stayed fast and cheap.

**The scale that bought:** roughly **950 Tinker runs** (≈500 training + ≈450 eval)
over **~33 days**, about **175** training-data builds, the BIT trace format revised
through **30-plus numbered versions** (up to V35), and **256** automated handoff
notes between agents. Small steps, kept cheap, run at volume.

**We had a strong early start — so we were hopeful, not going in blind.** Before we
leaned on a public CoT base, one of our own SFT-trace submissions already scored about
**78% overall** — which we *believe* placed us around **top 50** on the leaderboard at the
time — notably with BIT at only ~**22%**, so the five other puzzle types were already
carrying the score. That early traction kept us hopeful, and throughout we kept
doing **holistic looks at the errors** — reading what the model got wrong and distilling
each gap into a targeted mini-skill, rather than guessing. BIT was the obvious holdout,
which is what pushed us into the search-teaching experiments (which, on BIT specifically,
varied widely — from a worst of ~12% up to ~50–60% in better runs; same hard type).

**Where the time went, and what our final submission was.** Most of our effort went
into trying to teach the model a *different* search method for BIT (the bisection /
enumerate-test-prune work above) — and that bulk never became a finished submission.
Results **varied a lot**: our better BIT runs reached roughly **50–60%**, an early attempt
was ~**22%**, and the *worst* — a deterministic search that fit within budget — got only
~**12%**. In that ~12% case everything was **locally solvable** — like the alphabet: ask the
model each step alone ("what comes after A?", "after B?") and it's right essentially every
time; ask it to recite the whole alphabet and it drops to ~12%. It did each `A→B` step in
isolation but did not chain them — the same pattern as the dissociation above. (Our final
leaderboard entry was small and almost incidental: we were testing whether our cipher
rows added lift, and by the time we looked up we'd run out of time.)

**If we went again, we'd prototype on a small model first** — *for experiments, not for
submission.* The competition target is fixed at Nemotron-3-Nano-30B-A3B; we just mean a
cheap prototyping rig to iterate the search-teaching loop faster, then transfer the
lessons up to the fixed 30B (never submitting a different size). Iterating directly on a
30B is slow. Much of what *seems* to matter — trace shape, length budget, verifier-gated
trimming, where the model fakes a step — *looks* largely model-agnostic, so we suspect
those lessons would transfer up from a smaller model at a fraction of the cost. We can't
prove it; it's the bet we'd make next time. The unexplored transformation lane and
the STaR/GRPO loops are where we'd look first.

**SPECULATION (hindsight-inspired) — an architecture hunch about MoE routing.** *(All of
this is a hindsight-inspired hunch we could not confirm — us connecting our own routing
struggles, after the fact, with a noise result we only really weighed later.)*
Nemotron-3-Nano-30B-A3B is a mixture-of-experts: almost all of its capacity is in the
sparse *experts*, not the shared/attention layers. We reached about **0.76 relatively
easily with LoRA on the shared layers only — no expert layers**, which already gets the
bulk of the non-BIT types. Pushing further on BIT seemed to need the experts, and in
hindsight we *feel* the hard part was keeping expert **routing consistent across a long
trace**: a long enumerate-test-prune BIT trace only holds together if the same experts
fire reliably step after step, so any routing drift reads as noise. We *speculate* (purely
a guess) that some of that noise may even have been expert normalization. We're not the
only ones poking at noise here — we recall a public **gold-medal notebook** reporting that
simply *adding noise* improved their result, and *perhaps that's why*: if consistent
expert routing is the real bottleneck, a little noise (or a different normalization) might
help in a way none of us fully understands. We never pinned it down — flagging it in case
it's a thread someone can pull.

**Why share something that hasn't paid off?** It hasn't produced a reliable result
for us *so far* — BIT runs ranged from ~12% to ~50–60%. Whether that was a failure on our part or
something more fundamental, we honestly don't know. We suspect the missing piece may be
mundane (not enough training, too small a budget, or our cheap small-model *experiments*
not transferring up to the fixed 30B target — the submission model size is the
competition's, not ours to change). So we've
included lots of **training traces** (the 5,891-row mini-skill dump and the CoT
formats) to give others ideas to build on. And part of it is paying it forward: reading
other people's public notebooks gave *us* insights along the way (the noise observation
above is one), so contributing ours back is the least we can do. Putting it in the open
seems like the fastest way for someone to spot what we missed — that's the point of an
open contribution.
"""))

# ---- 7. repro -------------------------------------------------------------
cells.append(md(f"""## Reproduce / explore

- All six generators and the three interactive viewers: **{REPO_URL}**
- Live pages (guaranteed interactive): **{PAGES_URL}**

```bash
python3 generators/bit.py -n 5 --seed 0            # create 5 BIT puzzles
python3 generators/bit.py -n 1000 --jsonl > bit.jsonl
python3 generators/transformation.py --cipher -n 3 # symbol-disguised digits
```

**Acknowledgments.** Thanks to **Tong Huikang**, whose public Progress-Prize 0.85
solution ([github.com/tonghuikang/nemotron](https://github.com/tonghuikang/nemotron))
was the reference we compared our BIT approach against and learned from. Programming and
task management throughout were done with the help of AI coding agents — **OpenAI's
Codex** and **Anthropic's Claude**; Codex in particular drove the long-running Tinker
experiment loops end-to-end.

*Category entered: Best Data / Synthetic Data Method.*
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(HERE, "nemotron_data_method.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {out}  ({len(cells)} cells)")
