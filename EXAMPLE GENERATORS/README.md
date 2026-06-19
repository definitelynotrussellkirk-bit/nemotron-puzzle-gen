# EXAMPLE GENERATORS — a variety dump (for the write-up)

This folder is here to **show variety**, not to be a clean, runnable package.

It is the **full, as-is generator collection** from our working repo during the
NVIDIA Nemotron Reasoning Challenge — ~70 Python files, every puzzle type, every
custom CoT format we tried, plus the whole mini-skill framework. The competition
is over, so we're sharing all of it.

**What's the difference vs. the 6 clean generators at the repo root?**

- The root `generators/` folder is the **curated, standalone** set — one tidy
  *forward* sampler per puzzle type, no traces, no solvers, runs with only the
  standard library. That's the polished front door referenced by the write-up.
- **This** folder is the **raw working pile**. Many files here:
  - emit full reasoning **traces** (the CoT formats we experimented with),
  - import other modules from our private solver/training stack, so they
    **may not run standalone** on a fresh clone,
  - reference competition rows that were on our machine during the contest.

So treat this as a **museum of attempts**, not a library. It's the honest record
of how many different ways we tried to generate and shape data — useful if you
want ideas, formats, or starting points, not a `pip install`-and-go toolkit.

### Rough map of what's in here

- `microskill_framework.py`, `microskill_skills.py`, `microskill_bit_fluency.py`
  — the atomic-drill ("mini-skill") framework (the source behind the 6,000-row
  sample dump and the gallery in `miniskills.html`).
- `gen_bit_*.py`, `regen_bit.py`, `trace_perbit.py`, `trace_bit_program.py`,
  `bit_*.py` — BIT generators and CoT trace formats (3-stream, witness,
  ambiguous, const-trap, sliding-window, template, …).
- `gen_transform_*.py`, `gen_transform_cipher_*.py`, `trace_transform.py`,
  `transform_router.py` — the transformation lane, including the **cipher
  (cryptarithm)** generators behind the rows we actually shipped.
- `gen_symbol_*.py` — symbol-transformation variants.
- `encryption.py`, `gravitational.py`, `unit_conversion.py`,
  `number_conversion.py`, `transformation.py` — the working versions of the
  per-type generators (the root copies are the cleaned-up ones).
- `archive/` — older superseded generators, kept for reference.

Nothing here is required to read the write-up — it's supporting material to show
the breadth of the data-generation experiments.
