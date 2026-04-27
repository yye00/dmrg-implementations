# pdmrg-multi-gpu — flagged for follow-on investigation

**Status as of 2026-04-27**: flagged for a real GPU campaign after G1 (the
baseline N=10 rebench) lands. Not blocking the CPC submission.

**Why this deserves more work**: paper §6.7.1 reports partial 4-MI300X data
that is more interesting than the original "early investigation, not
competitive" framing suggested. The analytically-predicted crossover is
visible in the data — multi-GPU loses by 2--3× at $\chi=64$ and **wins by
2.5--3×** at $\chi=128, 256$ on long Heisenberg chains. The directional
agreement with the analytical bound is encouraging enough that a properly
statistically-pinned campaign is likely to yield a publishable positive
result in a clear regime.

---

## What we have

- **Code**: `gpu-rocm/pdmrg-multi-gpu/` (~3,191 LOC). Not built by
  `gpu-rocm/build_all.sh`, not in the campaign runner's variant list.
  Per the GPU audit, no `build_mi300x.sh` script — the scaffolding for a
  reproducible build is missing.
- **Backing data**: `benchmarks/paper_results/mi300x/challenge/pdmrg-multi-gpu_mi300x_challenge_20260407_partial.json`
  — 27 of 44 planned configs, single-rep, no Wilson CI, harness not
  statistically matched against the single-MI300X `pdmrg-gpu` numbers.
- **Comparison data points** (paper §6.7.1, `tab:multi_gpu`): four
  overlapping Heisenberg configurations against `pdmrg-gpu`, showing
  speedups $0.35\times$ ($L{=}50, \chi{=}64$), $3.03\times$ ($L{=}50, \chi{=}128$),
  $2.51\times$ ($L{=}50, \chi{=}256$), $0.44\times$ ($L{=}100, \chi{=}64$).

## What's missing for a defensible follow-on campaign

1. **Reproducible build script.** Add `gpu-rocm/pdmrg-multi-gpu/build_mi300x.sh`
   following the convention used by the three baseline variants. Add the
   variant to `gpu-rocm/build_all.sh` (or omit by design if it's
   multi-device-only and shouldn't auto-build with the others).
2. **Code review pass.** Apply the same review the baseline variants got
   in the pre-G1 audit (timing scope, hidden CPU paths, print statements
   in inner loop, CLI flag uniformity vs `pdmrg-gpu`). Fix anything that
   would invalidate $N{=}10$ measurements.
3. **CLI flag parity with `pdmrg-gpu`.** The single-MI300X variant accepts
   `--warmup`, `--polish`, `--segments`, `--cpu-svd`, `--rsvd`,
   `--n-recal`. The multi-GPU variant should accept the same flag surface
   so the campaign runner can drive it via the same config schema.
4. **Statistically-matched harness.** The current single-rep data was run
   with a different harness than the published single-MI300X numbers. A
   defensible comparison needs both variants run with the *same* warmup,
   polish, sweep budget, convergence tolerance, and chi-schedule. Use the
   JSON-driven campaign runner (`benchmarks/run_campaign.py`) with a new
   config `benchmarks/campaigns/multi_gpu_followon.json` once the variant
   is wired in.
5. **Multiple coordinated multi-MI300X windows.** A 4-device $N{=}10$
   campaign on the published challenge grid is roughly 4× the GPU-hour
   budget of the single-MI300X G1 rebench (per device, plus orchestration
   overhead). Need at least 12--16 multi-MI300X-hours in a single window
   to avoid VM-churn confounds (the current partial data was lost
   mid-run when the VM expired).
6. **Inter-device transfer instrumentation.** The analytical compute-vs-
   communication crossover (paper §6.7.1) predicts a clear $\chi$
   threshold; instrumenting actual inter-MI300X bandwidth (PCIe vs
   xGMI vs whatever the chassis exposes) would let the paper claim a
   *quantitative* crossover prediction, not just a *directional* one.

## Suggested investigation envelope

Same as G1 (Heisenberg + Josephson, $L \in \{32, 50, 100, 200\}$,
$\chi \in \{64, 128, 256\}$, 10 reps), plus three multi-GPU-specific
cells:

- $L{=}500, \chi{=}256$ Heisenberg (long chain, large per-segment work) ---
  the regime where multi-GPU should win biggest.
- $L{=}32, \chi{=}64$ across both 2-GPU and 4-GPU configurations ---
  isolates the per-segment work threshold.
- $L{=}100, \chi{=}512$ Heisenberg ($N_g{=}4$) --- the conjectured
  $\chi \gtrsim 1024$ regime is too expensive for a $N{=}10$ pass at this
  envelope, but $\chi{=}512$ is a reasonable upper-bound for the present
  hardware budget.

## Status of authority

This document is the **single point of truth** for pdmrg-multi-gpu
follow-on investigation. The corresponding `docs/PATH_B_FINISHING_PLAN.md`
status table (top of file) links here. Do not duplicate this content
elsewhere; update this file when state changes.

When the investigation is funded with GPU hours, the work order is:
(1) build script, (2) code review, (3) CLI parity, (4) campaign config,
(5) run, (6) analysis section in paper or follow-up note.

---

## Open questions for operator

- Is multi-GPU pdmrg in scope for this CPC submission as a separate
  results subsection (currently §6.7.1 is exploratory only, no headline
  claim) or a follow-up note?
- Is there appetite for a 4-MI300X window in the next 2--3 weeks?
- Should the multi-GPU pdmrg-multi-gpu variant share the
  `benchmarks/run_campaign.py` schema with the single-MI300X variants
  (clean), or get its own multi-device config schema (more flexible
  but doubles maintenance)?
