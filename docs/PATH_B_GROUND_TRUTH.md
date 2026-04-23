# Path B ground truth (LOCKED as of commit 6f45533)

**This file is the single source of truth for factual claims during Path B
execution. Do NOT re-audit the code. Cite this file.**

If you find anything in the code or data that contradicts what is written
here, STOP. Post a comment on the Path B execution tracker issue explaining
the contradiction. Do not silently "fix" the discrepancy by re-interpreting
either this file or the code — the whole point of locking the inventory is
to keep every agent in the revision loop anchored to the same baseline.

Updating this file requires a deliberate re-audit pass (new inventory agents,
explicit commit, new commit-pin header). It does not happen as a side effect
of any other work.

---

The following statements are CONFIRMED CORRECT vs the actual code/data, not
the paper or reviewers, as of commit `6f45533` on `main`.

## Algorithm reality (per-variant)
- **dmrg-gpu**: Lanczos only; default SVD = `rocsolver_gesvd_auto` (GPU); RSVD via opts_.rsvd; STRICT 1-site (no DMRG3S/noise/subspace expansion); timer starts AFTER build_initial_environments.
- **dmrg-gpu-opt**: Block-Davidson (b=4) HARDCODED, no flag; SVD = lapack_gesvd (CPU host roundtrip); STRICT 1-site; timer starts BEFORE build_initial_environments. opts_.device_k and opts_.rsvd are DECLARED BUT NEVER READ.
- **dmrg2-gpu**: Lanczos only; default SVD = rocsolver_gesvd_auto; RSVD via opts_.rsvd; timer starts AFTER env build.
- **dmrg2-gpu-opt**: Block-Davidson (b=4) hardcoded, no flag; SVD = lapack_gesvd (CPU); single stream; timer starts BEFORE env build. opts_.device_k + opts_.rsvd DECLARED BUT NEVER READ.
- **pdmrg-gpu**: Lanczos only; default SVD = rocsolver_gesvd_auto; CPU `accurate_svd` at boundaries; HAS recalibration phase (n_recal param, default off); CLAUDE.md compliant defaults (n_warmup=1, n_polish=0); per-segment streams.
- **pdmrg-gpu-opt**: Lanczos default, Davidson via --davidson; SVD = lapack_gesvd (CPU) ONLY; NO recalibration; NON-COMPLIANT defaults (n_warmup=3, n_polish=10 hardcoded with no CLI override).

## Newton-Schulz reality
- NS exists ONLY in `cpu/pdmrg-opt/pdmrg/numerics/linalg_utils.py` (CPU Python).
- ZERO GPU variants contain NS.
- The CPU NS NEVER produced any benchmark JSON.
- Therefore: paper's "Newton-Schulz diverges at chi >= 128" claim cannot be
  sourced from any benchmark; if NS was ever benchmarked, it was on the CPU
  and not recorded.

## CheFSI / Chebyshev reality
- NO source file matching `chefsi*`, `chebyshev*` exists anywhere in the repo.
- CheFSI appears only in `paper/main.tex` text and `paper/refs.bib` citation `Zhou2006`.
- The "1.9-11x slower" claim has no source code that produced it.

## SVD path reality
- Primary `-gpu` variants: GPU rocsolver default.
- `-opt` variants: CPU LAPACK (host roundtrip).
- Paper §2.6 says "SVD on CPU LAPACK" — true for `-opt` only, false for primary.
- Paper §5.6 Table 13 (97-98% SVD) — "for dmrg2-gpu Heisenberg L=64 chi=256"
  but NO JSON has this profile (ablation uses L=32 chi=128/256).
- Where the 97-98% number came from is unclear.

## Data reality
- 2 ablation JSONs have crash-counted-as-success (rc=-6 stored as ~7.8s and
  ~0.8s wall_s):
  - `benchmarks/data/gpu_ablation/20260420T181425Z/pdmrg-gpu/results.json` (superseded by 20260421T190910Z, but still on disk)
  - `benchmarks/data/gpu_ablation/20260421T004212Z/dmrg-gpu-opt/results.json` (NEVER re-run; paper Table 6 dmrg-gpu-opt cells stand on this corrupted data)
- 6 different git commits across ablation JSONs (no single commit produced full Table 6).
- Binary SHA drift within 1 hour same day for dmrg-gpu baseline.
- All JSONs that have `provenance.gpu` record `vendor=unknown` (rocm-smi parse failure).
- `paper_results/mi300x/results.json` (named in paper §D.A.) has NO provenance fields.
- `gpu_opt_bench.json` (also named in §D.A.) has NO provenance fields.
- `bench_opt_results.json` (source of Table 5) NOT named in §D.A.
- Table 5 number stale: paper says 53.13s, JSON says 53.57s.
- Tables 8, 9, 11, 12, 13 have NO findable backing JSON in audited tree.

## CPU baseline reality
- Paper §3.1 line 386: "host CPU is an AMD EPYC processor". Reality: Intel Xeon Platinum 8470 (MI300X host), 8480C (H100 host). CPU vendor in paper is WRONG.
- Paper §3.1 line 397: "thread counts of 1, 2, 4, 8, 12". Reality: `run_mi300x_challenge.py:382` hardcodes threads=1. The 1/2/4/8/12 sweep, if it ever ran, came from a script not in the repo.
- Paper §3.1 line 397: "with cotengra for path optimization". Reality: cotengra is NEVER imported by the quimb_runner. Paper claim is false.
- Paper says "OpenBLAS 0.3.28 source-built". Reality: NO build script exists, NO LD_LIBRARY_PATH set, NO rpath. Paper claim is unsubstantiated.
- quimb baseline: bond_dims=chi (FLAT, not graduated); cutoffs=1e-14; tol=1e-11. Not idiomatic quimb usage.

## CPU implementations reality
- `cpu/pdmrg/`, `cpu/pdmrg-cotengra/`, `cpu/pdmrg-opt/` all have warmup_sweeps default = 5 (VIOLATES CLAUDE.md ≤2).
- BUT: no CPU pdmrg results appear in published JSONs, so this doesn't contaminate paper numbers.
- `cpu/pdmrg/pdmrg/dmrg.py:612-618` has np=1 → quimb DMRG2 fallback (CONFIRMED).
- `cpu/a2dmrg/` ships with 30+ test_*.py at directory top, claude-progress.txt (51KB), TASK_ORIENTATION.md ("Current State: 58/73 features passing"). LLM workflow leakage.
- README:13 says "PDMRG-OPT: Specification only — no code written yet". CONTRADICTED by 1032 LOC in `cpu/pdmrg-opt/pdmrg/dmrg.py`.

## README reality (vs paper)
- README:83: "PDMRG np=1 → quimb DMRG2" CONFIRMED in CPU code; not in C++ binary; affects no published paper rows.
- README:85: "PDMRG Boundary Merge Optimization Disabled" — REFUTED. Code unconditionally runs the merge when n_segments >= 2 (`pdmrg_gpu_impl.h:2679, 2691`).
- README:120-122: names `heisenberg_benchmark.py` and `heisenberg_long_benchmark.py` as canonical entry points. NEITHER FILE EXISTS.
- README:133-137: references `cpu/pdmrg/venv/bin/python run_pdmrg_np1.py`. NEITHER VENV NOR SCRIPTS EXIST.

## Reference / bibliography reality
- 31 entries in `paper/refs.bib`; 22 cited in body; 9 uncited.
- `Liu1978`: journal "Numerical Algorithms in Chemistry: Algebraic Methods", volume 49 — NO SUCH JOURNAL. Fabricated.
- `Nakamura2013`: bib KEY is "Nakamura2013" but author field is "Nakatsukasa, Yuji and Higham, Nicholas J." (key/author mismatch — LLM artifact).
- `Nakamura2020`: missing volume and pages.
- `Nemes2014`: doi has invalid suffix ".surface".
- TeNPy/Hauschild-Pollmann: NOT IN BIB AT ALL.
- Cited but never \cite'd (uncited): Schollwoeck2005, Fishman2022 (ITensor), Zhai2021 (Block2), Hubig (DMRG3S 2015), Kantian2019, Ren2021, Lanczos1950.

## GpuOpts ablation flag wiring per variant
- Primary -gpu variants: all 6 flags wired.
- dmrg-gpu-opt: device_k NOT wired, rsvd NOT wired, lanczos_graph force-disabled (Block-Davidson incompat).
- dmrg2-gpu-opt: same as dmrg-gpu-opt.
- pdmrg-gpu-opt: device_k NOT wired through GpuOpts (uses local `use_rsvd_`), rsvd NOT wired through GpuOpts, lanczos_graph force-disabled when --davidson.
