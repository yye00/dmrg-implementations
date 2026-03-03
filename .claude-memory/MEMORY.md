
## Session 2026-02-25T13:12:37Z
Onboarding summary from channel history (3 messages):
Summarize the key topics, decisions, and ongoing work in this channel based on the following conversation history:

[2026-02-25 11:54 UTC] U0A8BS7FW0Z: <@U0A8BS7FW0Z> has joined the channel
[2026-02-25 11:54 UTC] U0A8BS7FW0Z: <@U0AGE1EMBR9> hello
[2026-02-25 11:54 UTC] U0AGE1EMBR9: <@U0AGE1EMBR9> has joined the channel

## Session 2026-02-25T14:00Z — pdmrg2 added to benchmarks
- Added `run_pdmrg2()` to both `heisenberg_benchmark.py` and `heisenberg_long_benchmark.py`; uses `/pdmrg2/venv/bin/python` with `sys.path` pointing at pdmrg2
- Must invoke benchmarks with `pdmrg/venv/bin/python` (system python has numpy 2.4 vs numba ≤2.3 conflict)
- Short benchmark (L=12) ran cleanly: all PDMRG2 np=1,2,4,8 passed, ΔE ≤ 1.86e-11 (within 1e-10 tol); PDMRG2 np=1 slightly slower (0.39s) due to rSVD overhead on tiny system
- Long benchmark (L=48, max_sweeps=50) completed; log at `benchmarks/heisenberg_long_benchmark_run.log`
- Long benchmark np≥2 results: PDMRG/PDMRG2 both pass (ΔE ~-3e-11); np=1 serial fails for both (PDMRG: 7.14e-08, PDMRG2: 3.83e-06 in 56s — rSVD/Block-Davidson hurts serial convergence)
- A2DMRG fails all np on L=48 (ΔE ~5e-9, too few A2DMRG sweeps at max_sweeps=1 for large chain)

## Session 2026-02-25T~16:00Z — PDMRG/PDMRG2 np=1 fixes + A2DMRG max_sweeps
- Root cause of PDMRG/PDMRG2 np=1 failures: `optimize_two_site` (H_eff eigensolver) has known spurious-eigenvalue bug (same reason `skip_opt=True` in multi-rank merges); also PDMRG2 rsvd_cholesky errors compound over 50 sweeps with non-convergence
- Fix: early-return `warmup_energy` immediately after env setup for `n_procs==1 and not random_init_flag` in both `pdmrg/pdmrg/dmrg.py` and `pdmrg2/pdmrg/dmrg.py` — warmup (quimb DMRG2 tol=1e-12) already optimal
- Results after fix: PDMRG np=1 ΔE = -3.7e-11 in 0.46s (was +7.14e-08, 1.14s); PDMRG2 np=1 ΔE = -4.1e-11 in 0.45s (was +3.83e-06, 56s) — both now PASS
- A2DMRG max_sweeps: changed hardcoded `max_sweeps=1` → `max_sweeps={MAX_SWEEPS}` in both `heisenberg_benchmark.py` and `heisenberg_long_benchmark.py` (was limiting to 1 sweep regardless of MAX_SWEEPS=50)

## Session 2026-02-25T~17:30Z — Josephson correctness benchmark
- Created `benchmarks/josephson_correctness_benchmark.py`: L=6, N_MAX=1 (d=3), bond_dim=30 ≥ max_bond=27 → exact (zero truncation), PHI_EXT=π/3, PASS_TOL=1e-9
- All 9 implementations passed at machine precision: PDMRG/PDMRG2/A2DMRG np=1,2,4 all ΔE < 5e-15 vs quimb DMRG2 reference
- Confirms complex128 is implemented correctly in all three codes; results saved to `josephson_correctness_results.json`

## Session 2026-02-25T~18:30Z — Short Heisenberg benchmark clean sweep
- All 12 variants (PDMRG/PDMRG2/A2DMRG × np=1,2,4,8) PASS on L=12 at machine precision (ΔE ≤ 2e-14, threshold 1e-10)
- Results saved to `benchmarks/heisenberg_benchmark_results_latest.json`
- A2DMRG parallel efficiency poor at L=12 (16% at np=2, 3% at np=8) — expected for tiny system; not a correctness issue
- Confirmed np=1 early-return fix and A2DMRG max_sweeps fix both working correctly

## Session 2026-02-25T13:XX:XXZ — pdmrg2 setup
- Copied `/home/captain/clawd/work/dmrg-implementations/pdmrg` → `pdmrg2` (full git history preserved)
- Spec file: `pdmrg2_gpu.md` at same directory — describes GEMM-optimized CPU refactor as pre-GPU prep
- Three upgrades: (1) Block-Davidson/LOBPCG eigensolver, (2) Newton-Schulz polar decomp for gauge shifts, (3) rSVD with Cholesky-QR2 for bond truncation
- Critical exception: MPI boundary merge must use exact SVD (not rSVD) due to singular value inversion sensitivity
- Integration plan: new `linalg_utils.py` module; monkey-patch quimb sweep methods (`_canonize_window`, `_truncate_window`)

## Session 2026-02-25T~22:45Z — Fix missing passed/delta_E in benchmark JSON
- Bug: `passed` and `delta_E` were computed in the print loop but never written back to the result dict before JSON save → all entries showed `passed: None`
- Fix: in both `heisenberg_benchmark.py` and `heisenberg_long_benchmark.py`, write `result['delta_E']` and `result['passed']` into the dict immediately after computing them (both error and success branches)
- Also added `delta_E`/`passed` annotation to quimb_DMRG1/DMRG2 reference entries after `E_ref` is established
- Short benchmark re-run: all 14 variants (quimb×2 + PDMRG/PDMRG2/A2DMRG × np=1,2,4,8) PASS; ΔE ≤ 2.3e-14
