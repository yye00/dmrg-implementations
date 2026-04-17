# Challenge MPO Benchmark Results — MI300X

**Run:** 2026-04-17 18:14Z
**Host:** enc1-gpuvm020 (AMD Instinct MI300X, ROCm 7.2+)
**Commit:** 752db99

## Configurations

- **J1-J2 Heisenberg** (J1=1, J2=0.5 — Majumdar–Ghosh point, OBC):
  L ∈ {50, 100, 200}, χ ∈ {64, 128}
- **2-leg Heisenberg ladder** (J_leg=J_rung=1, OBC, d=4 supersites):
  L_rungs ∈ {50, 100}, χ ∈ {64, 128}
- **Sweep schedules:**
  - dmrg-gpu (single-site): 10 sweeps
  - dmrg2-gpu (two-site): 5 sweeps
  - pdmrg-gpu: warmup=2 (1-site), segments=4, outer=6, local=2, polish=0

## J1-J2 timings (seconds)

| L | χ | dmrg1 | dmrg2 | pdmrg | E_final |
|---|---|-------|-------|-------|---------|
| 50  | 64  | 3.01  | 5.14  | 6.93  | -18.75 |
| 50  | 128 | 4.60  | 8.17  | 7.48  | -18.75 |
| 100 | 64  | 6.23  | 12.08 | 10.75 | -37.50 |
| 100 | 128 | 15.58 | 27.25 | 27.33 | -37.50 |
| 200 | 64  | 13.10 | 25.76 | 23.99 | -75.00 |
| 200 | 128 | 28.39 | 51.65 | 80.25 | -75.00 |

All three solvers converge to the MG dimer energy E = -3L/8 to machine precision. This problem turned out to be **too easy** — the J2=0.5 point has an exact MPS ground state (product of nearest-neighbor singlets in the thermodynamic limit; OBC is close) and converges in 1-2 sweeps regardless of solver.

## Ladder timings (seconds)

| L_rungs | χ | dmrg1 | dmrg2 | pdmrg |
|---------|---|-------|-------|-------|
| 50  | 64  | 5.87  | 30.41  | 12.35 |
| 50  | 128 | 12.31 | 73.63  | 36.18 |
| 100 | 64  | 13.99 | 66.65  | 27.65 |
| 100 | 128 | 29.04 | 163.23 | 76.07 |

Ladder energies (all within 1e-9 of each other across solvers):
- L_rungs=50, χ=128: E ≈ -57.47107364...
- L_rungs=100, χ=128: E ≈ -115.27538766...

## Verdict

**dmrg-gpu (single-site) wins speed on every single config.** Ratios (dmrg1 = 1.0):

| problem | dmrg1 | dmrg2 | pdmrg |
|---------|-------|-------|-------|
| J1-J2 avg | 1.0 | 1.9× | 2.1× |
| Ladder avg | 1.0 | 5.3× | 2.6× |

**Where pdmrg-gpu does win:** on the d=4 ladder, pdmrg is 2.0-2.4× faster than dmrg2. The d²=16 physical dimension blows up dmrg2's GEMM kernels, but pdmrg's single-site warmup sidesteps most of that cost. However, pdmrg still loses to dmrg1 on all 10 configs.

**Where pdmrg-gpu does NOT win:** nowhere vs dmrg1 on 1D ground-state problems tested so far. The J1-J2 Majumdar–Ghosh point is too easy (1-2 sweep convergence); the ladder is harder but dmrg1 still wins.

**Next steps to find pdmrg-gpu's niche:**
- J2/J1 ∈ {0.3, 0.4, 0.6, 0.7} — away from MG, harder to converge
- Larger chi (χ = 256, 512) — may amortize parallel segment overhead
- Long-range / power-law couplings — where bond dim stays high throughout
- Multi-target (excited states) or finite-T DMRG — where the segment parallelism gives real speedup per target
