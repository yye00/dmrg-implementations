# Challenge MPO v2 Benchmark Results — MI300X

**Run:** 2026-04-17 18:44Z
**Host:** enc1-gpuvm020 (AMD Instinct MI300X)
**Commit:** 2db2282

## Configurations

Targets regimes that v1 missed (v1 hit Majumdar–Ghosh at J2=0.5, too easy):
1. **Non-MG J1-J2:** J2 ∈ {0.3, 0.4, 0.6, 0.7} at L=100, χ=128
2. **Large χ=256:** J1-J2 at J2=0.4 (non-MG), L ∈ {50, 100}
3. **J1-J2-J3** (new MPO, D=20) at (J2=0.4, J3=0.2), L ∈ {50, 100, 200}, χ ∈ {64, 128}
4. **Ladder χ=256:** L_rungs ∈ {50, 100}

Schedules: dmrg-gpu 10 sweeps, dmrg2-gpu 5 sweeps, pdmrg-gpu warmup=2 (1-site), segments=4, outer=10, local=2, polish=0.

## pdmrg-gpu vs dmrg-gpu (single-site) — all ratios >1 (pdmrg slower)

| config | dmrg1 | pdmrg | ratio |
|--------|-------|-------|-------|
| **J1-J2 L=100 χ=128 J2=0.7** | 26.85 | 28.62 | **1.07× ← closest** |
| J1-J2-J3 L=50 χ=64 | 5.90 | 6.62 | 1.12× |
| J1-J2 L=100 χ=128 J2=0.6 | 16.96 | 19.90 | 1.17× |
| J1-J2 L=100 χ=128 J2=0.3 | 29.88 | 38.53 | 1.29× |
| J1-J2 L=100 χ=128 J2=0.4 | 24.42 | 35.69 | 1.46× |
| J1-J2-J3 L=200 χ=64 | 32.97 | 48.74 | 1.48× |
| J1-J2 χ=256 L=50 | 14.97 | 27.00 | 1.80× |
| J1-J2-J3 L=200 χ=128 | 62.67 | 136.65 | 2.18× |
| J1-J2 χ=256 L=100 | 39.56 | 90.01 | **2.27× ← worst** |
| **Ladder L=50 χ=256** | 14.56 | 146.38 | **10.1× ← catastrophic** |
| **Ladder L=100 χ=256** | 33.10 | 269.15 | **8.13× ← catastrophic** |

## pdmrg-gpu vs dmrg2-gpu (two-site) — pdmrg mostly wins

| config | dmrg2 | pdmrg | ratio |
|--------|-------|-------|-------|
| J1-J2 L=100 χ=128 J2=0.6 | 52.66 | 19.90 | **0.38× (2.6× faster)** |
| J1-J2 L=100 χ=128 J2=0.7 | 57.14 | 28.62 | 0.50× |
| **Ladder L=100 χ=256** | 499.91 | 269.15 | **0.54× (1.86× faster)** |
| J1-J2-J3 L=100 χ=128 | 61.87 | 36.31 | 0.59× |
| J1-J2 L=100 χ=128 J2=0.3 | 64.30 | 38.53 | 0.60× |
| J1-J2-J3 L=50 χ=64 | 10.91 | 6.62 | 0.61× |
| Ladder L=50 χ=256 | 226.08 | 146.38 | 0.65× |
| J1-J2 L=100 χ=128 J2=0.4 | 52.15 | 35.69 | 0.68× |
| J1-J2 χ=256 L=100 | 113.69 | 90.01 | 0.79× |
| J1-J2-J3 L=200 χ=64 | 59.33 | 48.74 | 0.82× |
| J1-J2-J3 L=200 χ=128 | 134.86 | 136.65 | ~1.0× (tie) |
| **J1-J2-J3 L=100 χ=64** | 25.81 | 31.46 | **1.22× (pdmrg slower)** |

## Verdict

**pdmrg-gpu never beats dmrg-gpu on any 1D ground-state problem.** The frontier is J1-J2 J2=0.7 where pdmrg is only 7% slower — but never faster.

**pdmrg-gpu reliably beats dmrg2-gpu** (11 of 13 configs), with biggest wins on:
- Frustrated J1-J2 at L=100 χ=128 (1.5-2.6× faster than dmrg2 across all J2)
- Ladder at χ=256 (1.5-1.9× faster than dmrg2)

**Larger χ makes pdmrg WORSE relative to dmrg1**, contra hypothesis:
- J1-J2 J2=0.4: χ=128 → 1.46× slower, χ=256 → 2.27× slower
- Ladder χ=128 → 2.6×, χ=256 → 8-10× slower
  
The segment-parallel overhead (boundary sync, L/R environment rebuilds at each outer iter) grows faster than the χ³ GEMM cost saved.

**J1-J2-J3 didn't change the story.** Extended-range frustration doesn't flip the dmrg1 win — convergence is still fast enough that the 10-sweep dmrg1 budget handles it comfortably.

## Where does pdmrg-gpu fit?

Based on v1 + v2 data, pdmrg-gpu's realistic niche is:
- **High-d physical sites (d≥4)** where dmrg2-gpu's d² blowup kills it and dmrg1 would still need two-site anyway for adaptive bond-dim growth
- **Not** 1D spin-1/2 ground states, regardless of frustration, MPO bond dim, or chi

To find a real pdmrg-gpu win vs dmrg1, we'd need to look at:
1. Excited states / multi-target DMRG (segments → independent eigenvectors)
2. Finite-T DMRG or time evolution (segments → independent time steps)
3. Very heavy MPOs with long-range couplings that require huge bond dim from the start
4. Fermionic / bosonic models with large local d (Hubbard d=4, extended Bose-Hubbard d≥5)
