# Follow-up Round 2 — Synthesis and Implementation Plan

**Opened:** 2026-04-10
**Parent:** Generate-and-critique loop over `docs/PROJECT_OVERVIEW.md` (9 REJECT, 1 REVISE)
**Inputs:** Four perplexity-backed research reports in this directory —
`research_A_hip_graph_rocblas.md`, `research_B_svd_frequency_reduction.md`,
`research_C_persistent_lanczos.md`, `research_D_beyond_pdmrg_a2dmrg.md`.

---

## Executive summary

Four independent research directions converged on one unifying primitive:
**Controlled Bond Expansion (CBE)** by Gleis, Li, von Delft (PRL 130, 246402,
2023; arXiv:2207.14712). CBE delivers "two-site DMRG accuracy at single-site
cost" and simultaneously:

1. Collapses the 97–98 % CPU-SVD wall-time bottleneck at `chi ≥ 128` (Research B)
   — by replacing the `O(d²·chi³·w)` two-site SVD with an `O(d·w·k·D²)`
   single-site update with bond expansion via RSVD + QR.
2. Unlocks a genuinely novel parallel DMRG scheme (Research D) — "CBE-TEBD-DMRG"
   — that beats both Stoudenmire PDMRG and Grigori-Hasan A2DMRG at
   `L ∈ [32, 128], chi ∈ [64, 256]` on 4× MI300X.
3. Has **zero public GPU implementation** (Research B, §3) — ITensor, SyTen,
   block2, TeNPy, quimb are all CPU-only for CBE/DMRG3S/3S. A GPU port is both
   the biggest single performance win on the table *and* publishable-novel.

Meanwhile:

- **Research A** delivered a critical go/no-go finding that reshapes Proposal 3:
  `rocblas_dgemm_batched` is **NOT** safe under HIP stream capture on ROCm 7.2
  (confirmed from AMD docs, ROCm/rocBLAS#1240, and release notes). The escape
  hatch is `rocblas_dgemm_strided_batched` + a mechanical refactor of
  `apply_heff_two_site`. If even that fails, fall back to a custom fused
  rocWMMA kernel.
- **Research C** designed a per-workgroup persistent Lanczos kernel that keeps
  `theta` + one Krylov vector in LDS across all Lanczos iterations for a bond.
  Expected 10–20 × Lanczos speedup at `chi ≤ 32` — exactly the regime where
  single-core CPU beats MI300X in 93 % of benchmark configs.

The round produces **four concrete follow-ups**, ordered by a dependency graph
that makes the biggest win (CBE-DMRG on GPU) the backbone and the smaller
optimizations composable extensions.

## 1. Unified finding: Controlled Bond Expansion is the linchpin

Both Research B (SVD frequency reduction) and Research D (novel parallel DMRG)
independently concluded that the project's real bottleneck is **two-site DMRG
itself**, not any specific rocBLAS call or stream scheduling. Two-site DMRG
buys better convergence per sweep by paying an `O(chi³)` SVD on every bond.
Single-site DMRG (1-site) is `O(chi²)` per bond but gets stuck in local minima
because it cannot expand the bond basis. CBE breaks this trade-off: it runs
single-site updates but expands the bond basis via a cheap rank-revealing QR
+ RSVD on a small `(d·w·k) × D` tangent-space matrix, *never* forming the
`(chi·d × chi·d)` theta.

This is the right lever for four reasons:

| Reason | Source |
|---|---|
| Bypasses `97–98 %` SVD dominance at `chi ≥ 128` | Research B §5.1 of overview |
| Variationally correct (monotone energy) | McCulloch/Osborne arXiv:2403.00562 |
| FLOP ratio vs 2-site: `O(d·w·k·D²) / O(d²·chi³·w)` ≈ 2–4 × at `chi=256` | Research B §2 |
| No GPU implementation exists | Research B §3, Research D §2 |

CBE is therefore the **backbone** of Round 2, and every other follow-up either
depends on it or composes with it.

## 2. The four follow-ups (ranked)

### Follow-up #R2-1 — CBE-DMRG GPU backbone (**highest priority**)

**What:** Port Controlled Bond Expansion single-site DMRG to GPU as a new
implementation `gpu-rocm/cbe-dmrg-gpu/` modeled after `dmrg-gpu-opt`. Use the
McCulloch/Osborne 2024 simplification (arXiv:2403.00562) — it replaces the
two-site projection with an RSVD on a small tangent-space matrix, which maps
cleanly to existing rocBLAS primitives and avoids the accuracy cliff that
killed the earlier randomized-SVD attempt (§4.2 of overview).

**Why:** 2–4× raw speedup at `chi ≥ 128` by construction (no SVD of the full
`theta`), and it is the prerequisite for every subsequent Round-2 follow-up.
This is also the single most publishable piece of work in the round — there
is no published GPU implementation of CBE as of 2025.

**Cost:** Medium. ~1500–2000 LOC. The kernels are a subset of what `dmrg-gpu`
already ships (single-site apply_heff + Lanczos + a small RSVD on a
low-dimensional matrix).

**Risks:**
- CBE's RSVD is on a `(d·w·k) × D` matrix where `D = chi_mpo · chi_L`. At
  `chi_L = 256, chi_mpo = 5` that's a 1280-wide matrix — exactly the size
  where rocSOLVER gesvd dies. Mitigation: the matrix has rank ≤ `k + p` by
  construction, so a small RSVD (power-iteration with `k + p` columns) is
  used, not full SVD.
- Convergence rate of single-site + CBE on Josephson arrays is not documented
  in the literature (published results are for Heisenberg, Hubbard, and
  ab-initio chemistry). Needs an early correctness gate on our three
  reference models before further investment.
- The existing tests (`test_dmrg_gpu`, `test_dmrg2_gpu`, `test_pdmrg_gpu`) do
  not have a CBE baseline. A CPU reference implementation (via a Python port
  of the Larsson 2024 pseudocode) is required for ΔE regression testing.

**Measurement plan:**
1. Python CBE reference in `cpu/cbe-reference/` against TeNPy for L=16 Heisenberg.
2. GPU port in `gpu-rocm/cbe-dmrg-gpu/` matching reference to 1e-10 on
   L∈{4,8,16} Heisenberg and Josephson.
3. Scaling sweep at L∈{16,32,64,128} × chi∈{64,128,256,512} against
   `dmrg2-gpu-opt` baseline. Target 2× at chi=128, 3× at chi=256.
4. CPU-SVD wall-time fraction should drop from 97 % to < 20 % after CBE ports
   cleanly — confirm with `PDMRG_PROFILE=1` traces.

---

### Follow-up #R2-2 — CBE-TEBD-DMRG novel parallel scheme (**flagship**)

**What:** A three-phase parallel DMRG algorithm, detailed in Research D §7,
built on top of #R2-1:

- **Phase 0 — Parallel iTEBD warmup.** Replace PDMRG's serial single-site
  warmup (`37 %` of L=64 chi=128 runtime per the Amdahl table) with
  embarrassingly-parallel even/odd TEBD gates + QR truncation (Krumnow/Eisert,
  arXiv:2212.09782 — 2700× GPU speedup reported for imaginary-time TEBD).
  Runs on `P` GPUs in parallel, no coupling required.
- **Phase 1.5 — Canonicalize + peer-broadcast.** Serial canonicalization
  (cheap, `~100 ms`) followed by peer-to-peer broadcast of MPS + environments
  across `P` GPUs via xGMI / NVLink. Makes environments replicated, not
  partitioned.
- **Phase 2 — Block-Jacobi CBE single-site sweeps with replicated envs.**
  Each GPU owns `L / P` contiguous sites and runs CBE single-site updates in
  parallel. Because environments are replicated (cheap on intra-node
  interconnects), there is no coupling step — Jacobi-style updates converge
  variationally in 2–3 extra iterations versus Gauss-Seidel but parallelize
  embarrassingly.
- **Phase 3 — Short serial CBE polish.** 1–2 serial CBE sweeps at the end.

**Why:** Attacks PDMRG's measured 1.22 × Amdahl ceiling directly. Every one
of the three serial phases (warmup 37 %, coupling 27 %, polish 32 %) is either
parallelized (Phase 0, Phase 2) or shrunk (Phase 3 becomes CBE, 2–4 × cheaper
than two-site polish). Expected speedup over PDMRG:

| L | chi | P | PDMRG (measured) | CBE-TEBD-DMRG (projected) |
|---|----|---|---|---|
| 64 | 128 | 4 | 1.22 × | **1.7 ×** |
| 64 | 256 | 4 | 1.25 × | **2.3 ×** |
| 64 | 256 | 8 | ~1.3 × | **2.8 ×** |

These are *over PDMRG*, not over `dmrg2-gpu-opt` directly. Stacked on the
~2 × from #R2-1, total speedup vs current `dmrg2-gpu-opt` is 3–5 × in the
target regime.

**Cost:** High. ~3000 LOC on top of #R2-1. Requires a TEBD gate kernel
(new), a canonicalization + broadcast stage (reuses `pdmrg-multi-gpu`
communication primitives), and a Jacobi scheduler.

**Risks:**
- Block-Jacobi CBE has no published convergence guarantee. Could degrade
  accuracy by 1–2 sweeps or fail to converge on critical-point Hamiltonians.
  Mitigation: gate via the existing `quimb-dmrg2` regression suite at
  `|ΔE| < 1e-10` on all three models, with an early-exit in Phase 2 if
  per-bond energy delta exceeds a threshold.
- Phase 0 iTEBD has its own accuracy concerns for non-translation-invariant
  Hamiltonians (Josephson with position-dependent fluxes). Mitigation: fall
  back to single-site warmup for these cases.
- Depends on #R2-1 shipping first.

**Measurement plan:** Identical grid to #R2-1 plus a dedicated
Amdahl-decomposition run that reports wall time for Phase 0, Phase 1.5,
Phase 2, Phase 3 separately. Success = measured total < 60 % of PDMRG
measured total at `L=64 chi=256 P=4`.

---

### Follow-up #R2-3 — Persistent Lanczos kernel at small chi

**What:** Per-workgroup persistent HIP kernel (Research C §6) that keeps
`theta` + one Krylov vector + workspace in 64 KB LDS and runs the entire
Lanczos outer loop for one bond inside a single `hipLaunchKernelGGL`. Archival
Krylov basis lives in L2 with SLC = 0 hints (best-effort retention — MI300X
has 256 MB Infinity Cache, so a 32 KB theta is trivially cacheable).

**Envelope:** `d = 2, chi ≤ 32` strictly (LDS hard limit); `chi ≤ 48` is
possible if the archival basis is forced to HBM. This is exactly the regime
where §5.2 single-core CPU wins are concentrated.

**Why:** HIP graph capture (Proposal 3) can reduce launch overhead but cannot
solve cache-residency. The persistent kernel is the only way to match
single-thread CPU L1-resident throughput on a GPU. Expected 10–20 × speedup
on the Lanczos portion at `chi ≤ 32`, which flips most of the
`L ∈ [32, 64], chi ∈ [32, 48]` CPU-win configs.

**Cost:** Medium-high. ~800 LOC of hand-written HIP + `rocWMMA` FP64 MFMA-16
intrinsics. No rocBLAS dependency (which sidesteps Research A's capture
problem entirely). AMD Composable Kernel (`composable_kernel`) tile operators
are the right building block — hipTENSOR has no persistent primitives.

**Risks:**
- Hand-rolled MFMA correctness vs rocBLAS. Mitigation: bit-exact regression
  against the existing `apply_heff_two_site` path on L = 8.
- Occupancy-1 per CU means weak latency hiding. Mitigation: prefetch
  reorthogonalization basis from L2 one iter ahead.
- Cache-control flags (GLC/SLC/DLC/NT) on `BUFFER_LOAD` are un-benchmarked on
  CDNA3 — we are assuming L2 retains `theta` across iterations. Microbench
  required before committing to this design.
- FP32 input / FP64 accumulate MFMA does not exist on CDNA3 (Research C §7),
  so the mixed-precision shortcut is not available.

**Measurement plan:**
1. Microbench `apply_heff` with persistent kernel vs existing path at
   `chi ∈ {16, 20, 24, 32, 48}`, Heisenberg `d=2, D=5`.
2. Full-sweep benchmark on the 29 CPU-win Heisenberg configs in
   `benchmarks/paper_results/mi300x/wins_cpu_vs_gpu.csv`. Target: flip ≥ 15.
3. L2-retention microbench first — if cache hint is a no-op, abort and
   redesign with full-LDS `theta` (caps envelope at `chi ≤ 20`).

**Composability:** This follow-up is orthogonal to #R2-1 and #R2-2 and can
land independently. It produces the biggest wins in the `chi ≤ 48` regime
while CBE wins in the `chi ≥ 128` regime.

---

### Follow-up #R2-4 — Refined Proposal 3 (HIP graph capture)

**What:** The narrowed, reviewed version of the sole survivor from the
generate-critique loop. See `followups/proposal_3_hip_graph_capture.md` for
full details.

**Critical update from Research A:** The Phase 0 go/no-go microbench must use
`rocblas_dgemm_strided_batched`, NOT `rocblas_dgemm_batched`. AMD explicitly
lists the `_batched` variant as unsupported under capture on ROCm 7.2. The
`_strided_batched` variant is the escape hatch.

**If strided-batched ALSO fails capture:** pivot to **Proposal 3-alt** (custom
fused HIP kernel via rocWMMA FP64 MFMA-16, Research A §3). This is a pure
superset of #R2-3's persistent kernel work — one fused kernel does
`apply_heff` Step 1 + Step 2 + Step 3 without any rocBLAS calls at all,
independent of graph-capture status, and portable to H100 via CUTLASS.

**Cost:** Low (graph capture variant, ~300 LOC). Medium (Proposal 3-alt
custom kernel variant, ~1000 LOC — but roughly half of that is shared with
#R2-3 and can be factored into a common `apply_heff_kernel` library).

**Risks / measurement:** Unchanged from `proposal_3_hip_graph_capture.md`.

**Composability:** Stacks with #R2-1 (CBE matvecs benefit from graph capture
just like two-site matvecs do) and is orthogonal to #R2-2 and #R2-3.

---

## 3. Dependency graph and recommended ordering

```
                 R2-1 CBE-DMRG backbone (6-8 weeks)
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
       R2-2       R2-3        R2-4
     flagship   persistent   graph
     parallel   Lanczos      capture
     (4-6 wks)  (3-4 wks)    (1 wk if GO,
                              2-3 wks if
                              Proposal 3-alt)
```

**Phase 0 (this week):** Run the Research A Phase-0 microbench for HIP graph
capture of `rocblas_dgemm_strided_batched`. Outcome determines whether R2-4 is
"tiny" or "medium" work. Also kick off the Python CBE reference
implementation in `cpu/cbe-reference/` in parallel — no GPU needed, can be
done locally.

**Phase 1 (weeks 1-8):** Ship R2-1 (CBE-DMRG backbone). This is the backbone
of everything else and the single biggest win on its own. Independently, R2-4
can land in parallel if the Phase-0 gate passed cleanly.

**Phase 2 (weeks 9-14):** Ship R2-2 (CBE-TEBD-DMRG flagship) on top of R2-1.
This is the novel publishable scheme.

**Phase 3 (weeks 15-18):** Ship R2-3 (persistent Lanczos kernel) targeting
the small-`chi` regime. Can run in parallel with Phase 2 if bandwidth allows.

**Phase 4 (ongoing):** CUDA / H100 port via CUTLASS for R2-3 and
Proposal 3-alt. R2-1 and R2-2 port via the existing `gpu-cuda/` mechanical
script.

## 4. Risk register

| Risk | Severity | Owner follow-up | Mitigation |
|---|---|---|---|
| CBE doesn't converge on Josephson arrays | High | R2-1 | Python reference gate before GPU port |
| rocBLAS strided-batched also fails capture | Medium | R2-4 | Fall back to Proposal 3-alt (rocWMMA fused kernel) |
| Block-Jacobi CBE has unacceptable convergence rate | High | R2-2 | Early-exit gate in Phase 2; fall back to Gauss-Seidel ordering |
| Persistent kernel cache-control flags are no-ops on CDNA3 | Medium | R2-3 | L2-retention microbench as first task |
| Phase 0 iTEBD fails on position-dependent Josephson fluxes | Medium | R2-2 | Retain classical single-site warmup as fallback |
| rocSOLVER RSVD accuracy cliff at tangent-space matrix | Medium | R2-1 | Use on-device power iteration, not rocSOLVER |

## 5. What Round 2 does NOT cover

- **Contraction path rewrite** (§9 item 1 of overview) — still open but
  independent of this round. Can be tackled after R2-1 ships.
- **Mixed-precision SVD** (§9 item 5) — superseded by R2-1 (no SVD to
  mix-precision).
- **rocSOLVER tuning at chi ≥ 512** (§9 item 6) — also superseded by R2-1.
- **Multi-state / excited-state DMRG** (rejected Proposal 10) — remains out
  of scope; target application is ground-state simulation.

## 6. Deliverables checklist

- [x] `docs/followups/proposal_3_hip_graph_capture.md` — refined Proposal 3
- [x] `docs/followups/research_A_hip_graph_rocblas.md` — go/no-go on rocBLAS capture
- [x] `docs/followups/research_B_svd_frequency_reduction.md` — CBE + AHC-DMRG
- [x] `docs/followups/research_C_persistent_lanczos.md` — persistent kernel design
- [x] `docs/followups/research_D_beyond_pdmrg_a2dmrg.md` — CBE-TEBD-DMRG scheme
- [x] `docs/followups/round_2_plan.md` — this document
- [ ] `cpu/cbe-reference/` — Python CBE reference (R2-1 prerequisite)
- [ ] `gpu-rocm/sandbox/bench_graph_capture.cpp` — Phase 0 gate (R2-4 prerequisite)
- [ ] `gpu-rocm/cbe-dmrg-gpu/` — R2-1 implementation
- [ ] `gpu-rocm/cbe-tebd-dmrg/` — R2-2 implementation
- [ ] `gpu-rocm/persistent-lanczos/` or kernel lib — R2-3 implementation
- [ ] `gpu-rocm/dmrg2-gpu-opt/` graph-capture patch — R2-4 integration

## 7. Citations

Primary CBE literature:
- Gleis, Li, von Delft. "Controlled Bond Expansion for DMRG Ground State
  Search at Single-Site Costs." Phys. Rev. Lett. 130, 246402 (2023).
  arXiv:2207.14712.
- McCulloch, Osborne. "Density Matrix Renormalization Group in the Age of
  Controlled Bond Expansion." arXiv:2403.00562 (2024).

Primary TEBD / QR-TEBD literature:
- Krumnow, Eisert. "Quantum-aware QR decomposition for efficient TEBD."
  arXiv:2212.09782 (2022).

Background (for Research B historical context):
- White. "Density matrix renormalization group for a highly entangled state."
  arXiv:cond-mat/0508709 (2005).
- Hubig, McCulloch, Schollwöck, Wolf. "Strictly single-site DMRG algorithm
  with subspace expansion." arXiv:1501.05504 (2015).
- Zhang. "An answer to an open question in the incremental SVD."
  arXiv:2204.05398 (2022).

All research reports in this directory contain full numbered citations with URLs.
