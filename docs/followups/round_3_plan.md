# Round 3 Plan — Post-Empirical Synthesis

**Status:** Final synthesis of 10 generate-critique pair agents run on 2026-04-10, with live MI300X empirical validation at `hotaisle@23.183.40.84` (VM `enc1-gpuvm019`, ROCm 7.2.0, rocBLAS 5.2.0.70200, rocSOLVER 7.2).
**Parent:** `round_2_plan.md` (ranked four follow-ups R2-1…R2-4).
**Date closed:** 2026-04-10.

---

## 0. Executive verdict

Round 2 had four follow-ups (R2-1 CBE backbone, R2-2 CBE-TEBD-DMRG parallel flagship, R2-3 persistent Lanczos, R2-4 HIP graph capture). Round 3 empirically validated, refined, or killed each of them and discovered **three independent fast-paths** that were not in round 2.

| Round 2 item | Round 3 verdict | Evidence |
|---|---|---|
| R2-1 CBE backbone | **KEEP — rescope to M/O CBE, chi ≥ 64 only, 600 LOC** | Pair 1 (algebra), Pair 2 (GPU kernel) |
| R2-2 CBE-TEBD-DMRG flagship | **DEFER — phase 0 killed, phase 2 needs CPU reference first** | Pair 3 (warmup), Pair 4 (block-Jacobi) |
| R2-3 persistent Lanczos | **KEEP — confirmed empirically, single-site target, 8–12× realistic** | Pair 5 (live LDS), Pair 6 (MFMA) |
| R2-4 HIP graph capture | **KILLED — both primary and alt empirically refuted** | Pair 7 (capture), Pair 8 (fused kernel), Pair 10 (integration) |

Three round-3-only fast-paths surfaced:

| New item | Ship window | Expected win | Evidence |
|---|---|---|---|
| **R3-F1 Step-3 strided-batched collapse** | 1–2 days | 30–50% `apply_heff` speedup at chi ≤ 48 | Pair 6 rocprof trace of current `dmrg2-gpu-opt` |
| **R3-F2 rocSOLVER `dgesvdj` drop-in** | 1 week | 2.6–3.8× GPU SVD at n ≥ 512, 30–50% full sweep at χ ≥ 256 | Pair 9 live benchmark |
| **R3-F3 AHC-DMRG skip gate** (rescued from Research B §6) | 2 weeks | 1.5–3× at chi ≥ 128 | Pair 10 integration review |

**The round 3 plan replaces round 2's "4 follow-ups" ordering with a phased schedule that ships three cheap wins before touching the CBE backbone.**

---

## 1. Pair-by-pair verdicts (evidence map)

All ten deliverables in `docs/followups/round_3_pair{01..10}_*.md`. One-line summary each:

| Pair | Topic | Verdict | Key finding |
|---|---|---|---|
| **01** | CBE algebraic correctness | READY (as M/O CBE) | "No 2-site SVD" claim was wrong — M/O CBE still forms two-site θ; what it kills is the **Lanczos loop** (20 matvecs → 1). Scope 1500–2000 → **600 LOC**, 6–8 → **3–4 weeks**. |
| **02** | CBE GPU kernel design | NEEDS REFINEMENT | StreamWorkspace for CBE is **55% of current dmrg2-gpu-opt** memory, not larger. Call graph is 4 strided-batched GEMMs + 1 QR, graph-safe *except* the rocSOLVER QR (device-writes). CBE **hurts at chi ≤ 32** — chi ≥ 64 only. |
| **03** | Parallel iTEBD warmup (R2-2 phase 0) | **KILLED** | β schedule off by 10–30× (need β≈23 for ε=10⁻³, not β=2). Wall-clock 14.5 s vs 8.8 s quimb DMRG1 — slower. Citation error in R2-2: arXiv:2212.09782 is Unfried/Hauschild/Pollmann, not Krumnow/Eisert, and only covers real-time quenches. Zero published TEBD→DMRG hybrid workflows. |
| **04** | Block-Jacobi CBE convergence (R2-2 phase 2) | NEEDS SAFER SYNC | No published block-Jacobi DMRG. A2DMRG (the closest relative, 2505.23429) had to add a global coarse-space correction. Stoudenmire PDMRG is Gauss-Seidel + V=Λ⁻¹, NOT Jacobi. Pessimistic 2.5× penalty: P=2 is a net loss, P=4 no faster than PDMRG. **Python CPU reference gate required before any GPU work.** |
| **05** | Persistent Lanczos LDS (live VM) | **ENVELOPE CONFIRMED at chi ≤ 48** | Measured: chi=16 LDS 12.5%, chi=32 40.6%, chi=48 **87.5%**, chi=64 compile fails (hard ceiling 153%). Zero register spills, bank-conflict-free `ds_read2st64_b64` in hot loop. Persistent **3.1× at chi=16, 3.2× at chi=48** vs 1-launch baseline. Realistic extrapolation vs real `dmrg-gpu` Lanczos: **8–12×**. Cooperative launch definitively unneeded. |
| **06** | MFMA FP64 utilization (live VM) | MFMA only inside R2-3 | **rocBLAS already uses FP64 MFMA on gfx942** (Tensile `Cijk_..._MI16x16x4x1_...`). Standalone custom MFMA kernel REJECTED. Padding to 16 REJECTED (rocBLAS is flat 6.7–7 µs host-time chi=8..48). Hand-rolled rocWMMA MFMA correct to 6.7e-16, 97.6% of FP64 peak. **Worth it only inside persistent Lanczos megakernel** (R2-3) where rocBLAS can't be called. |
| **07** | HIP graph capture live (R2-4) | **NO-GO** | Both `rocblas_dgemm` and `rocblas_dgemm_strided_batched` fail under capture on ROCm 7.2 with `rocblas_status_internal_error` + `hipErrorStreamCaptureInvalidated/Unjoined`. Smoking gun: `strings librocblas.so.5` reveals `_rocblas_handle::is_stream_in_capture_mode()` — rocBLAS intentionally refuses capture. All three capture modes fail identically. Uncaptured baseline is 1.74 µs/iter for 3 calls = 0.58 µs/launch; `hipGraphLaunch` overhead is already 2–3 µs, so a perfect graph cannot win. |
| **08** | Fused rocWMMA apply_heff (R2-4 alt) | **ABANDON** | LDS overflow 5.5× at chi=32, 22× at chi=64, 88× at chi=128. Live microbench: rocBLAS strided-batched 7.25 µs/call vs naive fused 22 µs at chi=32 (3× slower) and 161 µs at chi=64 (22× slower). Arithmetic intensity 22.8 FLOP/byte < MI300X ridge 30.7 — problem is memory-bound where fusion cannot help. 900 template specializations. 6–10 weeks not 1–2. |
| **09** | Mixed-precision SVD | PROMISING AS CONTINGENCY — but real win is `dgesvdj` drop-in | rocSOLVER `sgesvdj` is **QR + Jacobi on AᵀA** (squared condition number). On κ=1e6–1e8 matrices returns relative errors 2.5–4.3 — garbage for DMRG. Surprise: `rocsolver_dgesvdj` (pure FP64) is **2.6–3.8× faster than `dgesvd`** at n ≥ 512, no accuracy loss. 1-week ship, 30–50% full-sweep speedup at χ ≥ 256. |
| **10** | Cross-cutting integration risk | CUT R2-4, ADD R2-0 | R2-3 and R2-4 solve the same problem via incompatible mechanisms; in R2-4's window R2-1+R2-3 squeeze it to a 48-unit sliver. CBE at chi ≤ 32 reopens launch-overhead wound. Rescue AHC-DMRG skip gate from Research B §6 as 2-week fast path. Scope R2-1 as `-DCBE` flag on `dmrg2-gpu-opt`, not a new directory. Re-target R2-3 from two-site to single-site (chi ≤ 48 → ≤ ~80 at d=2). |

---

## 2. Refined round-3 follow-up set

The round-2 numbering is retired. New set:

### R3-F1 — Step-3 full strided-batched collapse  `[FAST-PATH, 1–2 DAYS]`

**Source:** Pair 6 rocprof trace of `dmrg2-gpu-opt/src/dmrg2_gpu_opt_impl.h:478-511`.

**Finding:** Current Step-3 batches one of three loop dimensions (D or d or d). Collapsing all three into a single `dgemm_strided_batched(batch = D·d·d = 20)` call eliminates ~19 kernel launches per Lanczos iter. This is pure refactor with no algorithmic change.

**Scope:** One file touched, ~50 LOC changed. Existing correctness tests (`test_dmrg2_gpu`) must pass unchanged.

**Expected:** **30–50% `apply_heff` speedup at chi ≤ 48**, independent of everything else. Ships in 1–2 days.

**Risk:** Trivial. Pointer-array setup is already GPU-side (A3). `hipError_t` checks exist.

---

### R3-F2 — `rocsolver_dgesvdj` drop-in  `[FAST-PATH, 1 WEEK]`

**Source:** Pair 9 live benchmark at `sandbox/pair09_mp_svd/`.

**Finding:** `rocsolver_dgesvdj` is 2.6–3.8× faster than `dgesvd` at n ≥ 512 with identical FP64 accuracy. At n=1024 on MI300X: 239 ms vs 2570 ms.

**Integration points:** All GPU SVD call sites:
- `gpu-rocm/dmrg2-gpu/src/dmrg2_gpu_impl.h:914`
- `gpu-rocm/dmrg-gpu/src/dmrg_gpu_impl.h` (SVD path)
- `gpu-rocm/pdmrg-gpu/src/` (SVD path — uses it in `V=Λ⁻¹` boundary path)

**Scope:** Three call sites, workspace query changes, ~150 LOC. Must preserve the energy delta < 1e-10 gate on all correctness targets.

**Expected:** **30–50% full-sweep wall-time reduction at χ ≥ 256**. Zero numerical risk — FP64 throughout.

**Risk:** rocSOLVER `dgesvdj` workspace query is a separate API; handle that correctly. Must benchmark vs CPU OpenBLAS 0.3.28 path (`--gpu-svd`) at chi ∈ {128, 256, 512} and confirm it wins before flipping default.

**NOT covered (explicitly rejected):** Any FP32 Jacobi or `sgesvdj` path. rocSOLVER's `sgesvdj` is QR + Jacobi on AᵀA and returns relative errors > 1 on DMRG-condition matrices. Mixed-precision SVD is only revived as a contingency if R2-1 CBE fails the Josephson correctness gate.

---

### R3-F3 — AHC-DMRG skip gate  `[FAST-PATH, 2 WEEKS]`

**Source:** rescued from Research B §6 by Pair 10.

**Finding:** Pair 10 identified that `round_2_plan.md` incorrectly folded AHC-DMRG (Adaptive Hot/Cold) into R2-1. It is independent: a `(Lanczos_residual < eps_res) ∧ (ΔF < eps_F) ∧ (prev_discarded_weight < eps_w)` gate that skips the 2-site SVD at bonds that are already converged, with a full-SVD flush every 5 sweeps for safety.

**Scope:** New gate logic in `dmrg2-gpu-opt` sweep loop. Per-bond state tracking (residual, ΔF, last discarded weight) added to `StreamWorkspace`. ~400 LOC.

**Expected:** **1.5–3× wall-clock at chi ≥ 128** (targets SVD dominance directly). Stacks with R3-F2: first-SVD-flush after 5 sweeps becomes a `dgesvdj` call.

**Risk:** Wrong gate thresholds can leak un-truncated bonds and degrade energy. **Must land with a regression suite checking energy delta < 1e-10 at L ∈ {16, 32, 64} Heisenberg and Josephson.**

---

### R3-1 — M/O CBE backbone (rescoped R2-1)  `[MAIN, 3–4 WEEKS]`

**Source:** Pairs 1, 2, 10.

**Critical correction from Pair 1:** What the round 2 plan called "CBE" and cited Gleis/Li/von Delft is actually the **McCulloch/Osborne 2024 simplification** (arXiv:2403.00562). These are **different algorithms**. The M/O variant:

- Still forms a two-site θ and calls `apply_heff_two_site` exactly **once per bond** (vs 20 matvecs for full Lanczos)
- Does **not** eliminate the two-site SVD (only the Lanczos loop)
- Is not "fully variational" — McCulloch/Osborne explicitly rebut Gleis et al.'s variational claim (Gleis concedes in 2501.12291)

The round-2 speedup target (2.5× vs two-site) remains correct (Pair 1 FLOP accounting: CBE 1.9e11 vs two-site 4.7e11 at L=64 χ=128).

**Scope refinement from Pair 10:** Implement as a `-DCBE` compile-time flag on `dmrg2-gpu-opt`, **not** a new directory. This avoids the `pdmrg/pdmrg-cotengra/pdmrg-opt` drift pattern and halves the eventual CUDA-port work.

**GPU call graph (Pair 2):**
```
apply_heff_with_CBE(bond b):
  1. rocblas_dgemm_strided_batched × 3  (Step 1 / Step 2 / Step 3 — same as today)
  2. Gaussian random block R: (d·χ, k+p=20) on device
  3. tangent projector M = (I − θ·θ†) · H · θ · R   (4 strided_batched GEMMs)
  4. rocsolver_dgeqrf on M  → Q  (second stream, hipEvent-gated)
  5. Augment bond: χ_new = χ + k, new single-site theta
  6. single-site Lanczos on augmented bond (reuses existing kernel)
  7. truncation SVD of B_{i+1} (reused from current path)
```

**StreamWorkspace:** +1.0 MB for tangent buffers (M_tan, Q, A_aug). Peak memory **55% of current dmrg2-gpu-opt** (single-site θ halves the dominant term).

**Hard prerequisites (non-negotiable):**
1. **CPU Python reference** at `cpu/cbe-reference/mo_cbe.py`, gated against quimb on L=16 Heisenberg + L=16 Josephson to energy delta < 1e-10. ~300 LOC, 1 week.
2. Use **double-pass classical Gram-Schmidt** in the projection step (Pair 1 numerical stability finding — single-pass loses orthogonality on long chains).
3. Oversampling `p = 10` (Halko/Martinsson/Tropp), not M/O's suggested `p = 5`, until numerics validated.

**Scope restriction:** **chi ≥ 64 only.** At chi ≤ 32, CBE's 3–4 extra kernel launches per bond harm the regime R2-3 is trying to close. The sweep must branch on bond chi and fall back to plain two-site `apply_heff` at chi ≤ 32.

**Fallback:** If CPU reference fails the Josephson gate (d=3 bosonic, d>4 regime has zero published CBE benchmarks), pivot to **Hubig/McCulloch 2015 DMRG3S** (arXiv:1501.05504) — ~200 LOC simpler, 10 years of validation, ~70% of the CBE speedup.

**Expected:** ~2× per-bond speedup at chi ≥ 64 on top of R3-F1 + R3-F2. Stacks multiplicatively with R3-F3 (the AHC skip gate).

---

### R3-2 — Persistent Lanczos single-site megakernel (rescoped R2-3)  `[MAIN, 4–5 WEEKS]`

**Source:** Pairs 5 (live empirical), 6 (MFMA integration), 10 (scope).

**Empirically validated envelope (Pair 5):** chi ≤ 48 at d=2 in a 3-tile LDS layout (theta + env_L + q_curr). chi = 64 hard-fails to compile (153% LDS). chi ≤ 16 if a full 20-vector Krylov basis must also fit in LDS.

**Scope refinement from Pair 10:** Target **single-site** `apply_heff` as in `dmrg-gpu`, not two-site. This extends the LDS envelope for d=2 from chi ≤ 48 to chi ≤ ~80 because the dominant term is `d·χ` rather than `d²·χ`.

**Critical integration with Pair 6:** The persistent kernel **cannot call rocBLAS** (host-only API), so the inner matvec must use hand-rolled `v_mfma_f64_16x16x4f64` via `rocwmma::mfma_sync`. Pair 6's validated microbench at `sandbox/pair06/pair06_mfma_bench.cpp` (6.7e-16 accuracy vs rocBLAS) is the prototype.

**Kernel structure:**
```
__global__ void persistent_lanczos_single_site(...)  // 1 workgroup per bond
  // LDS tiles: theta_in, theta_out, env_L_slice, q_curr, alpha/beta scalars
  // VGPR: MFMA accumulators (not LDS!)
  for k in 0..20:
    // 1. Matvec via inline MFMA (apply_heff)
    // 2. Orthogonalize q_curr vs q_{k-1} via MGS
    // 3. Compute alpha_k, beta_k on-chip
    // 4. Tridiag step (on-chip Thomas algorithm or single off-chip call)
  // Write result back to global
```

**Cooperative launch not needed** (Pair 5 definitive finding — `hipLaunchKernelGGL` per-workgroup is fine).

**Expected:** 8–12× Lanczos speedup at chi ≤ 48 vs current `dmrg-gpu` (which does ~8 rocBLAS launches per Lanczos iter). Directly targets the 93% CPU-win regime in `benchmarks/paper_results/mi300x/wins_cpu_vs_gpu.csv`.

**Hybrid fallback for chi ∈ [64, 256]:** rocBLAS matvec + fused housekeeping (orthogonalization, scalar updates). ~25% of the persistent speedup but covers the higher-chi band.

**Interactions with R3-1:** R3-1 CBE builds an *augmented* single-site theta, which this kernel is the perfect consumer for. Order: R3-2 can ship independently; R3-1 integration is a *feature addition* to R3-2's matvec.

**Stretch:** L2-retention hint experiment for "archival basis in L2" (the R2-3 design's unvalidated claim) — deferred to post-landing.

---

### R3-3 — Parallel CBE-DMRG (rescoped R2-2)  `[DEFERRED, CONTINGENT]`

**Source:** Pairs 3 (warmup killed), 4 (block-Jacobi risky).

**What dies from R2-2:**
- **Phase 0 parallel iTEBD warmup** — Pair 3 proved it's slower than the quimb DMRG1 warmup it would replace, the literature cites are wrong, and no published TEBD→DMRG hybrid workflow exists.

**What survives:**
- **Phase 2 block-Jacobi single-site CBE with replicated environments** — but with three mandatory fixes from Pair 4:
  1. **Re-add Stoudenmire's V=Λ⁻¹ boundary merge** (R2-2 dropped it — this is the only parallel-DMRG coupling scheme with an empirical track record).
  2. **Rollback-on-energy-increase** safety net.
  3. **CPU Python reference** at `cpu/cbe-reference/bj_cbe.py` gated against serial CBE on L ∈ {32, 64} Heisenberg + TFIM-critical + Josephson with P ∈ {2, 4, 8}. Speedup target: ≤ 1.8× serial sweep count.

**Schedule:** Do not touch this until R3-1 (M/O CBE backbone) has landed and the CBE CPU reference works. Then implement the block-Jacobi variant on top of the same Python reference and gate **on the Python reference passing** before any GPU code.

**Honest speedup expectation:** Pair 4 pessimistic analysis has P=2 as a net loss, P=4 at ~1.54× (barely above PDMRG's measured 1.22×). The 1.7× projection in R2-2 is aspirational.

**Measurement:** If the Python reference shows ≥ 1.5× over serial CBE at P=4 on all three models, graduate to GPU. Else abandon.

---

## 3. Ordered schedule

The new ordering reverses R2-2's "flagship first" framing in favor of cheap wins first, then algorithmic backbone, then (contingently) parallel scaling.

```
Week 1  ┌─ R3-F1  Step-3 batching collapse  (ship)
        ├─ R3-F2  rocsolver_dgesvdj drop-in  (start)
        └─ CBE Python reference  (start, cpu/cbe-reference/mo_cbe.py)

Week 2  ┌─ R3-F2  ship
        ├─ R3-F3  AHC-DMRG skip gate  (start)
        └─ CBE Python reference  (gate vs quimb)

Week 3  ┌─ R3-F3  ship
        ├─ R3-1   M/O CBE GPU backbone  (start in dmrg2-gpu-opt via -DCBE flag)
        └─ R3-2   persistent Lanczos single-site  (start in dmrg-gpu)

Week 4-5  R3-1 and R3-2 in parallel

Week 6-7  R3-1 CBE integration into R3-2 persistent kernel path

Week 8+   R3-3 Python block-Jacobi CBE reference (GATE)
           IF gate passes: schedule GPU block-Jacobi CBE
           ELSE: declare victory at R3-1 + R3-2 + R3-F* stack
```

**Total critical path:** ~7 weeks to an integrated CBE + persistent Lanczos + dgesvdj stack. That's the ship target.

---

## 4. Killed and why

Explicitly removed from the follow-up set:

| Item | Reason | Evidence |
|---|---|---|
| **R2-4 HIP graph capture of `apply_heff`** | Primary path fails: rocBLAS 5.2 on ROCm 7.2 intentionally refuses stream capture via `_rocblas_handle::is_stream_in_capture_mode()`. Alt fused rocWMMA path fails: LDS overflow 5.5–88× at chi ∈ {32, 64, 128}, rocBLAS is 3–22× faster than a naive fused kernel on the live VM, arithmetic intensity below MI300X ridge (memory bound). | Pair 7, Pair 8 |
| **R2-2 Phase 0 parallel iTEBD warmup** | Slower than the quimb DMRG1 warmup it would replace (14.5 s vs 8.8 s at L=64 χ=128 P=4). Citation error: arXiv:2212.09782 is real-time only. Zero published hybrid workflows. | Pair 3 |
| **FP32 Jacobi / `sgesvdj` SVD path** | rocSOLVER `sgesvdj` is QR + Jacobi on AᵀA (squared condition number). Returns relative errors 2.5–4.3 on DMRG-condition matrices. Same trap in cuSOLVER. | Pair 9 |
| **Standalone custom MFMA kernel for `apply_heff`** | rocBLAS already emits FP64 MFMA instructions on gfx942 (Tensile `MI16x16x4x1` verified in rocprof trace). Hand-rolled kernel is within 10% of rocBLAS GPU time. Launch overhead, not compute, is the floor. | Pair 6 |
| **Chi padding to multiples of 16** | rocBLAS is flat ~7 µs host time chi=8..48. Padding gains nothing. | Pair 6 |
| **Cooperative launch for persistent Lanczos** | Plain `hipLaunchKernelGGL` per-workgroup works. Cooperative is 10× slower per ROCm issue #3410. | Pair 5 |

---

## 5. Risk register (round 3)

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | CBE Python reference fails Josephson (d=3) gate | HIGH | Fallback: Hubig/McCulloch DMRG3S (arXiv:1501.05504), ~70% of CBE speedup, 10 years of validation |
| 2 | `rocsolver_dgesvdj` accuracy regression at specific shapes | MEDIUM | Correctness gate: ΔE < 1e-10 on all Heisenberg + Josephson targets before flipping default |
| 3 | Persistent Lanczos `v_mfma_f64` codegen break on ROCm 7.2 | MEDIUM | Pair 6's microbench already builds & runs on the live VM with 6.7e-16 accuracy — codegen is proven. Hybrid fallback at chi > 48. |
| 4 | R3-F3 AHC gate thresholds leak un-truncated bonds | MEDIUM | Energy-delta regression suite + forced full-SVD flush every 5 sweeps |
| 5 | Block-Jacobi CBE (R3-3) fails CPU reference gate | ACCEPTABLE | Don't ship it. R3-1 + R3-2 + R3-F* already covers the chi ∈ [64, 256] win target. |
| 6 | Compile-time `-DCBE` flag explosion (dual-mode `dmrg2-gpu-opt`) | LOW | Two build targets in CMakeLists: `dmrg2_gpu_opt` and `dmrg2_gpu_opt_cbe`. Shared source, test matrix runs both. |

---

## 6. Items explicitly NOT covered in round 3

- **Multi-state / excited-state DMRG** (still not touched)
- **rocSOLVER `gesvdj` parameter tuning beyond drop-in** (deferred)
- **Two-site persistent Lanczos** (rejected: Pair 10 showed single-site extends envelope)
- **cotengra contraction path reordering in `apply_heff`** (Pair 2 confirmed current 3-step order is optimal for the GEMM primitives; reordering only helps Python tensordot)
- **CUDA port of R3-F1 / R3-F2 / R3-1 / R3-2** (mechanical once ROCm side is green; scheduled as a separate round 4)
- **H100 benchmarking** (same — round 4)

---

## 7. Concrete deliverables checklist

### Documents (this round)
- [x] `round_3_pair01_cbe_algebra.md`
- [x] `round_3_pair02_cbe_gpu_kernel.md`
- [x] `round_3_pair03_cbe_tebd_warmup.md`
- [x] `round_3_pair04_block_jacobi_convergence.md`
- [x] `round_3_pair05_persistent_lanczos_empirical.md`
- [x] `round_3_pair06_mfma_fp64_small_chi.md`
- [x] `round_3_pair07_graph_capture_live.md`
- [x] `round_3_pair08_fused_rocwmma.md`
- [x] `round_3_pair09_mixed_precision_svd.md`
- [x] `round_3_pair10_integration_risk.md`
- [x] `round_3_plan.md` (this file)

### Implementation (pending — requires user authorization)
- [ ] **R3-F1**: refactor `gpu-rocm/dmrg2-gpu-opt/src/dmrg2_gpu_opt_impl.h:478-511` Step-3 to full strided-batched
- [ ] **R3-F2**: swap `rocsolver_dgesvd` → `rocsolver_dgesvdj` in three GPU SVD call sites
- [ ] **R3-F3**: AHC skip gate in `dmrg2-gpu-opt` sweep loop + per-bond state in `StreamWorkspace`
- [ ] **CBE Python reference**: `cpu/cbe-reference/mo_cbe.py` + pytest vs quimb on L=16 Heisenberg & Josephson
- [ ] **R3-1**: `-DCBE` compile flag on `dmrg2-gpu-opt` with M/O CBE apply_heff path
- [ ] **R3-2**: persistent Lanczos single-site megakernel in `dmrg-gpu` (not a new directory)
- [ ] **R3-3 Python block-Jacobi reference**: `cpu/cbe-reference/bj_cbe.py` (gated on R3-1 landing)

### Artifacts from round 3 empirical work
- `sandbox/pair05/bench_persistent_lanczos.hip` — persistent Lanczos microbench (on remote, compiles & runs)
- `sandbox/pair06/pair06_mfma_bench.cpp` — rocWMMA FP64 MFMA microbench (on remote, 6.7e-16 accuracy vs rocBLAS)
- `sandbox/bench_graph_capture.hip` / `bench_relaxed.hip` — HIP graph capture go/no-go (on remote, definitive NO-GO result)
- `sandbox/pair08/bench_fused_step1.cpp` — fused apply_heff sanity bench (on remote, confirmed slower than rocBLAS)
- `sandbox/pair09_mp_svd/pair09_bench_svd.hip` — rocSOLVER SVD benchmark (on remote, `dgesvdj` 2.6–3.8× win validated)

All sandbox binaries are re-runnable on `hotaisle@23.183.40.84` — reproduction requires no re-clone.

---

## 8. Citations (round 3 additions)

- Gleis, Li, von Delft 2023 — "Controlled bond expansion for DMRG ground state search at single-site costs," arXiv:2207.14712, PRL 130, 246402
- McCulloch, Osborne 2024 — DMRG with TPU CBE simplification, arXiv:2403.00562
- Gleis et al. 2025 reply — arXiv:2501.12291 (variational-claim concession)
- Hubig, McCulloch, Schollwöck, Wolf 2015 — DMRG3S single-site + subspace expansion, arXiv:1501.05504
- Grigori, Hasan 2025 — A2DMRG additive-Schwarz coarse correction, arXiv:2505.23429
- Unfried, Hauschild, Pollmann 2022 — real-time QR-TEBD (NOT imaginary-time, NOT a Krumnow/Eisert paper), arXiv:2212.09782
- Halko, Martinsson, Tropp 2011 — oversampling p = 10 for randomized SVD
- AMD rocBLAS Beta Features — stream capture support table (2026-04-10 snapshot)
- ROCm issue #3410 — cooperative launch performance regression (Pair 5 reference)

---

## 9. Next step (awaiting user authorization)

The natural first move is **R3-F1 (Step-3 full batching)** — it's one file, one afternoon, measurably improves `dmrg2-gpu-opt` at chi ≤ 48, and all the empirical infrastructure (live VM, benchmark grid, regression tests) is already in place. It also de-risks the other round-3 items by validating the fast iteration loop.

The second is **R3-F2 (dgesvdj drop-in)** — one week, 30–50% at χ ≥ 256, stacks with everything else.

Ship these two before touching R3-1 or R3-2.
