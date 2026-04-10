# Round 3, Pair 02 — CBE GPU kernel pipeline (refinement of R2-1)

**Date:** 2026-04-10
**Scope:** Concrete GPU call graph, buffer plan, microbench spec and HIP-graph
compatibility matrix for Controlled Bond Expansion (McCulloch/Osborne 2024
variant) on MI300X, built on top of the existing `dmrg2-gpu-opt` kernels.
**Parent:** `round_2_plan.md §2.1` (R2-1 CBE-DMRG GPU backbone).
**Author role:** GPU engineer + adversarial reviewer (same agent).

---

## 0. Notation used throughout

Let `site` be the currently optimized site and `bond = site` the bond to its
right (the one whose rank CBE is asked to expand). Write:

| Symbol | Meaning | Typical values |
|---|---|---|
| `d` | physical dimension | `2` (Heisenberg / TFIM), `3`–`5` (Josephson) |
| `D` = `D_mpo` | MPO bond dimension | `4`–`5` |
| `cL` | left bond dim = `bond_dims_[site]` | `32`–`256` |
| `cR` | right bond dim = `bond_dims_[site+1]` | `32`–`256` |
| `k` | CBE expansion rank | `16`–`32` |
| `kkeep` | # expansion directions kept after truncation | `k` itself |
| `s` | scalar size in bytes | `8` (double) / `16` (complex) |

Single-site `theta` tensor on device: shape `(cL, d, cR)`, row-major, dense,
size `cL*d*cR*s` bytes. This is identical to `dmrg-gpu` (the one-site
implementation) and **half** the storage of `dmrg2-gpu-opt`'s two-site theta
`(cL, d, d, cR)`.

---

## 1. Call graph of `apply_heff_with_CBE`

This is what Lanczos calls every iteration for the *expansion-and-eigen* step
of the CBE sweep. The McCulloch/Osborne 2024 variant (arXiv:2403.00562) is
logically three pieces: **(A)** a cheap, capture-friendly single-site
`apply_heff` that produces `H·psi` on the current basis; **(B)** a
**capture-friendly** expansion-direction projection that builds a small
tangent-space matrix `M ∈ R^{(d·cL)×(D·k)}` once per bond, not once per
Lanczos iteration; **(C)** a **deferred, out-of-capture** QR and
bond-augmentation stage that runs exactly once per bond between the eigensolve
and the shift to the next site.

The crucial architectural decision — the one that makes CBE safe under HIP
graphs at all — is: **the expansion / QR / augmentation machinery lives
outside `apply_heff`.** `apply_heff` stays a shape-stable, sync-free,
rocBLAS-strided-batched pipeline identical in structure to today's
`dmrg-gpu::apply_heff` (single-site). CBE's work is bolted on between the
eigensolve and `update_right_env`, not between Lanczos iterations.

### 1.1 Per-bond outer call graph

```
OPTIMIZE_BOND(site):
    // --- Part 1: classical single-site Lanczos, graph-captured -----------
    build_or_lookup_graph(site)            // host-only, O(µs)
    hipGraphLaunch(lanczos_graph)          // entire Lanczos loop replay
                                           //   *including* dot/nrm2 via
                                           //   device-pointer mode (A1)
    // The launched graph is equivalent to the old Lanczos loop:
    //   for j in 0..m-1:
    //     apply_heff_single_site(theta_j → Htheta_j)   // 3 GEMMs
    //     device-pointer-mode dot/nrm2/axpy/scal       // A1 scalars
    //     alpha_j, beta_j written to device scratch
    //   (tridiagonal coefficients read back once at end)

    // --- Part 2: tridiagonal solve on HOST (single sync) -----------------
    hipStreamSynchronize(stream)           // <-- the ONE sync per bond
    dstev(alphas, betas) → lowest_evec     // LAPACK, ~20 µs
    hipMemcpyAsync(d_evec_coeff, h_evec, m*s, H2D, stream)
    recombine_theta_kernel(V_krylov, d_evec_coeff, theta_new)
                                           // GPU gemv, capture-safe but
                                           // here we launch it eagerly

    // --- Part 3: CBE expansion, capture-friendly block -------------------
    // All of this is pre-compiled into a second HIP graph,
    // shape-keyed on (cL, cR, d, D, k):
    hipGraphLaunch(cbe_expand_graph)
    //   Step E1: build tangent matrix M  (one strided-batched GEMM)
    //     M[i,j] := (I - psi psi^T) · H · psi  projected onto non-psi dirs
    //     More precisely: M_{(s,a),(w,l)} = sum_b L_env[w,a,b]·theta_new[b,s',?]·...
    //     Practically: reuse T1_single from apply_heff's Step 1 as
    //     "H·psi without the right-env contraction", rearrange to an
    //     (d·cL) × (D·k_seed) matrix by contracting against a fixed
    //     random seed block R_seed.
    //   Step E2: Rescale / orthogonalize against current left isometry
    //     (one dense GEMM: A_L^H · M, one axpy to subtract, same gemm back).
    //   Step E3: store M in d_cbe_M_ at this bond.

    // --- Part 4: QR + augmentation (NOT graph-captured) ------------------
    rocsolver_dgeqrf(d_cbe_M_, lda=d*cL, M=d*cL, N=D*k,
                     d_tau_, &info)        // one launch
    rocsolver_dormqr(..., d_cbe_M_, d_Q_block_)   // materialize Q
    extract_first_kkeep_cols(d_Q_block_, d_expand_cols_)  // custom kernel
    augment_mps_column_kernel(d_mps_tensors_[site], d_expand_cols_,
                              d_mps_tensors_aug_[site])  // copy + zero-pad
    update_right_env_augmented(site+1)     // reuses dmrg2-gpu-opt path

    // --- Part 5: shift basis, ready for next site -----------------------
    bond_dims_[site+1] += kkeep
```

### 1.2 Per-Lanczos-iteration inner call graph (what gets captured)

All of this is **stream-capture-safe** provided the rocBLAS calls are
`strided_batched` (Research A finding) and all reductions use
**device pointer mode** (A1). This is the *only* part of CBE that runs at
Lanczos-iteration frequency.

```
apply_heff_single_site(theta_in → theta_out):    // ONE HIP graph, one launch
    // --- Step 1: batched L_env · theta, D*d independent GEMMs -------------
    hipLaunchKernel setup_heff_A_ptrs      (device-side ptr setup, A3)
    hipLaunchKernel setup_heff_B_ptrs      (device-side ptr setup, A3)
    hipLaunchKernel setup_heff_C_ptrs      (device-side ptr setup, A3)
    rocblas_dgemm_strided_batched          // <-- MUST be strided, not ptr-array
        op=(T, N), m=cL, n=cR, k=cL,
        A=L_env, strideA=cL*cL,   batch=D*d
        B=theta,  strideB=0,      (reuse theta across D*d = bad — see §5.4)
        C=T1,     strideC=cL*cR
    // --- Step 2: dense mid GEMM ----------------------------------------
    rocblas_dgemm
        m=cL*cR, n=d*D, k=D*d,
        A=T1, B=W_mat, C=T2
    // --- Step 3: batched T2 · R_env, D*d independent GEMMs ---------------
    rocblas_dgemm_strided_batched          // already done this way today
        op=(N, N), m=cL, n=cR, k=cR,
        A=T2, B=R_env, C=theta_out, batch=D*d

    // No dot / nrm2 / axpy in apply_heff itself — those live in Lanczos wrapper
```

Sync count inside the graph: **zero**. Host interaction during replay:
**zero**. Workspace pointers: **fixed** at `StreamWorkspace` allocation time,
so `hipGraphExecKernelNodeSetParams` is never called.

### 1.3 Total per-bond sync count

| Phase | Host syncs | Notes |
|---|---|---|
| apply_heff graph replay (×m Lanczos iters) | 0 | Device-pointer mode everywhere |
| Tridiagonal solve | **1** | `dstev` is host; this is the only true sync |
| CBE expansion graph replay | 0 | Same capture story as apply_heff |
| rocSOLVER `dgeqrf` | **1–2** | See §5.4 — rocSOLVER *does* internally sync for workspace query and info check |
| `update_right_env_augmented` | 0 | Same as dmrg2-gpu-opt today |

**Net: 2–3 host syncs per bond**, versus **1** in today's dmrg2-gpu-opt (only
the SVD path). The QR-induced sync is the one we have to kill in Round 4.
See §5.4 for the mitigation.

---

## 2. `StreamWorkspace` buffer allocation and peak memory

### 2.1 New buffers added for CBE

| Buffer | Lifetime | Size (bytes) | Purpose |
|---|---|---|---|
| `d_theta_single_` | per-stream | `max(cL*d*cR)·s` | single-site theta |
| `d_T1_` | per-stream | `D·d·max(cL*cR)·s` | Step 1 output (reused from dmrg-gpu) |
| `d_T2_` | per-stream | `D·d·max(cL*cR)·s` | Step 2 output (reused from dmrg-gpu) |
| `d_krylov_V_` | per-stream | `m·cL·d·cR·s` | Krylov basis, `m=20` |
| `d_cbe_M_` | per-stream | `(d·cL)·(D·k)·s` | tangent-space expansion matrix |
| `d_cbe_R_seed_` | persistent | `(cR)·(D·k)·s` | fixed random seed block (RSVD) |
| `d_tau_` | per-stream | `min(d·cL, D·k)·s` | geqrf Householder scalars |
| `d_qr_workspace_` | per-stream | `query_size(rocsolver_dgeqrf)` | see §5.4 — NOT known until first call |
| `d_Q_block_` | per-stream | `(d·cL)·(D·k)·s` | materialized Q from geqrf |
| `d_expand_cols_` | per-stream | `cL·d·kkeep·s` | first-kkeep Q columns, reshaped |
| `d_mps_tensors_aug_[site]` | per-site | `cL·d·(cR+kkeep)·s` | temporary augmented site tensor |
| `d_cbe_evec_coeff_` | per-stream | `m·s` | tridiag eigenvector |
| `d_const_one/zero` | persistent | `16` | device scalars for pointer-mode BLAS (already present) |

**Reused from dmrg-gpu** (not new): `d_L_envs_[]`, `d_R_envs_[]`, `d_W_left_[]`,
`d_W_right_[]`, `d_batch_A/B/C_` (although the last three become unnecessary
once all apply_heff paths move to strided_batched — see §5.2).

### 2.2 Peak per-stream workspace (L=64, D=5, d=2, k=32, m=20 Krylov)

Numbers from `python3 -c ...` arithmetic, in **MB**:

| chi | theta | T1+T2 | Krylov | M_tan | Q_cbe | Aaug | **TOTAL per-stream** | dmrg2-gpu-opt (for comparison) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 32  | 0.02 | 0.16 | 0.3  | 0.08 | 0.08 | 0.03 | **0.7 MB** | 1.0 MB |
| 64  | 0.07 | 0.66 | 1.3  | 0.16 | 0.16 | 0.10 | **2.5 MB** | 4.1 MB |
| 128 | 0.26 | 2.62 | 5.2  | 0.33 | 0.33 | 0.33 | **9.1 MB** | 16.3 MB |
| 256 | 1.05 | 10.49 | 21.0 | 0.66 | 0.66 | 1.18 | **35.0 MB** | 65.0 MB |

**CBE peak workspace is ~55 % of dmrg2-gpu-opt** at every chi we care about.
The savings come from single-site theta being half the size, and T1/T2 being
half because their `d²` factor drops to `d`. The CBE-specific overheads
(`M_tan + Q + Aaug`) add back less than 2 MB even at chi=256.

Chain-wide environment buffers scale identically for both implementations
(`2·L·D·chi·chi·s`); at L=64, chi=256 this is ~336 MB and is the dominant
allocator, independent of whether we use CBE or two-site.

**Verdict on memory**: CBE does NOT double the memory. It **halves** the
per-stream workspace and leaves the chain-wide env allocation untouched. No
configs are killed on memory grounds, including the 4-GPU PDMRG layout at
L=128 chi=256 (which already fits in dmrg2-gpu-opt's budget and we now take
half of that).

### 2.3 PDMRG multi-stream consideration

In `pdmrg-gpu-opt` each segment owns a `StreamWorkspace`. At `n_segments=4`
the peak is `4 × 35 MB = 140 MB`. Trivially within the 192 GB of an MI300X.
The expansion buffers `d_cbe_R_seed_` and the per-bond `d_cbe_M_` storage are
per-workspace, so there is **no shared state** across segments. This matters
for Phase 4 (PDMRG integration) — CBE does not introduce any new
inter-segment coupling beyond what dmrg2-gpu-opt already has.

---

## 3. Fast-path validation microbench (what to build first)

The single highest-risk claim in the CBE plan is: **"the expansion projection
`M = P⊥ · H · theta` is cheap."** This is the thing we have to validate
before touching any of the Lanczos, QR or augmentation code. Everything else
is classical single-site DMRG.

### 3.1 Build target: `bench_cbe_projection.cpp`

Drop in `gpu-rocm/sandbox/bench_cbe_projection.cpp`. No dependency on any
DMRG class. Pure rocBLAS + a small custom projection kernel.

```cpp
// Inputs: dummy L_env (D,cL,cL), theta (cL,d,cR), R_seed (cR,D,k), W (D,d,d,D)
// Outputs: M = apply_heff(theta) restricted via R_seed:
//   H_theta[w,a,s,b] = sum_{c,s',n,e} L[w,a,c] W[w,s,s',n] theta[c,s',e] R[e,n,b]
//   M[(s,a),(w,l)]  = sum_b H_theta[w,a,s,b] · R_seed[b,w,l]
//
// Pipeline (four GEMM calls, all strided_batched):
//   T1 := L_env ·_(a=c) theta      strided over (w,s,s') [standard Step 1]
//   T2 := T1    ·_(W)    W          single dense GEMM      [standard Step 2]
//   T3 := T2    ·_(b)    R_seed     batched over (w,s)    [new]
//   M  := reshape(T3) into (d*cL, D*k)

// Runs: warm up 3, time 100 iters, compare against a plain
//       apply_heff_two_site at the same (cL, cR, d, D) that already exists
//       in dmrg2-gpu-opt.
```

### 3.2 Expected runtime (MI300X, `~10 TF/s` sustained with rocBLAS overhead)

FLOP counts from `python3`:

| chi | single-site matvec | two-site matvec | CBE extras (projection + QR) |
|:---:|:---:|:---:|:---:|
| 32  | 0.001 G   | 0.003 G   | 0.002 G |
| 64  | 0.011 G   | 0.024 G   | 0.007 G |
| 128 | 0.087 G   | 0.181 G   | 0.020 G |
| 256 | 0.684 G   | 1.395 G   | 0.065 G |

At 10 TF/s sustained:

| chi | 1-site matvec | 2-site matvec | CBE projection+QR | **Expected speedup per matvec** |
|:---:|:---:|:---:|:---:|:---:|
| 32  | 0.15 ms | 0.34 ms | 0.12 ms | 2.26 × (dispatch-limited; CBE eats the savings) |
| 64  | 1.13 ms | 2.42 ms | 0.64 ms | 2.14 × |
| 128 | 8.72 ms | 18.09 ms | 2.09 ms | **2.07 ×** |
| 256 | 68.4 ms | 139.5 ms | 6.5 ms  | **2.04 ×** |

The matvec alone is the **top line we can see** — CBE's expected wall-clock
win over dmrg2-gpu-opt is **≥ 2 × per matvec at chi ≥ 64**. Minus the once-per-
bond QR (amortized over `m` Lanczos iters = 20 × cheaper), the effective
steady-state should stay above 1.9 × at chi ≥ 128.

**Microbench success criterion**: the four-step CBE-projection pipeline at
chi=128 must run in **< 11 ms** on MI300X (loosely: 2× the 1-site apply_heff
time measured today in `dmrg-gpu` ≈ 5 ms), and must produce an `M` matrix
whose left kernel matches the projected H·psi computed via a reference
`numpy.tensordot` call on a small test case.

**Microbench failure modes to catch:**
- Launch-overhead dominated at chi ≤ 64 (CBE wins nothing if the 4-GEMM
  pipeline doesn't go through HIP graph capture). Gate: if un-captured
  runtime at chi=64 > 2 ms, immediately build the graph-capture version (§4).
- Projection correctness against CPU reference: **bit-exact within 1e-11**.
  If we miss this, either the reshape is wrong or the R_seed is not on the
  right axis.
- Workspace query for `rocsolver_dgeqrf` returns > 64 KB. If so, pre-allocate
  at chi=512 worst case (~4 MB) and never call the query again.

### 3.3 What we do NOT build in the microbench

- Lanczos wrapper (already exists in `dmrg-gpu`)
- Full `augment_mps_column_kernel` (copy + zero-pad is trivial)
- The `update_right_env_augmented` path (identical to dmrg2-gpu-opt's env
  update after the bond dim has grown)

The microbench is *purely* the four-GEMM projection and the QR. Nothing else.

---

## 4. HIP graph capture compatibility matrix

| CBE sub-call | Capture-safe? | Evidence | If no, fallback |
|---|:---:|---|---|
| `apply_heff` Step 1 `rocblas_dgemm_strided_batched` | **YES** | AMD beta-features doc lists only `dot/asum/nrm2` (level-1, host-ptr) and **device-ptr-mode level-3** as unsupported. Strided-batched level-3 in **host-ptr mode** is fine. R2-4 Phase-0 microbench will confirm. | rocWMMA fused kernel (Proposal 3-alt) |
| `apply_heff` Step 2 `rocblas_dgemm` | YES (host-ptr) | Same source; level-3 host-ptr is supported | same |
| `apply_heff` Step 3 `rocblas_dgemm_strided_batched` | YES | same | same |
| Device-pointer-mode `dot/nrm2/axpy/scal` (Lanczos) | **NO** | rocBLAS beta-features: "BLAS Level-3 and BLAS-EX functions in pointer mode device do not support HIP Graph" — and also Level-1 reductions in pointer mode host are explicitly unsupported | **Lanczos outer loop is NOT captured** in R2-1 (same scoping decision as R2-4). Only `apply_heff` itself is captured. |
| CBE expansion `rocblas_dgemm_strided_batched` (T3 := T2·R_seed) | YES | same as Step 1/3 | same |
| Custom `augment_mps_column_kernel` | YES | Pure HIP kernel launch, no sync | — |
| `rocsolver_dgeqrf` | **PROBABLY NO** | rocSOLVER docs do not mention stream capture at all. rocSOLVER routines internally call rocBLAS *and* (worse) have an `info` output which in many implementations forces a host-visible write. GitHub search shows no examples of `rocsolver_*` under `hipStreamBeginCapture`. | **Run QR out-of-capture** — it is once-per-bond, not once-per-matvec, so we can afford a separate stream launch. |
| `rocsolver_dormqr` | PROBABLY NO | same | same |
| `recombine_theta_kernel` | YES | custom HIP kernel | — |

### 4.1 Capture scope decision

The only things we actually try to put inside a HIP graph are:
1. **`apply_heff_single_site`** — 4 rocBLAS calls + 3 pointer-setup kernels,
   same structure as dmrg2-gpu-opt's two-site apply_heff under Proposal 3.
2. **CBE expansion projection** — 4 rocBLAS calls + 1 reshape kernel, built as
   a *separate* graph, launched once per bond (not once per Lanczos iter).

Neither of these uses any of the "unsupported" rocBLAS functions. Crucially,
**we do NOT put the Lanczos outer loop inside the graph**, because the
outer loop needs device-pointer-mode `dot/nrm2` (A1), which AMD explicitly
lists as capture-unsafe. This matches R2-4's scoping decision word-for-word.

### 4.2 Per-segment graph cache sizing (PDMRG)

PDMRG `n_segments=4`, sites per segment ~16. Bond shapes grow geometrically
from `(1, 2)` to `(chi/2, chi)` and back. Unique `(cL, cR)` shapes per
segment: ≤ 10 growing + 10 shrinking = 20. Plus 2 for each of the 4 segments
= 80 distinct apply_heff graphs. Plus 80 CBE-expansion graphs at the same
shapes. **Total 160 graphs**, each ~2–4 KB of metadata → ~500 KB per
`StreamWorkspace`. Capture cost at ~500 µs/graph × 160 = 80 ms one-time, only
on first sweep. Amortizes across ≥ 3 sweeps and is negligible against the
multi-second-per-sweep steady state.

---

## 5. Adversarial review findings

### 5.1 Hidden CPU sync audit

| Candidate hidden sync | Status |
|---|---|
| `rocblas_dgemm_batched` host-ptr-array upload | **Eliminated** — the design forbids `_batched` and uses `_strided_batched` throughout. Same reason as R2-4. |
| Per-Lanczos-iter `h_alpha`/`h_beta` writeback | **Eliminated** by device-pointer mode (A1). The tridiagonal coefficients land in `d_alpha_dev`/`d_beta_dev` and are only copied back to host **once** at the end of the bond. |
| `rocsolver_dgeqrf` internal sync | **NOT eliminated.** rocSOLVER returns an `info` status that in practice requires a host-visible write. This is the sync I called out in §1.3 and §4. Mitigation: fire-and-forget the QR on a **second** stream, then `hipEventRecord` + `hipStreamWaitEvent` onto the compute stream before the augmentation kernel. One stream event, no host sync. If that still doesn't work, fall back to a custom 16-iter modified-Gram-Schmidt kernel on LDS (see §5.4 below). |
| Pinned memory for `M` upload to host | **Not applicable** — `M` never leaves the GPU. Only `evec_coeff` crosses back after the tridiagonal solve, and that is a one-time 160-byte copy on the `hipMemcpyAsync` call after `dstev`. Well within the §5.3 pinned-memory race envelope because the host side of `alphas/betas/evec_coeff` is written once per bond and read only once. |
| `rocsolver_dgesvd` lurking in an "optional fallback" path | **Explicitly forbidden.** The R2-1 plan says to NOT call rocSOLVER SVD for CBE. I am repeating it here because that is how these bugs come back. The only rocSOLVER call is `dgeqrf` (and possibly `dormqr`). No `dgesvd`, no `dgesdd`. |

**Verdict:** The only sync I cannot eliminate in the pipeline as currently
designed is the one inside `rocsolver_dgeqrf`. It is once per bond (not once
per Lanczos iter), but it is real.

### 5.2 Does the buffer plan double memory?

No. Per §2.2, CBE uses ~55 % of dmrg2-gpu-opt's per-stream workspace at every
chi we care about. Chain-wide storage (envs, MPS tensors, W matrices) is
unchanged. **No configs are killed.**

The one place we grow storage is the transient `d_mps_tensors_aug_[site]`
while augmentation happens — it is `cL·d·(cR + kkeep)·s` bytes, at chi=256
kkeep=32 that is 1.18 MB. Peanuts.

### 5.3 Does QR serialize at chi ∈ [64, 256]?

The tangent matrix has shape `(d·cL) × (D·k) = (2·cL) × (5·32) = (2·cL)×160`.
At:

- chi=64: `128 × 160` — very tall-skinny. `geqrf` cost `2·m·n² − (2/3)·n³ ≈
  6.5 MFLOPs`. On MI300X this is ~100 µs of actual work, almost
  pure dispatch latency. **Not a serialization problem per se, but it IS
  the dispatch-latency regime.**
- chi=128: `256 × 160`. `~13 MFLOPs`, ~200 µs.
- chi=256: `512 × 160`. `~26 MFLOPs`, ~400 µs.

None of these are "serializing" in the FLOP sense — they are "fixed-cost 200
µs launches that happen once per bond". Across one sweep (126 bonds at
L=64) that is ~25 ms per sweep from QR alone. Compared to the ~1–2 seconds
of CBE apply_heff work per sweep at chi=128, the QR is **~1.5 %** overhead.
Acceptable.

**However**: at chi ≤ 32, where the rest of the pipeline is itself only
0.15 ms/matvec, the QR's 50-100 µs overhead becomes a measurable fraction.
See §5.5 for the conclusion on chi ≤ 32.

### 5.4 rocSOLVER `dgeqrf` usability

Three concerns, ranked:

1. **Workspace query requires a call.** `rocsolver_dgeqrf` uses internal
   workspace that the user must allocate; the query is a separate API call
   and may itself sync. Mitigation: at `StreamWorkspace` construction time,
   query once at worst-case `(d·cL_max, D·k)` and never query again. This
   moves the sync out of the hot path.
2. **Stream capture compatibility is undocumented.** rocSOLVER docs do not
   mention HIP graphs *at all*. Our working assumption: not safe. We run
   geqrf on a separate stream and use `hipEventRecord` + `hipStreamWaitEvent`
   to chain it into the main stream. Total extra latency: ~10 µs (event
   overhead). **Acceptable because it is amortized over `m` Lanczos iters.**
3. **Custom QR for chi ≤ 64?** A Householder QR on `128 × 160` does ~6.5 M
   flops in `160` sequential Householder reflections. On an MI300X CU with
   LDS-resident panel (8 KB per column slice), we could match this in an
   LDS-persistent kernel similar to R2-3's Lanczos design. Crossover
   estimate: when `cL ≤ 64`, `m_geqrf = 2·cL ≤ 128`, which fits in 64 KB LDS
   (2 KB per column × 160 columns = 320 KB does NOT fit, hmm).

   **Revised crossover**: a custom LDS QR is only viable if we chunk it into
   **blocked Householder** with panel width `b = 16`, so the LDS stores a
   `(2·cL) × 16` panel at once (at `cL=64` that is 16 KB — fine). Then we do
   10 panel passes, each a sequential column loop. Total work per panel ≈
   `2·16·2·cL·16 = 65 KB flops`. Ten panels = 650 KB flops. On a single CU at
   ~1 TF/s that is ~1 µs. **So the custom LDS QR wins from chi ≥ 16
   upward**, and is strictly better than rocSOLVER up to the point where
   `(2·cL) × 160` no longer fits in the panel scheme (chi ≥ 512).

   **Recommendation:** start with `rocsolver_dgeqrf` on a separate stream
   for correctness. Write the custom LDS blocked-Householder QR as a Phase-2
   optimization. It is essentially the same engineering effort as R2-3's
   persistent Lanczos and the code can share the `composable_kernel` tile
   infrastructure.

### 5.5 Does CBE help at chi ≤ 32, where CPU currently wins?

**No. It hurts there.** Arithmetic:

- At chi=32, CBE's extra work per bond (projection + QR + augmentation) is
  about `0.12 ms + 0.05 ms + 0.01 ms ≈ 0.18 ms`.
- Classical single-site matvec at chi=32 is `0.15 ms`.
- So CBE roughly **doubles the per-bond time at chi=32**, while only
  converging slightly faster per sweep.

The whole `chi ≤ 50` regime is dispatch-latency-bound, and CBE **adds**
three per-bond rocBLAS calls plus a QR plus an augmentation kernel.
Single-core CPU AHC-DMRG (or just classical 1-site DMRG with White
perturbation) will win decisively here.

**Explicit scope restriction for R2-1**: CBE is advertised as a **chi ≥ 64**
optimization. For chi ≤ 48 we fall back to `dmrg-gpu` single-site (the
existing implementation), and for chi ≤ 32 we fall back to single-core CPU
(either scipy or the eventual R2-3 persistent Lanczos kernel). We do **not**
claim any chi ≤ 50 CPU-win config flips from CBE. Any such claims must be
stricken from the R2-1 description in `round_2_plan.md §2.1`.

### 5.6 Variational correctness gate

CBE's variational correctness is proven only for the case where the
expansion is done with an **isometric** `Q` block. Our plan uses rocSOLVER's
`dgeqrf` → `dormqr` to materialize `Q` from the Householder reflectors. The
resulting `Q` is orthonormal to floating-point precision (~1e-14 on the
identity test). Good.

**But**: if we later switch to a rank-revealing QR with column pivoting
(`rocsolver_dgeqp3`), the variational property still holds *only if* we keep
all of the `Q` columns, not just the first `kkeep`. Dropping pivoted
columns is equivalent to rank truncation and is **not** variational. The
McCulloch/Osborne paper is explicit about this.

**Action**: in the R2-1 CPU reference (`cpu/cbe-reference/`), add a test
that disables the column-pivot path entirely. The GPU port must NOT use
`dgeqp3` and must mirror that restriction.

### 5.7 What this refinement adds to R2-1 that was not in round_2_plan.md

- Call-graph-level placement of the QR outside the capturable block (§1.3)
- Concrete `StreamWorkspace` buffer list and per-chi peak memory (§2.2)
- Actual FLOP counts per pipeline phase (§3.2)
- Rejection of `rocsolver_dgeqp3` on variational grounds (§5.6)
- Explicit scope restriction out of the chi ≤ 48 regime (§5.5)
- Microbench specification that can be written this week, independent of
  the Lanczos or augmentation code (§3)

---

## 6. VERDICT

**NEEDS REFINEMENT.**

The plan is implementable and the memory / FLOP budgets check out, but three
concrete blockers must be resolved before R2-1 implementation can start in
earnest:

### 6.1 Blockers

1. **[BLOCKER-1] Validate strided-batched capture first.** The entire
   apply_heff capture story piggybacks on R2-4's Phase-0 `bench_graph_capture.cpp`
   microbench (rocBLAS strided_batched under `hipStreamBeginCapture`).
   If this fails, we pivot to rocWMMA fused kernels for **both** R2-1 and
   R2-4 — which is fine but doubles the LOC in R2-1. R2-1 cannot proceed
   without this answer. It is a **shared gate** with R2-4.

2. **[BLOCKER-2] CPU CBE reference must exist before any GPU port.** The
   `cpu/cbe-reference/` Python implementation against TeNPy is stated as a
   prerequisite in `round_2_plan.md §2.1` but is not yet built. Without it,
   there is no regression gate for the GPU port and no way to validate the
   tangent-space projection formulas. This can be done locally, no GPU
   required, in ~2 days.

3. **[BLOCKER-3] rocSOLVER `dgeqrf` workspace query behavior under
   pre-allocated workspace.** Confirm on the remote MI300X that the pattern
   "query once at allocation, never query again" actually works with
   `rocsolver_set_workspace`-style pre-allocation — rocSOLVER does NOT have a
   `rocsolver_set_workspace` equivalent to rocBLAS's, so the workspace path
   may need a manual pre-allocation via `hipMalloc` of the queried size. One
   small bench program confirms this in ~30 minutes of remote time.

### 6.2 Non-blocking concerns (track, don't gate)

- Custom LDS blocked-Householder QR is attractive but not on the critical
  path. Schedule for Phase 2 alongside R2-3.
- CBE at chi ≤ 48 is a non-goal. Document this in the R2-1 implementation's
  README and skip all benchmarks in that regime.
- Column-pivoted QR is banned for variational reasons (§5.6); write this
  into the code as a comment next to the `rocsolver_dgeqrf` call.

### 6.3 Memory and FLOP verdict

**Memory**: READY. CBE is ~55 % of dmrg2-gpu-opt at every chi. No configs
killed.

**FLOPs**: READY. CBE's matvec is ~2× cheaper than the two-site matvec from
chi=64 up, and the per-bond CBE overhead (projection + QR + augment) is
<10 % of one Lanczos matvec at chi ≥ 128.

**Graph capture**: CONDITIONAL on R2-4 Phase-0 microbench.

**Sync count**: READY at 2–3 host syncs per bond, down from the ~100 sync
stalls we had before A1. Same scale as dmrg2-gpu-opt today.

---

## 7. Concrete next action

**This week:**

1. **Ship R2-4 Phase-0 microbench** (shared gate with R2-1). File:
   `gpu-rocm/sandbox/bench_graph_capture.cpp`. Outcome determines whether
   R2-1 uses rocBLAS strided_batched or pivots to rocWMMA fused kernels. One
   remote-MI300X day of work.

2. **Write the CPU CBE reference**. File: `cpu/cbe-reference/cbe_dmrg.py`.
   ~400 LOC of numpy, modeled on the McCulloch/Osborne 2024 pseudocode.
   Validate against TeNPy's DMRG2 ground-state energies on L=16 Heisenberg
   to 1e-10. Two days, purely local — no GPU required. **This is the single
   prerequisite that unblocks everything else.**

3. **rocSOLVER pre-allocation bench**. File:
   `gpu-rocm/sandbox/bench_rocsolver_dgeqrf_workspace.cpp`. Confirms (or
   denies) the "query once, allocate, never query again" pattern on ROCm
   7.2. 30 minutes of remote time.

**Next week (contingent on all three above):**

4. **Write `bench_cbe_projection.cpp`** (§3.1) as the first GPU-side CBE
   code. One day. First numerical correctness gate against the Python
   reference.

5. **Begin the R2-1 scaffold** (`gpu-rocm/cbe-dmrg-gpu/`), modeled after
   `dmrg-gpu`, with apply_heff in strided_batched form and the CBE
   expansion/QR/augmentation stages as separate methods. Two weeks to first
   passing Heisenberg L=8 target.

The total critical-path time from today to "first green CBE-DMRG sweep on
MI300X L=16 Heisenberg" is approximately **3 weeks** with no parallelism and
~**2 weeks** if the CPU reference, the capture microbench and the rocSOLVER
bench run in parallel the first week.

---

## 8. Cross-references

- `round_2_plan.md §2.1` — parent R2-1 plan
- `research_B_svd_frequency_reduction.md §3` — CBE literature + absence of GPU implementations
- `proposal_3_hip_graph_capture.md §3.2` — graph cache structure, reused here
- `research_A_hip_graph_rocblas.md` — the strided-batched-only rule
- `docs/PROJECT_OVERVIEW.md §4.1 A1/A3/A7` — existing sync-elimination
  optimizations that CBE inherits
- `docs/PROJECT_OVERVIEW.md §5.2` — honest framing of the chi ≤ 50 regime
  (where CBE does NOT help)
- `docs/PROJECT_OVERVIEW.md §5.3` — pinned-memory race (why we keep `M` on
  device)
- `gpu-rocm/dmrg-gpu/src/dmrg_gpu_impl.h:370` — existing single-site
  apply_heff, the closest template for CBE
- `gpu-rocm/dmrg2-gpu-opt/src/dmrg2_gpu_opt_impl.h:427` — two-site apply_heff
  with strided-batched Step 3 (the pattern to replicate for Steps 1 & 3)
- `gpu-rocm/dmrg-gpu/src/scalar_traits.h` — A3 pointer-setup kernels (reused
  without change)
