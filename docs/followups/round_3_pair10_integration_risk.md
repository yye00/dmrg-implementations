# Round 3 — Pair 10: Cross-cutting integration risk across R2-1 / R2-2 / R2-3 / R2-4

**Opened:** 2026-04-10
**Pair:** 10 of 10, round 3 of the DMRG-GPU follow-up refinement loop.
**Scope:** Are the four R2 follow-ups actually composable, or does their
"clean stack" diagram paper over real kernel-level, data-structure, and
convergence conflicts? Where are the redundancies and the contradictions?
**Agent role:** generator + adversarial reviewer (same voice, no separation).

Reviewed artefacts:
- `docs/followups/round_2_plan.md`
- `docs/followups/proposal_3_hip_graph_capture.md`
- `docs/followups/research_A_hip_graph_rocblas.md`
- `docs/followups/research_B_svd_frequency_reduction.md`
- `docs/followups/research_C_persistent_lanczos.md`
- `docs/followups/research_D_beyond_pdmrg_a2dmrg.md` (via round_2_plan summary)
- `gpu-rocm/dmrg2-gpu-opt/src/dmrg2_gpu_opt_impl.h` (1682 LOC)
- `gpu-rocm/pdmrg-gpu-opt/src/pdmrg_gpu_opt.h` — `StreamWorkspace` definition
- `docs/PROJECT_OVERVIEW.md` §4, §5

**Verdict up front:** the round_2_plan stack is **not** clean. Two of the four
follow-ups attack the same bottleneck via incompatible paths, one depends on
the other in a way the dependency graph understates, and the StreamWorkspace
that all four want to extend is already over-stuffed. The plan needs to cut
one follow-up, re-order another, and introduce an explicit 2-week fast-path.

---

## 1. Kernel-level overlap table

Current shared kernels in `dmrg2-gpu-opt/src/dmrg2_gpu_opt_impl.h`:

| Kernel (file:line)          | R2-1 CBE     | R2-2 CBE-TEBD   | R2-3 persistent Lanczos | R2-4 graph capture |
|-----------------------------|--------------|-----------------|-------------------------|---------------------|
| `apply_heff_two_site` (427) | **REPLACES** with `apply_heff_single_site` + tangent-space projector; never calls this path at all in steady state | uses R2-1 version inside Block-Jacobi Phase 2 | **REPLACES** with fused LDS-resident kernel at `chi ≤ 48`; at `chi > 48` leaves it alone | **CAPTURES** this exact call as a 3-node sub-graph |
| `update_left_env` (519)     | Still called, but on `d·chi·D → d·chi_new·D` tensor (single-site shape, not `d²·chi`) — existing routine reusable | Phase 1.5 canonicalization path uses it; Phase 2 calls it per segment | Untouched (environment update lives outside the persistent kernel) | Capture-eligible only if batched-pointer refactor lands (Research A) |
| `update_right_env` (594)    | same as left  | same as left    | untouched               | same as left        |
| Lanczos outer loop (726)    | Still called; CBE does not change the eigensolver topology | same  | **REPLACES** entirely at `chi ≤ 48`; becomes a single `hipLaunchKernelGGL` | **NOT CAPTURED** (Proposal 3 §3.1 explicitly excludes); `apply_heff_two_site` inside is captured node-by-node |
| SVD split (`svd_split_*`)   | **DELETED** on internal bonds (CBE has no full two-site SVD); small tangent RSVD replaces it | same  | untouched (different regime)  | not captured (LAPACK / rocSOLVER) |
| `form_theta_two_site`       | **DELETED** (CBE never forms `theta` on internal bonds) | same  | still used at `chi ≤ 48` in R2-3 regime | captured inside apply_heff capture region if fused, otherwise separate node |
| `newton_schulz_left/right`  | Needed for CBE's QR of tangent matrix | same  | untouched | capture-eligible (pure GEMM) |
| `apply_heff_single_site` (dmrg-gpu) | **This is the new hot path** for CBE; must be pulled into dmrg2-gpu-opt or a new cbe-dmrg-gpu/ | Phase 0 iTEBD warmup bypasses it; Phase 2 calls it | R2-3 could target single-site Lanczos instead of two-site — actually a better match (theta is `d·chi` not `d²·chi`, LDS budget 4× looser) | capture-eligible |

**Observations from the table**:

1. **R2-1 effectively deletes three of the hottest kernels** in
   `dmrg2-gpu-opt` (`apply_heff_two_site`, `form_theta_two_site`,
   `svd_split`). R2-3 and R2-4 both plan to refactor these same kernels. If
   R2-1 ships first and becomes the default, **R2-3 and R2-4 are fighting for
   code paths that no longer exist in the flagship**.

2. **R2-3 should target `apply_heff_single_site`, not
   `apply_heff_two_site`.** Single-site theta at `d=2, chi=32` is 128 doubles
   = 1 KB (vs 8 KB for two-site). The LDS budget analysis in Research C §1
   assumes two-site (`d² · chi · chi`), which is pessimistic. If R2-3 targets
   single-site instead, the envelope extends to `chi ≤ ~80` for `d=2` with
   full Krylov basis in LDS — that's essentially the entire `chi ≤ 128`
   competitive regime, not just the `chi ≤ 48` one. **This also collapses
   naturally onto R2-1, since CBE *is* single-site.**

3. **R2-4 (graph capture) amortises across a regime where R2-1 removes the
   call entirely.** Graph capture buys 2–4× dispatch reduction at `chi ∈
   [64, 128]` (per proposal_3). In that regime, CBE replaces the expensive
   Step 1 + Step 3 batched GEMMs with a smaller tangent-space RSVD. The
   captured graph becomes a different shape (`single-site apply_heff` +
   `tangent-space GEMM`) and the cached shapes from the pre-CBE
   `apply_heff_two_site` are **all thrown away**. R2-4's graph cache will be
   rebuilt from scratch the first time CBE sweeps run.

---

## 2. StreamWorkspace conflict analysis

The current `StreamWorkspace` in `pdmrg-gpu-opt/src/pdmrg_gpu_opt.h` (lines
101–176) already holds **≈40 named buffers** across six optimization paths:

- Contraction intermediates (`d_theta`, `d_T1`, `d_T2`, `d_heff_result`)
- Batched GEMM pointer arrays (6 pointers)
- Block-Davidson basis + host buffers (~8 fields)
- Lanczos fallback workspace (~13 fields)
- Newton-Schulz workspace (4 fields)
- SVD workspace (CPU + GPU, ~12 fields)
- NS-split eigendecomp host workspace (4 fields)
- rSVD (Halko-Martinsson-Tropp) workspace (7 fields)

The three Round-2 follow-ups that touch `StreamWorkspace` want to add:

**R2-1 (CBE)** — at minimum:
- `d_tangent_matrix` — `(d·w·k) × D` rectangular buffer for the tangent-space
  projection. At `d=2, w=5, k=20, D=chi·D_mpo=640` that's 256 KB.
- `d_cbe_Y`, `d_cbe_Q`, `d_cbe_B` — RSVD power-iter workspace on the tangent
  matrix. Another 3 × 256 KB = 768 KB.
- `d_cbe_expansion_vectors` — staging for the bond-expanded `U`, `V^T`.
  Variable size.
- Host buffer for the small RSVD core (k × k SVD).
- Essentially a parallel copy of the `d_rsvd_*` buffers that already exist,
  at a different size, with different semantics. **Collision risk: both need
  to be live across a sweep.**

**R2-3 (persistent Lanczos)** — paradoxically wants *less* workspace, because
the whole point is to keep state in LDS / registers. But it does need:
- `d_persistent_archive_basis` — the L2-resident Krylov basis for
  reorthogonalization (per Research C §6). 20 × 32 KB = 640 KB per bond,
  but this must live in global memory with SLC=0 hints — **not** a per-stream
  buffer, it's a per-workgroup-per-bond buffer dispatched at kernel-launch
  time. Ownership is ambiguous: is it part of StreamWorkspace or part of the
  dispatch?
- `d_persistent_scratch` for the on-chip tridiagonal eigensolve (CPU host
  round-trip replaced by on-chip QL) — trivially small.

**R2-4 (graph capture)** — needs:
- `ApplyHeffGraphCache cache` with up to 32 `hipGraphExec_t` entries per
  workspace (per Proposal 3 §3.2). `hipGraphExec_t` is an opaque handle
  (~pointer-sized), but each instantiated graph reserves ~1–2 MB of driver
  memory internally, so 32 graphs × 4 segments × 2 sweep directions =
  **~256 MB of driver memory just for cached graphs** on pdmrg-gpu-opt.
  This is not visible from the StreamWorkspace byte count but is real.

### Do they fit in the same struct?

**Bytes-wise yes, semantics-wise no.**

- R2-1 buffers are **owned by the sweep**: they need to survive from
  `optimize_bond(site, ...)` call to call. Fine.
- R2-3 persistent-kernel buffers have **dispatch-time** ownership: the
  archive basis is read/written by a kernel that holds all state internally
  and returns only the converged eigenvector. The StreamWorkspace does not
  actually need to name these buffers — they can be pool-allocated on first
  call.
- R2-4 graph cache is **per-stream cross-sweep state**: it has to survive
  sweep-to-sweep, but its contents (captured graphs) are invalidated any
  time any of the buffer pointers change address. R2-1 **changes the buffer
  layout** (CBE doesn't need `d_T1`/`d_T2` at all — they live in a smaller
  tangent-space buffer), so every graph R2-4 captures against the pre-CBE
  layout is invalid after R2-1 ships.

### Concrete collision scenarios

| Scenario                                             | Result |
|------------------------------------------------------|--------|
| R2-4 lands first, R2-1 lands second                  | All cached graphs invalidated, R2-4 rebuilds cache on first CBE sweep. Net: R2-4 ships, delivers its win for ~6 weeks, then silently degrades until CBE is also in the graph path. |
| R2-1 lands first, R2-4 lands second                  | R2-4 targets CBE's new 3–4 GEMM call pattern. Captured graphs have fewer nodes, savings per sweep are smaller (CBE already eliminated the biggest calls). R2-4's expected speedup drops from 1.3–1.5× to ~1.1×. |
| R2-3 persistent Lanczos lands inside R2-4 graph      | See §5 — this is a new hazard. |
| R2-1 + R2-2 land, R2-3 never lands                   | Fine, no collision. |
| R2-1 + R2-3 land, R2-4 never lands                   | Fine, and arguably the cleanest outcome. |

**Recommendation**: do not try to make all four coexist in the same
`StreamWorkspace`. Factor CBE state into a **new** `CbeWorkspace` struct that
only exists when `use_cbe_` is true. R2-3's persistent buffers should pool-
allocate at first use. R2-4's graph cache should be gated on `!use_cbe_`.

---

## 3. Build-target recommendation: new dir vs `-DCBE=ON` flag

Round_2_plan §2 says R2-1 should be a new `gpu-rocm/cbe-dmrg-gpu/` directory
"modeled after `dmrg-gpu-opt`." That's the wrong call for three reasons:

1. **`dmrg2-gpu-opt/src/dmrg2_gpu_opt_impl.h` is 1682 lines** and has
   accumulated every `-opt` experiment (Block-Davidson, Chebyshev, rSVD,
   NS-split, batched sweep, Lanczos fallback). Cloning it into
   `cbe-dmrg-gpu/` means duplicating 1682 lines and immediately losing
   sync with future bug fixes or perf tuning to the base. This is
   **exactly** the mistake the project made with the `pdmrg`/`pdmrg-cotengra`/
   `pdmrg-opt` split — the CPU side now has three copies of the same file
   with drift.

2. **A feature flag is cheaper.** The CBE path replaces three functions
   (`apply_heff_two_site`, `form_theta_two_site`, `svd_split`) and adds one
   new one (`cbe_expand_and_update`). A `bool use_cbe_` member plus an
   `if (use_cbe_) { ... } else { ... }` branch in `optimize_bond` is ~50
   lines of dispatcher plus ~800 lines of CBE kernels in a new header
   included from `dmrg2_gpu_opt_impl.h`.

3. **A new directory doubles the CUDA port work.** `gpu-cuda/` currently has
   six mirrored directories. A seventh means porting + maintaining another
   18 KLOC worth of near-duplicate code. A flag means a single `#ifdef` or
   `if constexpr` path in both branches of the existing port.

**Counter-argument**: the "kitchen sink" accretion in `-opt` is already
unreviewable. Adding CBE makes it worse. The precedent in this codebase is
three-tier (`*-base/`, `*-gpu/`, `*-gpu-opt/`) — CBE is a genuinely new
algorithm, not a tuning knob, so it arguably deserves its own tier.

**Compromise recommendation**:

- Add CBE as a **compile-time feature flag** inside `dmrg2-gpu-opt`:
  `template<typename Scalar, bool UseCbe>`. Two explicit instantiations.
  Zero runtime branching overhead, shared BLAS-3 plumbing, single binary
  per instantiation.
- Test burden: the existing `test_dmrg2_gpu_opt.cpp` harness is extended
  with a new test matrix covering `UseCbe = true` on all three models.
  No new CMake target beyond an extra binary.
- If the CBE path grows to > 500 LOC of kernels, factor those into a
  separate header (`cbe_kernels.h`) but keep the sweep driver in the
  existing file. No new directory.

---

## 4. Convergence-interaction hazards

These are the places where the follow-ups interact through numerics, not code:

### 4.1 CBE inner Lanczos convergence rate (R2-1 → R2-3)

**Claim in round_2_plan**: CBE is "two-site accuracy at single-site cost",
which people informally read as "same eigenvector quality." But the McCulloch/
Osborne 2024 paper (arXiv:2403.00562) actually reports that **CBE's inner
Lanczos typically converges in 10–20 iterations at chi ≥ 128, same as
two-site**. So R2-3's assumption that 15–20 Krylov iters fit in the L2-
resident archive basis still holds. **No conflict here.**

**But**: CBE at small chi (≤ 48, R2-3 target regime) runs on a
`d·chi × chi` single-site theta that has `d=2, chi=32` → 128 rows, 32 cols.
The Lanczos spectrum on such a small operator converges in **5–8 iterations,
not 15–20**. This is actually good news for R2-3: the LDS archive basis
shrinks from 20 × 8 KB = 160 KB to 8 × 8 KB = 64 KB, which *barely fits in
64 KB LDS without spilling to L2*. R2-3's whole "spill basis to L2" complexity
(Research C §6) becomes unnecessary in the CBE + small-chi regime. **R2-1 +
R2-3 together simplify R2-3.**

### 4.2 Block-Jacobi parallel sync vs persistent Lanczos (R2-2 ↔ R2-3)

**Hazard**: R2-2 Phase 2 is block-Jacobi: each of `P` GPUs owns `L/P`
contiguous sites and runs CBE single-site updates in parallel, with a
peer-to-peer barrier at the end of each Jacobi round. R2-3's persistent
Lanczos kernel holds a single workgroup hostage for the duration of the
Lanczos outer loop (~1 ms to 10 ms). **Does the Jacobi barrier block while
the persistent kernel is running?**

Yes, and this is actually **fine**: the Jacobi round only synchronizes
*after* every bond in the segment has finished. The persistent kernel is
launched per bond and the barrier waits for the whole sweep. The persistent
kernel is not held across the barrier — it completes before the barrier is
reached. **No conflict.**

**The real hazard** is that R2-2 expects each segment to sweep the same number
of bonds in the same time (for load balance). R2-3's persistent kernel has
very different latency at `chi ≤ 32` (where it wins) vs `chi ∈ [48, 128]`
(where it degrades or is disabled). If the segments have different chi
profiles — e.g., boundary segments grow to chi = 128 faster than bulk
segments — then the persistent-kernel segments finish much earlier and the
block-Jacobi round is bottlenecked on the slowest segment. **This is a real
load-imbalance risk that neither R2-2 nor R2-3 addresses.**

**Mitigation**: disable R2-3 entirely during R2-2's Phase 2 (single-mode
scheduling wins over mode-switching). R2-3 only activates during Phase 3
(serial CBE polish) where there's a single segment and no load-balance
concerns.

### 4.3 CBE dispatch overhead at chi ≤ 32 (R2-1 vs R2-3 target regime)

**Adversarial finding**: the round_2_plan paints CBE as a pure win. But CBE
at small chi **adds kernel calls**: the tangent-space projection is an extra
GEMM, the RSVD power iteration is 2–4 GEMMs, the QR of the small
expansion basis is another call. Net: **CBE at `chi = 32` makes apply_heff
~3–4 GEMMs more expensive** than the single-site Lanczos path that already
ships in `dmrg-gpu`.

At `chi = 32`, kernel launch overhead is the dominant cost (this is the
whole premise of R2-3). Adding 3–4 more kernel launches per bond
**directly reopens the CPU-wins wound that R2-3 is trying to close**.
The round_2_plan §2 R2-1 table says CBE wins by "FLOP ratio" at `chi=256`
— but FLOP ratio is not the bottleneck at `chi=32`, **launch count** is.

**This is a real conflict**: R2-1's per-bond call count > `dmrg-gpu` single-
site call count, and at `chi ≤ 32` that directly harms the 93%-CPU-wins
regime. **R2-1 should be disabled at `chi ≤ 32` in favor of the vanilla
single-site path.** The plan does not mention this crossover.

### 4.4 Phase 0 iTEBD vs CBE warmup basis consistency (R2-2 internal)

Not strictly a cross-cutting conflict but worth noting: R2-2 Phase 0 is
iTEBD warmup, which produces a left-canonical MPS. R2-2 Phase 2 is CBE
single-site, which *assumes* a mixed-canonical form around the active site.
The Phase 1.5 canonicalization step handles this, but **iTEBD's QR truncation
does not match CBE's bond-expansion truncation weight**. There's a risk that
Phase 1.5 "corrects" iTEBD's basis enough that Phase 2 needs extra sweeps to
recover the tangent-space subspace CBE wants. Needs measurement in the
R2-2 success criteria, which currently only checks final energy delta.

---

## 5. Graph-capture × persistent-kernel compatibility

**Question from the brief**: can R2-3's persistent kernel be launched inside
an R2-4 captured graph?

**Answer**: yes **mechanically**, but the resulting graph is a single node
containing a single kernel launch — i.e., **graph capture adds nothing** when
the persistent kernel is present, because the persistent kernel has already
fused what graph capture was trying to amortize. This is the most important
finding in this report.

Concretely:

- Regular (non-persistent) apply_heff: 3 rocBLAS calls × 15 Lanczos iters =
  **45 kernel launches per bond**. Graph capture reduces dispatch to 1 graph
  launch (plus ~45 intra-graph transitions at ~1 ns each — negligible).
  Net launch savings: ~44 × ~5 µs = **~220 µs/bond** in the `chi ≤ 64` regime.
- Persistent apply_heff (R2-3): **1 kernel launch per bond**. Graph capture
  reduces this to 1 graph launch. Net savings: **0**.

R2-3 and R2-4 are **solving the same problem** — kernel launch overhead at
small chi — via two incompatible mechanisms:

| | R2-3 persistent kernel | R2-4 graph capture |
|---|---|---|
| Launch count per bond | 1 | 1 (captured) or 45 (uncaptured) |
| Data locality | `theta` in LDS, basis in L2 | `theta` in HBM, basis in HBM |
| chi envelope | `≤ 48` (hard LDS limit) | `∈ [64, 128]` (per Proposal 3 §6) |
| Dev effort | ~800 LOC hand-rolled MFMA | ~300 LOC rocBLAS-graph wrapper |
| Risk | High (MFMA correctness, cache-hint no-ops) | Medium (Research A pivot) |

**The regimes are almost disjoint** — R2-3 targets `chi ≤ 48`, R2-4 targets
`chi ∈ [64, 128]`. They "could" coexist in principle. But:

1. The `chi ∈ [48, 64]` gap is exactly where **graph capture becomes
   marginal** (dispatch is no longer the bottleneck) and **persistent kernel
   is infeasible** (LDS doesn't fit). Neither follow-up wins this regime.

2. **If R2-1 (CBE) lands**, the `chi ≤ 32` regime uses single-site CBE which
   has smaller `theta` (`d·chi × chi`, not `d²·chi × chi`), pushing the
   persistent-kernel LDS envelope up to `chi ≤ ~80`. At that point R2-3
   covers the entire `chi ≤ 80` regime and **R2-4's target window
   (`chi ∈ [64, 128]`) shrinks to `chi ∈ [80, 128]`, a 48-unit sliver.**

3. Proposal 3 §7 already concedes R2-4 wins nothing at `chi ≤ 32` ("CPU
   cache residency wall") and nothing at `chi ≥ 256` ("SVD dominates"). If
   we also concede the `chi ≤ 80` regime to R2-3-post-CBE, **R2-4 is
   competing for `chi ∈ [80, 128]` only**, which is maybe 5 % of the
   benchmark configs.

**Adversarial verdict**: R2-4 is **dead on arrival** in a world where R2-1
and R2-3 both land. R2-4 should be cut *or* R2-3 should be cut. They cannot
both justify their 3–4 week cost together.

---

## 6. Adversarial findings — conflicts and redundancies

### 6.1 R2-2 depends on R2-1 more strongly than the graph shows

Round_2_plan §3 has R2-2 "4-6 wks" as its own edge in the dependency graph,
implying it starts once R2-1 is "done." In reality:

- R2-2 Phase 0 (iTEBD warmup) is **independent** of CBE and could ship
  standalone as a PDMRG warmup replacement.
- R2-2 Phase 2 (Jacobi + replicated envs) is **independent of the local
  bond update algorithm**. It could wrap `apply_heff_single_site` from
  `dmrg-gpu` instead of CBE.
- R2-2 Phase 3 (polish) depends on whatever flagship bond update is in
  place.

**Observation**: R2-2 doesn't depend on R2-1, R2-2 depends on **having a
better single-site bond update than what ships today**. CBE is one such
update, but plain-old single-site from `dmrg-gpu` with a subspace-expansion
mixer is another. If R2-1 slips to 10+ weeks, R2-2 can still ship on top of
vanilla single-site and deliver most of its parallel-speedup claim. The
round_2_plan obscures this.

**Impact on ordering**: if R2-1 is blocked or slipping, R2-2 should be
unblocked to proceed on top of the existing `dmrg-gpu` single-site path,
not waiting.

### 6.2 Research B's AHC-DMRG and R2-1 are not the same follow-up

This isn't in the round_2_plan but worth flagging: Research B (§6) proposes
**Adaptive Hot/Cold DMRG** (AHC-DMRG) as an orthogonal win on top of
classical two-site DMRG — a 1.5–3× sweep speedup from skipping converged
bond SVDs. Round_2_plan completely subsumes this into R2-1 ("CBE replaces
SVD, so AHC is moot"). **This is wrong**: AHC-DMRG has a much lower
implementation cost (~400 LOC, pure residual-gated skip) and ships a win at
every bond dimension, not just `chi ≥ 128`. It should be a **separate
follow-up** that can ship in 2-3 weeks independently of everything else.

**Recommendation**: rescue AHC-DMRG from round_2_plan as **R2-0** (new
numbering), scope it to a weekend PoC + 1 week integration, and ship it
before anything else in the round.

### 6.3 R2-3 and R2-4 are solving the same problem

Covered in §5 above. One of them must die.

### 6.4 The "flagship" label on R2-2 is risky

R2-2 is labeled "flagship" and "most publishable," with a projected 1.7×
speedup stacked on R2-1's 2×. But:

- Block-Jacobi convergence on critical-point Hamiltonians has **no published
  guarantee** (round_2_plan itself flags this as "High" risk).
- The projected numbers in the speedup table are **not measured**, they are
  estimated from Amdahl arithmetic on PDMRG's current breakdown.
- R2-2 requires R2-1 AND new TEBD kernels AND canonicalization/broadcast
  AND a Jacobi scheduler. Total LOC estimate: 3000. This is 8–10 weeks of
  work.
- The "publishable" novelty is CBE-TEBD-DMRG as a *scheme*, but the paper
  would be much cleaner with R2-1 CBE-DMRG GPU port on its own — which is
  already "first public GPU CBE."

**Adversarial verdict**: R2-2 should be **deferred** until R2-1 has shown
measurable wins on the reference models. The flagship label is aspirational,
not earned.

---

## 7. Revised ordering and scoping

**Cuts and defers** (ruthless):

- **CUT: R2-4 (HIP graph capture)** unless R2-3 is also cut. The two target
  overlapping regimes and R2-3 is strictly more powerful where both apply.
  If Research A Phase 0 microbench shows strided-batched works, and CBE+R2-3
  slips beyond 12 weeks, R2-4 can be revived as a fast tactical win — but
  not as a Round 2 deliverable competing for the same engineering weeks.
- **DEFER: R2-2 (CBE-TEBD-DMRG flagship)** until R2-1 has landed AND showed
  ≥ 1.5 × speedup over `dmrg2-gpu-opt` on at least one reference model.
  Then revisit with measured R2-1 numbers instead of projected ones.
- **ADD: R2-0 (AHC-DMRG skip gate)** — rescue from Research B §6. Low LOC,
  low risk, ships a win at every `chi ≥ 128` config without touching any
  of the R2-1 through R2-4 refactors.

**Revised sequence:**

```
Week 0 (now):
  - R2-0 AHC-DMRG PoC on dmrg2-gpu-opt (2 day spike)
  - Research A Phase-0 microbench for rocBLAS strided-batched capture
  - Python CBE reference in cpu/cbe-reference/ (can be parallel)

Weeks 1-2:   R2-0 AHC-DMRG integration + benchmark (fast-path win, see §8)
Weeks 3-10:  R2-1 CBE-DMRG backbone (as -DCBE flag on dmrg2-gpu-opt, not
             a new directory — see §3)
Weeks 11-12: Measurement gate. Did R2-1 deliver ≥ 1.5 × on Heisenberg +
             Josephson + TFIM at chi ∈ {128, 256}? If no, stop and
             reconsider. If yes, unblock R2-2.
Weeks 13-18: R2-2 CBE-TEBD-DMRG, scoped down to Phase 0 (iTEBD warmup) +
             Phase 2 (block-Jacobi) only, on top of R2-1.
Weeks 19+:   R2-3 persistent Lanczos IF AND ONLY IF R2-1 + R2-2 have not
             closed the small-chi CPU-wins gap via CBE's smaller theta.
             Measurement-gated.
```

**Ordering changes from the round_2_plan:**

| | round_2_plan | revised |
|---|---|---|
| Phase 0 fast win | — | R2-0 AHC-DMRG (2 weeks) |
| Backbone | R2-1 CBE-DMRG (6-8 wks) | R2-1 as -DCBE flag (6-8 wks) |
| Parallel flagship | R2-2 immediately after R2-1 | R2-2 **measurement-gated** on R2-1 |
| Small-chi kernel | R2-3 in parallel with R2-2 | R2-3 **gated** on R2-1 not solving it already |
| Graph capture | R2-4 in parallel with R2-1 | **CUT** |

---

## 8. Fast-path proposal: what can ship in 2 weeks?

**The round_2_plan gives no 2-week win.** R2-1 is 6-8 weeks minimum. R2-2 is
gated on R2-1. R2-3 is 3-4 weeks plus L2-retention microbench. R2-4 is 1 week
only in the best case and delivers marginal wins even then.

**R2-0 AHC-DMRG skip gate** (rescued from Research B §6) is the only
deliverable that ships a measurable win in ≤ 2 weeks. Scope:

**Week 1:**
- Day 1-2: Add per-bond metadata cache to `DMRG2GPUOpt`:
  `struct BondState { double sigma_tail, discarded_weight,
  theta_frobenius, lanczos_residual; }` — one per bond, one per sweep.
- Day 3-4: Implement the composite gate in `optimize_bond`:
  `skip = (lanczos_residual < 1e-8) && (|theta - prev_theta|_F / |theta|_F <
  1e-5) && (discarded_weight < 1e-12)`.
- Day 5: Cold-bond path = canonicalization-only update (apply saved `U`,
  `Vh` without re-SVDing). This is pure GEMM and already works.

**Week 2:**
- Day 1-3: Correctness testing against `quimb-dmrg2` on the existing
  `test_dmrg2_gpu_opt` suite. Require `|ΔE| < 1e-10` at every
  `L ∈ {8, 16, 32}, chi ∈ {64, 128}` config.
- Day 4-5: Benchmark on `Heisenberg L=64 chi=256` and `L=64 chi=512`,
  measure skip-rate and wall-clock improvement. Target: ≥ 1.5 × sweep
  speedup at `chi ≥ 128`, zero regression at `chi ≤ 64`.

**Expected win**: 1.5–3 × speedup on the `chi ≥ 128` configs where
CPU-SVD is already 97 % of runtime. This is a *real* two-week win on the
current flagship before CBE or anything else ships.

**Cost**: ~400 LOC, no new kernels, no refactor. Zero interaction with
R2-1 / R2-2 / R2-3 / R2-4. Composes with all of them and particularly
clean with R2-1 (CBE already avoids the SVD, AHC skips it *and* the
projection update at converged bonds).

**Why this is not in the round_2_plan**: because the plan concluded CBE
subsumes AHC, which is factually incorrect — CBE eliminates the full-theta
SVD but still does per-bond Lanczos + tangent-space GEMMs at every bond
every sweep. AHC skips the entire bond. They are complementary, not
redundant.

---

## 9. Summary of cuts

| Follow-up | Decision | Reason |
|---|---|---|
| **R2-0 AHC-DMRG skip gate** | **ADD** (new, rescued from Research B §6) | Two-week fast-path win, complements everything else, low risk |
| **R2-1 CBE-DMRG backbone** | KEEP, scope as `-DCBE` flag on existing `dmrg2-gpu-opt` (not new dir) | Biggest single win, real publishable novelty, but test burden of new dir is unjustified |
| **R2-2 CBE-TEBD-DMRG flagship** | DEFER past an R2-1 measurement gate | "Flagship" label is aspirational until R2-1 numbers land; scope is too large for simultaneous dev |
| **R2-3 persistent Lanczos** | KEEP conditionally, **re-target to single-site `apply_heff` not two-site**, gate on whether R2-1 already closes the small-chi gap | Orthogonal regime from CBE, but redundant with R2-4 |
| **R2-4 HIP graph capture** | **CUT** — subsumed by R2-3 in its target regime, and by R2-1 in the larger-chi regime | Redundant with R2-3 wherever both apply; R2-3 is strictly more powerful |

---

## 10. Open questions this report does not answer

1. **Does the Phase 0 iTEBD produce a good enough initial basis for CBE
   Phase 2?** Needs measurement on Josephson specifically.
2. **Does CBE at `chi = 32` actually add the 3-4 extra kernel launches I
   claimed in §4.3, or does the McCulloch/Osborne simplification fuse
   enough of them to match single-site `apply_heff`?** Needs Python
   CBE reference profiling.
3. **Does the AHC skip rate survive on Josephson (d=5) as well as
   Heisenberg (d=2)?** Research B §6 flags this as a thresholding risk.
4. **What does Pair 7 (graph capture GO/NO-GO) and Pair 8 (fused kernel
   feasibility) conclude?** If Pair 7 is NO-GO, R2-4 is already dead and
   the cut in §9 is free. If Pair 8 is GO on fused kernels at
   `chi ≤ 64`, then R2-3 becomes a strictly-better implementation of the
   same idea and R2-4 is doubly dead.

## 11. Relevant code pointers

- `/home/captain/clawd/work/dmrg-implementations/gpu-rocm/dmrg2-gpu-opt/src/dmrg2_gpu_opt_impl.h` — `apply_heff_two_site` at line 427, env updates at 519/594, sweep driver at 1571-1588
- `/home/captain/clawd/work/dmrg-implementations/gpu-rocm/pdmrg-gpu-opt/src/pdmrg_gpu_opt.h` — `StreamWorkspace` struct at line 101 (the "everything already lives here" argument)
- `/home/captain/clawd/work/dmrg-implementations/gpu-rocm/dmrg-gpu/src/` — the single-site `apply_heff_single_site` that R2-3 should target instead of two-site (per §1 observation 2)
- `/home/captain/clawd/work/dmrg-implementations/docs/followups/research_B_svd_frequency_reduction.md` §6 — the AHC-DMRG algorithm that becomes the new R2-0
- `/home/captain/clawd/work/dmrg-implementations/docs/followups/round_2_plan.md` §3 — the dependency graph that this report argues is wrong
