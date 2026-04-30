# Vertical review — pdmrg family — round 12 — 2026-04-30

## Charter

Audit the pdmrg family across pdmrg-gpu, pdmrg-gpu-opt, and pdmrg-multi-gpu
for tier conformity, J1 (Stoudenmire `accurate_svd_gpu` mandatory at every
boundary merge), J2 (-opt = strict superset of -gpu, Block-Davidson default),
the round-11 cleanup regression list (commits bc3fcd0, c3d3e50), and the
round-10 H10-multi-WW guard. Length budget 800 words.

Note: charter says to compare across (-base, -gpu, -gpu-opt) but the brief I
was handed names (-gpu, -gpu-opt, -multi-gpu); I followed the brief.

## Charter proof — techniques applied

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | per-symbol grep on private members of all three variants; 5 dead in -multi-gpu |
| B. Behavioral diff | DONE | run() signatures + hot-path methods compared across the three |
| C. Docstring verification | DONE | J1 docstrings + class-level claims checked against code |
| D. clangd filter | SKIPPED | clangd not invokable for ROCm headers in this sandbox; fell back to A |
| E. Absence-naming brief | FOLLOWED | tier checklist applied to each variant |
| F. Workspace-aliasing audit | DONE | d_dav_work / d_dav_work2 / d_T1 / d_T2 / d_lanczos_v reviewed |
| G. Sibling fix-propagation | DONE | regression-watch list checked against -multi-gpu sibling |

## CRITICALS — block GPU run / paper submission

- **PDMRG-rules violation in pdmrg-multi-gpu (3 of the 4 mandatory rules)**.
  Project CLAUDE.md mandates: warmup `≤ 2`, polish `≤ 2`, polish single-site,
  zero-warmup/zero-polish supported.
  - `[pdmrg-multi-gpu: pdmrg_multi_gpu_impl.h:2552]` polish phase hard-codes
    `int n_polish = 10;` (Rule 3 violation, hard-coded > 2).
  - `[pdmrg-multi-gpu: pdmrg_multi_gpu_impl.h:2554-2556]` polish loop calls
    `sweep_LR_full()` / `sweep_RL_full()` — **two-site polish** (Rule 2
    violation; Rule 1's mirror — must be `_1site` variants).
  - `[pdmrg-multi-gpu: pdmrg_multi_gpu.h:39]` `run()` signature is
    `run(int, int, int n_warmup = 3)` — default `n_warmup = 3 > 2` (Rule 3
    violation), and there is no `n_polish` parameter, so the user cannot pass
    `--polish 0` (Rule 4 violation: zero-polish must be supported).
  Pdmrg-gpu and pdmrg-gpu-opt both have rule-compliant polish (1site, n_polish
  CLI-controlled, default 0). The -multi-gpu sibling was never brought into
  compliance after the 2026-04-15 rule lock — same pattern as the round-8/11
  "fix never propagated" defect class. Recommend mirroring pdmrg-gpu's run()
  signature + 1site polish + CLI-driven counts.

## HIGHS — fix before next major event

- **Dead host-pinned batch pointer arrays in pdmrg-multi-gpu**
  `[pdmrg-multi-gpu: pdmrg_multi_gpu.h:113-115; pdmrg_multi_gpu_impl.h:339-341]`
  `h_batch_A_pinned`, `h_batch_B_pinned`, `h_batch_C_pinned` are declared on
  every `DeviceContext`, set to `nullptr` in `allocate_device_resources`, and
  never `hipHostMalloc`'d, never written, never read, never freed. Pure dead
  infrastructure — exact same defect class as round-11 c3d3e50 cleanup
  (h_dav_V_copy / h_rsvd_B / h_rsvd_U_small in pdmrg-gpu-opt). G-sibling
  miss: the cleanup commit touched -gpu-opt but skipped -multi-gpu.

- **Dead host-RSVD legacy buffers in pdmrg-multi-gpu**
  `[pdmrg-multi-gpu: pdmrg_multi_gpu.h:158-159; pdmrg_multi_gpu_impl.h:430-431]`
  `h_rsvd_B` and `h_rsvd_U_small` are `std::vector<Scalar>` members that get
  resized in `allocate_device_resources` and are never read or written
  thereafter (exactly 1 grep hit each, the `resize()` call). Same defect
  class as round-11 c3d3e50; same G-sibling miss as above. Comment claims
  "kept for fallback paths" — there is no fallback path that touches them.
  Drop both members and the resize calls.

## MEDIUMS — fix when convenient

- **Multi-gpu polish phase rebuild has no `n_polish == 0` short-circuit**
  `[pdmrg-multi-gpu: pdmrg_multi_gpu_impl.h:2549-2566]` Even after the
  CRITICAL above is addressed, the entire `if (n_segments_ > 1) { ... }`
  block needs an explicit `&& n_polish > 0` guard mirroring pdmrg-gpu's
  `[pdmrg_gpu_impl.h:2757]`. Otherwise the gather/scatter cost is paid even
  when the user requested zero polish.

## NITS — cosmetic

- `[pdmrg-multi-gpu: pdmrg_multi_gpu.h:158-159]` Comment on the dead host
  buffers reads "legacy host buffer (kept for fallback paths)". When the
  buffers are removed (HIGH above), drop the misleading comment too.

## FALSE POSITIVES VERIFIED

- pdmrg-gpu `h_svd_tmp` / `d_const_*` / `d_ones_D` look candidate-dead but
  are live: h_svd_tmp services the CPU-SVD fallback, const scalars feed
  Lanczos Rayleigh-Ritz GEMVs, d_ones_D is the Step-3 reduction vector.
- pdmrg-gpu-opt davidson-toggle symmetric setter `[pdmrg_gpu_opt.h:61-72]`
  intact — bc3fcd0 (dmrg-gpu-opt) did not regress the round-7 sister fix.
- pdmrg-gpu-opt h_dav_V_copy / h_rsvd_B / h_rsvd_U_small are gone (c3d3e50).
- Legacy `gpu-rocm/pdmrg-gpu/src/accurate_svd.h` is gone; no live include,
  no stale CMake reference; surviving mentions are commit-history comments.
- H10-multi-WW guard intact `[pdmrg_multi_gpu_impl.h:684]`.
- J1 lock holds: `accurate_svd_gpu<Scalar>(...)` at
  `[pdmrg_gpu_impl.h:2491; pdmrg_gpu_opt_impl.h:3350; pdmrg_multi_gpu_impl.h:2277]`.
- J2 lock holds: `use_davidson_ = true` at `[pdmrg_gpu_opt_impl.h:204]`.

## SUMMARY

The J1 (Stoudenmire) and J2 (Davidson default) locks hold across all three
pdmrg variants; round-11 cleanup is clean for pdmrg-gpu and pdmrg-gpu-opt;
the H10-multi-WW guard and round-7 canonical-Vh swap survive; the
davidson-toggle symmetric setter survived bc3fcd0 untouched. The major
finding is pdmrg-multi-gpu lagging — both the 2026-04-15 PDMRG-rules lock
(CRITICAL: 10-sweep two-site polish hard-coded, default warmup = 3, no
n_polish CLI) and the round-11 dead-buffer cleanup (HIGH×2: pinned-batch
arrays + legacy h_rsvd_*) never propagated. Sibling-fix-propagation pattern
is the same defect class that triggered the techniques F+G addition in
round 8. Address the CRITICAL by porting pdmrg-gpu's run() signature and
1site-polish loop into pdmrg-multi-gpu before the next GPU window.
Technique D was skipped (no clangd / no ROCm headers in sandbox); A
subsumes the most important diagnostics for that channel.
