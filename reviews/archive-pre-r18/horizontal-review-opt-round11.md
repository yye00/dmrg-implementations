# Horizontal review — `-gpu-opt` tier — round 11 — 2026-04-28

HEAD `1d44d89` (review: round-10 conformity report). Gating round; the
charter asks for verification that round-10's two net-new findings
(H10-multi-WW-leak, M-opt-rsvd-env) are fixed and that no further
propagation gaps exist in the same class.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan       | DONE        | dmrg-gpu-opt has 12 dead Lanczos sync-free + R3-F1 buffers — pre-existing carry-over (round-10 deferred H10 covers). dmrg2-gpu-opt and pdmrg-gpu-opt clean. |
| B. Behavioral diff         | DONE        | -gpu vs -opt async-primitive counts consistent (88→146, 80→134, 172→282). One naming nit in dmrg-gpu-opt (`svd_fallback`). |
| C. Docstring verification  | DONE        | One drift in `accurate_svd_gpu.h` (comment says MAX_DEPTH default 5; code uses 3). Cosmetic. |
| D. clangd filter           | N-A         | No ROCm headers on host. |
| E. Absence-naming brief    | FOLLOWED    | All required -opt features present (pad_mfma16, chi_max_user_, Block-Davidson + tiny fallback, batched Step-3, public setters, pdmrg-only worker pool + chebyshev + batched-segment). |
| F. Workspace-aliasing audit| DONE        | `d_dav_work_/work2_` sizing verified (round-9 H-new1 fix in place across all 3 -opt). `d_T1_/T2_` sized `D*dd*chi_max²`, fits both Step-1 V layout and Step-2 U layout. `d_xs_batch_*` (`pdmrg-gpu-opt`) sized `n_segments*D*dd`, total batch ≤ that bound. accurate_svd_gpu `d_block[depth]` reused sequentially across 6a/6b, sized `max(sz_mk, sz_kn)` — OK. |
| G. Sibling fix-propagation | DONE        | Round-10 H10-multi-WW-leak fix verified in pdmrg-multi-gpu:684; sibling guard pattern present in dmrg2-gpu-base, dmrg2-gpu, dmrg2-gpu-opt, pdmrg-gpu-base, pdmrg-gpu, pdmrg-gpu-opt at the corresponding `set_mpo` / `precompute_fused_mpo` call sites. M-opt-rsvd-env fix `use_rsvd_ = opts_.rsvd` present in all 3 -opt ctors at lines 62, 60, 205. **One new propagation gap found: lonely `set_use_davidson` lanczos_graph round-trip — see HIGHS / N-opt-davidson-toggle.** |

## CRITICALS

None.

## HIGHS

None new. The round-10 H10-multi-WW-leak (HIGH) and round-9 H-new1
(d_dav_work sizing) are verified fixed.

## MEDIUMS

**M-opt-davidson-toggle (NEW, technique G — lonely fix).**
`gpu-rocm/dmrg-gpu-opt/src/dmrg_gpu_opt.h:83`. The set_use_davidson
setter is the simple one-line version:

```cpp
void set_use_davidson(bool use_dav) { use_davidson_ = use_dav; }
```

But dmrg2-gpu-opt (`dmrg2_gpu_opt.h:73-81`) and pdmrg-gpu-opt
(`pdmrg_gpu_opt.h:61-72`) have the round-trip-preserving variant that
re-enables `opts_.lanczos_graph` when Davidson is toggled off (it was
disabled at construction because Davidson's variable output pointer is
graph-incompatible). Net effect: a benchmark or test that calls
`dmrg_gpu_opt.set_use_davidson(false)` to switch to Lanczos will run
without the graph cache that the user originally requested via
`DMRG_GPU_OPT_LANCZOS_GRAPH=1`. Performance regression for that toggle
path; correctness unaffected. Severity MEDIUM (matches the round-7
fix-class severity in the sibling variants).

Suggested fix (one line, mirroring dmrg2-gpu-opt's pattern, plus a new
private member `lanczos_graph_was_user_enabled_`):

```cpp
void set_use_davidson(bool use_dav) {
    use_davidson_ = use_dav;
    if (use_dav && opts_.lanczos_graph) {
        opts_.lanczos_graph = false;
        lanczos_graph_was_user_enabled_ = true;
    } else if (!use_dav && lanczos_graph_was_user_enabled_) {
        opts_.lanczos_graph = true;
    }
}
```

Pre-existing carry-overs (none net-new):

- **MED-dmrg-gpu-opt-dead-lanczos-scalars**: 8 device buffers
  (`d_neg_alpha_`, `d_neg_overlap_`, `d_inv_nrm_`, `d_alpha_dev_`,
  `d_beta_dev_`, `d_neg_beta_scalars_`, `d_dot_result_`,
  `d_nrm2_result_`) allocated in ctor at
  `dmrg_gpu_opt_impl.h:174-181`, freed at `:361-368`, never used
  elsewhere. The Lanczos eigensolver instead allocates a local
  `d_neg_alpha_scr` (`:1025-1188`) and uses host-resident
  `h_alpha`/`h_beta`. The header explicitly states "Lanczos α/β remain
  host-resident in this variant" (line 199). The dead buffers are
  scaffolding for a planned sync-free port deferred per round-10's
  "H10 Lanczos α/β device-pointer-mode port". **NOT net-new** — covered
  by the round-10 deferred list.
- **MED-dmrg-gpu-opt-dead-constants**: `d_const_one_`,
  `d_const_zero_`, `d_const_neg_one_`, `d_ones_D_` (lines 183-197) are
  set up for device-pointer-mode rocBLAS / R3-F1 collapse but never
  used. dmrg-gpu-opt's apply_heff Step-3 uses strided_batched +
  per-(wp,sp) loops, not the R3-F1 GEMV collapse the buffers were
  intended for. Same dead-infrastructure class. NOT net-new — same
  H10-deferred bundle.
- **MED-pdmrg-opt-{1,2,3}** (round-10 carry-over): dead host buffers
  in pdmrg-gpu-opt (`h_rsvd_B`, `h_rsvd_U_small`, `h_dav_V_copy`).
  Cosmetic.

## NITS

- **N1-svd-name-drift**: dmrg-gpu-opt method named `svd_fallback`
  (`dmrg_gpu_opt.h:277`, `_impl.h:1265`) but the comment at
  `_impl.h:1259` explicitly says it is no longer a fallback (replaces
  host-LAPACK). Rename to `svd_and_update_mps` for naming parity with
  dmrg-gpu sibling. Cosmetic.
- **N2-asvd-doc-drift**: `gpu-rocm/common/accurate_svd_gpu.h:32`
  comment says "MAX_DEPTH (default 5)" but `:124` defines
  `static constexpr int MAX_DEPTH = 3`. Update comment.

## FALSE POSITIVES VERIFIED

- Round-10 H10-multi-WW-leak: VERIFIED FIXED at
  `gpu-rocm/pdmrg-multi-gpu/src/pdmrg_multi_gpu_impl.h:684`
  (`if (devices_[k].d_WW[bond]) HIP_CHECK(hipFree(...));` before the
  hipMalloc).
- Round-10 M-opt-rsvd-env: VERIFIED FIXED in all three -opt
  ctors (`dmrg_gpu_opt_impl.h:62`, `dmrg2_gpu_opt_impl.h:60`,
  `pdmrg_gpu_opt_impl.h:205`); each places
  `use_rsvd_ = opts_.rsvd;` after `opts_.load_from_env()`.
- Round-9 H-new1 d_dav_work sizing: VERIFIED at
  `dmrg_gpu_opt_impl.h:310-313` (sized `max(theta·b + max_sub·b,
  max_sub²)` to cover both regions concurrently) plus the same
  pattern at `dmrg2_gpu_opt_impl.h:267-270` and pdmrg-gpu-opt
  ctor.
- Round-9 M4-W (W-buffer guards in -opt set_mpo): VERIFIED at
  `dmrg_gpu_opt_impl.h:516,547,551`; same at
  `dmrg2_gpu_opt_impl.h:463,494,498` and pdmrg-gpu-opt
  `:577,607,611`. WL/WW nnz guards present too.
- Round-9 H1-ext-gpu (nonblocking flag): VERIFIED at
  `dmrg_gpu_opt_impl.h:110-111`, `dmrg2_gpu_opt_impl.h:107-108`,
  `pdmrg_gpu_opt_impl.h:125,138`. Round-10 self-audit additionally
  propagated the flag to `dmrg-gpu-base`, `dmrg2-gpu-base`,
  `radam-gpu`, `rlbfgs-gpu` (verified at the cited lines).
- Round-8 CR-D1 dav_work_sz: VERIFIED (same sites as H-new1).
- Round-7 C2/H6 (Block-Davidson on-device syev): VERIFIED via
  `d_dav_eigvals_`, `d_dav_E_`, `d_dav_info_` device buffers used in
  all three -opt block_davidson_eigensolver paths (live, not dead).
- pdmrg-gpu-opt worker stream pool actually used: technique-A scan
  shows 28 hits across `n_workers_`, `worker_streams_`,
  `worker_done_events_`, `step_done_events_` — live infrastructure.
- J1 Stoudenmire lock: pdmrg-gpu-opt
  `pdmrg_gpu_opt_impl.h:3346,3353` calls `accurate_svd_gpu<Scalar>`
  at boundary; pdmrg-gpu-opt header `:13` includes
  `../../common/accurate_svd_gpu.h`. dmrg-gpu-opt and dmrg2-gpu-opt
  immune (no segment boundaries).
- pad_mfma16 idempotent at zero-mod-16: `(x + 15) & ~15` — for x=16
  yields 16. OK.
- chi_max_ vs chi_max_user_ consistency: ctors initialize
  `chi_max_(pad_mfma16(chi_max)), chi_max_user_(chi_max)`; bond_dims
  capped at `chi_max_user_` (verified `pdmrg_gpu_opt_impl.h:115`).

## SUMMARY

The round-10 fixes are intact and correctly placed. Both H10 and
M-opt-rsvd-env are verified at the cited file:line for every sibling
variant in scope. All round-7/8/9 watch-list items remain in good
order.

**One new finding** — M-opt-davidson-toggle, MEDIUM, technique G —
exposes a lonely-fix pattern that survived 10 prior rounds. The round-7
batch-2 commit (`e8cdd91`) added the lanczos_graph round-trip
preservation to dmrg2-gpu-opt and pdmrg-gpu-opt's `set_use_davidson`,
but the corresponding fix in dmrg-gpu-opt's `set_use_davidson` was
never made. Severity is MEDIUM rather than HIGH because (a) it's a
performance regression on a non-default toggle path, not the default,
and (b) most benchmark scripts construct fresh objects rather than
calling `set_use_davidson(false)` post-construction.

**Trend confirmation**: round 11 is *not* zero net-new — it surfaced
one MEDIUM. The methodology-hardening trend continues to find the
sibling-fix-propagation class even after rounds 7/8/9/10 each closed
several. The two rounds-clean criterion for G1 has therefore not been
met yet; suggest a round-12 after the M-opt-davidson-toggle fix
lands, with the additional charter ask "verify dmrg-gpu-opt
set_use_davidson now matches dmrg2/pdmrg variants".

The two carry-over MEDIUMs (`MED-dmrg-gpu-opt-dead-lanczos-scalars`,
`MED-dmrg-gpu-opt-dead-constants`) are not new — they fall under the
round-10 deferred "H10 Lanczos α/β device-pointer-mode port" bundle.
Worth flagging now so the H10 follow-up commit also cleans up the
12-buffer dead-infrastructure that came along for the ride.

### Overall verdict

Block GPU run? **NO** for correctness. The new MEDIUM is a performance
toggle hazard that does not affect benchmark numbers in the default
configuration (Block-Davidson on; lanczos_graph off in dmrg-gpu-opt
unconditionally — the toggle is exotic).

Action: fix M-opt-davidson-toggle (single-line + new private member),
then re-run round 12 for the two-rounds-clean G1 gate.
