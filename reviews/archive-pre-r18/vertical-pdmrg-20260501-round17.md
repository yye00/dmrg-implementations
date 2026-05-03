# Vertical review — pdmrg family — round 17 — 2026-05-01

## Charter

Audit pdmrg-gpu-base, pdmrg-gpu, pdmrg-gpu-opt, pdmrg-multi-gpu for
J1 (Stoudenmire), J2 (-opt = Block-Davidson default + ctor gate),
PDMRG-rules-2026-04-15 lock, the round-15 fix list, and the round-16
single-site Step-1 device-pointer fix. Regression-watch baseline is
`f40140d` per orchestrator brief; round-16 closure of 187fddf-msg
CRITICAL is the new bar.

## Charter proof — techniques applied

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | `t_absorb_` STILL DEAD in -gpu/-opt; `t_env_update_` STILL DEAD in multi-gpu |
| B. Behavioral diff | DONE | pdmrg-multi-gpu apply_heff_two_site Step 3 diverges from -gpu sibling — see HIGH |
| C. Docstring verification | DONE | Stale "Fix plan" comment at -opt:2283-2292 still narrates a CRITICAL that's now FIXED — see NIT |
| D. clangd filter | SKIPPED | no ROCm headers in sandbox |
| E. Absence-naming brief | FOLLOWED | three round-16 lonely-sibling HIGHs unaddressed |
| F. Workspace-aliasing audit | DONE | `dav_work_sz=max(θ·b, max_sub²)` envelope at opt:274 INTACT; new ws.d_batch_A/B/C reuse on Step-1 device-kernel path is single-region per call |
| G. Sibling fix-propagation | DONE | 3 round-15 lonely-sibling gaps unfixed; 1 round-16-related new HIGH (multi-gpu Step 3 host loops) |

## CRITICALS — block GPU run / paper submission

None. Round-16's CRITICAL (pdmrg-gpu-opt apply_heff_single_site Step 1
stack `Scalar* h_A[256]` + 3× H2D in sparse_s1 and dense branches) is
FIXED at opt:2321-2334 (sparse) and 2337-2350 (dense) via
`setup_batch_ptrs_wd[_sparse]` kernel launches from
`common/batch_ptrs_kernels.h`. Step 3 fallback (opt:2395-2410) and
strided-batched fast path (opt:2382-2391) remain device-side. Registry
is clean: the only D2 hit is a comment narrating the now-fixed defect.

## HIGHS — fix before next major event

- **pdmrg-multi-gpu: apply_heff_two_site Step 3 uses triple-nested
  host loop over (s1p, s2p, n) firing D·d² serial GEMMs**
  `[pdmrg-multi-gpu: pdmrg_multi_gpu_impl.h:802-816]` Sibling
  pdmrg-gpu collapses Step 3 to a single batched-GEMM
  (`setup_heff_ts_step3_full_ptrs` + one `gemm_batched`,
  gpu_impl.h:1010). multi-gpu fires `4·D` rocBLAS calls per Lanczos
  iter per bond per device — pure launch overhead. Round-16
  horizontal-gpu flagged this class. Shared kernels already in
  `common/batch_ptrs_kernels.h`; multi-gpu imports them for Step 1
  / single-site Step 3. Two-site Step 3 was missed.

- **pdmrg-gpu-opt: block_davidson_eigensolver has NO PhaseTimer**
  `[opt:1479-1786]` Unfixed from round-16. With
  `use_davidson_=true` default, every optimize_bond:2254 and
  batched_segment:3315 dispatch is UNTIMED. Lanczos fallback at
  opt:1792 has the pair; Davidson does not. Round-15 (69da5b4)
  wired `t_lanczos_` into dmrg-gpu-opt's and dmrg2-gpu-opt's
  block_davidson; pdmrg-gpu-opt sibling still not patched.
  Lonely-sibling, two cycles open.

- **pdmrg-gpu-opt: apply_heff_single_site has NO PhaseTimer pair**
  `[opt:2275-2413]` Unfixed from round-16. Default warmup/polish
  path silent in panel. pdmrg-gpu sibling has the pair at
  gpu:1935/1954/2090 (69da5b4). Lonely-sibling, two cycles open.

- **pdmrg-multi-gpu: apply_heff_single_site has NO PhaseTimer pair**
  `[multi:1645-1705]` Unfixed from round-16. Two-site twin at
  multi:757/818 IS instrumented. Single-site outlier.

## MEDIUMS — fix when convenient

- **pdmrg-multi-gpu: blocking `hipMemcpy` for new_k D2H per SVD**
  `[multi:1398, 1763]` Reconfirmed from round-16. Synchronous, no
  stream. Siblings use `hipMemcpyAsync` (gpu:1646/2157, opt:1388/
  2506). Violates no-host-roundtrips-2026-04-27.

- **`t_absorb_` STILL DEAD** `[gpu:215, 2824, 2845; opt:253, 3589,
  3608]` Three rounds open. Wire at S·Vh-into-MPS absorb or drop.

- **pdmrg-multi-gpu: `t_env_update_` STILL DEAD** `[multi.h:216,
  222, 240]` Three rounds open. `update_left/right_env` (multi:826,
  899) have zero begin/end; pdmrg-gpu wires at :1153/1215 and
  :1229/1287.

- **pdmrg-gpu-opt: per-iter `hipStreamSynchronize` in Davidson CGS**
  `[opt:1713]` Reconfirmed; CGS norm read on host-pointer-mode in
  inner orthogonalization. Pre-existing.

## NITS — cosmetic

- **pdmrg-gpu-opt: stale "Fix plan" comment**
  `[pdmrg_gpu_opt_impl.h:2283-2292]` Comment narrates the
  round-16-CRITICAL stack `Scalar* h_A[256]` defect and ends
  with "Fix plan: port device-side ptr-setup kernels to this path."
  The fix has shipped (lines 2321-2350 use kernel launches), but
  the comment block lives on. Remove or update to "Round-16 fix:
  device-side setup kernels installed."

## FALSE POSITIVES VERIFIED

- **8abb6e7 D6 INTACT**: pdmrg-gpu has zero local `setup_batch_ptrs_*`
  definitions (only variant-specific `setup_lenv/renv/heff_ss/ts_*`
  at gpu_impl.h:31-128); pdmrg-gpu-opt has zero duplicates. Both
  `#include "../../common/batch_ptrs_kernels.h"` (gpu:22, opt:19).

- **0efe96d D12 immune for pdmrg**: pdmrg-gpu/-opt Lanczos already
  device-pointer-mode pre-baseline (gpu:2202, opt:1786 — PointerMode
  Guard + `d_alpha_dev`/`d_beta_dev`/`d_nrm2_result` device scratch).

- **J1**: `accurate_svd_gpu<Scalar>` at base:1276, gpu:2452, opt:3331,
  multi:2214 — all four live.

- **J2 + ctor gate**: `use_davidson_=true` at opt:191; ctor :222-227
  disables `lanczos_graph` symmetrically; `set_use_davidson()` mirrors.
  `use_chebyshev_` / `use_batched_sweep_` read at opt:2251, 3472, 3485.

- **PDMRG-rules-2026-04-15**: `n_warmup=1, n_polish=0` defaults on all
  four `run()`; warmup/polish bodies dispatch only through
  `sweep_*_full_1site`.

- **pdmrg-multi-gpu round-15/16 fixes preserved**: PointerModeGuard at
  multi:1172/1187/1207/1318; `use_rsvd_=opts_.rsvd` at :163;
  init_mps_product/neel at multi.h:47-48 / impl:596, 613.

- **F aliasing**: `dav_work_sz=max(θ·b, max_sub²)` at opt:274 INTACT.
  `ws.d_batch_A/B/C` reuse for Step-1 device kernels in
  apply_heff_single_site is single-region per call (Step 1 finishes
  before Step 3 reuses at :2397-2410), no concern.

## SUMMARY

Round-16's CRITICAL is closed: pdmrg-gpu-opt apply_heff_single_site
Step 1 now uses device-side `setup_batch_ptrs_wd[_sparse]` kernels
in both sparse and dense branches, killing the stack `Scalar*
h_A[256]` + 3× H2D pattern on the default warmup/polish path. J1/J2/
PDMRG-rules-2026-04-15 locks hold across all four variants; D6
shared-header fix and D12 Lanczos pointer-mode are intact. Four
HIGHs remain — three are direct round-16 carry-overs (Davidson and
single-site PhaseTimers in -opt; single-site PhaseTimer in
multi-gpu — all lonely-sibling), and one new round-17 finding
surfaces from technique B: pdmrg-multi-gpu's apply_heff_two_site
Step 3 still uses the triple-nested host loop (D·d² serial GEMMs)
that pdmrg-gpu replaced with a single batched-GEMM full-collapse —
firing per Lanczos iter per bond per device. Recommend bundling all
four HIGHs into one multi-gpu/-opt cleanup PR.
