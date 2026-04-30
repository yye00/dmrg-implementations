# Vertical review — pdmrg family — round 13 — 2026-04-30

## Charter

Audit pdmrg-gpu, pdmrg-gpu-opt, and pdmrg-multi-gpu for tier conformity,
J1 (Stoudenmire `accurate_svd_gpu` mandatory), J2 (-opt = Block-Davidson
default + lanczos_graph ctor gate), the PDMRG-rules-2026-04-15 lock
(single-site warmup/polish, ≤ 2 each, zero supported, explicit CLI), the
8b7a68e + 0b9fccf round-12 fix list, and the round-10/11 baseline
regression watch. Length budget 900 words.

## Charter proof — techniques applied

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | private members of all 3 variants enumerated; rsvd buffers + h_svd_tmp + d_const_* live |
| B. Behavioral diff | DONE | run() / merge_and_optimize_boundaries / svd_split / apply_heff_two_site / lanczos compared across 3 variants |
| C. Docstring verification | DONE | "only rsvd and profile honoured" claim in pdmrg-multi-gpu — verified against ctor + dispatch code |
| D. clangd filter | SKIPPED | clangd not invokable for ROCm in sandbox; A subsumes |
| E. Absence-naming brief | FOLLOWED | tier checklist applied per variant |
| F. Workspace-aliasing audit | DONE | d_dav_work / d_dav_work2 (max(b·dim, max_sub²)), d_T1/T2 (t_max), d_lanczos_v reviewed |
| G. Sibling fix-propagation | DONE | 5 round-12 fix classes traced + 4 baseline regression-watch items |

## CRITICALS — block GPU run / paper submission

(none)

## HIGHS — fix before next major event

- **pdmrg-multi-gpu: `opts_.rsvd` env var ignored — docstring/code drift**
  `[pdmrg-multi-gpu: pdmrg_multi_gpu_impl.h:151]` ctor sets
  `use_rsvd_ = false` and never copies `opts_.rsvd → use_rsvd_`. The
  0b9fccf class docstring (header :24) declares "only `rsvd` and
  `profile` are honoured", but `DMRG_GPU_OPT_RSVD=1` only writes
  `opts_.rsvd` via `opts_.load_from_env()` (impl :159), and the dispatch
  in `optimize_bond` reads `use_rsvd_` (impl :1628) — env var silently
  dropped. Same defect class as round-10 M-opt-rsvd-env, fixed in
  pdmrg-gpu-opt with `use_rsvd_ = opts_.rsvd;` at
  `pdmrg_gpu_opt_impl.h:205` and **never propagated to pdmrg-multi-gpu**
  by 8b7a68e or 0b9fccf. Technique-G miss; HIGH because round-12
  introduced the false docstring. One-line fix at impl :152.

## MEDIUMS — fix when convenient

- **pdmrg-multi-gpu: `n_recal` parameter missing from run()**
  `[pdmrg-multi-gpu: pdmrg_multi_gpu.h:53]` pdmrg-gpu/-opt both accept
  `int n_recal = 0` for periodic full-chain recalibration. multi-gpu
  drops it. Out-of-scope per the multi-device charter but the API gap
  should be documented alongside the existing "only rsvd and profile
  honoured" disclaimer.

- **pdmrg-multi-gpu: t_absorb_ / t_env_update_ are dead timers**
  `[pdmrg-multi-gpu: pdmrg_multi_gpu.h:210-211, 228-231]` Both
  PhaseTimer instances are init'd and reported, but never `.begin/.end`'d
  — output will always be 0.00 ms / 0 calls. 0b9fccf's commit message
  flagged "forward compatibility" but the panel prints them
  unconditionally. Print only when calls() > 0, or remove from report.

## NITS — cosmetic

- **pdmrg-multi-gpu: docstring sub-sentence about `set_use_davidson`**
  `[pdmrg-multi-gpu: pdmrg_multi_gpu.h:24-25]` reads "only `rsvd` and
  (implicitly via `set_use_davidson` in pdmrg-gpu-opt) `lanczos_graph`
  are honoured here" — but pdmrg-multi-gpu has neither
  `set_use_davidson` nor any Davidson path (it's Lanczos-only). The
  parenthetical describes -opt's setter, not multi-gpu's. Reword to
  drop the cross-reference or make the multi-gpu immunity explicit.

## FALSE POSITIVES VERIFIED

- **J1 lock**: `accurate_svd_gpu<Scalar>(...)` at `pdmrg_gpu_impl.h:2491`,
  `pdmrg_gpu_opt_impl.h:3367`, `pdmrg_multi_gpu_impl.h:2210`.
- **J2 lock in pdmrg-gpu-opt**: `use_davidson_ = true` at impl :204;
  ctor-time `if (opts_.lanczos_graph && use_davidson_)` gate at :220-226
  matches dmrg-gpu-opt:73 + dmrg2-gpu-opt:69. Round-12 H-opt-pdmrg fix
  verified present.
- **PDMRG-rules-2026-04-15 lock in pdmrg-multi-gpu**:
  `run(...n_warmup=1, n_polish=0)` header :53; warmup uses
  `sweep_*_full_1site` (impl :2396-2397); polish single-site gated on
  `n_polish > 0` (impl :2483-2495); two-site sweep_*_full removed (stub
  at :2021); test driver has `--polish` CLI (:377) defaulting to 0.
  All four rules satisfied. Round-12 CRITICAL verified fixed.
- **PointerModeGuard RAII in multi-gpu**: 4 RAII scopes at impl :1168,
  :1183, :1203, :1314; zero `rocblas_set_pointer_mode` direct calls.
- **Dead-buffer cleanup verified**: h_batch_{A,B,C}_pinned, h_rsvd_B,
  h_rsvd_U_small, d_boundary_staging, copy_boundary_mps_to_device — all
  gone; comment stub at impl :1144-1145 documents the removal.
- **API parity restored**: `initialize_mps_product` (:592),
  `initialize_mps_neel` (:609) added per 0b9fccf.
- **PhaseTimer panel wired**: init in ctor (:160), report in run
  epilogue (:2504), .begin/.end at apply_heff_two_site (:753/:814),
  lanczos (:1161/:1330), svd_split (:1343/:1475). The `if (di == 0)`
  gate is load-bearing — PhaseTimer's std::vector::push_back is not
  thread-safe, so single-device gating is a correctness fence, not
  just a perf choice.
- **H10-multi-WW-leak guard intact**: free-then-malloc pattern at
  pdmrg_multi_gpu_impl.h:710-711 preserved.
- **Legacy `gpu-rocm/pdmrg-gpu/src/accurate_svd.h`** gone; no surviving
  include.
- **Davidson workspace aliasing OK**: d_dav_work / d_dav_work2 sized
  `max(theta_size_max·b, max_sub²)`; residuals (n_new·dim ≤ b·dim) and
  H_proj (k² ≤ max_sub²) both fit.

## SUMMARY

J1 (Stoudenmire), J2 (Davidson default + lanczos_graph ctor gate), and
the PDMRG-rules-2026-04-15 lock now hold across all three pdmrg
variants — the round-12 CRITICAL + 3 HIGHs from 8b7a68e and the 4
mediums from 0b9fccf are all verified in place. One net-new HIGH
remains: pdmrg-multi-gpu's `use_rsvd_` is never bound to `opts_.rsvd`,
so the round-12 docstring claim that "only `rsvd` and `profile` are
honoured" is half a lie — `DMRG_GPU_OPT_RSVD=1` env will not enable
the RSVD path. Same defect class as round-10 M-opt-rsvd-env, missed by
both 8b7a68e and 0b9fccf in the propagation set. Two mediums (n_recal
API gap + dead forward-compat timers) and one nit (multi-gpu docstring
mentions a setter that doesn't exist on multi-gpu) round out the
report. Technique D skipped (no clangd / ROCm headers); A covers the
most important channel.
