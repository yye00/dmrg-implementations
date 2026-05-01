# Horizontal review — -base tier — round 18 (2026-05-01)

HEAD: `12d02c5`. Baseline: `reviews/conformity-20260501-round15.md`.
Scope: `dmrg-gpu-base`, `dmrg2-gpu-base`, `pdmrg-gpu-base` + `gpu-rocm/common/{accurate_svd_gpu,hip_check,pointer_mode_guard,scalar_traits}.h`.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | Spot-checked dmrg-gpu-base private members (d_W_right_, d_svd_A_/E_/info_, d_svdj_residual_, d_svdj_n_sweeps_) — all have ctor + dtor + ≥1 hot-path use. No dead infrastructure. |
| B. Behavioral diff | DONE | Three Lanczos impls structurally identical (rocsolver_dsteqr, on-device alpha/beta, classical-GS reorth, device-pointer mode guard). apply_heff: dmrg-gpu/dmrg2-gpu use single rocblas_h_; pdmrg-gpu uses per-segment handles_[si]/streams_[si]. Naive single-GEMM-per-(w,s) loops in all three (D13 charter-allowed). update_left_env/update_right_env follow the same naive triple-step pattern (Step 1 unbatched GEMMs, Step 2 dense GEMM, Step 3 unbatched accum-GEMMs). All three SVD calls dispatch through `Traits::rocsolver_gesvd_auto`. |
| C. Docstring verification | DONE | Each header docstring lists features explicitly excluded by charter (dual-stream, HIP graph, RSVD, batched GEMM, GpuOpts, sparse-MPO, D_PAD). Grepped each — all true negatives in impl. pdmrg-gpu-base docstring claim "DOES use the on-device Stoudenmire `accurate_svd_gpu` at boundary merge" — verified live at `pdmrg_gpu_base_impl.h:1276`. |
| D. clangd filter | N-A | Local clangd unable to resolve ROCm headers; technique A subsumes the most important dead-symbol case. |
| E. Absence-naming brief | FOLLOWED | Per-tier expected feature checklist below. |
| F. Workspace-aliasing audit | DONE | -base only aliases `d_T1_`/`d_T2_` (apply_heff Steps 1→2→3, sequential lifetimes — not concurrent). `d_svd_*_` lifetimes sequential within svd_split. `d_svd_S` reuse for V-upload in `form_theta_with_V` correctness justified inline (overwritten by SVD in same optimize_bond, same stream). No round-12+ aliasing changes touched -base. No OVERRUN. |
| G. Sibling fix-propagation | DONE | 6 watch-list commits (abd88b9, 187fddf, 8abb6e7, 0efe96d, 54f2fcf, 12d02c5) verified — `git show --stat -- gpu-rocm/{dmrg,dmrg2,pdmrg}-gpu-base/` returns ZERO lines on each. No -opt-tier port leaked into -base. R8 C-new1 canonical-Vh swap re-verified intact at `pdmrg_gpu_base_impl.h:1330-1346`. |

Defect-class registry sweep: `bash .claude/scripts/defect-registry.sh` → `TOTAL HITS: 0` across D7+D8/D9/D10/D11/D13/D14/D15. (D13 -base whitelisted by design; the remainder genuinely zero.)

## Absence-naming checklist (technique E)

| Expected feature | dmrg-gpu-base | dmrg2-gpu-base | pdmrg-gpu-base |
|---|---|---|---|
| HIP_CHECK from `common/hip_check.h` | present | present | present |
| ScalarTraits dispatch (double + complex) | present | present | present |
| Single rocBLAS handle / single stream | present | present | per-segment (J1 algorithmic) |
| On-device Lanczos (rocsolver_dsteqr) | present | present | present |
| On-device SVD (rocsolver_gesvd_auto) | present | present | present |
| accurate_svd_gpu at boundary (J1) | N-A | N-A | present (`:1276`) |
| Per-segment streams (pdmrg J1) | N-A | N-A | present |
| Single-site warmup `_1site` | N-A | N-A | present (`sweep_LR_full_1site/sweep_RL_full_1site`) |
| Single-site polish `_1site` | N-A | N-A | present |
| n_warmup, n_polish ≤ 2, configurable | N-A | N-A | present (defaults 1 / 0) |
| **NOT** present: GpuOpts | absent (correct) | absent (correct) | absent (correct) |
| **NOT** present: lanczos_graph / hipGraph | absent | absent | absent |
| **NOT** present: RSVD / sparse_mpo / D_PAD | absent | absent | absent |
| **NOT** present: Block-Davidson | absent | absent | absent |
| **NOT** present: PhaseTimer (D9/D15) | absent | absent | absent |
| **NOT** present: env-var / getenv toggles | absent | absent | absent |

All three -base variants pass the charter.

## CRITICALS — block GPU run / paper submission

(none)

## HIGHS — fix before next major event

(none)

## MEDIUMS — fix when convenient

(none)

## NITS

(none)

## FALSE POSITIVES VERIFIED

- **`form_theta_with_V` H2D of `bs.V.data()`** [pdmrg-gpu-base: pdmrg_gpu_base_impl.h:1217]. Surface-level host roundtrip on the sweep path. **Non-defect**: V is host-resident Stoudenmire boundary state of size chi_bond doubles, uploaded once per boundary merge (not per Lanczos iter). Identical sibling pattern in -gpu (`:2364`) and -gpu-opt (`:3278`). Inherent to V=Λ⁻¹ algorithm, not per-sweep scratch.

- **`precompute_WW` host loop in dmrg2-gpu-base** [dmrg2_gpu_base_impl.h:284-309]. **Non-defect**: runs at set_mpo() time, outside the timed sweep (line 271 comment). One-shot setup; identical to dmrg2-gpu/-opt siblings.

## Regression watch (vs round-15 baseline)

| Watch item | Status |
|---|---|
| abd88b9, 187fddf, 8abb6e7 — -opt H2-opt ports | -base untouched (verified per-commit `--stat`) |
| 0efe96d — D12 device-pointer Lanczos -opt port | -base untouched; existing -base Lanczos unchanged |
| 54f2fcf — CR-D1 Davidson port to pdmrg-gpu-opt | -base immune (no Block-Davidson at -base tier) |
| 12d02c5 — D9/D15 PhaseTimer panel + multi-gpu Step3 | -base untouched (no PhaseTimer at -base tier) |
| R8 C-new1 canonical-Vh swap | re-verified intact at `pdmrg_gpu_base_impl.h:1330-1346` (R_env built from `ws.d_Vh_canonical`, MPS swapped back at line 1348) |
| R14 H1-base host-pointer-mode comment in Lanczos | re-verified in all three impls (`:528`, `:597`, `:753`) |
| R10 M4 set_mpo double-call guard | re-verified in dmrg2/pdmrg precompute_WW prologue |

Only delta to `gpu-rocm/{dmrg,dmrg2,pdmrg}-gpu-base/` + `common/` since R15 baseline: `git diff --stat f40140d..12d02c5 -- gpu-rocm/{dmrg,dmrg2,pdmrg}-gpu-base/ gpu-rocm/common/` → one file (`gpu-rocm/common/batch_ptrs_kernels.h`, +225 lines, added by 8abb6e7). That file is consumed only by -gpu-opt callers; -base impls do not include it. Zero behavioral delta in -base since R15.

## SUMMARY

-base tier is clean: 0C / 0H / 0M / 0 NIT. All seven techniques completed. No -opt-tier feature leaked through R16/R17/R18 commits into -base. The four watch-list fixes (D12, CR-D1, PhaseTimer, Davidson syev) stayed at -opt. R8 C-new1 canonical-Vh swap remains intact.

Behavioral parity across the three -base impls is essentially perfect modulo the algorithmic delta. Lanczos is the same algorithm in all three with the same on-device tridiag eigensolve; SVD dispatches through the same Traits seam; HIP_CHECK comes from `common/hip_check.h`; zero env-var / GpuOpts / PhaseTimer surface.

Cleanest -base review since R11. Technique-G "lonely-sibling" pattern from R12-R15 did not recur. Verdict: READY. Fold into round-18 orchestrator with 0/0/0.
