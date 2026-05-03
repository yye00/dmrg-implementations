# Horizontal review — -gpu-opt tier — 2026-04-28 round-10

HEAD `db7dcdf` (round-10 self-audit fixes — H1 + M4-W remainders).
Scope: `gpu-rocm/{dmrg,dmrg2,pdmrg}-gpu-opt/src/*` plus matching -gpu siblings,
plus `gpu-rocm/common/*`.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | dual-stream + worker-pool symbols all live |
| B. Behavioral diff | DONE | hot-path divergences classified |
| C. Docstring verification | DONE | no claim/code drift |
| D. clangd filter | N-A | no ROCm headers locally |
| E. Absence-naming brief | FOLLOWED | -opt feature checklist all `present` |
| F. Workspace-aliasing audit | DONE | 6 shared buffers checked, 0 OVERRUN |
| G. Sibling fix-propagation | DONE | 4 recent fixes traced, 1 MISSING in pdmrg-multi-gpu |

---

## CRITICALS

None.

## HIGHS

### H10-multi-WW-leak — pdmrg-multi-gpu `precompute_fused_mpo` missing M4-W guard for `d_WW`

**Class**: round-9 M4-W / round-10 M4-W-multi propagation gap caught by
technique G.

Round-10 commit `db7dcdf` extended the M4-W fix to pdmrg-multi-gpu’s
`set_mpo` body, guarding `d_mpo[i]`, `d_W_left[i]`, `d_W_right[i]` with
`if-then-hipFree` before `hipMalloc`
([pdmrg_multi_gpu_impl.h:629,633,637]). But `set_mpo` calls
`precompute_fused_mpo(h_mpo_tensors)` at line 644, which in turn does
unconditional `hipMalloc(&devices_[k].d_WW[bond], …)` at line 683 —
**no guard.** All single-host -gpu / -gpu-opt siblings DO have the guard
on `d_WW_[bond]` (dmrg2-gpu impl line 651, dmrg2-gpu-opt line 544,
pdmrg-gpu line 869, pdmrg-gpu-opt line 705).

Same defect class as M4-W: a second `set_mpo()` call (e.g. swapping the
Hamiltonian on a hot multi-GPU object) leaks one `d_WW[bond]` allocation
per device per bond. Severity HIGH (matches the original M4-W).

Out of strict -opt scope but the round-10 commit explicitly extended the
fix into pdmrg-multi-gpu's `set_mpo` and stopped short of
`precompute_fused_mpo` — exactly the kind of "lonely fix" that technique
G is meant to catch.

**Site**: `gpu-rocm/pdmrg-multi-gpu/src/pdmrg_multi_gpu_impl.h:683`.
**Fix**: insert `if (devices_[k].d_WW[bond]) HIP_CHECK(hipFree(devices_[k].d_WW[bond]));`
before line 683.

## MEDIUMS

### M-opt-rsvd-env — `opts_.rsvd` env var silently ignored in all three -opt variants

dmrg-gpu reads `opts_.rsvd` directly (impl lines 322, 333, 1260) so the
shell `DMRG_GPU_OPT_RSVD=1` path works there. All three -opt variants
load `opts_.rsvd` via `opts_.load_from_env()` and then never consult it
— rsvd in -opt is gated only on the `use_rsvd_` member, which the public
`set_rsvd()` setter flips and which the ctor initializes to `false`
(dmrg-gpu-opt impl line 233 already comments around the init-vs-setter
issue, but doesn’t propagate the env var). Net effect: a user who runs
`DMRG_GPU_OPT_RSVD=1 ./test_dmrg_gpu_opt …` gets the print line saying
rsvd is on but the run never enters the rsvd path.

Severity MEDIUM (parity gap, not correctness).

**Sites**:
- `dmrg-gpu-opt/src/dmrg_gpu_opt_impl.h:1278` (only `use_rsvd_` consulted)
- `dmrg2-gpu-opt/src/dmrg2_gpu_opt_impl.h` (same)
- `pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h:205,2260,3223` (init false, no
  env propagation; only `use_rsvd_` consulted)

**Fix**: in each -opt ctor, after `opts_.load_from_env()`, add
`use_rsvd_ = opts_.rsvd;` (and similarly initialize from the env-loaded
`opts_` for any other -opt-specific member that mirrors a -gpu env flag).

## NITS

None new — round-9 carryovers (stale comments, no-op `set_quiet`)
unchanged.

## FALSE POSITIVES VERIFIED

### Watch-list verification

| Watch item | Status | Evidence |
|---|---|---|
| Round-9 H-new1 — pdmrg-gpu-opt `d_dav_work` `max(theta_max*b, max_sub²)` | INTACT | `pdmrg_gpu_opt_impl.h:256-262`. d_dav_work hosts H_proj (k², ≤ max_sub²) sequentially with residuals (b·dim) + W; d_dav_work2 hosts eigvecs/overlap, both ≤ max_sub². Allocation covers both. |
| Round-9 M4-W (W-family guards across 9 variants) — three -opt | INTACT | dmrg-gpu-opt L511,542,546,579,585. dmrg2-gpu-opt L461,492,496,544,576,582. pdmrg-gpu-opt L577,607,611,642,648,705,735,741. Plus all -gpu siblings carry the guard (J2 propagation OK). |
| Round-9 H1-ext-gpu — `hipStreamCreateWithFlags(hipStreamNonBlocking)` | INTACT | dmrg-gpu-opt L105-106, dmrg2-gpu-opt L105-106, dmrg-gpu L200-201, dmrg2-gpu L198,201, pdmrg-gpu L288, pdmrg-gpu-opt L125,138. (Round-10 extended to base + radam/rlbfgs/multi.) |
| Round-8 CR-D1 — `dav_work_sz = max(theta_max*b + max_sub*b, max_sub²)` | INTACT | dmrg-gpu-opt L305-310, dmrg2-gpu-opt L265-270. Concurrent residuals + overlap on d_dav_work_; covers round-7 H6 syev port aliasing. |
| Round-7 fixes (C2/C4/C5/C6, H1/H2/H6/H8, M5/M7/M8/M9) | INTACT | sampled at cited sites; no regression. |

### Workspace-aliasing audit (technique F)

| Buffer | Variant | Regions | Lifetime | Required size | Allocated size | Verdict |
|---|---|---|---|---|---|---|
| `d_dav_work_` | dmrg-gpu-opt | residuals (b·dim) + overlap (k·n_new ≤ max_sub·b); restart X_keep (dim·keep ≤ b·dim) | concurrent inner; sequential restart | `b·theta_max + max_sub·b` | `max(b·theta_max + max_sub·b, max_sub²)` | OK |
| `d_dav_work2_` | dmrg-gpu-opt | H_proj/eigvecs (k², ≤ max_sub²); restart X_keep target overwrites residuals on d_dav_work_ | sequential | `max_sub²` | same as work_ | OK |
| `d_dav_work_` | dmrg2-gpu-opt | residuals (b·dim) + overlap (k·n_new); restart writes (dim·keep) | concurrent | `b·theta_max + max_sub·b` | same formula | OK |
| `d_dav_work2_` | dmrg2-gpu-opt | H_proj/eigvecs (k², ≤ max_sub²) | sequential, never holds overlap | `max_sub²` | same formula | OK |
| `ws.d_dav_work` | pdmrg-gpu-opt | H_proj (k²) → host-copied, then residuals (b·dim), then W; restart X_keep (dim·keep) | sequential — H_proj copied to host BEFORE residual writes | `max(b·theta_max, max_sub²)` | `max(b·theta_max, max_sub²)` | OK |
| `ws.d_dav_work2` | pdmrg-gpu-opt | eigvecs uploaded (k²) → x0 GEMV (line 1612), then overlap k·n_new (line 1682) trampling eigvecs; restart re-uploads eigvecs (line 1732) | sequential — eigvecs dead before overlap write | `max_sub²` | same formula | OK |
| `d_T1_/d_T2_` | dmrg-gpu-opt / dmrg2-gpu-opt | apply_heff Step1 V / Step2 U | sequential, single role each | `D·d·chi²` (1site) or `D·d²·chi²` (2site) | matches `t_max` | OK |
| `d_batch_*_` vs `d_batch_*_env_` | dmrg-gpu-opt / dmrg2-gpu-opt | Step-1 batched-GEMM pointer tables | concurrent (main vs env stream) | distinct allocations per stream | distinct | OK |

### Sibling fix-propagation (technique G)

| Recent fix | dmrg-gpu-opt | dmrg2-gpu-opt | pdmrg-gpu-opt | -gpu siblings | -base siblings | radam/rlbfgs | pdmrg-multi-gpu |
|---|---|---|---|---|---|---|---|
| H1 nonblocking stream flag | fixed | fixed | fixed | fixed | fixed (round-10) | fixed (round-10) | fixed (round-9) |
| M4-W set_mpo guards | fixed | fixed | fixed | fixed | fixed | n/a | **partial — `precompute_fused_mpo` `d_WW[bond]` unguarded** |
| Round-9 H-new1 d_dav_work sizing | fixed (round-8 CR-D1 already covered) | fixed (CR-D1) | fixed (round-9) | n/a (no Davidson in -gpu) | n/a | n/a (no Davidson) | n/a (multi has its own davidson? — out of scope) |
| Round-7 C2 syev on-device | fixed | fixed | **NOT yet ported** (round-9 deferred) | n/a | n/a | n/a | n/a |

The H10-multi-WW-leak finding above is the lone MISSING entry; one row of
the table — exactly the failure mode technique G is meant to catch.

### Other verifications (technique B, C, E)

- Hot-path structural diff (technique B): all three -opt have
  Davidson-default + Lanczos-fallback dispatch
  (`use_davidson_ ? block_davidson_eigensolver(...) : lanczos_eigensolver(...)`)
  in their respective `optimize_*` functions; tiny-system fallback
  (`dim ≤ 2*b → lanczos`) present in all three. Dual-stream pattern
  (`event_canon_ready_` → wait → `update_*_env` → `event_env_done_` →
  drain at sweep end) intact in dmrg-gpu-opt L1851-1876 and
  dmrg2-gpu-opt L1760-1786. pdmrg-gpu-opt uses the per-segment +
  worker-pool pattern instead — symbol scan confirms all worker-pool
  members are consumed (n_workers_=8 hits, worker_streams_=7,
  worker_done_events_=8, step_done_events_=7).
- Docstring verification (technique C): no class-level claim contradicts
  code at the cited sites in any of the three -opt headers.
- Absence-naming (technique E): every -opt expected feature
  (pad_mfma16, chi_max_user_, Block-Davidson, batched Step-3, public
  setters, worker-pool / batched-sweep / Chebyshev for pdmrg) is
  `present` and live. `use_chebyshev_` consulted at impl line 2251 and
  `use_batched_sweep_` consulted at impl L3437,3494,3507.

## SUMMARY

Round-10 head `db7dcdf` is **clean for the three -opt variants in the
strict horizontal scope** — all four round-9 watch-list items
(H-new1-pdmrg-opt, M4-W, H1-ext-gpu, CR-D1) verified intact, and
techniques A-G ran in full with no -opt-internal CRITICAL or HIGH
finding.

The one HIGH new this round is a sibling-propagation gap caught by
technique G in pdmrg-multi-gpu (`d_WW[bond]` in `precompute_fused_mpo`
unguarded — same defect class as round-9 M4-W and round-10 M4-W-multi
that the round-10 self-audit caught everywhere except this one
function). The MEDIUM is a longstanding rsvd env-var/setter divergence
across all three -opt variants — does not affect default-path
correctness.

**Top action before MI300X G1 window**: add the M4-W guard to
`pdmrg-multi-gpu/src/pdmrg_multi_gpu_impl.h:683` (one-line `if-hipFree`
ahead of the existing `hipMalloc(&devices_[k].d_WW[bond], …)`). The
RSVD env-var alignment can wait — it is benign on the default path.
