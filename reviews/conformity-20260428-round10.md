# Full conformity review — 2026-04-28 (round-10, post round-9 fixes)

Pre-G1 GPU window gating audit. Six sub-reviews dispatched in parallel
against commit `db7dcdf` (post round-10 self-audit). Round-10 found
**2 net-new findings**; both fixed in commit `4d8924d`.

## Charter proof — sub-review status

| Sub-review | Status | Findings (net-new vs round-9) |
|---|---|---|
| vertical-review-dmrg     | OK | **0 critical, 0 high, 0 medium, 0 nit** |
| vertical-review-dmrg2    | OK | 0 critical, 0 high, 0 medium, 1 nit (cosmetic comment) |
| vertical-review-pdmrg    | OK | 0 critical, 0 high, 3 medium (dead host buffers in -opt) |
| horizontal-review-base   | OK | 0 critical, 0 high, 0 medium, 1 nit (`lanczos_use_1site_` foot-gun) |
| horizontal-review-gpu    | OK | **0 critical, 0 high, 0 medium, 0 nit** — first clean tier review |
| horizontal-review-opt    | OK | **1 net-new HIGH** (H10-multi-WW-leak), **1 net-new MEDIUM** (M-opt-rsvd-env) |

All six sub-reviews ran A-G in full.

## NET-NEW findings (both fixed in `4d8924d`)

### 1. H10-multi-WW-leak (HIGH, technique G) — pdmrg-multi-gpu

`pdmrg-multi-gpu/src/pdmrg_multi_gpu_impl.h:683` —
`hipMalloc(&devices_[k].d_WW[bond], …)` inside `precompute_fused_mpo`
without the `if (devices_[k].d_WW[bond]) hipFree(...)` guard. The
round-10 self-audit closed `set_mpo` in pdmrg-multi-gpu for `d_mpo`,
`d_W_left`, `d_W_right`, but stopped short of the `precompute_fused_mpo`
call site where `d_WW[bond]` is allocated.

**Same defect class** as round-9 H-new1 (M4-W) — the `set_mpo` fix
covered three buffers but not the WW fused-MPO buffer in the
companion function. This is the **lonely-fix pattern** technique G
exists to catch.

All single-host -gpu/-gpu-opt siblings DO have this guard. Net effect:
double-call to `set_mpo` would leak `d_WW[bond]` × L bonds × n_devices.

**Fix**: added the guard at `precompute_fused_mpo:683`.

*(found by: horizontal-review-opt)*

### 2. M-opt-rsvd-env (MEDIUM) — silent env-var ignore

All three -opt variants (dmrg-gpu-opt, dmrg2-gpu-opt, pdmrg-gpu-opt)
load `opts_.rsvd` from the `DMRG_GPU_OPT_RSVD` env var via
`opts_.load_from_env()` in the constructor, but never propagate it to
the `use_rsvd_` member that gates the actual RSVD code path.

Net effect: `DMRG_GPU_OPT_RSVD=1` was silently ignored at -opt tier.
The flag worked at -gpu tier (which reads `opts_.rsvd` directly).

**Fix**: added `use_rsvd_ = opts_.rsvd;` after `opts_.load_from_env()`
in all three -opt ctors.

*(found by: horizontal-review-opt)*

## CRITICALS

None.

## HIGHS

H10-multi-WW-leak (fixed).

## MEDIUMS

- **M-opt-rsvd-env** (fixed).
- **MED-pdmrg-opt-{1,2,3}**: dead host buffers in pdmrg-gpu-opt
  (`h_rsvd_B`, `h_rsvd_U_small`, `h_dav_V_copy`) and a leftover
  dead source file `pdmrg-gpu/src/accurate_svd.h`. All caught by
  technique A. Same dead-infrastructure class as round-9 MED-base-1.
  Cosmetic — defer to post-G1 cleanup.
- Pre-existing carry-overs: dmrg-gpu local DmrgPointerModeGuard,
  stale `// h_batch_*_pinned` comment, `set_quiet` no-op,
  H7-ext deferral.

## NITS

- vertical-review-dmrg2 cosmetic comment improvement on
  `dav_work_sz` documentation.
- horizontal-review-base N2: `lanczos_use_1site_` is a non-atomic
  member shared across `parallel_sweep` threads in pdmrg-gpu-base.
  Currently safe (only set on stream 0 outside parallel regions), but
  a foot-gun if the call graph changes. Promote to StreamWorkspace
  if any future refactor pulls single-site optimization into a
  per-segment path.

## FALSE POSITIVES VERIFIED

All previously-listed round-7, round-8, and round-9 fixes verified
intact at the cited file:lines. J1 Stoudenmire lock holds in all four
pdmrg variants. D_PAD R-env identity at `D_mpo_actual_-1`. Round-6
dual-stream env-update overlap intact. The CR-D1 / d_dav_work sizing
in dmrg-gpu-opt and dmrg2-gpu-opt verified by technique F regions
audit. The round-9 H-new1-pdmrg-opt fix verified.

## SUMMARY VERDICT

### Block GPU run? **NO** (after `4d8924d`).

Round-10 produced 2 net-new findings (1 HIGH + 1 MEDIUM) — both
caught by technique G (sibling fix-propagation) and both in the
extension/multi-GPU surface that prior rounds didn't cover at depth.

### Pattern across rounds — diminishing returns

| Round | Net-new findings | Methodology |
|---|---|---|
| 6 | 1 (dead stream_env_) | technique A informal |
| 7 | many (~20) | A-E formal |
| 8 | 2 (CR-D1, C-new1) | A-E mandatory |
| 9 | 4 (M4-W, H1-ext-gpu, pdmrg-opt dav_work, dead d_svd_work_) | A-G mandatory + self-audit catches 4 |
| 10 | 2 (H10-multi-WW, M-opt-rsvd-env) | A-G + self-audit catches 2 |

**Round-10 is the first round where two of six sub-reviews returned
zero net-new findings** (vertical-review-dmrg + horizontal-review-gpu).
The trend: methodology hardening + each round closing more propagation
gaps → fewer findings each round → eventually zero.

### Recommendation

Run round 11 to confirm the trend. If round 11 returns zero net-new
findings across all 6 sub-reviews → two consecutive clean rounds →
ready for G1.

The cost of one more round (6 sub-agents, ~5 min) is trivial compared
to a wasted GPU minute. The user has explicitly said GPU time is
expensive.

### What's left to fix even if round 11 finds nothing

The deferred mediums are all post-G1 cleanup, none affect the
benchmark numbers:
- pdmrg-gpu-opt block_davidson host syev (deferred per task #103)
- dmrg-gpu-opt apply_heff host pointer-tables (paper-excluded variant)
- H10 Lanczos α/β device-pointer-mode port
- H11 sparse-MPO Step-3 batch
- Various dead host buffers in pdmrg-gpu-opt
- Stale comments and unused `set_quiet` no-ops
