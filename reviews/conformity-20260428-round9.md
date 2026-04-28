# Full conformity review — 2026-04-28 (round-9, post round-8 self-audit)

Pre-G1 GPU window gating audit. Six sub-reviews dispatched in parallel
against commit `85492c9` (post round-8 self-audit). Round-9 found
**4 net-new findings** that the prior 8 rounds had missed; all fixed
in commit `d252907`.

## Charter proof — sub-review status

| Sub-review | Status | Findings (net-new vs round-8) |
|---|---|---|
| vertical-review-dmrg     | OK | 0 critical, **1 new HIGH** (M4-W incomplete W-buffer guard), pre-existing carryovers |
| vertical-review-dmrg2    | OK | 0 critical, 0 new highs (CR-D1 fix verified) |
| vertical-review-pdmrg    | OK | 0 critical, 0 new highs (all round-8 fixes verified intact) |
| horizontal-review-base   | OK | 0 critical, 0 high, **1 new MEDIUM** (dead d_svd_work_), 1 new MEDIUM (M4-W parallel) |
| horizontal-review-gpu    | OK | 0 critical, **1 new HIGH** (H1-ext-gpu nonblocking flag missing) |
| horizontal-review-opt    | OK | 0 critical, **1 new HIGH** (pdmrg-gpu-opt d_dav_work undersized) |

All six sub-reviews ran A-G in full. Techniques F (workspace-aliasing)
and G (sibling fix-propagation) are now mandatory after round-8.

## NET-NEW findings — all fixed in commit `d252907`

### 1. M4-W (HIGH, propagation gap) — caught by technique G

The round-7 M4 + round-8 M4-ext fix guarded only `d_mpo_tensors_[i]` in
`set_mpo()`. The same `set_mpo` body unconditionally `hipMalloc`s
`d_W_left_`, `d_W_right_`, `d_WL_nnz_*`, `d_WW_*`, `d_WW_nnz_*` —
identical defect class, leak on double-call. Affected ALL 9 variants
(dmrg / dmrg2 / pdmrg × base / gpu / gpu-opt).

**Fix**: applied `if (d_X_[i]) hipFree(d_X_[i]);` before each hipMalloc
across all 9 variants for every W-family buffer.

*(found by: vertical-review-dmrg + horizontal-review-base)*

### 2. H1-ext-gpu (HIGH, propagation gap) — caught by technique G

dmrg-gpu and dmrg2-gpu created `stream_` and `stream_env_` with bare
`hipStreamCreate` (default = blocking). Round-7 H1 fixed pdmrg-gpu-opt
streams; round-8 self-audit added pdmrg-multi-gpu. The dual-stream-pair
siblings (4 variants) were never updated.

**Fix**: switched all 4 affected variants (dmrg-gpu, dmrg-gpu-opt,
dmrg2-gpu, dmrg2-gpu-opt) to
`hipStreamCreateWithFlags(hipStreamNonBlocking)`.

*(found by: horizontal-review-gpu)*

### 3. H-new1-pdmrg-opt (HIGH, workspace-aliasing) — caught by technique F

pdmrg-gpu-opt per-StreamWorkspace `d_dav_work` / `d_dav_work2` sized at
`theta_size_max * b`. Inner Rayleigh-Ritz writes `H_proj` (k×k, k ≤
max_sub) into the same buffer; required size = `max(theta_max*b,
max_sub²)`. Underrun on smoke tests with `theta_size_max < 256`. Same
defect class as round-8 CR-D1 but sequential rather than concurrent
regions.

**Fix**: bumped allocation in `allocate_stream_workspaces` to the max.

*(found by: horizontal-review-opt)*

### 4. MED-base-1 (MEDIUM, dead infrastructure) — caught by technique A

dmrg2-gpu-base allocates and frees `d_svd_work_` but the two-site
`svd_split` writes scaled rows of Vh and scaled cols of U directly into
the destination MPS tensors via `scale_rows_by_diag_kernel` /
`scale_cols_by_diag_kernel` — never touches `d_svd_work_`. Same dead-
infrastructure pattern as round-7 dmrg2-gpu dead-stream miss.

**Fix**: removed declaration, alloc, free.

*(found by: horizontal-review-base)*

## CRITICALS

None. All six sub-reviews returned 0 criticals.

## HIGHS (all fixed in `d252907`)

See net-new findings above. All three of:
- M4-W (W-buffer guards across 9 variants)
- H1-ext-gpu (nonblocking flag in 4 dual-stream variants)
- H-new1-pdmrg-opt (d_dav_work sizing)

## MEDIUMS (carry-over + 1 new fixed)

- MED-base-1 (dead d_svd_work_) — fixed.
- M-carry: dmrg-gpu local DmrgPointerModeGuard not migrated to
  common/. Pre-existing M1-ext deferral.
- M-carry: pdmrg-gpu-opt block_davidson host syev not yet ported
  (deferred per task #103).
- M-carry: stale `// h_batch_*_pinned` comment in pdmrg-gpu impl.
- M-carry: `set_quiet(bool)` no-op across all -base variants.
- M-carry: pdmrg-gpu-opt H7 raw `set_pointer_mode` toggles in lanczos
  (deferred per round-8 H7-ext).

## NITS

- pdmrg-gpu loop-bound style inconsistency at parallel-segment join.
- Stale comments noted by sub-reviewers.

## FALSE POSITIVES VERIFIED (preserved to avoid re-discovery)

- All round-7 fixes (C2/C3/C4/C5/C6, H1/H2/H3/H4/H7/H8/H9, M1-M14) at
  cited file:lines.
- Round-8 fixes (CR-D1 dav_work_sz, C-new1 canonical-Vh swap) intact.
- Round-8 self-audit fixes (M4-ext for d_mpo_tensors_, C-new1-ext for
  pdmrg-multi-gpu, H1-ext for pdmrg-multi-gpu, H2-ext for
  pdmrg-multi-gpu) intact.
- pdmrg J1 lock: `accurate_svd_gpu` confirmed in all four pdmrg
  tiers (base, gpu, opt, multi-gpu).
- D_PAD R-env identity slot at `D_mpo_actual_-1` across all variants.
- Round-6 dual-stream env-update intact in dmrg-gpu, dmrg2-gpu,
  dmrg-gpu-opt, dmrg2-gpu-opt.

## SUMMARY VERDICT

### Block GPU run? **NO** (after `d252907`).

All four net-new round-9 findings are fixed. The orchestrator with
techniques F and G newly enforced caught defects that the prior 8
rounds (each of which said "ready") missed:

- **F (workspace-aliasing)** caught the pdmrg-gpu-opt d_dav_work
  underrun — same class as round-8 CR-D1 in a different variant.
  Without F, smoke tests at small chi would have corrupted memory
  on the GPU run.
- **G (sibling fix-propagation)** caught both the M4-W buffer-set
  propagation gap (one fix → many siblings still vulnerable) and
  the H1 nonblocking-flag propagation gap. Without G, the dual-
  stream env-update overlap in dmrg-gpu / dmrg2-gpu would have run
  with implicit blocking against the legacy null stream.

### What changed since the round-8 baseline

The methodology update (commit `f5a0e61`) added F and G as mandatory
techniques. Round-9 is the first full orchestrator run with all
seven techniques enforced. Result: 4 new findings the prior
methodology missed. **This is exactly the failure mode F and G were
designed to catch.**

### Pattern across rounds

| Round | "Ready" claimed? | Net-new found | Cause |
|---|---|---|---|
| 6 | yes | 1 (dead stream_env_ in dmrg2-gpu) | A wasn't formal yet |
| 7 | yes | many | first formal A-E run |
| 8 | yes | 2 (CR-D1 + C-new1) | F, G missing |
| 9 | yes | 4 (M4-W + H1-ext-gpu + pdmrg-opt dav_work + dead d_svd_work_) | F, G now mandatory; pre-commit-self-audit ran (caught 4 more before this) |

The trend is the methodology improving. The pre-commit-self-audit
caught 4 propagation gaps BEFORE this orchestrator ran (saved a
round). The orchestrator caught 4 more that the self-audit missed
(F-class buffer sizing in pdmrg-opt, M4-W defect class extension,
dead-infra pattern). All fixed.

### Top-3 actions before MI300X G1 window

All three already done in `d252907`:

1. ✓ Fix the 4 round-9 findings.
2. ✓ Apply M4-W guard pattern uniformly across 9 variants.
3. ✓ Switch all dual-stream variants to nonblocking flag.

### Recommended next step

**Compile + run standalone `test_*` correctness binaries on the remote
MI300X BEFORE starting the G1 benchmark sweep.** That catches
compile-time bugs from round-7/8/9 changes; small-fixture tests catch
correctness regressions. Without compile-test access locally, that's
the only remaining filter.

### Methodology learnings remaining

The fact that round-9 found 4 things after round-8 said "clean"
suggests the seven techniques (A-G) are necessary but the orchestrator
may still need to be run iteratively until two consecutive rounds
produce no net-new findings. **Recommend: re-run orchestrator after
the GPU compile/test passes; if it surfaces nothing, then proceed
with G1.**
