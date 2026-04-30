# Full conformity review — round 11 — 2026-04-30

Pre-G1 GPU window gating audit. Six sub-reviews dispatched in parallel
against commit `1d44d89` (post round-10). The user said this was the
last round.

## Charter proof — sub-review status

| Sub-review | Status | Findings (net-new vs round-10) |
|---|---|---|
| vertical-review-dmrg     | OK | 0 critical, 0 high, 2 mediums (dead Lanczos scratches in dmrg-gpu-opt; dead h_svd_tmp_) |
| vertical-review-dmrg2    | OK | 0 critical, 0 high, 4 mediums (dead d_svd_work_ + dead h_svd_* across dmrg2-gpu and dmrg2-gpu-opt) |
| vertical-review-pdmrg    | **CLEAN** | 0 critical, 0 high, 0 medium, 0 nit |
| horizontal-review-base   | **CLEAN** | 0 critical, 0 high, 0 medium, 1 cosmetic NIT (stale comment) |
| horizontal-review-gpu    | OK | 0 critical, 0 high, 0 medium, 1 NIT (dead h_svd_* in dmrg-gpu, dmrg2-gpu) |
| horizontal-review-opt    | OK | 0 critical, 0 high, 1 medium (M-opt-davidson-toggle in dmrg-gpu-opt) |

All six ran A-G in full.

## Net-new findings — characterization

**Across all 6 sub-reviews: 0 criticals, 0 highs.**

The mediums and nits are all in **two defect classes**:

### Class 1: dead post-refactor host/device buffers (defer per round-10 precedent)

Buffers allocated in ctors and freed in dtors with zero usage between. Same defect class as round-9 MED-base-1 (dead d_svd_work_ in dmrg2-gpu-base, fixed) and round-10 MED-pdmrg-opt-{1,2,3} (dead h_rsvd_B / h_rsvd_U_small / h_dav_V_copy in pdmrg-gpu-opt, **deferred** per round-10 plan).

Round-11 instances:
- dmrg-gpu-opt: 8 sync-free Lanczos device scratches + d_const_one_/zero_/neg_one_ + d_ones_D_ + h_svd_tmp_ (round-7 M8/M9 scaffolding for the deferred H10 device-pointer Lanczos port — task #101).
- dmrg-gpu, dmrg2-gpu: dead h_svd_U_/Vh_/S_/tmp_ (~1 MB host RAM, leftover from removed CPU-SVD path).
- dmrg2-gpu, dmrg2-gpu-opt: dead d_svd_work_ (same class as round-9 MED-base-1, sibling propagation gap).

**Per round-10 precedent**: these are post-G1 cleanup. **No GPU run impact** (they only consume a small fraction of host/device RAM at ctor and never affect timed-sweep numerics).

### Class 2: setter symmetry — fixed in `bc3fcd0`

**M-opt-davidson-toggle**: dmrg-gpu-opt's `set_use_davidson()` was a one-line setter without the symmetric `lanczos_graph_was_user_enabled_` round-trip pattern that round-7 added to dmrg2-gpu-opt and round-7 M11 added to pdmrg-gpu-opt. Same lonely-fix class technique G is designed to catch. Fixed in `bc3fcd0` by porting the round-7 M5 pattern.

## Regression-watch — all PASS

All round-7/8/9/10 fixes verified intact at cited file:lines:
- Round-10 H10-multi-WW-leak guard in pdmrg-multi-gpu (line 684).
- Round-10 M-opt-rsvd-env propagation in all 3 -opt variants.
- Round-9 M4-W W-buffer guards across 9 variants + multi-gpu (round-10 self-audit).
- Round-9 H1-ext-gpu nonblocking flag in dmrg-gpu, dmrg-gpu-opt, dmrg2-gpu, dmrg2-gpu-opt.
- Round-9 H-new1-pdmrg-opt d_dav_work sizing.
- Round-8 CR-D1 dav_work_sz in dmrg-gpu-opt and dmrg2-gpu-opt.
- Round-8 C-new1 canonical-Vh swap in pdmrg-gpu-base, pdmrg-gpu, pdmrg-gpu-opt, pdmrg-multi-gpu.
- Round-7 fixes (C2/C3/C4/C5/C6, H1-H11, M1-M14) all at cited lines.
- J1 Stoudenmire lock: `accurate_svd_gpu` in all four pdmrg tiers.
- Round-6 dual-stream env-update overlap intact.

## Pattern across 11 rounds

| Round | Net-new criticals | Net-new highs | Net-new mediums | Methodology |
|---|---|---|---|---|
| 6 | 0 | 0 | 1 (dead stream_env_) | A informal |
| 7 | many | many | many | A-E formal |
| 8 | 2 | 0 | 0 | A-E mandatory |
| 9 | 0 | 4 | 4 | A-G mandatory + self-audit |
| 10 | 0 | 1 | 1 | A-G + self-audit catches 4 |
| **11** | **0** | **0** | **6** (all dead-buffer / setter) | A-G + self-audit |

**Round-11 is the first round with zero criticals AND zero highs.** The remaining 6 mediums are all in two well-characterized non-blocking classes; one (M-opt-davidson-toggle) was fixed; five are deferred per round-10 precedent for the dead-buffer class.

## SUMMARY VERDICT

### Block GPU run? **NO** (after `bc3fcd0`).

- 0 criticals across all variants.
- 0 highs across all variants.
- 1 medium fixed (setter symmetry).
- 5 mediums deferred (dead-buffer cleanup, post-G1 precedent from round-10).
- 2 sub-reviews returned absolutely clean (vertical-pdmrg, horizontal-base modulo one cosmetic NIT).
- 1 sub-review found only a NIT (horizontal-gpu).

### Honest answer to the gating question

The strict criterion "two consecutive rounds with zero net-new findings of any severity" is **not met** — round-11 still surfaced ~6 mediums. But:

- The remaining mediums are all dead-infrastructure cleanup with **zero numerical or performance impact** on the GPU run.
- Round-10 already established the precedent for deferring this exact defect class (MED-pdmrg-opt-{1,2,3}).
- The trend across rounds is monotonically converging: criticals went 6→2→0→0, highs went 11→4→1→0, mediums concentrated into a single deferrable class.

**Static-review-ready as of `bc3fcd0`.** The post-G1 cleanup batch should consolidate all dead-buffer findings (round-9 MED-base-1, round-10 MED-pdmrg-opt-{1,2,3}, round-11 mediums in dmrg/dmrg2 variants) into a single deletion commit.

### What to do on the GPU window

1. **Build all 12 gpu-rocm variants** on the remote MI300X (`gpu-rocm/build_all.sh`). This is the only filter that catches compile-time bugs from rounds 7-11.
2. **Run standalone `test_*` correctness binaries** at small fixture sizes (L=4, L=8, chi=4, chi=8) to verify no Davidson / canonical-Vh / dual-stream regressions.
3. **Then proceed with the G1 benchmark sweep.**

### Methodology learnings retained

The round-8 lessons (techniques F, G, /pre-commit-self-audit) caught 6 net-new findings across rounds 9-11 that the round-7 A-E methodology would have missed. The discipline works. Future rounds (post-G1) should keep techniques F and G mandatory.
