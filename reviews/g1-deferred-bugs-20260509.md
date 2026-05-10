# G1 deferred bugs — 2026-05-09

## D-G1-1: dmrg2-gpu-opt hangs at χ≥128

**Status**: Fix proposed 2026-05-10 on branch
`claude/fix-dmrg2-gpu-opt-chi-hang`. Deferred from the in-flight G1
campaign via `VARIANT_SKIP=dmrg2-gpu-opt`; remove the skip after the
fix is verified at χ=128/256 on a fresh GPU window.

**Root cause (identified 2026-05-10 by static analysis)**: the
"eager RSVD allocation" block in
`gpu-rocm/dmrg2-gpu-opt/src/dmrg2_gpu_opt_impl.h:226-246` `hipFree`s
the four standard-SVD buffers `d_svd_{S,E,U,Vh}` and reallocates them
sized to `r_max = chi_max + RSVD_OVERSAMPLE_` (= chi_max + 10).
That's correct in single-site (`dmrg-gpu-opt`) where
`svd_max_k = chi_max`, so `r_max ≥ svd_max_k`. In two-site,
`svd_max_k = chi_max · d`, so for d=2 and chi_max ≥ 11 the realloc
**shrinks** the buffers. When `use_rsvd_=false` (the default — launcher
doesn't export `DMRG_GPU_OPT_RSVD=1`), the standard
`rocsolver_gesvd_auto` path writes `full_k = min(m,n) = chi_max·d`
singular values into a `chi_max+10` buffer. At χ=128: 256 svals into a
138-element buffer = 245 KB out-of-bounds writes per buffer (S, U, Vh,
E), corrupting whatever rocsolver/HIP allocated next. GPU drops to 0%
because the corrupted state takes a downstream call into a
non-progressing kernel launch / sync-with-no-work state, not because of
a Davidson-specific bug.

`χ=64` survives because the overrun (128 − 74 = 54 svals worth) lands
in slack rather than critical adjacent state.

`dmrg2-gpu` (the non-`-opt` sibling) has the same pattern but gates the
realloc on `if (opts_.rsvd)`; since the launcher doesn't set
`DMRG_GPU_OPT_RSVD=1`, that gate is false and the bug never fires there.

**Fix**: replace
```cpp
int r_max = chi_max_ + RSVD_OVERSAMPLE_;
```
with
```cpp
int r_max = std::max(chi_max_ + RSVD_OVERSAMPLE_, svd_max_k);
```
in both `dmrg2-gpu-opt/src/dmrg2_gpu_opt_impl.h` (always-on realloc) and
`dmrg2-gpu/src/dmrg2_gpu_impl.h` (rsvd-gated realloc, defensive
sibling fix). Memory delta: a few hundred KB on a 192 GB GPU.

**Verification plan**:
1. `cd gpu-rocm/dmrg2-gpu-opt && bash build_mi300x.sh`
2. `./build/dmrg2_gpu_opt 50 128 20` — must complete with sane E, GPU
   stays >50% utilized.
3. `./build/dmrg2_gpu_opt 16 256 15 --josephson` — must complete.
4. Re-run smoke and confirm 0 watchdog kills on dmrg2-gpu-opt.
5. Remove `VARIANT_SKIP=dmrg2-gpu-opt` from
   `.claude/commands/g1-poll.md` smoke-clean and chi512 mop-up
   triggers.

**Failure pattern (smoke run 20260509-1314)**:

5 watchdog KILLs in 20 minutes, all dmrg2-gpu-opt:

| time (UTC)        | model      | L  | χ   | sweeps | elapsed | gpu% |
|-------------------|------------|----|----|--------|---------|------|
| 2026-05-09T15:46  | heisenberg | 50  | 128 | 20    | 127s    | 0    |
| 2026-05-09T15:59  | josephson  | 16  | 128 | 20    | 126s    | 0    |
| 2026-05-09T16:02  | josephson  | 16  | 256 | 15    | 137s    | 0    |
| 2026-05-09T16:04  | josephson  | 32  | 128 | 20    | 132s    | 0    |
| 2026-05-09T16:06  | josephson  | 32  | 256 | 15    | 137s    | 0    |

Plus 3 configs returned `E=-0.0000000000` in <25s (zombie results from
killed runs that the launcher recorded as "ok" anyway):

- config 93: dmrg2-gpu-opt heisenberg L=50 χ=256 sw=15  (t=11s, E=0)
- config 96: dmrg2-gpu-opt heisenberg L=100 χ=256 sw=15 (t=22s, E=0)
- config 105: dmrg2-gpu-opt heisenberg L=50 χ=256 sw=15 (t=19s, E=0, 2nd run)

**Pattern**: GPU drops to 0% mid-run on χ≥128. χ=64 completes cleanly
(t≈10s for L=50, recorded with sane E values).

**Reproduction config**:
```bash
ssh hotaisle@<vm-ip>
cd ~/dmrg-implementations
gpu-rocm/dmrg2-gpu-opt/build/dmrg2_gpu_opt 50 128 20
# expect: GPU usage drops to 0% within ~30s of start, hangs until killed
```

**History**:
- 2026-05-08: First observed. Triggered the watchdog regex fix (commit
  `1d9665f` and surrounding) and hang_watcher.sh redesign. Watchdog now
  catches it; the underlying hang was assumed fixed by round-8 CR-D1
  (Davidson buffer overrun in dmrg-gpu-opt + dmrg2-gpu-opt).
- 2026-05-09: Watchdog catches confirm the hang **still occurs** at
  challenge sizes (χ≥128). Round-8 CR-D1 fix was incomplete or
  regressed.

**Suspected root cause**: Davidson eigensolver path on dmrg2-gpu-opt
has a state-dependent stall when the local Hamiltonian dimension grows
past a threshold (likely χ²·d² > 128²·4 = 65k). Possibilities:
1. rocsolver_dsyevd workspace miscalculation (similar to round-8 but
   in a different code path).
2. Stream synchronization missing between Davidson restart batch and
   the next iteration's GEMM.
3. Pointer-mode RAII regression.

**Action for user**:
1. Diff dmrg2-gpu-opt against dmrg-gpu-opt around the Davidson + WW
   precompute paths — dmrg-gpu-opt does NOT hang.
2. Run with `HIP_LAUNCH_BLOCKING=1` and `AMD_LOG_LEVEL=4` to identify
   the stalled call.
3. If the hang is in `block_davidson_eigensolver`, check whether the
   recent host-syev → rocsolver_dsyevd port (round-9 H6) propagated
   correctly to dmrg2-gpu-opt vs only landed in pdmrg-gpu-opt.

**Impact on G1 results**: dmrg2-gpu-opt produces NO usable data this
campaign. The remaining 8 variants (dmrg-gpu-{base,std,opt},
dmrg2-gpu-{base,std}, pdmrg-gpu-{base,std,opt}) cover the paper's
lead claims.
