# G1 deferred bugs — 2026-05-09

## D-G1-1: dmrg2-gpu-opt hangs at χ≥128

**Status**: Deferred. Excluded from --full via `VARIANT_SKIP=dmrg2-gpu-opt`
in `/g1-poll`'s smoke-clean trigger. User picks up next time online.

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
