# Round 3 Generative Refinement — gen_05

**Slice**: LDS layout and memory staging for the persistent Lanczos kernel
**Target**: MI300X (gfx942 / CDNA3), 64 KB LDS per CU, 32 banks × 4 B, 4 MB
L2 per XCD (shared across 38 CUs), 256 MB Infinity Cache (GPU-wide)
**Depends on**: Round 2 R2-3 (`research_C_persistent_lanczos.md`)

## 1. Byte budgets per envelope (FP64 throughout)

`theta` size = `chi_L · d² · chi_R · 8 B`. For the two-site fused case we
use `theta[chi, d², chi] = theta[chi, d, d, chi]`. `WW = W_L ⊗ W_R` is
`D² · d⁴ · 8 B` where `D` is the MPO bond (`D=5` Heisenberg, `D≈6`
Josephson). `L_env`, `R_env` are `chi · D · chi · 8 B`. Krylov basis is
`m_max = 20` vectors, each same shape as `theta`.

### chi = 16, d = 2 ("full-basis-in-LDS" envelope)

| Buffer            | Formula             | Bytes   | Running |
|-------------------|---------------------|---------|---------|
| `theta_lds`       | 16·4·16·8           |  8 192  |  8 192  |
| `q_curr` (Lanczos)| 16·4·16·8           |  8 192  | 16 384  |
| `w_scratch`       | 16·4·16·8           |  8 192  | 24 576  |
| Krylov basis v[0..19] | 20·8 192        | 163 840 | **overflow** |

Even at chi=16 the full 20-vector basis (160 KB) does not fit. **Envelope
correction**: we keep `theta`, `q_curr`, `w_scratch`, plus the **four most
recent basis vectors** in a ring buffer for re-orthogonalization against
the immediate past (CGS-2 practice keeps β*q_{k-1} for the 3-term
recurrence and one extra for reorthogonalization); the rest spill to L2.

| Buffer                    | Bytes  | Running |
|---------------------------|--------|---------|
| `theta_lds`               |  8 192 |  8 192  |
| `q_curr`                  |  8 192 | 16 384  |
| `q_prev`                  |  8 192 | 24 576  |
| `w_scratch`               |  8 192 | 32 768  |
| basis ring (2 extra)      | 16 384 | 49 152  |
| `L_env` (16·5·16·8)       |  6 400 | 55 552  |
| `R_env` (16·5·16·8)       |  6 400 | 61 952  |
| `WW` (25·16·8)            |  3 200 | **overflow by 1088** |

Resolution: `WW` is tiny (3.2 KB) and is **read-shared across all Lanczos
iterations**, so place it in a constant pool in scalar L1 (sgpr path via
`s_load_dwordx4`) and not in LDS. Final chi=16 budget: **55 552 B / 65 536**
— 15% headroom for Lanczos α/β scalars, the 20×20 tridiagonal, and MFMA
staging registers. Full basis (20 vectors, 160 KB) lives in L2, which is
still under 5% of one XCD's 4 MB L2.

### chi = 32, d = 2 (basis to L2, hot working set in LDS)

| Buffer          | Formula        | Bytes  | Running |
|-----------------|----------------|--------|---------|
| `theta_lds`     | 32·4·32·8      | 32 768 | 32 768  |
| `q_curr`        | 32·4·32·8      | 32 768 | **overflow** |

Two 32 KB buffers already eat 100% of LDS. **We tile theta** across LDS:
keep a 32×2×32 slab (16 KB) resident, stream the other slab from L2 each
microstep. Or, equivalently, factor `theta[chi_L, d, d, chi_R]` along the
physical index and keep one physical leg's slice in LDS at a time. Updated
budget:

| Buffer                              | Bytes  | Running |
|-------------------------------------|--------|---------|
| `theta_slab` (chi·d·chi, 1 phys leg)| 16 384 | 16 384  |
| `q_curr_slab`                       | 16 384 | 32 768  |
| `w_scratch_slab`                    | 16 384 | 49 152  |
| `L_env` (32·5·32·8)                 | 12 800 | **overflow** |

Fallback: `L_env` and `R_env` go to L2 (both fit in 4 MB L2 trivially,
12.8 KB each), accessed through prefetched 512 B cache lines. Final LDS
resident set:

| Buffer                      | Bytes  | Running |
|-----------------------------|--------|---------|
| `theta_slab` (2 phys legs)  | 32 768 | 32 768  |
| `q_curr_slab`               | 16 384 | 49 152  |
| `w_scratch_slab`            | 16 384 | 65 536  |

Exactly full. The 20-vector Krylov basis (20·32 KB = 640 KB) lives in L2
(16% of one XCD's L2). `L_env`, `R_env`, `WW` all L2-resident. No
prefetch slots in LDS — we double-buffer `w_scratch_slab` into the
`theta_slab` region as soon as that slab has been consumed.

### chi = 48, d = 2 (minimal LDS, everything to L2)

`theta = 48·4·48·8 = 73 728 B` — already exceeds LDS alone. We must tile.

| Buffer                              | Bytes  | Running |
|-------------------------------------|--------|---------|
| `theta_slab` (chi·1·chi double-buf) | 36 864 | 36 864  |
| `q_curr_slab` double-buf            | 18 432 | 55 296  |
| `T_ij` tridiag scratch (20×20)      |  3 200 | 58 496  |
| CGS-2 reorthogonalization dots      |    640 | 59 136  |
| MFMA accumulator staging            |  2 048 | 61 184  |

**Total 61 184 / 65 536**, 4.3 KB headroom. `L_env`, `R_env`, `WW`, full
basis all in L2. Basis is 20·73 728 = 1.44 MB, which is **36% of a single
XCD's 4 MB L2** — safe, but now the per-bond Lanczos must not be
interleaved with SVD or env-update kernels in the same sweep, or they
will pollute L2. The persistent-per-bond dispatch model guarantees this.

**L2 capacity confirmation**: CDNA3 has 8 XCDs; each XCD has its own
**4 MB L2** shared across its 38 CUs (not "4 MB per CU pair" — that
phrasing is incorrect for CDNA3). The 256 MB Infinity Cache sits below L2
as a memory-side victim cache, GPU-wide. At chi=48 the 1.44 MB basis
comfortably fits in L2; worst case a CU whose L2 is evicting will refetch
from Infinity Cache at ~218 ns, not HBM at ~400 ns.

## 2. Bank-conflict map (32 banks × 4 B, FP64 = 2 adjacent banks)

Each FP64 load hits 2 adjacent 4 B banks. A wave of 64 lanes issuing
concurrent FP64 loads hits 128 banks worth of traffic across 4 cycles.
**Conflict-free rule for FP64**: lane `l` must access address
`base + l · stride · 8` with `stride` odd, OR with `stride` = 1 and
`base % 128 == 0`.

### Explicit LDS offset table (chi=32 slab envelope)

```
offset   size   buffer            alignment
------   ----   ----------------  ---------
0x0000   8192   theta_slab_A      128-byte (bank 0)
0x2000   8192   theta_slab_B      128-byte (bank 0)   [double-buf pair]
0x4000   8192   q_curr_slab       128-byte (bank 0)
0x6000    128   α/β scalars       64-byte
0x6080   8192   w_scratch_slab    128-byte (bank 0)
0x8080    384   T_ij (20×20 tri)  64-byte
0x8200    512   reduction scratch 64-byte
0x8400   ...    free (MFMA stg)
```

`theta_slab_A` and `q_curr_slab` both start on bank 0, but they are
**accessed in orthogonal phases**: `theta` is read during step-1 GEMM,
`q_curr` is written during step-3 store-back, never overlapping. The
problematic pair is `theta_slab` vs `WW` broadcast reads during step-2.
`WW` lives in sgpr/constant pool (not LDS), so no conflict.

During step-1 (`L_env × theta` producing `temp1`), 64 lanes of a wave
load consecutive `theta[i, :, :]` slices; stride along the `d=2` axis
means lanes stride by 16 bytes = 4 banks, periodic after 8 lanes, so we
get **8-way bank conflict**. Fix: add a 4-byte pad row every 32 doubles,
i.e. allocate `theta_slab` as `32 × 33 × 8 B` instead of `32 × 32 × 8 B`.
Cost: 256 extra bytes, eliminates the 8-way conflict (standard CK-tile
XOR-shift trick also works but the pad is simpler for a hand-written
kernel).

## 3. Double-buffering for L2→LDS streaming

`__builtin_amdgcn_global_load_lds` is the relevant intrinsic (available
since ROCm 6.x on gfx942) — it issues an async `global_load` with
direct-to-LDS destination, bypassing VGPR staging. While the current
slab is being consumed by `v_mfma_f64_16x16x4f64`, we issue the next
slab's global_load into the opposing half of the double buffer. The
issuer waits on `s_waitcnt vmcnt(0)` at the slab boundary. This gives us
compute-memory overlap with zero VGPR pressure cost.

Non-temporal hint: the spilled Krylov basis uses `nt=0 slc=0 glc=0` to
bias toward L2 retention; the one-shot `theta` load uses `nt=1` because
it is consumed once per Lanczos iteration and then rewritten.

## 4. Register pressure and MFMA scheduling

`v_mfma_f64_16x16x4f64` per issue: 4 A-regs + 4 B-regs + 16 C-regs (each
is a VGPR pair for FP64) = **48 VGPRs per in-flight MFMA**. gfx942 has
512 VGPRs per SIMD lane shared by resident waves. At single-wave
occupancy (512 VGPR budget) we can hold **up to 10 concurrent MFMAs**
before spill — in practice we target **4 concurrent** (192 VGPR for
MFMA, ~150 VGPR for address arithmetic / reductions / Lanczos scalars,
~170 VGPR slack). 4 concurrent MFMAs are enough to saturate the dual
MFMA pipes on a CDNA3 SIMD through back-to-back issue.

## 5. Occupancy and mapping

**Waves per CU = 4 (one workgroup of 256 threads)**. Rationale:

- LDS = ~56–65 KB → exactly 1 workgroup/CU regardless of wave count.
- Single-wave persistent (64 threads) would be lowest launch overhead
  but forces serialization of the 4 MFMA pipes and leaves 3 SIMDs idle.
- 4 waves (one per SIMD) give one MFMA pipe per SIMD and halve the
  intra-workgroup reduction latency (`__syncthreads` across 256 vs 64
  threads is ~3 cycles slower, negligible).

**Mapping**: **one workgroup per `apply_heff` call**. For serial two-site
DMRG, that is one workgroup total per matvec, launched on one CU.
Parallel Stoudenmire segments dispatch one workgroup per segment across
distinct CUs, up to the number of CUs in one XCD (38). We do NOT map WG
per site — sites are processed serially because they share `L_env`
updates.

## 6. Measurable prediction

At chi=32 d=2 Heisenberg, the current stream-of-rocBLAS-gemms matvec
measures **~48 µs/matvec** end-to-end (15 iter sweep × 48 µs = 720 µs
per bond, matching the "current 700–1500 µs/bond" from Research C).
Compute floor at chi=32: one matvec = ~1.6 MFLOPs, MI300X FP64 MFMA
throughput = 81.7 TF, so **~20 ns compute**. LDS bandwidth floor for
`theta_slab` streaming: 64 KB at 17 TB/s aggregate LDS = **~4 ns**.
Realistic floor including reductions and CGS-2 reorthogonalization
against the L2-resident basis: **~1.5 µs/matvec**.

**Prediction**: persistent Lanczos at chi=32 completes 1 matvec in
**≤ 2 µs** (2000 ns), vs current stream-of-gemms at **~48 µs**
(48 000 ns). Per-bond cost drops from ~720 µs to ~30 µs (20 Lanczos iter
× 1.5 µs), a **24× speedup in the Lanczos portion** at chi=32. At
chi=16 the speedup widens to ~40× because launch overhead dominates
even more of the current path.

**Validation gate**: run the kernel on a single bond, measure with
`rocprof --stats`, require matvec latency < 2.5 µs and energy delta
from rocBLAS baseline < 1e-12.

---

### 3-sentence summary

At chi=16 the byte budget (theta 8 KB + q_curr/prev/w 24 KB + envs 12.8 KB
+ 4-vec ring 16 KB = ~55.5 KB) fits in 64 KB LDS with WW held in sgpr
constant pool, while chi=32 requires slab-tiling theta along the
physical leg (32 KB slab + 16 KB q_curr + 16 KB w = 64 KB full) with the
20-vector basis spilled to the 4 MB per-XCD L2, and chi=48 retreats
further to a 36 KB double-buffered slab + tridiagonal scratch totaling
~61 KB LDS with a 1.44 MB L2 basis footprint. Bank conflicts on the 32-bank
LDS are eliminated by padding `theta_slab` from `32×32` to `32×33`
doubles and by keeping `WW` out of LDS entirely; double-buffering uses
`__builtin_amdgcn_global_load_lds` to overlap L2 streaming with
`v_mfma_f64_16x16x4f64` at 4 concurrent MFMAs per wave (192/512 VGPR),
mapping one workgroup (256 threads, 4 waves, occupancy 1) per
`apply_heff` call. The measurable prediction: persistent Lanczos at
chi=32 completes one matvec in ≤2 µs versus the current stream-of-gemms
path at ~48 µs, a 24× speedup driven by eliminating 4 rocBLAS launches,
20 HBM round-trips for theta, and per-iteration dot/axpy/nrm2 kernel
launches inside the 20-iteration outer loop.
