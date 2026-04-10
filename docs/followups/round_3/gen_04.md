# Round 3 — Gen 04: Cross-GPU MPS + Environment Broadcast Protocol (R2-2 Phase 1.5)

**Scope.** Define the exact wire protocol, chunking, primitive choice, and
expected wall time for shipping the canonicalized MPS plus left/right
environments from GPU 0 to GPUs 1…P-1 at the R2-2 Phase 1 → Phase 2 handoff.
Verify or refine research D's ~100 ms estimate.

## 1. Live hardware probe

The `hotaisle@23.183.40.84` VM exposes only **one** MI300X device
(`/dev/dri/card0`; `rocm-smi --showtopo` returns a 1×1 matrix; `rocminfo`
reports a single `gfx942` agent). The host has 8 render nodes but the VM is
single-GPU, so `hipDeviceCanAccessPeer` and `rccl-tests` cannot be exercised
against peers from this instance. Numbers below therefore come from the
published MI300X OAM/UBB spec and from the `rocm-smi` firmware field
(`TA XGMI firmware version: 32.00.00.20`), not in-VM measurement.

## 2. xGMI topology assumption (4-GPU subset)

MI300X OAM modules expose **7 xGMI links × 64 GB/s/link = 448 GB/s**
aggregate per device (bidirectional). In the standard UBB 8-GPU platform this
forms a fully-connected mesh: every GPU has a direct xGMI path to every other
GPU. A 4-GPU partition of that same fabric is therefore also fully connected
— every GPU pair has ≥1 direct xGMI link of 64 GB/s peak (128 GB/s bidi).

Realistic sustained unidirectional bandwidth per link, based on published
rccl-tests numbers for MI300X (and matching hipMemcpyPeerAsync measurements),
is **50–55 GB/s** — i.e. 80–85% of peak, limited by packet overhead and the
SDMA engine scheduling. I will use **50 GB/s** as the conservative design
number and **55 GB/s** as the optimistic one.

## 3. Payload inventory (L=64)

For a CBE/DMRG state at site dimension `d=2`, Heisenberg MPO `D=5`,
doubles (or 16 B complex doubles where noted):

| Object            | Shape                   | Bytes @ χ=128 (double) | Bytes @ χ=256 (double) |
|-------------------|-------------------------|------------------------|------------------------|
| MPS tensor × L    | `(χ, d, χ)`             | `64·128·2·128·8` = **16.8 MB** | `64·256·2·256·8` = **67.1 MB** |
| Left env × L      | `(χ, D, χ)`             | `64·128·5·128·8` = **42.0 MB** | `64·256·5·256·8` = **168 MB** |
| Right env × L     | `(χ, D, χ)`             | **42.0 MB**            | **168 MB**             |
| Fused WW MPO × L-1| `(D, d², D, d²)` doubles| `63·5·4·5·4·8` ≈ **0.20 MB** | same (χ-independent) |
| **Total**         |                         | **≈ 101 MB**           | **≈ 403 MB**           |

For complex doubles the totals double: **202 MB** (χ=128) and **806 MB**
(χ=256). The 100 ms budget in research D implicitly assumed real doubles and
a ~1 TB/s effective broadcast rate; that needs refinement (see §6).

## 4. Primitive choice

Three candidates:

1. **raw `hipMemcpyPeerAsync`** — lowest overhead, direct SDMA engine driving
   xGMI. One submit per (src,dst,buffer) triple. Must schedule the
   P-1 fan-out ourselves.
2. **RCCL `ncclBroadcast`** — portable, handles rings/trees automatically,
   but RCCL ring broadcast is optimized for *all-reduce-shaped* large
   messages, not for O(L) medium messages. Incurs a per-call launch
   ~15–25 µs plus a ring step penalty of `(P-1)/P ≈ 0.75×` the single-link
   bandwidth.
3. **Custom fan-out** — for P=4 on a fully connected mesh, the optimal
   broadcast is not a ring but a **1-step fan-out from GPU 0 to 1, 2, 3 in
   parallel**, each over its own dedicated xGMI link. This *triples*
   effective bandwidth vs. a ring because all 3 target copies proceed
   simultaneously on disjoint links.

**Decision:** hand-rolled 1-hop fan-out using `hipMemcpyPeerAsync` on three
dedicated streams (one per destination GPU), with a single
`hipEventRecord`/`hipStreamWaitEvent` fence at the end. RCCL is a fallback.
The fan-out assumes GPU 0 can drive 3 × 50 GB/s = 150 GB/s aggregate out of
the 448 GB/s egress capacity — well within budget.

## 5. Chunking

Benchmark sweet spots on MI300X for single-peer xGMI transfers (from public
rccl-tests data): efficiency is ≥95% of peak for messages ≥4 MB, and levels
off above 16 MB. Below 1 MB the SDMA launch cost dominates.

**Chunk size: 8 MB.** For L=64 χ=128 the largest single tensor is an env
slice of 655 kB — too small individually. So we coalesce: pack MPS[0..L-1]
into one contiguous 16.8 MB device buffer, L-env into 42 MB, R-env into
42 MB, then issue 3 peer copies per destination = **9 peer copies total**
(3 destinations × 3 buffers). Each destination unpacks locally via a small
indexing kernel. This amortizes launch overhead to <0.5% of the transfer
time.

## 6. Protocol pseudocode

```
// Phase 1.5 handoff, called on GPU 0 only
fn broadcast_phase_1_5(mps, L_env, R_env, WW_mpo, peers=[1,2,3]):
    // (a) Pack into 3 contiguous device buffers on GPU 0
    pack_mps_buf     = device_alloc(total_mps_bytes)
    pack_lenv_buf    = device_alloc(total_lenv_bytes)
    pack_renv_buf    = device_alloc(total_renv_bytes)
    launch pack_kernel<<<...>>>(mps, L_env, R_env, pack_*)
    hipStreamSynchronize(default_stream)

    // (b) Per-destination stream so copies run in parallel on disjoint links
    for dst in peers:
        s = streams[dst]
        hipMemcpyPeerAsync(dst_mps[dst],  dst, pack_mps_buf,  0, mps_bytes,  s)
        hipMemcpyPeerAsync(dst_lenv[dst], dst, pack_lenv_buf, 0, lenv_bytes, s)
        hipMemcpyPeerAsync(dst_renv[dst], dst, pack_renv_buf, 0, renv_bytes, s)
        hipMemcpyPeerAsync(dst_ww[dst],   dst, ww_buf,        0, ww_bytes,   s)
        hipEventRecord(done_evt[dst], s)

    // (c) Fence — caller on GPU 0 waits; each destination GPU also waits on its event
    for dst in peers:
        hipEventSynchronize(done_evt[dst])

    // (d) On each destination, launch unpack_kernel to place tensors in canonical layout
    for dst in peers:
        launch_on(dst, unpack_kernel, dst_buffers -> dst_mps_list, dst_lenv_list, dst_renv_list)
```

Total primitive count: **9 hipMemcpyPeerAsync + 3 events + 3 unpack kernels**
— well under any scheduling concern.

## 7. Overlap with Phase 1 (TEBD tail)

Phase 1 (TEBD warmup) ends when the final even half-step finishes on GPU 0.
The canonicalization (left-to-right QR sweep) that follows is *serial* and
takes an estimated ~3 ms at L=64 χ=128 on MI300X. We can overlap by
*double-buffering*: as soon as site `k` is canonicalized, pack `MPS[k]` and
`L_env[k]` into the packed buffer slot, and start a peer copy for *that
slice* on a background stream. Because canonicalization is strictly
left-to-right, the L-envs arrive progressively; R-envs are built on-the-fly
in a second right-to-left sweep that happens to be ~3 ms, so those ship
second. Net effect: the broadcast **hides behind the canonicalization
sweeps** for all but the last ~L/P = 16 sites. Expected overlap savings:
20–40%.

For the conservative "hard barrier" design we use the figure in §8 as-is.

## 8. Expected wall time — measurable prediction

Per-destination transfer time at 50 GB/s sustained:

| Case                    | Payload/dest | 1 dest | 3 dests (parallel) | w/ overlap |
|-------------------------|--------------|--------|---------------------|------------|
| L=64 χ=128 double       | 101 MB       | 2.0 ms | **2.0 ms**          | ~1.4 ms    |
| L=64 χ=128 complex128   | 202 MB       | 4.0 ms | **4.0 ms**          | ~2.8 ms    |
| L=64 χ=256 double       | 403 MB       | 8.1 ms | **8.1 ms**          | ~5.6 ms    |
| L=64 χ=256 complex128   | 806 MB       | 16.1 ms| **16.1 ms**         | ~11 ms     |
| L=128 χ=256 complex128  | 1.61 GB      | 32 ms  | **32 ms**           | ~22 ms     |

Add a fixed **~200 µs** for the 9 `hipMemcpyPeerAsync` launches plus pack /
unpack kernels.

**Headline prediction:** the R2-2 flagship target (L=64, χ=256, complex
doubles) will complete Phase 1.5 broadcast in **≈16 ms** without overlap and
**≈11 ms** with canonicalization overlap — **~6–9× faster** than research
D's ~100 ms estimate. The research D number is consistent only if you assume
an uncoalesced per-site broadcast over a ring topology, which our fan-out
avoids. Even the 1.6 GB L=128 χ=256 extreme fits comfortably inside the
100 ms envelope.

## 9. Correctness

Each destination receives the exact bytes on GPU 0; no reduction, no FP
rounding, so all GPUs start Phase 2 bit-identical. To harden against
silent corruption, stamp a 64-bit xxhash over each packed buffer on GPU 0,
copy it as a tail word, and verify on destination with a tiny reduction
kernel (cost ~10 µs). Mismatch → §10.

## 10. Failure recovery

Failure modes for `hipMemcpyPeerAsync`:
- `hipErrorPeerAccessNotEnabled` → call `hipDeviceEnablePeerAccess` at
  program start for every pair; assert at init.
- Transient ECC or link-retrain event → `hipMemcpy*` returns an async
  error on the next `hipStreamSynchronize`. Strategy: catch, log, retry the
  **single failed chunk** up to 3× on the same stream, then fall back to
  staging through host pinned memory (`hipMemcpyAsync` via `hipHostAlloc`
  buffer), cost ~4× slower but always works.
- Hash mismatch → retry chunk once; on second mismatch abort and re-run
  Phase 1.5 from a GPU-0 snapshot (kept for exactly this reason until all
  destinations acknowledge).

## 11. Summary of the refinement

- Topology: **full mesh**, every GPU pair has a direct xGMI link.
- Primitive: **hand-rolled 1-hop fan-out** via `hipMemcpyPeerAsync` on
  dedicated streams — beats RCCL broadcast by ~3× for this payload class.
- Chunking: coalesce MPS / L-env / R-env into **3 packed buffers of ≤170 MB**;
  no need for 8 MB sub-chunks because the full buffers already sit in the
  high-efficiency regime.
- **Predicted wall time: 4 ms (χ=128 cplx), 16 ms (χ=256 cplx)**; overlap
  with canonicalization brings this to 3 / 11 ms respectively.
- Research D's 100 ms budget is **6–9× pessimistic** and gives us plenty of
  headroom for safety, retries, and a larger χ=512 future flagship.

---

### 3-sentence summary (return value)

A 4-GPU MI300X partition is a full xGMI mesh, so GPU 0 can fan broadcast to
peers 1–3 in parallel on disjoint 50 GB/s links — yielding a packed
hipMemcpyPeerAsync protocol that ships the R2-2 Phase 1.5 payload (MPS +
L/R envs + WW MPO) as 9 peer copies plus a hash check. Predicted wall time
is **≈4 ms for L=64 χ=128 complex** and **≈16 ms for L=64 χ=256 complex**,
both well under the ~100 ms budget in research D and overlap-hideable behind
the ~3 ms canonicalization sweep. Failure recovery falls back to host-pinned
staging on the individual failed chunk, keeping Phase 1.5 robust without
sacrificing bit-exact replication across GPUs.
