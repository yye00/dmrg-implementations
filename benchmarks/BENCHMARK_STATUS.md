# Benchmark Status

**Total results: 417**

## Results by Implementation

| Implementation | Total | Success | Failed |
|---|---|---|---|
| pdmrg | 110 | 102 | 8 |
| pdmrg2 | 69 | 66 | 3 |
| quimb-dmrg1 | 122 | 119 | 3 |
| quimb-dmrg2 | 116 | 107 | 9 |

## GPU Results: 0
**No GPU results collected yet.** GPU phase had not started.
GPU implementations have been updated to default to GPU (rocsolver) SVD.

## Hybrid MPI+threads: 99 results

## Key Findings
- **CPU SVD bug**: OpenBLAS LAPACK SVD causes divergence at chi>=65 in two-site DMRG
- **GPU SVD works**: rocsolver SVD has no such issue, now the default for all GPU implementations
- **Affected files**: dmrg2-gpu, pdmrg-gpu, pdmrg2-gpu all updated to `use_cpu_svd_ = false`

## What Remains
- GPU benchmarks (pdmrg-gpu, pdmrg2-gpu) need running with GPU SVD (old CPU SVD results were purged)
- Any timed-out cases at large L/chi may need revisiting
- pdmrg2 hybrid cases were partially collected
