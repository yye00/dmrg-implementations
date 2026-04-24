# Quarantined: 20260420T181425Z pdmrg-gpu (48/96 crashed reps)

## Reason for quarantine

48 of 96 reps completed with `returncode != 0` (SIGABRT, rc=-6).
The `wall_s` field was populated from the "Total wall time:" line printed
before abort, so crashed reps passed the original `is not None` filter and
contaminated `median_wall_s` throughout this JSON.  The JSON does **not**
set `partial=true`.

## Status

**Superseded** by `benchmarks/data/gpu_ablation/20260421T190910Z/pdmrg-gpu/results.json`,
which was re-run from the same commit with all reps completing successfully.

This file is kept here for forensic / paper-rebuttal purposes only.
It MUST NOT be used to populate any paper table.

## Provenance

- `git_commit`: c26f0bd39b573f4960c5f360daf8beca8a69d57b
- `binary_sha256`: 0faa038344aed8ea6b457dcfb9d08bbaff1ce3b3154214cd1aca0e7f74bf7018
- `timestamp`: see results.json `timestamp` field
- Crashed reps: 48 / 96 (50%)

## Binary-drift note

This file shares `git_commit c26f0bd` with
`20260421T004212Z/dmrg-gpu-opt/results.json` (also quarantined), yet the
two binaries have **different sha256 values** (`0faa038…` vs `98e1ab0…`),
indicating ad-hoc rebuilds with mutated CMake flags within the same day.
This is the binary-drift problem documented in cluster H §5.
