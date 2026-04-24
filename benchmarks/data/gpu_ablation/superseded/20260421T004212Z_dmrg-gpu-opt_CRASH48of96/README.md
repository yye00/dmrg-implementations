# Quarantined: 20260421T004212Z dmrg-gpu-opt (48/96 crashed reps)

## Reason for quarantine

48 of 96 reps completed with `returncode != 0` (SIGABRT, rc=-6).
The `wall_s` field was populated from the "Total wall time:" line printed
before abort, so crashed reps passed the original `is not None` filter and
contaminated `median_wall_s`.  The JSON does **not** set `partial=true`.

## Status

**NEVER re-run.**  This file is the SOLE backing data for
`dmrg-gpu-opt` rows in paper Table 6 (ablation study).
Those cells stand on contaminated data and must be treated as unreliable
until a clean re-run lands.

This file is kept here as evidence of the provenance failure described
in the paper's limitations section and needed for any formal rebuttal.

## Provenance

- `git_commit`: c26f0bd39b573f4960c5f360daf8beca8a69d57b
- `binary_sha256`: 98e1ab093905ef74bd2535021cac4090646baf459f8518fee91ad47af6f65ec5
- `timestamp`: see results.json `timestamp` field
- Crashed reps: 48 / 96 (50%)

## Binary-drift note

Shares `git_commit c26f0bd` with `20260420T181425Z/pdmrg-gpu/results.json`
(also quarantined) but has a **different sha256** (`98e1ab0…` vs `0faa038…`),
indicating a separate rebuild with different CMake flags on the same day —
the binary-drift problem documented in cluster H §5.

## Action required

Before paper resubmission:
1. Wire `opts_.device_k` and `opts_.rsvd` into `dmrg-gpu-opt` call sites
   (or explicitly mark those ablation columns "n/a (Block-Davidson)").
2. Re-run from a single campaign-tagged commit with `--reps 10` and the
   binary-drift manifest guard enabled.
3. Update paper Table 6 dmrg-gpu-opt rows from the new clean JSON.
