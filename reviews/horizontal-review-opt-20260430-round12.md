# Horizontal review — -opt tier — round 12 (2026-04-30, post-cleanup)

## Charter

Review the optimized tier across all three families:
- `gpu-rocm/dmrg-gpu-opt/`
- `gpu-rocm/dmrg2-gpu-opt/`
- `gpu-rocm/pdmrg-gpu-opt/`

J2 contract: each -opt is a strict superset of its -gpu sibling. J2 lock:
Block-Davidson default (`use_davidson_ = true`). Verify regression watch
list since round-11 baseline `2d55d90`: bc3fcd0 (M-opt-davidson-toggle in
dmrg-gpu-opt) and c3d3e50 (spotless dead-buffer cleanup).

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | members of each -opt header counted vs impl; no DEAD members. cleanup commit verified to remove only orphans |
| B. Behavioral diff | DONE | three-way diff of `block_davidson_eigensolver`, `apply_heff(_two_site)`, sweep functions, dual-stream/per-segment env-update pattern |
| C. Docstring verification | DONE | setter comments / ctor comments cross-checked against code; one drift found (see HIGH-1) |
| D. clangd filter | N-A | clangd not invokable in sandbox |
| E. Absence-naming brief | FOLLOWED | required -opt features enumerated; one MISSING (HIGH-1) |
| F. Workspace-aliasing audit | DONE | `d_dav_work_/work2_` traced in all 3 -opt; sizing OK in all 3 |
| G. Sibling fix-propagation | DONE | bc3fcd0, c3d3e50, round-8 CR-D1, round-9 H-new1, round-7 C5+C6, round-10 M-opt-rsvd-env all traced |

## J2 lock check

| Variant | `use_davidson_` default | Site |
|---|---|---|
| dmrg-gpu-opt | `true` (in-class init) | `dmrg_gpu_opt.h:252` |
| dmrg2-gpu-opt | `true` (in-class init) | `dmrg2_gpu_opt.h:228` |
| pdmrg-gpu-opt | `true` (ctor body) | `pdmrg_gpu_opt_impl.h:204` |

J2 lock holds in all three.

## CRITICALS — block GPU run / paper submission

(none)

## HIGHS — fix before next major event

### HIGH-1. Ctor-time `lanczos_graph` gate missing in pdmrg-gpu-opt (G — sibling propagation gap)

`[pdmrg-gpu-opt: pdmrg_gpu_opt_impl.h:83-211]`

`set_use_davidson()` in pdmrg-gpu-opt has the symmetric pattern (lines
61-72 of `pdmrg_gpu_opt.h`) and asserts that LANCZOS_GRAPH is
incompatible with Davidson because `apply_heff_two_site` is called
with `AV + j*dim` — a per-subspace-column variable output pointer
that breaks HIP graph capture. The Davidson code path at
`pdmrg_gpu_opt_impl.h:1526, 1759, 1770` confirms the variable-output-
pointer pattern is real and reachable.

dmrg-gpu-opt and dmrg2-gpu-opt fix this at construction:

- `dmrg_gpu_opt_impl.h:73-79`: `if (opts_.lanczos_graph && use_davidson_) { warn; opts_.lanczos_graph = false; lanczos_graph_was_user_enabled_ = true; }`
- `dmrg2_gpu_opt_impl.h:69-75`: same pattern.

pdmrg-gpu-opt's ctor (`pdmrg_gpu_opt_impl.h:83 .. 211`) calls
`opts_.load_from_env()` at line 83 (which can set
`opts_.lanczos_graph=true` from `DMRG_GPU_OPT_LANCZOS_GRAPH=1`) and
then sets `use_davidson_ = true` at line 204, but never gates the two
together. With both flags on by default, `block_davidson_eigensolver`
will invoke `apply_heff_two_site` under graph capture and either hang
on replay (per the setter's own comment) or write through the stale
captured pointer. This is the exact failure mode the comment warns
against.

The setter alone is not a substitute, because env-var-driven
construction never goes through the setter — `lanczos_graph_was_user_
enabled_` stays `false`, `opts_.lanczos_graph` stays `true`, Davidson
runs with capture, hang. This is also why a benchmark driver that
later calls `set_use_davidson(false)` will NOT re-enable
`opts_.lanczos_graph` (the tracking flag is false), breaking the
symmetric round-trip the setter advertises.

Round-7 M11 / round-11 M-opt-davidson-toggle were both about the
SETTER. The CTOR gate is the same defect class on a different code
path, never propagated. Fix: add the same `if (opts_.lanczos_graph &&
use_davidson_) { warn; disable; lanczos_graph_was_user_enabled_ = true; }`
block to pdmrg-gpu-opt's ctor immediately after `use_davidson_ = true`
at line 204 (or near `opts_.load_from_env()` at line 83 once
`use_davidson_` is determined).

## MEDIUMS — fix when convenient

(none)

## NITS — cosmetic

(none)

## FALSE POSITIVES VERIFIED

- **`d_dav_work_` sizing in pdmrg-gpu-opt differs from dmrg/dmrg2-opt** —
  initially looked like a CR-D1-class miss. Verification: pdmrg-gpu-opt's
  Davidson aliases differently. Residuals live in
  `ws.d_dav_work[0 .. n_new*dim)` (`pdmrg_gpu_opt_impl.h:1637, 1674`) and
  the orthogonalization overlap matrix lives in `ws.d_dav_work2`
  (`:1679, 1684`), NOT at an offset into `d_dav_work` as
  dmrg/dmrg2-opt do. Therefore the round-9 H-new1 sizing
  `max(theta*b, max_sub²)` is correct for pdmrg's aliasing pattern.
  Round-8 CR-D1 sizing `max(theta*b + max_sub*b, max_sub²)` IS
  preserved in `dmrg_gpu_opt_impl.h:281-284` and
  `dmrg2_gpu_opt_impl.h:265-268`. All three OK.

- **`d_T2_` / `d_T1_env_` / `d_T2_env_` / `d_lanczos_v_` look like
  alloc/free-only buffers** — initial grep showed only ctor+dtor hits in
  -opt. Verification: each is reassigned to a renamed local pointer at
  the use site (e.g., `Scalar* U = d_T2_;` at
  `dmrg_gpu_opt_impl.h:604`; `Scalar* V = d_T1_env_;` at `:761`; same
  pattern in dmrg2-gpu-opt). Live, not dead.

- **pdmrg-gpu-opt missing `env_update_pending_` / `stream_env_`** — not
  a J2 violation. pdmrg uses per-segment streams (`streams_[si]`) and
  `worker_streams_[seg][w]` instead of a side-channel env stream. The
  dual-stream env-update overlap pattern of dmrg/dmrg2-opt is
  algorithm-specific and does not apply.

- **No `d_batch3_` named array in any -opt** — the brief's name is
  loose; the Step-3 batched/strided GEMMs are present in all three
  via `Traits::gemm_strided_batched` calls (e.g.,
  `dmrg_gpu_opt_impl.h:702, 803, 897`). Feature satisfied.

## Regression-watch summary (vs round-11 baseline 2d55d90)

| Item | Status |
|---|---|
| bc3fcd0 — symmetric `set_use_davidson` | **propagated to all 3 -opt setters** (verified header lines: dmrg-opt:89, dmrg2-opt:73, pdmrg-opt:61). Ctor-side gate present in dmrg/dmrg2-opt; **MISSING in pdmrg-opt** → HIGH-1 above. |
| c3d3e50 — dead-buffer cleanup | clean. Zero stale references to any of the 16 removed names across all -opt impls. No new dead infrastructure introduced (technique A scan). |
| Round-8 CR-D1 — `d_dav_work_` overrun | sizing preserved in dmrg-opt:281-284 and dmrg2-opt:265-268. pdmrg-opt uses different aliasing; sizing correct for that pattern. |
| Round-9 H-new1 — pdmrg-gpu-opt `d_dav_work` sizing | preserved at pdmrg-opt:257-260. |
| Round-7 C5+C6 — pdmrg-opt `n_recal` + `d_Vh_canonical` | both present. `n_recal` in `run()` signature (h:48), driver at impl:3529. `d_Vh_canonical` allocated impl:303, used impl:3408-3418. |
| Round-10 M-opt-rsvd-env — `use_rsvd_ = opts_.rsvd` | present at pdmrg-opt:205. |

## SUMMARY

One HIGH finding: pdmrg-gpu-opt's ctor lacks the `(opts_.lanczos_graph
&& use_davidson_)` gate that its dmrg-opt and dmrg2-opt siblings have.
Setting `DMRG_GPU_OPT_LANCZOS_GRAPH=1` with default Davidson will hit
the exact incompatibility the setter's own comment warns about
(variable output pointer per subspace column, captured graph replays
to stale address). This is a sibling-propagation gap from round 11
M-opt-davidson-toggle which fixed only the setter side of the family.
All other regression-watch items hold; the c3d3e50 cleanup left no
stale references and introduced no new dead infrastructure; J2
default lock holds in all three variants. Recommend a one-line ctor
fix in pdmrg-gpu-opt before the next benchmark window that exercises
LANCZOS_GRAPH.
