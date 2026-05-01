# Vertical review — pdmrg family — round 14 — 2026-05-01

## Charter

Audit pdmrg-gpu, pdmrg-gpu-opt, pdmrg-multi-gpu (+ pdmrg-gpu-base for
sibling propagation only) for tier conformity, J1 (Stoudenmire
`accurate_svd_gpu`), J2 (-opt = Block-Davidson default + ctor-time
`lanczos_graph` gate), the PDMRG-rules-2026-04-15 lock, and the
round-13 fix list (ee653f0, f5c0617).

## Charter proof — techniques applied

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | worker_streams/handles/events live in -opt; t_absorb_ DEAD in -gpu and -opt; all 5 PhaseTimers DEAD in -opt |
| B. Behavioral diff | DONE | apply_heff/lanczos/svd_split/update_*_env compared -gpu vs -opt vs multi-gpu — instrumentation gap surfaced |
| C. Docstring verification | DONE | "only rsvd and profile honoured" claim re-verified after round-13 H13 binding fix; n_recal scope verbal-only |
| D. clangd filter | SKIPPED | no ROCm headers in sandbox |
| E. Absence-naming brief | FOLLOWED | tier checklist applied per variant |
| F. Workspace-aliasing audit | DONE | d_dav_work/d_dav_work2 (max(b·dim, max_sub²)), d_T1/T2 (D·dd·χ²) verified |
| G. Sibling fix-propagation | DONE | 5 round-13 fixes traced; **2 propagation gaps surfaced** |

## CRITICALS — block GPU run / paper submission

(none)

## HIGHS — fix before next major event

- **pdmrg-gpu-opt: PhaseTimer panel is fully inert** —
  technique-G lonely-sibling
  `[pdmrg-gpu-opt: pdmrg_gpu_opt_impl.h:3593-3597, 3611-3615]` All five
  timers (`t_lanczos_`, `t_apply_heff_`, `t_svd_`, `t_absorb_`,
  `t_env_update_`) init'd at ctor :3593-3597 and printed at :3611-3615,
  but `grep 't_.*\.(begin|end)'` on the impl returns ZERO. Round-13
  ee653f0 fixed exactly this in pdmrg-gpu (5 sites: 947/1135,
  1144/1211, 1219/1283, 1325/1567, 1575/1735) but did not propagate
  to pdmrg-gpu-opt, which has the identical 5-phase surface
  (`apply_heff_two_site` :799, `update_left_env` :976,
  `update_right_env` :1063, `lanczos_eigensolver` :1805, `svd_split`
  :1316). With `PDMRG_GPU_OPT_PROFILE=1` the panel prints
  `0.00 ms / 0 calls` for every phase. Same defect-class as
  H13-pdmrg-phase-timer-dead — fresh lonely-sibling reappeared.

- **pdmrg-gpu: t_apply_heff_ event-leak on lanczos_graph cache hit**
  `[pdmrg-gpu: pdmrg_gpu_impl.h:959, 980, 1135]` Round-13's
  `t_apply_heff_.begin()` at 959 fires before the early `return;` at
  line 980 (graph cache-hit path inside `if (opts_.lanczos_graph)`).
  The matching `.end()` at 1135 is bypassed every cache hit.
  PhaseTimer truncates totals to `min(starts, stops)` so timing is
  silently low, but every cache hit creates one `hipEvent_t` that
  never gets destroyed (~PhaseTimer iterates `min` pairs, unmatched
  starts pile up). Thousands of cache hits per sweep with
  `LANCZOS_GRAPH=1` → measurable event-creation overhead and per-bond
  leak. Fix: add `if (si == 0) t_apply_heff_.end(streams_[si]);`
  before line 980. Round-13-introduced; HIGH because the fix's stated
  purpose (correct profiling) breaks on the default -gpu code path
  when lanczos_graph is on.

## MEDIUMS — fix when convenient

- **pdmrg-gpu: dead-timer guard not propagated**
  `[pdmrg-gpu: pdmrg_gpu_impl.h:2822-2837]` Round-13 added
  `if (t.calls() == 0) return;` in multi-gpu's `report_timers()` at
  pdmrg_multi_gpu.h:228. pdmrg-gpu has the same situation: `t_absorb_`
  init'd (:2817) and printed (:2835) but never `.begin/.end`'d — will
  always print `absorb : 0.00 ms (0 calls)`. Lonely-sibling.

- **pdmrg-gpu-opt: 8 raw `rocblas_set_pointer_mode` pairs in
  lanczos_eigensolver** `[pdmrg-gpu-opt: pdmrg_gpu_opt_impl.h:1819,
  1821, 1834, 1838, 1852, 1918, 1988, 2001]` Round-13 M1-base-prop
  adopted PointerModeGuard across -base variants. pdmrg-gpu uses RAII
  already; multi-gpu has zero raw calls. pdmrg-gpu-opt's lanczos
  still uses 4 manually paired blocks; a ROCBLAS_CHECK failure
  between set-device and set-host leaves the handle in device mode.
  Same defect class.

- **pdmrg-multi-gpu: `n_recal` API gap not documented**
  `[pdmrg-multi-gpu: pdmrg_multi_gpu.h:24-31, 50-54]` Round-13 said
  "documented as out-of-scope per the multi-device charter." The
  scope paragraph at :24-31 lists what's honoured but never mentions
  the missing `n_recal` argument that pdmrg-gpu/-opt both accept.
  One-sentence add to :50-54.

## NITS — cosmetic

(none new — previous nits resolved)

## FALSE POSITIVES VERIFIED

- **J1 lock** holds: `accurate_svd_gpu<Scalar>` at
  pdmrg_gpu_impl.h:2505, pdmrg_gpu_opt_impl.h:3367,
  pdmrg_multi_gpu_impl.h:2214, pdmrg_gpu_base_impl.h:1271.
- **J2 lock** in pdmrg-gpu-opt: `use_davidson_ = true` impl :204;
  ctor `if (opts_.lanczos_graph && use_davidson_)` gate :220-226.
- **PDMRG-rules-2026-04-15 lock**: defaults `n_warmup=1, n_polish=0`
  on all four signatures; warmup/polish through `sweep_*_full_1site`.
- **Round-13 H13-multi-rsvd-prop**: `use_rsvd_ = opts_.rsvd;` at
  pdmrg_multi_gpu_impl.h:163, after load_from_env at :158.
- **Round-13 PhaseTimer in pdmrg-gpu** wired at 5 sites with si==0
  gating (apart from cache-hit early-return — see HIGH).
- **Round-13 dead-timer guard** in multi-gpu at :227-236.
- **Round-13 docstring nit**: multi-gpu :24-31 no longer references
  `set_use_davidson`.
- **Davidson workspace aliasing**: `max(theta_size_max·b, max_sub²)`
  at pdmrg_gpu_opt_impl.h:274-276 — round-9 H-new1 envelope intact.
- **pdmrg-gpu-base** immune to round-13 #1/#2/#3 (no GpuOpts/timer).

## SUMMARY

Round-13's two HIGH fixes propagated correctly into pdmrg-multi-gpu
(use_rsvd_ binding) and pdmrg-gpu (PhaseTimer instrumentation), and
the J1/J2/PDMRG-rules locks all hold across the 4-variant family.
However, technique G surfaces two propagation gaps that the round-13
batch missed: (1) pdmrg-gpu-opt's PhaseTimer panel is identically
inert to what pdmrg-gpu was before ee653f0 — same 5-phase surface,
same `si`-keyed worker scheme, fully uninstrumented; and (2) the
round-13 instrumentation in pdmrg-gpu's `apply_heff_two_site` calls
`.begin()` before the lanczos-graph cache-hit early-return at line
980 and never `.end()`s, leaking a hipEvent on every cache hit when
profile + lanczos_graph are both enabled. Three mediums — dead
absorb-timer printing in pdmrg-gpu, raw rocblas_set_pointer_mode
pairs in pdmrg-gpu-opt's lanczos, and an undocumented n_recal scope
gap in pdmrg-multi-gpu's docstring — round out the report. The
axis-3 lesson recurs: a fix delivered to one variant is half-fixed
until propagated to its sibling; round-13 closed one such gap and
opened another.
