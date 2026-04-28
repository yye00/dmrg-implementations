# GPU-rocm review methodology

Reusable methodology for the seven family/tier review commands in
`.claude/commands/{vertical,horizontal,conformity}-review-*.md`.
Encapsulates lessons from rounds 4-8:
- Round 7 caught the dmrg2-gpu dead-infrastructure miss → led to
  technique A.
- Round 7 caught dual-stream env-update overlap missing in -opt
  variants despite -gpu sibling having it → led to technique B.
- Round 7 caught the false `accurate_svd_gpu` claim in pdmrg-gpu-base
  docstring → led to technique C.
- Round 8 caught a regression I introduced (CR-D1: Davidson buffer
  overrun from new aliasing without ctor resize) → leads to
  **technique F** (workspace-aliasing audit).
- Round 8 caught a fix that wasn't back-ported to a sibling variant
  (C-new1: round-7 C6 fix in pdmrg-gpu-opt missed pdmrg-gpu-base) →
  leads to **technique G** (sibling fix-class propagation).

The seven techniques (A-G) are MANDATORY. A review that skips any of
them returns invalid even if it finds real defects, because the
techniques exist to surface different *classes* of defect.

---

## Technique A — Symbol-usage scan

**Purpose**: catch declared-but-unused private members. Resources that
are allocated in the ctor and freed in the dtor without ever being
read or written between is the exact failure mode that hid the
dead dmrg2-gpu dual-stream infrastructure for three review rounds.

**Procedure**: for each variant in scope, list every private member
declared in the class header. For each member, run a grep across the
implementation file:

```
grep -c '\bMEMBER\b' impl.h
```

Subtract the ctor and dtor sites (typically one each: alloc + free).
Any member with zero remaining hits is flagged as **dead
infrastructure**. Look especially carefully at concurrency primitives
(streams, events, handles, "_pending_" flags) and scratch buffers.

**Output**: a table per variant — `Member | Total hits | Ctor/dtor
hits | Other hits | Verdict`. Verdict is `live` (≥1 other hit) or
`DEAD` (0 other hits).

---

## Technique B — Behavioral diff between paired variants

**Purpose**: catch silent feature-omission across siblings (a function
exists in one variant and is missing or simpler in another). The
audit pattern that would have surfaced "dmrg-gpu has 6 event/wait
calls in sweep_left_to_right; dmrg2-gpu has 0."

**Procedure**: for each sibling pair within a family or tier, pick
the hot-path functions:

- `optimize_site` / `optimize_bond` / `optimize_segment`
- `sweep_left_to_right` / `sweep_right_to_left` / `batched_segment_sweep`
- `update_left_env` / `update_right_env`
- `svd_fallback` / `svd_split` / `svd_split_fallback`
- `apply_heff` / `apply_heff_two_site`
- `lanczos_eigensolver` / `block_davidson_eigensolver`
- `build_initial_environments`

For each function present in either variant, dump its source side by
side and report the structural delta:

- Number of kernel launches, GEMMs, batched GEMMs.
- Number of `hipStream*` / `hipEvent*` / `hipMemcpy*` calls.
- Number of pointer-mode toggles, `hipStreamSynchronize` sites.
- Functions called.
- Stream membership of each call (`stream_` vs `stream_env_` vs
  per-segment streams).

A divergence is a candidate finding. The reviewer must classify each
as **intentional** (with rationale) or **defect**.

---

## Technique C — Docstring claim verification

**Purpose**: catch doc/code drift. Header docstrings and class
comments often promise behavior the implementation has lost or never
had. The pdmrg-gpu-base "uses plain rocsolver_gesvd_auto" miss
(corrected in round 6) was a docstring/code drift defect.

**Procedure**: extract every "we do X" / "this variant has Y" /
"compared to Z, this omits W" statement from class-level docstrings
and from the file's leading comment block. For each claim, try to
locate the corresponding code site:

- "lanczos with HIP graph capture" → grep for `hipStreamBeginCapture`
- "dual-stream pipeline" → grep for `hipEventRecord` and
  `hipStreamWaitEvent` in the sweep functions
- "Stoudenmire accurate SVD at boundaries" → grep for
  `accurate_svd_gpu` in the merge function

Any claim with zero matching code is a **claim defect**. Either the
docstring is wrong or the implementation is incomplete; both are
defects.

---

## Technique D — clangd diagnostic filter

**Purpose**: surface real warnings buried in unavoidable noise. Local
clangd produces a flood of `pp_file_not_found` / `unknown_typename`
errors because no ROCm headers are on the host. Real signals
(unused-private-field, unused-parameter, dead-store) get drowned out
and reviewers train themselves to ignore the whole channel.

**Procedure**: where clangd is available, run with a filter that
strips the known-noise categories:

```
clangd --check=<file> 2>&1 \
  | grep -v 'pp_file_not_found' \
  | grep -v 'unknown_typename' \
  | grep -v 'no_member.*std::is_same' \
  | grep -v 'typename_nested_not_found' \
  | grep -v 'undeclared_var_use.*\(blockIdx\|blockDim\|threadIdx\|__global__\)'
```

Any remaining diagnostic is a **real signal**: investigate. Special
attention to `unused-private-field`, `unused-variable`,
`dead-initialization`, `useless-cast`.

If clangd is not invokable, fall back to grep-based dead-symbol
analysis (technique A subsumes the most important case).

---

## Technique E — Absence-naming brief

**Purpose**: shift reviewer mindset from "is this code correct" to
"is this code complete and reachable." The two questions surface
different defect classes, and the second is rarely asked unless the
brief explicitly demands it.

**Procedure**: every review brief MUST contain the literal sentence:

> Identify what should be present in this variant but is not.

Plus a checklist of expected features for the tier or family:

- For -base: HIP_CHECK macro, ScalarTraits dispatch, single-stream
  correctness, opts-free inner loop. For pdmrg-base specifically:
  per-segment streams, accurate_svd_gpu at boundary (J1).
- For -gpu: GpuOpts env-var surface, lanczos_graph cache, RSVD
  workspace + opt-in path, sparse_mpo compaction, dual-stream
  env-update overlap, batched GEMM, D_PAD, on-device fused-MPO
  precompute, accurate_svd_gpu (pdmrg only).
- For -opt: superset of -gpu (J2 contract). Block-Davidson default
  with Lanczos fallback. MFMA-16 padding. Strided/batched Step-3
  GEMMs. Public setter API parity.

The reviewer reports each expected feature as `present` / `MISSING`
/ `DEAD` (declared but unused — caught by technique A).

---

## Technique F — Workspace-aliasing audit

**Purpose**: catch buffer-sizing bugs when a single device buffer
holds two or more concurrent regions. This is the round-8 CR-D1
class: my round-7 H6 syev port routed the residual loop's overlap
matrix into `d_dav_work_ + n_new*dim`, but the constructor sized
`dav_work_sz` for residuals only. The aliasing logic was correct;
the buffer was 128 Scalars short. Smoke tests with `dim < 256` ride
the 1024-Scalar floor and pass; benchmark-sized runs corrupt the
adjacent allocation.

**Procedure**: identify every device buffer that is used as scratch
in a hot-path function. For each:

1. List the regions the buffer hosts during the function's execution
   (e.g., `d_dav_work_` hosts: residuals W [0, n_new*dim), overlap
   matrix [n_new*dim, n_new*dim + k*n_new), and possibly a third
   region in the restart path).
2. For each region pair, determine if their lifetimes are concurrent
   (both live at the same point in execution) or sequential.
3. Compute the required total size:
   - **Concurrent regions**: `total = sum of region sizes`.
   - **Sequential regions** (one dead before next is born):
     `total = max of region sizes`.
4. Find the buffer's allocation in the constructor (or
   `allocate_*_workspaces` for per-stream variants). Verify the
   allocation matches the required total.

Apply this to every shared scratch buffer touched by recent code
changes. The most common offenders:

- `d_dav_work_` / `d_dav_work2_` (Block-Davidson — residuals,
  overlap, projected H, eigvecs).
- `d_T1_` / `d_T2_` (apply_heff — multiple roles per call).
- `d_svd_work` / `d_svd_A` (SVD — input, output, scratch).
- `d_rsvd_*` (randomized SVD — Y, Q, B, U_small).
- `d_batch_*_` / `d_batch_*_env_` (batched-GEMM pointer tables —
  must not be shared between concurrent streams).

**Special attention** when a recent commit changed how a buffer is
sliced — that is exactly where size requirements drift.

**Output**: a table per shared buffer — `Buffer | Regions | Lifetime
(concurrent/sequential) | Required size | Allocated size | Verdict`.
Verdict is `OK` (allocated ≥ required) or `OVERRUN` (allocated <
required by N elements).

---

## Technique G — Sibling fix-class propagation

**Purpose**: catch defects that were fixed in one variant but never
back-ported to its siblings. This is the round-8 C-new1 class:
round-7 fixed C6 (canonical-Vh swap before R_env build) in
pdmrg-gpu-opt — but pdmrg-gpu-base had the exact same defect and was
never touched. Three rounds went by without anyone asking "if -opt
needs this fix, do -base and -gpu need it too?"

**Procedure**: for each defect class fixed in a recent commit:

1. **Read the commit log** since the last conformity baseline (or
   the most recent `reviews/conformity-*.md` if one exists).
2. For each fix, identify the **defect class**, not just the file:
   - "canonical-Vh swap before R_env" (defect class).
   - "rocsolver_syevd replaces host LAPACK" (defect class).
   - "non-blocking stream flag" (defect class).
3. For each defect class, enumerate the **sibling variants** in
   scope of the current review:
   - Vertical review: -base, -gpu, -gpu-opt of the family.
   - Horizontal review: same-tier siblings of other families.
4. For each sibling, verify ONE of:
   - **Already fixed**: the same fix was applied (cite file:line).
   - **Genuinely immune**: the variant lacks the relevant feature
     (e.g., -base has no Davidson, so the syev port doesn't apply).
     Document the immunity.
   - **Needs fixing**: the defect is present in the sibling and was
     missed. **Flag as a CRITICAL or HIGH (severity matches the
     original).**

The "genuinely immune" leg is what distinguishes G from B. Technique
B compares current code; technique G specifically asks "did the
recent fix propagate, or did it stop at one variant?"

**Output**: a table per recent fix — `Fix | Variant | Status (fixed
/ immune / MISSING)`. Any MISSING entry is a finding.

This technique is mandatory for vertical reviews (which see all
three tiers of one family) and horizontal reviews (which see all
three families of one tier). The orchestrator's brief embeds the
last conformity report's "what was fixed in the most recent batch"
list to make this concrete.

---

## Output template

Every review command emits a Markdown report with this structure:

```markdown
# <Review name> — <date>

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE / SKIPPED | n unused members, n dead |
| B. Behavioral diff | DONE / SKIPPED | n divergences |
| C. Docstring verification | DONE / SKIPPED | n unverified claims |
| D. clangd filter | DONE / SKIPPED / N-A | n real warnings |
| E. Absence-naming brief | FOLLOWED | — |
| F. Workspace-aliasing audit | DONE / SKIPPED | n shared buffers checked, n OVERRUN |
| G. Sibling fix-propagation | DONE / SKIPPED | n recent fixes traced, n MISSING in siblings |

A review with any technique SKIPPED that is not N-A is INVALID.

## CRITICALS — block GPU run / paper submission
- <finding> [variant: file.h:line]

## HIGHS — fix before next major event
- <finding> [variant: file.h:line]

## MEDIUMS — fix when convenient
- <finding> [variant: file.h:line]

## NITS — cosmetic
- <finding> [variant: file.h:line]

## FALSE POSITIVES VERIFIED
- <finding> — verification: <evidence why it's not a defect>

## SUMMARY
<one paragraph: overall verdict, what to act on first>
```

---

## Severity definitions

- **CRITICAL** — incorrectness on the default code path, or a
  J1/J2 lock violation, or a missing feature whose absence
  invalidates the variant's stated purpose.
- **HIGH** — performance regression, dead infrastructure that was
  meant to provide a feature, header-claim that the implementation
  contradicts.
- **MEDIUM** — duplication that should be consolidated, unused
  scaffolding, opportunistic optimization not taken.
- **NIT** — naming, comment, formatting; cosmetic only.
- **FALSE POSITIVE** — finding from a sub-agent that on closer
  inspection is correct or out-of-scope; preserved in output for
  transparency so we don't re-discover the same false positive next
  round.

---

## Reviewer mindset

Reviewers default to "is this present and correct." The seven
techniques force the reviewer to also ask:

- "is this complete and used" (A, E),
- "does this match its sibling" (B),
- "does the doc match the code" (C),
- "what real signal is in the noise" (D),
- "does the workspace size match the workspace usage" (F),
- "did the recent fix propagate to all variants that need it" (G).

If a finding feels uncomfortable to write up because the code "looks
fine" — write it up anyway. The dmrg2-gpu dead-stream miss looked
fine for three rounds. The CR-D1 buffer overrun looked fine because
the aliasing logic was correct — only the size was wrong, in a
different file. The C-new1 missing canonical-Vh swap looked fine
because pdmrg-gpu-base had always done it that way — but its
sibling -opt had been fixed in the same commit cycle and the -base
sibling never got the same audit.

**Specifically watch for "regression-by-the-fix":** a fix in one
variant that introduces aliasing or restructuring without updating
the corresponding sizing or sibling code. Round-8 CR-D1 is the
canonical example. Technique F catches this. **And specifically
watch for "lonely fix": a defect-class fix in one variant whose
siblings still have the defect.** Round-8 C-new1 is the canonical
example. Technique G catches this.
