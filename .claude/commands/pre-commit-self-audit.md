---
description: Developer-side gate before claiming "ready". Forces an honest pass over my own recent changes — workspace aliasing, sibling propagation, and the regression watch list — using the lessons from rounds 6/7/8 as a checklist. Run this BEFORE invoking /conformity-review-full.
---

This is the gate the developer (me) runs before declaring a batch
"ready" or invoking `/conformity-review-full`. Round-8 caught a
regression I introduced (CR-D1) and a sibling-propagation miss
(C-new1) that this self-audit would have caught BEFORE the
orchestrator burned 6 sub-reviewer agents on it.

## When to use

- After a multi-fix batch, before pushing for review.
- Before invoking `/conformity-review-full`.
- Before telling the user "we're ready for the GPU run."

## Procedure

### 1. Enumerate my own recent changes

```
git diff --stat HEAD~N..HEAD -- gpu-rocm/
git log --oneline HEAD~N..HEAD -- gpu-rocm/
```

Where `N` is the count of fix-commits since the last conformity
baseline. List the files I touched and the defect classes I claimed
to fix.

### 2. Workspace-aliasing pass (technique F applied to my own work)

For every file I touched in `gpu-rocm/*-opt/src/*_impl.h`:

- Did I change how a shared scratch buffer is sliced? Look for new
  pointer offsets like `ptr + n_new*dim`, `ptr + offset`, `&ptr[k]`,
  or `Scalar* alias = ws.something + ...`.
- For each new aliasing site:
  - **List the regions** the buffer now hosts simultaneously.
  - **Compute the required total size** (sum if concurrent, max if
    sequential).
  - **Find the buffer's allocation** in the constructor (or
    `allocate_*_workspaces`).
  - **Verify allocation ≥ required.** If not, FIX BEFORE COMMIT.

The round-8 CR-D1 failure mode: I added an aliasing site
(`overlap = d_dav_work_ + n_new*dim`) without auditing the buffer's
total-size requirement. The fix is in a different file from the
aliasing site, which is easy to miss.

**Trip-wire grep**:

```
git diff HEAD~N..HEAD | grep -E '^\+.*Scalar\* \w+ = \w+ \+ '
```

For each match, I must produce evidence the buffer is large enough.

### 3. Sibling fix-propagation pass (technique G applied to my own work)

For each defect class I fixed in this batch:

- List the variant where I applied the fix (e.g., "C6 in pdmrg-gpu-opt").
- Enumerate the **sibling variants** that have the same algorithm:
  - Same family (-base, -gpu, -gpu-opt of pdmrg).
  - Same tier across families (pdmrg-gpu-base vs dmrg-gpu-base vs
    dmrg2-gpu-base, etc.).
- For each sibling, decide ONE of:
  - **Already had the fix** (cite file:line).
  - **Genuinely immune** (e.g., -base lacks Davidson). Document why.
  - **Has the same defect, MUST FIX BEFORE COMMIT.**

The round-8 C-new1 failure mode: I fixed C6 in pdmrg-gpu-opt
without auditing whether pdmrg-gpu-base had the same defect. It
did. The orchestrator caught it three rounds later.

**Trip-wire question**: for each fix in my batch, can I name all
sibling variants that have or could have the same defect? If I
can't list them, I haven't done G.

### 4. Regression watch list

Read the most recent `reviews/conformity-*.md` baseline. For each
fix listed there:

- **Run a quick grep** to confirm the fix is still present in the
  cited file:line. If round-N fixed something at `foo.h:123`, my
  recent edits may have moved or removed it.
- If something I touched accidentally reverts a prior fix, fix
  before commit.

### 5. Build verification

Where possible:

- Run any available host-side compile (`make` if the project has it,
  or per-variant `cmake --build`). Filter clangd noise (no ROCm
  headers locally).
- If GPU access is available and authorized: build on the remote
  MI300X (`tmux send-keys -t test_remote ...`).
- If neither, document explicitly: "Build verification deferred to
  GPU window — static review only."

### 6. Self-audit verdict

Emit a structured verdict before declaring ready:

```markdown
# Pre-commit self-audit — <date> — batch <N>

## Files touched
- <file>: <fix description>

## Technique F (workspace-aliasing)
| Buffer | New aliasing sites | Required total | Allocated | Verdict |
|---|---|---|---|---|
| <buffer> | <regions> | <sum/max> | <ctor size> | OK / OVERRUN |

## Technique G (sibling fix-propagation)
| Fix | Variant fixed | Siblings | Status |
|---|---|---|---|
| <fix> | <variant> | <sibling> | fixed / immune / MISSING |

## Regression watch
| Prior fix | File:line | Status |
|---|---|---|
| <fix> | <site> | intact / reverted / moved |

## Build verification
- Status: ran / deferred / failed
- Errors: <none / list>

## Verdict: READY / NOT READY
```

If any line in any table is OVERRUN, MISSING, reverted, or failed —
**NOT READY**. Fix before commit. Do not invoke
`/conformity-review-full` until READY. Do not tell the user "ready
for GPU run" until READY.

### 7. Honesty note

The dmrg2-gpu-opt CR-D1 buffer overrun (round-8) and the
pdmrg-gpu-base C-new1 (round-8) would BOTH have been caught by this
self-audit if it had existed when I worked on round-7 batches 5 and
2 respectively.

The orchestrator's 6 sub-reviewers caught them in round-8. That's
correct but expensive. The self-audit catches them in round-7
*before* the orchestrator runs.

**Use this every time. No exceptions.**
