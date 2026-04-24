# Path B operator instructions

One-way channel from the operator (the human) to the running Path B loop.

## How it works

The orchestrator reads this file at the start of every tick (before
inspecting PR state). It honors unchecked instructions from the "Active"
section in top-down order, moves processed ones to "Processed" with a
timestamp and outcome, commits the update, and then continues the normal
tick.

Unknown or malformed instructions are left unchecked and flagged with a
comment on the tracking issue. The orchestrator never guesses intent —
if a verb is not in the DSL below, it does nothing.

## Usage

1. Append an unchecked bullet to "Active instructions" using the DSL below.
2. Commit and push. (Optional if you'd prefer the orchestrator's next
   `git pull` to pick it up at tick start.)
3. Wait for the next `/path-b-tick`. The orchestrator will honor it and
   move it to "Processed".

To steer urgently, `git pull` on the remote, commit your edit, push.
Next tick honors it. Typical latency: up to 30 min (one loop cycle).

## Active instructions

<!-- Add unchecked bullets below. One instruction per bullet. -->
<!-- Example:
- [ ] merge #7
- [ ] priority E
- [ ] note: booking Hot Aisle for Saturday 10am
-->


## Processed

- [x] gpu-unlock — PROCESSED 2026-04-24T18:30:00Z — F and H flipped from `waiting_gpu` → `not_started`. Implementer dispatched for cluster F (wave-2 GPU campaign). Change recorded in tracker change log.
- [x] merge #10 — PROCESSED 2026-04-24T17:50:00Z — MERGED: PR #10 (`review-clean`). Merge commit `26753726`. Cluster E → `merged`. Wave-1 sandbox-scope complete: 8/10 clusters merged; F+H remain `waiting_gpu` (GPU-gated).
- [x] merge #9 — PROCESSED 2026-04-24T17:29:07Z — MERGED: PR #9 (`review-clean`). Merge commit `69ecf22`. Cluster C → `merged`. paper/main.tex serialization constraint lifted; E now eligible for dispatch.

<!-- Orchestrator appends here. Do not edit by hand. -->
- [x] merge #8 — PROCESSED 2026-04-24T17:02:25Z — MERGED: PR #8 (`review-clean`). Merge commit `bb7a879`. Cluster B → `merged`. paper/main.tex serialization constraint lifted; C and E now eligible for dispatch.
- [x] merge #7 — PROCESSED 2026-04-24T15:00:00Z — MERGED: PR #7 (`review-clean`). Merge commit `954bdcf`. Cluster I → `merged`. paper/main.tex serialization constraint lifted; B, C, E now eligible for dispatch.
- [x] merge #6 — PROCESSED 2026-04-24T14:32:26Z — MERGED: PR #6 (`review-clean`). Merge commit `cd162a2`. Cluster J → `merged`. paper/main.tex serialization constraint lifted; cluster I now unblocked.
- [x] merge #6 — PROCESSED 2026-04-24T14:10:00Z — SKIPPED: PR #6 label is `needs-review`, not `review-clean`. Flagged on tracking issue. Will honor once reviewer clears.
- [x] merge #6 — PROCESSED 2026-04-23T23:50:00Z — SKIPPED: PR #6 label is `needs-review`, not `review-clean`. Flagged on tracking issue. Will honor once reviewer clears.
- [x] note: pre-authorized PR #6 merge; sandbox queue should drain B/C/E/I next. — PROCESSED 2026-04-23T23:50:00Z — comment posted on issue #2.

## DSL reference

All instructions start with a verb. Arguments are space-separated.

| Verb | Arguments | Effect |
|--|--|--|
| `merge` | `#<pr-number>` | Merge the PR if its label is `review-clean`. If it is not, flag and leave unchecked. |
| `priority` | `<cluster-id>` | Bump this cluster to the top of the dispatch queue for the next tick that would dispatch an implementer. |
| `skip` | `<cluster-id>` | Do not dispatch this cluster until a `resume <cluster>` instruction or the user manually flips its status. |
| `resume` | `<cluster-id>` | Revert a prior `skip` for that cluster. |
| `pause` | *(none)* | Stop dispatching new implementers. Reviewer and fixer dispatches still fire. |
| `resume` | *(none)* | Opposite of `pause`. Resumes implementer dispatches. |
| `gpu-unlock` | *(none)* | Orchestrator awareness that `PATH_B_GPU_AUTH=1` has been set by the operator. Flip F and H from `waiting_gpu` → `not_started`. Record in the tracker change log. |
| `gpu-lock` | *(none)* | Flip F and H back to `waiting_gpu`. Used if the Hot Aisle window ends or is canceled. |
| `note` | `: <free text>` | Post the text as a comment on the tracking issue. No side effect beyond the comment. |

Anything else: the orchestrator posts a tracking-issue comment naming the
instruction and why it was rejected (unknown verb, missing argument,
referenced PR or cluster does not exist, etc.), and leaves the bullet
unchecked.

## Adding a new verb

To extend the DSL, patch `.claude/agents/orchestrator.md` (Step 0
section). The file is read at every tick, so restart-of-loop is only
needed if your Claude Code session caches the agent definition at
session start.

## Audit trail

Every processed instruction lands in git history as a commit authored
by the orchestrator with message `ops: process N operator instruction(s)`.
The "Processed" section below is the human-readable log; `git log
docs/OPERATOR_INSTRUCTIONS.md` is the authoritative audit trail.
