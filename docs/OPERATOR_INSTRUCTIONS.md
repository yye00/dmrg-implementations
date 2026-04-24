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

- [ ] merge #6

## Processed

<!-- Orchestrator appends here. Do not edit by hand. -->
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
