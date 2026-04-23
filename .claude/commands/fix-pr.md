---
description: Fire the fixer subagent on a review-blocked Path B PR. Usage /fix-pr <pr-number>
argument-hint: <pr-number>
---

Spawn the `fixer` subagent on PR `$ARGUMENTS`.

Brief to pass to the agent:

> Address unresolved reviewer comments on PR #$ARGUMENTS per your agent
> definition. Read `docs/PATH_B_GROUND_TRUTH.md`, fetch unresolved
> threads, count previous fixer passes (stop if ≥3), make one targeted
> follow-up commit, reply to and resolve each addressed thread, and
> re-label from `review-blocked` to `needs-review`. Report pass number,
> threads resolved, and commit SHA.
