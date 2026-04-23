---
description: Fire the reviewer subagent on a specific Path B PR. Usage /review-pr <pr-number>
argument-hint: <pr-number>
---

Spawn the `reviewer` subagent on PR `$ARGUMENTS`.

Brief to pass to the agent:

> Review PR #$ARGUMENTS on `yye00/dmrg-implementations` per your agent
> definition. Read `docs/PATH_B_GROUND_TRUTH.md`, fetch the PR diff and
> the brief from the PR body, run the full checklist, post a single
> GitHub review with either APPROVE (label review-clean) or
> REQUEST_CHANGES (label review-blocked). Report PASS/FAIL and the
> review URL.
