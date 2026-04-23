---
description: Run one Path B supervisor tick — dispatch the next role for in-flight PRs, refresh the tracking issue.
---

Spawn the `orchestrator` subagent to run one supervisory tick.

Brief to pass to the agent:

> Run one Path B tick per your agent definition. Read
> `docs/PATH_B_GROUND_TRUTH.md`, locate the "Path B execution tracker"
> issue on `yye00/dmrg-implementations`, inspect each cluster's PR state,
> dispatch at most one role this tick (reviewer, fixer, or implementer),
> update the tracking issue body, and report back in ≤10 lines.

After the agent returns, print its report verbatim. Do not re-interpret.
