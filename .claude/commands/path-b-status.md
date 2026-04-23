---
description: Show the current Path B cluster status from the tracking issue.
---

Fetch the body of the "Path B execution tracker" issue on
`yye00/dmrg-implementations` and print:

1. The status table verbatim.
2. Any cluster whose row status is `human-review-required`, `waiting_gpu`,
   or `needs-author-input`, with its PR URL.
3. Any cluster that has not been touched in more than 48 hours (compute from
   the `updated_at` on the referenced PR).

Keep the output to ≤30 lines. Do not spawn any subagents — this is a
read-only status command.
