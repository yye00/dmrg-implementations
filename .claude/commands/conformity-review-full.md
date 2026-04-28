---
description: Orchestrator — runs all 6 vertical + horizontal review commands in parallel, dedupes overlapping findings, synthesizes a single criticals/highs/mediums report. Used before major events (GPU runs, paper deadlines).
---

Run the full conformity review across all DMRG implementations:
3 vertical (per-family, per-tier completeness) + 3 horizontal
(per-tier, cross-algorithm conformity) = 6 sub-reviews in parallel.
Synthesize into a single deduplicated report.

## When to use

Before major events:
- An MI300X GPU window is about to open (G1 baseline campaign).
- A paper revision is being prepared for submission.
- A round-N audit is being kicked off (every audit pass).

## Procedure

0. **Build the regression-watch list FIRST.** Before dispatching the
   sub-reviews, build a list of fixes since the last conformity
   baseline. This list MUST be embedded in each sub-reviewer's brief
   so technique G (sibling fix-propagation) has concrete fixes to
   trace. Procedure:

   - Find the most recent `reviews/conformity-*.md` baseline (the
     report from the previous orchestrator run).
   - `git log --oneline <baseline-commit>..HEAD -- gpu-rocm/` to
     enumerate fixes since.
   - For each fix, identify the **defect class** (canonical-Vh swap
     missing; Davidson `d_dav_work_` aliasing; pointer-mode RAII
     missing; etc.), not just the file. Defect classes are what
     siblings can also have.
   - Pass the list to each sub-reviewer in their brief as the
     "regression watch list" they must verify did NOT regress AND
     did propagate to siblings.

1. **Spawn 6 sub-reviews in parallel** — issue 6 Agent calls in a
   single message:

   - `/vertical-review-dmrg`
   - `/vertical-review-dmrg2`
   - `/vertical-review-pdmrg`
   - `/horizontal-review-base`
   - `/horizontal-review-gpu`
   - `/horizontal-review-opt`

   Each sub-review reads `.claude/review-methodology.md` and
   follows techniques A-G. Each emits a standard Markdown report
   with criticals / highs / mediums / nits / false-positives.

2. **Collect the 6 reports.** They will overlap — for example, a
   defect in dmrg-gpu-opt may surface in BOTH vertical-review-dmrg
   and horizontal-review-opt. That overlap is intentional (different
   review angles catch different aspects of the same defect).

3. **Deduplicate findings.** A finding is "the same" if it cites
   the same file:line AND the same root cause. When duplicates are
   found, merge them into a single entry preserving:
   - The strictest severity from any sub-review.
   - The clearest description.
   - Both review angles in a "found by:" tag.

4. **Synthesize.** Produce a single Markdown report:

```markdown
# Full conformity review — <date>

## Charter proof — sub-review status

| Sub-review | Status | Findings |
|---|---|---|
| vertical-review-dmrg | OK / FAILED | n criticals, n highs |
| vertical-review-dmrg2 | OK / FAILED | n criticals, n highs |
| vertical-review-pdmrg | OK / FAILED | n criticals, n highs |
| horizontal-review-base | OK / FAILED | n criticals, n highs |
| horizontal-review-gpu | OK / FAILED | n criticals, n highs |
| horizontal-review-opt | OK / FAILED | n criticals, n highs |

A sub-review is FAILED if any of its A-E techniques was SKIPPED.

## CRITICALS (deduplicated, sorted by file)
- [variant: file:line] description (found by: <reviews>)

## HIGHS (deduplicated)
- ...

## MEDIUMS (deduplicated)
- ...

## NITS (deduplicated)
- ...

## FALSE POSITIVES (cross-review verified)
- ...

## SUMMARY VERDICT

- **Block GPU run / paper submission?** YES / NO — count of
  CRITICALS.
- **Top-3 actions before next major event.**
- **What was checked vs. last conformity review:** if a previous
  full-conformity report exists in the repo, link it and note any
  regressions (findings that re-appear) — those are the most
  serious.
```

5. **Save the synthesized report** to
   `reviews/conformity-<YYYYMMDD>.md` so future sub-reviews can
   diff against it.

## Notes

- The 6 sub-reviews are intentionally redundant — a defect that
  evades vertical review may be caught by horizontal review and vice
  versa. This is the correct cost-of-overlap tradeoff for a final
  pre-event audit.
- If running ad-hoc mid-development, prefer a single
  vertical-review or horizontal-review depending on what changed.
  The orchestrator is for major-event gating, not routine review.
- Report length budget for the synthesized output: ≤ 2000 words.
