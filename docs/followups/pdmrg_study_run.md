# PDMRG-GPU Performance Study — Warmup/Polish Sweep

**Date:** 20260415T130847Z
**Grid:** ULTRA_TRIM (18 configs/model × 3 models = 54 configs)
**Repeats:** 3
**Impl:** pdmrg-gpu

## Variants

| Tag | --warmup | --polish | Purpose |
|-----|----------|----------|---------|
| w0p0 | 0 | 0 | Pure PDMRG kernel — no serial overhead |
| w1p0 | 1 | 0 | Minimal warmup, no polish |
| w1p1 | 1 | 1 (1-site) | Light warmup + 1 single-site polish |
| w1p2 | 1 | 2 (1-site) | Light warmup + 2 single-site polish |

## Log

| variant | model | start | end | walltime | status |
|---------|-------|-------|-----|----------|--------|
