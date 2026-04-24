"""
benchmarks/lib/stats.py — Statistical utilities for Path B paper revision.

Implements:
  - Wilson score CI for proportions (win-rate analysis)
  - Paired-bootstrap speedup CI
  - Per-variant noise-floor win threshold

Ground truth: docs/PATH_B_GROUND_TRUTH.md (locked at 6f45533).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for proportion k/n.

    Returns (lo, hi).  Handles n=0 by returning (0.0, 1.0).
    """
    if n == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(centre - half), float(centre + half))


def paired_bootstrap_speedup(
    t_base: "np.ndarray | list[float]",
    t_opt: "np.ndarray | list[float]",
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap over reps. Returns (median_speedup, lo95, hi95).

    Uses the shorter of the two arrays to keep sampling balanced.
    """
    rng = np.random.default_rng(seed)
    t_base = np.asarray(t_base, dtype=float)
    t_opt = np.asarray(t_opt, dtype=float)
    n = min(len(t_base), len(t_opt))
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_ratios = np.median(t_base[idx], axis=1) / np.median(t_opt[idx], axis=1)
    return (
        float(np.median(boot_ratios)),
        float(np.percentile(boot_ratios, 2.5)),
        float(np.percentile(boot_ratios, 97.5)),
    )


def per_variant_win_threshold(cv: float, k: float = 2.0) -> float:
    """Return the win threshold = 1 + k*CV.

    k=2 gives ~95% one-sided confidence that an observed speedup exceeds noise.
    For dmrg2-gpu RSVD (CV≈17%) this is 1.34×.
    For pdmrg-gpu LANCZOS_GRAPH (CV<1%) this is effectively 1.02×.
    """
    return 1.0 + k * cv


def coefficient_of_variation(times: "np.ndarray | list[float]") -> float:
    """CV = std / mean for a set of valid timing reps."""
    arr = np.asarray(times, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return float("nan")
    return float(np.std(arr, ddof=1) / np.mean(arr))


__all__ = [
    "wilson_ci",
    "paired_bootstrap_speedup",
    "per_variant_win_threshold",
    "coefficient_of_variation",
]
