#!/usr/bin/env python3
"""
analysis/statistical_summary.py — Statistical summary for Path B paper revision.

Consumes post-quarantine ablation JSONs, applies crash filter, computes per-cell
stats (median, CV, Wilson CI on win-rate, paired-bootstrap speedup CI), and
emits a CSV + Markdown table per variant.

Usage:
    python analysis/statistical_summary.py \\
        --ablation-root benchmarks/data/gpu_ablation \\
        --commit-pin <sha> \\
        --out reports/mi300x/stats_<sha>/

Ground truth: docs/PATH_B_GROUND_TRUTH.md (locked at 6f45533).

UNWIRED FLAG POLICY (per ground truth §"GpuOpts ablation flag wiring"):
  - dmrg-gpu-opt: device_k NOT wired, rsvd NOT wired, lanczos_graph force-disabled
  - dmrg2-gpu-opt: same as dmrg-gpu-opt
  - pdmrg-gpu-opt: device_k NOT wired through GpuOpts, rsvd NOT wired through GpuOpts
Cells whose config exercises an unwired flag are marked "unwired_flag=True" and
excluded from win-rate aggregation. They appear in output with a NOTE column.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# Allow running from repo root.
_BENCH_LIB = Path(__file__).resolve().parent.parent / "benchmarks" / "lib"
sys.path.insert(0, str(_BENCH_LIB))
from stats import (
    wilson_ci,
    paired_bootstrap_speedup,
    per_variant_win_threshold,
    coefficient_of_variation,
)

# Flags known to be unwired per ground truth §"GpuOpts ablation flag wiring"
UNWIRED_FLAGS: dict[str, set[str]] = {
    "dmrg-gpu-opt": {"DEVICE_K", "RSVD", "LANCZOS_GRAPH"},
    "dmrg2-gpu-opt": {"DEVICE_K", "RSVD", "LANCZOS_GRAPH"},
    "pdmrg-gpu-opt": {"DEVICE_K", "RSVD"},
}


def _is_unwired(variant: str, config: str) -> bool:
    """Return True if the config label exercises an unwired flag for this variant."""
    unwired = UNWIRED_FLAGS.get(variant, set())
    for flag in unwired:
        if flag in config.upper():
            return True
    return False


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"WARNING: cannot read {path}: {exc}", file=sys.stderr)
        return None


def _valid_walls(reps_data: list[dict]) -> list[float]:
    """Return wall_s values for valid (non-crashed) reps only."""
    return [
        r["wall_s"]
        for r in reps_data
        if r.get("valid", r.get("returncode") == 0)
        and r.get("energy") is not None
        and r.get("wall_s") is not None
    ]


def summarize_variant(
    variant: str,
    results: list[dict],
    commit_pin: str | None,
) -> list[dict]:
    """Compute per-cell summary rows for one variant's results list."""
    rows = []
    # Extract baseline wall times per problem for speedup computation.
    baseline_times: dict[str, list[float]] = {}
    for res in results:
        if res["config"] == "baseline":
            walls = _valid_walls(res["reps"])
            baseline_times[res["problem"]] = walls

    for res in results:
        problem = res["problem"]
        config = res["config"]
        reps_data = res.get("reps", [])
        walls = _valid_walls(reps_data)
        n_attempted = len(reps_data)
        n_valid = len(walls)
        n_failed = n_attempted - n_valid

        import numpy as np
        walls_arr = np.asarray(walls)
        median_s = float(np.median(walls_arr)) if walls else math.nan
        mean_s = float(np.mean(walls_arr)) if walls else math.nan
        cv = coefficient_of_variation(walls)

        ci95_lo = ci95_hi = math.nan
        if walls:
            # 95% CI via percentile bootstrap on the median.
            rng = np.random.default_rng(42)
            if n_valid >= 2:
                idx = rng.integers(0, n_valid, size=(10_000, n_valid))
                boot_medians = np.median(walls_arr[idx], axis=1)
                ci95_lo = float(np.percentile(boot_medians, 2.5))
                ci95_hi = float(np.percentile(boot_medians, 97.5))
            else:
                ci95_lo = ci95_hi = median_s

        # Speedup vs baseline.
        spd_median = spd_ci_lo = spd_ci_hi = math.nan
        win_threshold = math.nan
        is_win = False
        base_walls = baseline_times.get(problem, [])
        if base_walls and walls and config != "baseline":
            spd_median, spd_ci_lo, spd_ci_hi = paired_bootstrap_speedup(
                base_walls, walls
            )
            base_cv = coefficient_of_variation(base_walls)
            if not math.isnan(base_cv):
                win_threshold = per_variant_win_threshold(base_cv)
                is_win = (spd_ci_lo > win_threshold) if not math.isnan(spd_ci_lo) else False

        unwired = _is_unwired(variant, config)
        note = "unwired_flag" if unwired else ""

        row = {
            "variant": variant,
            "problem": problem,
            "config": config,
            "commit_pin": commit_pin or "",
            "n_attempted": n_attempted,
            "n_valid": n_valid,
            "n_failed": n_failed,
            "data_quality": res.get("data_quality", "OK" if n_failed == 0 else "DEGRADED"),
            "median_s": round(median_s, 4) if not math.isnan(median_s) else "",
            "mean_s": round(mean_s, 4) if not math.isnan(mean_s) else "",
            "cv": round(cv, 4) if not math.isnan(cv) else "",
            "ci95_lo": round(ci95_lo, 4) if not math.isnan(ci95_lo) else "",
            "ci95_hi": round(ci95_hi, 4) if not math.isnan(ci95_hi) else "",
            "speedup_vs_baseline": round(spd_median, 4) if not math.isnan(spd_median) else "",
            "speedup_ci_lo": round(spd_ci_lo, 4) if not math.isnan(spd_ci_lo) else "",
            "speedup_ci_hi": round(spd_ci_hi, 4) if not math.isnan(spd_ci_hi) else "",
            "win_threshold": round(win_threshold, 4) if not math.isnan(win_threshold) else "",
            "is_win": is_win,
            "unwired_flag": unwired,
            "note": note,
        }
        rows.append(row)
    return rows


def aggregate_win_rate(rows: list[dict]) -> tuple[int, int, float, float]:
    """Compute overall win-rate + Wilson 95% CI excluding unwired-flag cells and baseline."""
    eligible = [
        r for r in rows
        if not r["unwired_flag"] and r["config"] != "baseline" and r["is_win"] in (True, False)
    ]
    k = sum(1 for r in eligible if r["is_win"])
    n = len(eligible)
    lo, hi = wilson_ci(k, n)
    return k, n, lo, hi


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], variant: str, k: int, n: int, lo: float, hi: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Statistical summary: {variant}",
        "",
        f"Win-rate: {k}/{n} eligible configs declare a win.",
        f"Wilson 95% CI: [{lo:.3f}, {hi:.3f}]",
        "",
        "| problem | config | n_valid/attempted | median_s | cv | speedup | spd_ci_lo | spd_ci_hi | win_threshold | is_win | note |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        n_str = f"{r['n_valid']}/{r['n_attempted']}"
        spd = r.get("speedup_vs_baseline", "")
        lines.append(
            f"| {r['problem']} | `{r['config']}` | {n_str} | {r['median_s']} "
            f"| {r['cv']} | {spd} | {r['speedup_ci_lo']} | {r['speedup_ci_hi']} "
            f"| {r['win_threshold']} | {r['is_win']} | {r['note']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ablation-root", required=True, type=Path,
                        help="Root of gpu_ablation data dir (e.g. benchmarks/data/gpu_ablation).")
    parser.add_argument("--commit-pin", default=None,
                        help="Expected git commit SHA for all input JSONs. "
                             "Refuses to process JSONs with a different commit_sha.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output directory for per-variant CSV + Markdown.")
    args = parser.parse_args()

    ablation_root = args.ablation_root
    if not ablation_root.is_dir():
        print(f"ERROR: --ablation-root {ablation_root} does not exist.", file=sys.stderr)
        sys.exit(1)

    # Discover variant directories: ablation_root/<timestamp>/<variant>/results.json
    # Collect all results.json files, grouped by variant.
    variant_results: dict[str, list] = {}
    skipped_commit: list[Path] = []

    for ts_dir in sorted(ablation_root.iterdir()):
        if ts_dir.name.startswith("_") or ts_dir.name == "superseded":
            # Skip quarantine/superseded dirs.
            continue
        for variant_dir in sorted(ts_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            results_path = variant_dir / "results.json"
            if not results_path.exists():
                continue
            data = _load_json(results_path)
            if data is None:
                continue
            # Commit-pin check.
            if args.commit_pin:
                json_commit = (
                    data.get("provenance", {}).get("git_commit")
                    or data.get("commit_sha")
                )
                if json_commit and json_commit != args.commit_pin:
                    skipped_commit.append(results_path)
                    continue
            variant = variant_dir.name
            if variant not in variant_results:
                variant_results[variant] = []
            results_list = data.get("results", [])
            if results_list:
                variant_results[variant].extend(results_list)

    if skipped_commit:
        print(f"WARNING: Skipped {len(skipped_commit)} JSON(s) with commit_sha != {args.commit_pin}:")
        for p in skipped_commit:
            print(f"  {p}")

    if not variant_results:
        print("ERROR: No valid results found. Check --ablation-root and --commit-pin.", file=sys.stderr)
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)
    aggregate_rows: list[dict] = []

    for variant, results in sorted(variant_results.items()):
        rows = summarize_variant(variant, results, args.commit_pin)
        if not rows:
            continue
        k, n, lo, hi = aggregate_win_rate(rows)
        csv_path = args.out / f"{variant}_stats.csv"
        md_path = args.out / f"{variant}_stats.md"
        write_csv(rows, csv_path)
        write_markdown(rows, variant, k, n, lo, hi, md_path)
        print(f"{variant}: {len(rows)} cells, win-rate {k}/{n} Wilson CI [{lo:.3f},{hi:.3f}] → {csv_path}")
        aggregate_rows.extend(rows)

    # Write combined aggregate CSV.
    combined_csv = args.out / "all_variants_stats.csv"
    write_csv(aggregate_rows, combined_csv)
    print(f"\nCombined: {combined_csv}")


if __name__ == "__main__":
    main()
