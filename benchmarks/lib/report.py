"""
Report generation from benchmark results.

Produces formatted tables and markdown reports.
"""

import json
from datetime import datetime


def format_energy(e):
    """Format energy value."""
    if e is None:
        return "N/A"
    return f"{e:.12f}"


def format_time(t):
    """Format time value."""
    if t is None:
        return "N/A"
    if t < 1.0:
        return f"{t*1000:.0f}ms"
    return f"{t:.2f}s"


def print_results_table(results, title="Benchmark Results"):
    """Print a formatted results table to stdout.

    Args:
        results: list of dicts with keys: impl, model, size, np, threads, energy, time, success
        title: table header
    """
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"{'Implementation':<20} {'Model':<12} {'Size':<8} {'np':<4} {'Thr':<4} "
          f"{'Energy':<20} {'Time':<10} {'Status':<8}")
    print(f"{'-'*90}")

    for r in results:
        status = "PASS" if r.get("success") else "FAIL"
        np_str = str(r.get("np", "-"))
        thr_str = str(r.get("threads", "-"))
        print(f"{r['impl']:<20} {r['model']:<12} {r.get('size', '-'):<8} "
              f"{np_str:<4} {thr_str:<4} "
              f"{format_energy(r.get('energy')):<20} "
              f"{format_time(r.get('time')):<10} {status:<8}")

    print(f"{'='*90}\n")


def save_results(results, output_path, metadata=None):
    """Save results to JSON file.

    Args:
        results: list of result dicts
        output_path: path to write JSON
        metadata: optional extra metadata dict
    """
    output = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    if metadata:
        output["metadata"] = metadata

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_path}")


def generate_markdown_report(results, gold_standard=None):
    """Generate a markdown report from benchmark results.

    Args:
        results: list of result dicts
        gold_standard: optional dict of reference energies {(model, size): energy}

    Returns:
        Markdown string
    """
    lines = [
        f"# DMRG Benchmark Report",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
    ]

    # Group by model
    models = sorted(set(r["model"] for r in results))
    for model in models:
        model_results = [r for r in results if r["model"] == model]
        lines.append(f"## {model.title()} Model")
        lines.append("")
        lines.append("| Implementation | Size | np | Threads | Energy | Time | Status |")
        lines.append("|---|---|---|---|---|---|---|")

        for r in model_results:
            status = "PASS" if r.get("success") else "FAIL"
            lines.append(
                f"| {r['impl']} | {r.get('size', '-')} | {r.get('np', '-')} | "
                f"{r.get('threads', '-')} | {format_energy(r.get('energy'))} | "
                f"{format_time(r.get('time'))} | {status} |"
            )
        lines.append("")

    return "\n".join(lines)
