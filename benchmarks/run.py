#!/usr/bin/env python3
"""
DMRG Benchmark Suite - Unified Entry Point

Usage:
    ./run.py validate              Quick correctness check (all impls, small systems)
    ./run.py benchmark             Full timing suite (all impls, multiple sizes)
    ./run.py scale                 Scaling study (np and thread scaling)
    ./run.py report                Generate report from saved results
    ./run.py generate-data         Regenerate binary MPS/MPO test data
    ./run.py list                  List available implementations

Common flags:
    --impl NAME[,NAME]      Filter to specific implementation(s)
    --model NAME             heisenberg or josephson
    --size NAME              small, medium, or large
    --np N[,N]               MPI ranks (CPU) or HIP streams (GPU)
    --threads N[,N]          OPENBLAS_NUM_THREADS (CPU implementations)
    --output PATH            Save results to JSON file

Examples:
    ./run.py validate --impl quimb-dmrg2,pdmrg
    ./run.py benchmark --model heisenberg --size medium --np 2,4
    ./run.py scale --impl pdmrg --np 1,2,4,8 --threads 1
    ./run.py benchmark --impl dmrg-gpu,dmrg2-gpu --size large
"""

import argparse
import sys
import os

# Ensure the repo root is on sys.path so that 'benchmarks.lib' imports work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_list(value):
    """Parse a comma-separated string into a list."""
    if value is None:
        return None
    return [x.strip() for x in value.split(",")]


def parse_int_list(value):
    """Parse a comma-separated string into a list of ints."""
    if value is None:
        return None
    return [int(x.strip()) for x in value.split(",")]


def add_common_args(parser):
    """Add flags shared across subcommands."""
    parser.add_argument("--impl", type=str, default=None,
                        help="Comma-separated list of implementations to test")
    parser.add_argument("--model", type=str, default=None,
                        help="Model: heisenberg, josephson (default: both)")
    parser.add_argument("--np", type=str, default=None, dest="np_values",
                        help="Comma-separated MPI ranks (CPU) or HIP streams (GPU)")
    parser.add_argument("--threads", type=str, default=None, dest="thread_values",
                        help="Comma-separated OPENBLAS_NUM_THREADS values (CPU)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")


def cmd_validate(args):
    """Run quick correctness validation."""
    from benchmarks.validation.validate import validate

    impl_names = parse_list(args.impl)
    models = parse_list(args.model)
    np_values = parse_int_list(args.np_values)
    thread_values = parse_int_list(args.thread_values)

    results = validate(
        impl_names=impl_names,
        models=models,
        np_values=np_values,
        thread_values=thread_values,
    )

    if args.output:
        from benchmarks.lib.report import save_results
        save_results(results, args.output)

    # Exit with error if any failures
    if any(not r.get("success") for r in results):
        sys.exit(1)


def cmd_benchmark(args):
    """Run full timing benchmarks."""
    from benchmarks.performance.benchmark import benchmark

    impl_names = parse_list(args.impl)
    models = parse_list(args.model)
    sizes = parse_list(args.size) if args.size else None
    np_values = parse_int_list(args.np_values)
    thread_values = parse_int_list(args.thread_values)

    benchmark(
        impl_names=impl_names,
        models=models,
        sizes=sizes,
        np_values=np_values,
        thread_values=thread_values,
        output=args.output,
    )


def cmd_scale(args):
    """Run scaling study."""
    from benchmarks.performance.scaling import scale

    impl_names = parse_list(args.impl)
    models = parse_list(args.model)
    np_values = parse_int_list(args.np_values)
    thread_values = parse_int_list(args.thread_values)

    scale(
        impl_names=impl_names,
        models=models,
        size=args.size or "medium",
        np_values=np_values,
        thread_values=thread_values,
        output=args.output,
    )


def cmd_report(args):
    """Generate report from saved results."""
    import json
    from benchmarks.lib.report import generate_markdown_report

    if not args.input:
        print("Error: --input is required for report generation")
        sys.exit(1)

    with open(args.input) as f:
        data = json.load(f)

    results = data.get("results", data)
    report = generate_markdown_report(results)
    print(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")


def cmd_generate_data(args):
    """Regenerate binary MPS/MPO test data."""
    from benchmarks.data.generate import main as generate_main
    # Pass through to generate.py's argparse
    sys.argv = ["generate.py", "--all", "--seed", str(args.seed or 42)]
    if args.output:
        sys.argv.extend(["--output-dir", args.output])
    generate_main()


def cmd_list(args):
    """List available implementations."""
    from benchmarks.lib.registry import IMPLEMENTATIONS

    print(f"\n{'Implementation':<20} {'Type':<16} {'np':<6} {'Threads':<8} Description")
    print(f"{'-'*80}")
    for name, impl in sorted(IMPLEMENTATIONS.items()):
        np_flag = "yes" if impl["supports_np"] else "-"
        thr_flag = "yes" if impl["supports_threads"] else "-"
        print(f"{name:<20} {impl['type']:<16} {np_flag:<6} {thr_flag:<8} "
              f"{impl['description']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="DMRG Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # validate
    p_validate = subparsers.add_parser("validate",
                                        help="Quick correctness validation")
    add_common_args(p_validate)

    # benchmark
    p_bench = subparsers.add_parser("benchmark",
                                     help="Full timing benchmarks")
    add_common_args(p_bench)
    p_bench.add_argument("--size", type=str, default=None,
                         help="Size category: small, medium, large (comma-separated)")

    # scale
    p_scale = subparsers.add_parser("scale",
                                     help="Scaling study (np / thread scaling)")
    add_common_args(p_scale)
    p_scale.add_argument("--size", type=str, default="medium",
                         help="Problem size for scaling (default: medium)")

    # report
    p_report = subparsers.add_parser("report",
                                      help="Generate report from results JSON")
    p_report.add_argument("--input", type=str, required=False,
                          help="Input results JSON file")
    p_report.add_argument("--output", type=str, default=None,
                          help="Output markdown file")

    # generate-data
    p_gen = subparsers.add_parser("generate-data",
                                   help="Regenerate binary MPS/MPO test data")
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--output", type=str, default=None,
                       help="Output directory")

    # list
    subparsers.add_parser("list", help="List available implementations")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "validate": cmd_validate,
        "benchmark": cmd_benchmark,
        "scale": cmd_scale,
        "report": cmd_report,
        "generate-data": cmd_generate_data,
        "list": cmd_list,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
