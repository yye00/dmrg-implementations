"""
Quick correctness validation for DMRG implementations.

Runs each implementation on small systems and checks energy
against gold standard reference values.
"""

import json
import sys
from pathlib import Path

from benchmarks.lib.registry import (
    IMPLEMENTATIONS, SIZES, VALIDATION_TOL, CONVERGENCE_TOL,
    get_size, list_impls,
)
from benchmarks.lib.dispatch import run_implementation
from benchmarks.lib.report import print_results_table, format_energy

GOLD_STANDARD_PATH = Path(__file__).parent / "gold_standard.json"


def load_gold_standard():
    """Load gold standard reference energies."""
    if not GOLD_STANDARD_PATH.exists():
        print(f"Warning: gold standard file not found at {GOLD_STANDARD_PATH}")
        return {}
    with open(GOLD_STANDARD_PATH) as f:
        return json.load(f)


def validate(impl_names=None, models=None, np_values=None, thread_values=None):
    """Run validation suite.

    Args:
        impl_names: list of implementation names to test (None = all)
        models: list of model names to test (None = all)
        np_values: list of np values to test (None = [2])
        thread_values: list of thread counts (None = [1])

    Returns:
        list of result dicts, each with pass/fail status
    """
    if impl_names is None:
        impl_names = list(IMPLEMENTATIONS.keys())
    if models is None:
        models = ["heisenberg", "josephson"]
    if np_values is None:
        np_values = [2]
    if thread_values is None:
        thread_values = [1]

    gold = load_gold_standard()
    results = []
    passed = 0
    failed = 0

    for model in models:
        size_params = get_size(model, "small")

        for impl_name in impl_names:
            impl = IMPLEMENTATIONS[impl_name]

            # Determine np values for this implementation
            if impl["supports_np"]:
                nps = np_values
            else:
                nps = [1]

            # Determine thread values
            if impl["supports_threads"]:
                threads_list = thread_values
            else:
                threads_list = [1]

            for np_val in nps:
                for threads in threads_list:
                    print(f"  Validating {impl_name} on {model} "
                          f"(np={np_val}, threads={threads})...", end=" ", flush=True)

                    try:
                        result = run_implementation(
                            impl_name=impl_name,
                            model=model,
                            L=size_params["L"],
                            chi=size_params["chi"],
                            max_sweeps=size_params["max_sweeps"],
                            tol=CONVERGENCE_TOL,
                            np_count=np_val,
                            threads=threads,
                            n_max=size_params.get("n_max", 2),
                        )
                    except Exception as e:
                        result = {
                            "impl": impl_name, "model": model,
                            "energy": None, "time": None,
                            "success": False, "error": str(e),
                        }

                    result["size"] = "small"
                    result["np"] = np_val
                    result["threads"] = threads

                    # Check against gold standard
                    if result.get("success") and result.get("energy") is not None:
                        # Look for reference energy in gold standard
                        ref_energy = _find_reference(gold, model, size_params)
                        if ref_energy is not None:
                            error = abs(result["energy"] - ref_energy)
                            result["energy_error"] = error
                            if error > VALIDATION_TOL:
                                result["success"] = False
                                result["error"] = (
                                    f"Energy error {error:.2e} > tolerance {VALIDATION_TOL:.0e} "
                                    f"(got {format_energy(result['energy'])}, "
                                    f"ref {format_energy(ref_energy)})"
                                )
                        print(f"E={format_energy(result['energy'])} "
                              f"({'PASS' if result['success'] else 'FAIL'})")
                    else:
                        print(f"FAIL: {result.get('error', 'unknown error')[:60]}")

                    if result.get("success"):
                        passed += 1
                    else:
                        failed += 1

                    results.append(result)

    print_results_table(results, title="Validation Results")
    print(f"Summary: {passed} passed, {failed} failed, {passed + failed} total")

    return results


def _find_reference(gold, model, size_params):
    """Try to find a reference energy in the gold standard data."""
    if not gold:
        return None

    # The gold standard format varies; try common patterns
    L = size_params["L"]

    # Try direct lookup
    for key in gold:
        if isinstance(gold[key], dict):
            entry = gold[key]
            if entry.get("model") == model and entry.get("L") == L:
                return entry.get("energy")
            # Also check nested "results" key
            if "results" in entry:
                for r in entry["results"]:
                    if r.get("model") == model and r.get("L") == L:
                        return r.get("energy")

    return None
