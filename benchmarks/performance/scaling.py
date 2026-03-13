"""
Scaling studies: measure how performance varies with np, threads, or problem size.
"""

from benchmarks.lib.registry import (
    IMPLEMENTATIONS, SIZES, CONVERGENCE_TOL,
    get_size, list_impls,
)
from benchmarks.lib.dispatch import run_implementation
from benchmarks.lib.report import print_results_table, save_results, format_time


def scale(impl_names=None, models=None, size="medium",
          np_values=None, thread_values=None, output=None):
    """Run scaling study.

    Tests how performance changes as np or threads increase for each implementation.

    Args:
        impl_names: implementations to test (None = all parallel)
        models: models to test (None = ['heisenberg'])
        size: problem size to use
        np_values: np values to sweep (None = [1, 2, 4, 8])
        thread_values: thread values to sweep (None = [1, 2, 4, 8])
        output: path to save results JSON
    """
    if impl_names is None:
        # Default to parallel implementations for scaling study
        impl_names = list_impls("cpu-parallel") + list_impls("gpu-parallel")
    if models is None:
        models = ["heisenberg"]
    if np_values is None:
        np_values = [1, 2, 4, 8]
    if thread_values is None:
        thread_values = [1, 2, 4, 8]

    results = []

    for model in models:
        size_params = get_size(model, size)
        print(f"\n=== Scaling Study: {model.title()} {size} "
              f"(L={size_params['L']}, chi={size_params['chi']}) ===")

        for impl_name in impl_names:
            impl = IMPLEMENTATIONS[impl_name]
            print(f"\n  {impl_name}:")

            # np scaling
            if impl["supports_np"]:
                print(f"    np scaling (threads=1):")
                for np_val in np_values:
                    print(f"      np={np_val}...", end=" ", flush=True)
                    try:
                        result = run_implementation(
                            impl_name=impl_name, model=model,
                            L=size_params["L"], chi=size_params["chi"],
                            max_sweeps=size_params["max_sweeps"],
                            tol=CONVERGENCE_TOL, np_count=np_val, threads=1,
                            n_max=size_params.get("n_max", 2),
                        )
                    except Exception as e:
                        result = {
                            "impl": impl_name, "model": model,
                            "energy": None, "time": None,
                            "success": False, "error": str(e),
                        }

                    result["size"] = size
                    result["np"] = np_val
                    result["threads"] = 1
                    result["scaling_type"] = "np"

                    if result.get("success"):
                        print(f"{format_time(result.get('time'))}")
                    else:
                        print(f"FAIL")

                    results.append(result)

            # Thread scaling (CPU only)
            if impl["supports_threads"]:
                default_np = 2 if impl["supports_np"] else 1
                print(f"    Thread scaling (np={default_np}):")
                for threads in thread_values:
                    print(f"      threads={threads}...", end=" ", flush=True)
                    try:
                        result = run_implementation(
                            impl_name=impl_name, model=model,
                            L=size_params["L"], chi=size_params["chi"],
                            max_sweeps=size_params["max_sweeps"],
                            tol=CONVERGENCE_TOL, np_count=default_np,
                            threads=threads,
                            n_max=size_params.get("n_max", 2),
                        )
                    except Exception as e:
                        result = {
                            "impl": impl_name, "model": model,
                            "energy": None, "time": None,
                            "success": False, "error": str(e),
                        }

                    result["size"] = size
                    result["np"] = default_np
                    result["threads"] = threads
                    result["scaling_type"] = "threads"

                    if result.get("success"):
                        print(f"{format_time(result.get('time'))}")
                    else:
                        print(f"FAIL")

                    results.append(result)

    print_results_table(results, title="Scaling Results")

    if output:
        save_results(results, output)

    return results
