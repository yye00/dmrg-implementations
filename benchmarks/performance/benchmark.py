"""
Full timing benchmark suite.

Runs implementations across problem sizes and reports timing results.
"""

from benchmarks.lib.registry import (
    IMPLEMENTATIONS, SIZES, CONVERGENCE_TOL,
    get_size, list_impls,
)
from benchmarks.lib.dispatch import run_implementation
from benchmarks.lib.report import print_results_table, save_results


def benchmark(impl_names=None, models=None, sizes=None,
              np_values=None, thread_values=None, output=None):
    """Run timing benchmarks.

    Args:
        impl_names: implementations to test (None = all)
        models: models to test (None = all)
        sizes: size categories to test (None = ['small', 'medium'])
        np_values: np values for parallel impls (None = [2, 4])
        thread_values: thread counts for CPU impls (None = [1, 4])
        output: path to save results JSON (None = print only)

    Returns:
        list of result dicts
    """
    if impl_names is None:
        impl_names = list(IMPLEMENTATIONS.keys())
    if models is None:
        models = ["heisenberg", "josephson"]
    if sizes is None:
        sizes = ["small", "medium"]
    if np_values is None:
        np_values = [2, 4]
    if thread_values is None:
        thread_values = [1, 4]

    results = []

    for model in models:
        for size_name in sizes:
            try:
                size_params = get_size(model, size_name)
            except KeyError:
                continue

            print(f"\n--- {model.title()} {size_name} "
                  f"(L={size_params['L']}, chi={size_params['chi']}) ---")

            for impl_name in impl_names:
                impl = IMPLEMENTATIONS[impl_name]

                if impl["supports_np"]:
                    nps = np_values
                else:
                    nps = [1]

                if impl["supports_threads"]:
                    threads_list = thread_values
                else:
                    threads_list = [1]

                for np_val in nps:
                    for threads in threads_list:
                        label = f"{impl_name} (np={np_val}, t={threads})"
                        print(f"  Running {label}...", end=" ", flush=True)

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

                        result["size"] = size_name
                        result["np"] = np_val
                        result["threads"] = threads

                        if result.get("success"):
                            t = result.get("time")
                            t_str = f"{t:.2f}s" if t else "?"
                            print(f"{t_str}")
                        else:
                            print(f"FAIL: {result.get('error', '?')[:60]}")

                        results.append(result)

    print_results_table(results, title="Benchmark Results")

    if output:
        save_results(results, output)

    return results
