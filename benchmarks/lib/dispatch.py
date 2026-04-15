"""
Dispatch layer: maps implementation names to their runners.

This is the glue between the registry and the runner modules.
"""

from benchmarks.lib.registry import get_impl, IMPLEMENTATIONS


def run_implementation(impl_name, model, L, chi, max_sweeps=30, tol=1e-11,
                       np_count=None, threads=1, n_max=2, **kwargs):
    """Run a single implementation and return results.

    Args:
        impl_name: registered implementation name (e.g. 'quimb-dmrg2', 'pdmrg-gpu')
        model: 'heisenberg' or 'josephson'
        L, chi, max_sweeps, tol: DMRG parameters
        np_count: MPI ranks or HIP streams (None = use default)
        threads: BLAS threads per rank (CPU only)
        n_max: charge truncation for Josephson

    Returns:
        dict with at least: energy, time, success (plus impl-specific keys)
    """
    impl = get_impl(impl_name)
    runner_name = impl["runner"]

    # Set defaults for np
    if np_count is None:
        np_count = 2 if impl["supports_np"] else 1

    # Dispatch to the right runner
    if runner_name == "quimb":
        from benchmarks.lib.runners.quimb_runner import run
        result = run(
            model=model, L=L, chi=chi, max_sweeps=max_sweeps, tol=tol,
            algorithm=impl["algorithm"], threads=threads, n_max=n_max, **kwargs,
        )

    elif runner_name == "pdmrg":
        from benchmarks.lib.runners.pdmrg_runner import run
        result = run(
            model=model, L=L, chi=chi, max_sweeps=max_sweeps, tol=tol,
            np_count=np_count, threads=threads,
            package=impl["package"], entry=impl["entry"], function=impl["function"],
            n_max=n_max, **kwargs,
        )

    elif runner_name == "a2dmrg":
        from benchmarks.lib.runners.a2dmrg_runner import run
        result = run(
            model=model, L=L, chi=chi, max_sweeps=max_sweeps, tol=tol,
            np_count=np_count, threads=threads, n_max=n_max, **kwargs,
        )

    elif runner_name == "gpu":
        from benchmarks.lib.runners.gpu_runner import run
        result = run(
            model=model, L=L, chi=chi, max_sweeps=max_sweeps,
            executable=impl["executable"], np_count=np_count,
            n_max=n_max, **kwargs,
        )

    elif runner_name == "radam":
        from benchmarks.lib.runners.radam_runner import run
        result = run(
            model=model, L=L, chi=chi, max_sweeps=max_sweeps, tol=tol,
            threads=threads, n_max=n_max, **kwargs,
        )

    else:
        raise ValueError(f"Unknown runner type: {runner_name}")

    # Tag result with implementation info
    result["impl"] = impl_name
    result["model"] = model
    result["L"] = L
    result["chi"] = chi
    result["np"] = np_count
    result["threads"] = threads

    return result
