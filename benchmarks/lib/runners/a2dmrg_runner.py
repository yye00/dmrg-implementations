"""
Runner for A2DMRG (Additive Two-Level DMRG).

Thin wrapper around pdmrg_runner since A2DMRG uses the same MPI launch pattern
but calls a2dmrg_main instead of pdmrg_main.
"""

from benchmarks.lib.runners.pdmrg_runner import run as pdmrg_run


def run(model, L, chi, max_sweeps=30, tol=1e-11, np_count=2, threads=1,
        n_max=2, **kwargs):
    """Run A2DMRG via MPI. See pdmrg_runner.run for parameter docs."""
    return pdmrg_run(
        model=model, L=L, chi=chi, max_sweeps=max_sweeps, tol=tol,
        np_count=np_count, threads=threads,
        package="a2dmrg", entry="a2dmrg.dmrg", function="a2dmrg_main",
        n_max=n_max, **kwargs,
    )
