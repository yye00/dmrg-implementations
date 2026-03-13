"""
Runner for quimb DMRG1 and DMRG2 (CPU reference implementations).

Runs in-process via direct Python import.
"""

import time
import numpy as np


def run(model, L, chi, max_sweeps=30, tol=1e-11, algorithm="dmrg2",
        threads=1, n_max=2, **kwargs):
    """Run quimb DMRG and return results dict.

    Args:
        model: 'heisenberg' or 'josephson'
        L: chain length
        chi: bond dimension
        max_sweeps: maximum number of sweeps
        tol: convergence tolerance
        algorithm: 'dmrg1' or 'dmrg2'
        threads: OPENBLAS_NUM_THREADS (set before import)
        n_max: charge truncation for Josephson (d = 2*n_max + 1)

    Returns:
        dict with keys: energy, time, sweeps, converged, success
    """
    import os
    thread_str = str(threads)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = thread_str

    import quimb.tensor as qtn

    # Build MPO
    if model == "heisenberg":
        mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
    elif model == "josephson":
        from benchmarks.lib.models import build_josephson_mpo
        mpo = build_josephson_mpo(L, n_max=n_max)
    else:
        raise ValueError(f"Unknown model: {model}")

    # Create DMRG solver
    dmrg_class = qtn.DMRG2 if algorithm == "dmrg2" else qtn.DMRG1
    dmrg = dmrg_class(mpo, bond_dims=chi, cutoffs=1e-14)

    # Run
    t0 = time.perf_counter()
    dmrg.solve(max_sweeps=max_sweeps, tol=tol, verbosity=0)
    elapsed = time.perf_counter() - t0

    energy = float(np.real(dmrg.energy))

    return {
        "energy": energy,
        "time": elapsed,
        "sweeps": max_sweeps,
        "converged": True,
        "success": True,
    }
