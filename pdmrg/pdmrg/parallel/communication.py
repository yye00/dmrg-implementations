"""MPI communication utilities for PDMRG.

All inter-rank communication is point-to-point (rank ±1 only).
Uses blocking sendrecv to avoid deadlock (paper recommends blocking).
"""


def safe_exchange(comm, rank, neighbor, my_data):
    """Exchange data with a neighbor rank using sendrecv (deadlock-free).

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator.
    rank : int
        This rank's ID.
    neighbor : int
        Neighbor rank to exchange with.
    my_data : dict
        Data to send.

    Returns
    -------
    neighbor_data : dict
        Data received from neighbor.
    """
    neighbor_data = comm.sendrecv(my_data, dest=neighbor, source=neighbor)
    return neighbor_data


def check_convergence(E_local, E_prev, tol, comm, rtol=1e-12):
    """Check if all ranks have converged.

    Each rank contributes its best energy estimate from boundary merges.
    The merge energy from any bond IS the total system energy (since
    environments encode everything else). At convergence all merge
    energies agree, so we broadcast rank 0's value for consistency.

    Uses numpy.allclose-style convergence:
        converged = |dE| < atol + rtol * |E|
    
    This combines absolute tolerance (for small energies) with relative
    tolerance (for large energies), following standard numerical practice.

    Parameters
    ----------
    E_local : float
        Local energy estimate from this rank (from boundary merge).
    E_prev : float
        Previous global energy.
    tol : float
        Absolute convergence tolerance (atol).
    comm : MPI.Comm
    rtol : float, optional
        Relative tolerance, default 1e-12 (near float64 machine precision).

    Returns
    -------
    converged : bool
    E_global : float
        Current global energy.
    """
    from mpi4py import MPI

    n_procs = comm.Get_size()

    if n_procs == 1:
        E_global = E_local
    else:
        # Gather all energies and use the minimum active merge energy.
        # Each merge produces the total system energy; at convergence they
        # agree.  Use MIN of active merge energies (None for idle ranks
        # that had no merge on a given side).
        all_E = comm.allgather(E_local)
        merge_energies = [e for e in all_E if e is not None]
        if merge_energies:
            E_global = min(merge_energies)
        else:
            E_global = E_local if E_local is not None else 0.0

    # numpy.allclose style: |dE| < atol + rtol * |E|
    # This combines absolute tolerance (good for small E) with
    # relative tolerance (prevents over-iteration at machine precision)
    dE = abs(E_global - E_prev)
    effective_tol = tol + rtol * abs(E_global)
    
    converged = dE < effective_tol
    # All ranks must agree on convergence decision
    converged = comm.bcast(converged, root=0)

    return converged, E_global
