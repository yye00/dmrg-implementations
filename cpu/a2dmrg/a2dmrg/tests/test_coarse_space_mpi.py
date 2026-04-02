"""MPI (mpi4py) integration tests for coarse-space matrix construction.

These tests are meant to be run under mpirun, e.g.:

  mpirun -np 2 python -m pytest -q a2dmrg/tests/test_coarse_space_mpi.py

They validate that the distributed row-wise build (assigned_sites + Allreduce)
reproduces the serial result.
"""

import numpy as np
import pytest

from a2dmrg.mpi_compat import MPI, HAS_MPI

import quimb.tensor as qtn

from a2dmrg.parallel.coarse_space import build_coarse_matrices
from a2dmrg.mps.mps_utils import create_random_mps


@pytest.mark.mpi
@pytest.mark.skipif(not HAS_MPI, reason="MPI not available")
def test_build_coarse_matrices_distributed_matches_serial():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    # This test requires >1 rank.
    if size < 2:
        pytest.skip("need mpirun -np >= 2")

    L = 6
    d_plus_1 = 7  # number of candidates

    mpo = qtn.MPO_ham_heis(L, j=1.0, bz=0.0, cyclic=False)

    # Build candidates on rank 0 and broadcast a compact serialization so every
    # rank contracts the *same* states (avoid RNG / backend nondeterminism).
    if rank == 0:
        np.random.seed(0)
        states = [
            create_random_mps(L, 4, phys_dim=2, dtype="float64", canonical="left")
            for _ in range(d_plus_1)
        ]
        payload = []
        for mps in states:
            tensors = []
            for t in mps.tensors:
                tensors.append((np.asarray(t.data), t.inds, list(t.tags)))
            payload.append({
                "L": mps.L,
                "site_tag_id": mps.site_tag_id,
                "tensors": tensors,
            })
    else:
        payload = None

    payload = comm.bcast(payload, root=0)

    states = []
    for m in payload:
        tn = qtn.TensorNetwork([qtn.Tensor(data=data, inds=inds, tags=tags) for data, inds, tags in m["tensors"]])
        states.append(
            qtn.MatrixProductState.from_TN(
                tn,
                site_tag_id=m["site_tag_id"],
                site_ind_id="k{}",
                cyclic=False,
                L=m["L"],
            )
        )

    # Serial reference computed on rank 0
    if rank == 0:
        H_serial, S_serial = build_coarse_matrices(states, mpo, comm=None)
    else:
        H_serial = None
        S_serial = None

    # Row-wise distribution across ranks (excluding row 0, which rank 0 also does)
    assigned_sites = [i for i in range(d_plus_1 - 1) if (i % size) == rank]

    H_dist, S_dist = build_coarse_matrices(
        states,
        mpo,
        comm=comm,
        assigned_sites=assigned_sites,
        filter_redundant=False,
    )

    if rank == 0:
        np.testing.assert_allclose(H_serial, H_dist, atol=1e-12)
        np.testing.assert_allclose(S_serial, S_dist, atol=1e-12)
