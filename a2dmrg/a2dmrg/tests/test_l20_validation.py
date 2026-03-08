"""Test #35: L=20 validation against quimb DMRG."""

import pytest
import numpy as np
from a2dmrg.mpi_compat import MPI
from quimb.tensor import SpinHam1D, DMRG2
from a2dmrg.dmrg import a2dmrg_main

pytestmark = pytest.mark.mpi


def test_l20_matches_quimb():
    """Test #35: A2DMRG matches quimb for L=20."""
    L = 20
    bond_dim = 50

    # Step 1-2: Run quimb DMRG2
    builder = SpinHam1D(S=1/2)
    builder += 1.0, "X", "X"
    builder += 1.0, "Y", "Y"
    builder += 1.0, "Z", "Z"
    mpo = builder.build_mpo(L)

    dmrg = DMRG2(mpo, bond_dims=bond_dim)
    dmrg.solve(tol=1e-10, verbosity=1)
    E_serial = dmrg.energy
    print(f"\nQuimb DMRG2 energy: {E_serial:.15f}", flush=True)

    # Step 3-4: Run A2DMRG
    energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=30,
        bond_dim=bond_dim,
        tol=1e-10,
        comm=MPI.COMM_WORLD,
        dtype=np.float64,
        one_site=True,
        verbose=True
    )
    print(f"A2DMRG energy:      {energy:.15f}", flush=True)

    # Step 5: Verify match
    diff = abs(energy - E_serial)
    print(f"Difference:         {diff:.3e}", flush=True)

    assert diff < 1e-10, f"Energies differ by {diff}"


if __name__ == "__main__":
    test_l20_matches_quimb()
