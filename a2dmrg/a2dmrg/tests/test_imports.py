"""
Test that all dependencies can be imported.

This basic test ensures the development environment is correctly configured.
"""

import pytest
import numpy as np
import scipy
import quimb


def test_numpy_import():
    """Test that NumPy is available."""
    assert np.__version__ is not None
    print(f"NumPy version: {np.__version__}")


def test_scipy_import():
    """Test that SciPy is available."""
    assert scipy.__version__ is not None
    print(f"SciPy version: {scipy.__version__}")


def test_quimb_import():
    """Test that quimb is available."""
    assert quimb.__version__ is not None
    print(f"Quimb version: {quimb.__version__}")


def test_mpi4py_import():
    """Test that mpi4py is available."""
    try:
        from a2dmrg.mpi_compat import MPI
        version = MPI.Get_version()
        assert version is not None
        print(f"MPI version: {version}")
    except (ImportError, RuntimeError) as e:
        # Skip if mpi4py or MPI library is not available
        # RuntimeError happens when libmpi.so cannot be found
        pytest.skip(f"MPI not available: {e}")


def test_a2dmrg_import():
    """Test that a2dmrg package can be imported."""
    import a2dmrg
    assert a2dmrg.__version__ is not None
    print(f"A2DMRG version: {a2dmrg.__version__}")


def test_basic_quimb_mps():
    """Test creating a basic MPS with quimb."""
    import quimb.tensor as qtn

    # Create a simple MPS
    mps = qtn.MPS_rand_state(L=4, bond_dim=2, dtype='float64')

    # Verify structure
    assert mps.L == 4
    assert len(mps.tensors) == 4

    # Check each tensor
    for i in range(4):
        tensor = mps[i]
        assert tensor.data is not None
        assert tensor.data.dtype == np.float64

    print("✓ Basic quimb MPS creation works")


def test_basic_quimb_mpo():
    """Test creating a basic MPO with quimb."""
    import quimb.tensor as qtn

    # Create Heisenberg Hamiltonian
    H = qtn.MPO_ham_heis(L=4, j=1.0, bz=0.0, cyclic=False)

    # Verify structure
    assert H.L == 4
    assert len(H.tensors) == 4

    print("✓ Basic quimb MPO creation works")


def test_mps_energy():
    """Test that we can use quimb's built-in DMRG."""
    import quimb.tensor as qtn

    # Create simple problem
    L = 4
    H = qtn.MPO_ham_heis(L=L, j=1.0, cyclic=False)

    # Use quimb's built-in DMRG (for validation reference later)
    dmrg = qtn.DMRG2(H, bond_dims=[10])
    dmrg.solve(tol=1e-6, verbosity=0)

    # Verify we got a result
    energy = dmrg.energy
    assert isinstance(energy, (int, float, complex, np.number))
    assert energy < 0  # Ground state of Heisenberg is negative
    print(f"✓ Quimb DMRG works: E = {float(np.real(energy)):.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
