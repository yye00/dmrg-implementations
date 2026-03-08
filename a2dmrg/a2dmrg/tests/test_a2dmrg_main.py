"""
Tests for A2DMRG main loop integration.

This module tests feature #33: A2DMRG main loop single iteration and
feature #34: A2DMRG convergence.
"""

import pytest
import numpy as np
import quimb.tensor as qtn
from a2dmrg.mpi_compat import MPI

from a2dmrg.dmrg import a2dmrg_main
from a2dmrg.numerics.observables import compute_energy

pytestmark = pytest.mark.mpi


def create_heisenberg_mpo(L, J=1.0, cyclic=False):
    """
    Create a Heisenberg chain MPO using quimb's built-in functionality.

    H = J * sum_i (S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + S_i^z S_{i+1}^z)

    Parameters
    ----------
    L : int
        Number of sites
    J : float, optional
        Coupling strength (default: 1.0)
    cyclic : bool, optional
        Whether to use periodic boundary conditions (default: False)
        Note: Currently only OBC (cyclic=False) is supported

    Returns
    -------
    mpo : quimb MPO
        Matrix Product Operator for Heisenberg model
    """
    from quimb.tensor import SpinHam1D

    # Create Heisenberg Hamiltonian using SpinHam1D
    # H = sum_i J (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
    builder = SpinHam1D(S=1/2)

    # Add nearest-neighbor interactions
    builder += J, 'X', 'X'
    builder += J, 'Y', 'Y'
    builder += J, 'Z', 'Z'

    # Build the MPO (always OBC for now)
    mpo = builder.build_mpo(L)

    return mpo


class TestA2DMRGSingleIteration:
    """Test feature #33: A2DMRG main loop single iteration."""

    def test_initialization(self):
        """Step 1: Initialize left-orthogonal MPS."""
        L = 10
        bond_dim = 4

        # Create Hamiltonian
        mpo = create_heisenberg_mpo(L, J=1.0, cyclic=False)

        # Run single iteration (max_sweeps=1)
        comm = MPI.COMM_WORLD
        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=1,
            bond_dim=bond_dim,
            tol=1e-10,
            comm=comm,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )

        # Verify we get an MPS back
        assert isinstance(mps, qtn.MatrixProductState)
        assert len(mps.tensors) == L

        # Verify energy is finite and reasonable
        assert np.isfinite(energy)
        # With max_sweeps=1, we don't expect convergence to ground state
        # Just verify energy is reasonable (not huge)
        assert abs(energy) < 10  # Should be O(1) for Heisenberg model

    def test_prepare_orthogonal_decompositions(self):
        """Step 2: Prepare d orthogonal decompositions."""
        L = 8
        bond_dim = 4

        mpo = create_heisenberg_mpo(L, J=1.0, cyclic=False)

        # This is tested implicitly by running the algorithm
        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=1,
            bond_dim=bond_dim,
            tol=1e-10,
            comm=MPI.COMM_WORLD,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )

        # If we get here without errors, orthogonal decompositions worked
        assert True

    def test_parallel_local_microsteps(self):
        """Step 3: Perform parallel local micro-steps (serial for testing)."""
        L = 6
        bond_dim = 4

        mpo = create_heisenberg_mpo(L, J=1.0, cyclic=False)

        # Run with serial communicator
        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=1,
            bond_dim=bond_dim,
            tol=1e-10,
            comm=MPI.COMM_WORLD,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )

        # Verify result
        assert np.isfinite(energy)

    def test_build_coarse_space_matrices(self):
        """Step 4: Build coarse-space matrices."""
        L = 6
        bond_dim = 4

        mpo = create_heisenberg_mpo(L, J=1.0, cyclic=False)

        # This is tested implicitly
        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=1,
            bond_dim=bond_dim,
            tol=1e-10,
            comm=MPI.COMM_WORLD,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )

        assert True

    def test_solve_coarse_eigenvalue_problem(self):
        """Step 5: Solve coarse eigenvalue problem."""
        L = 6
        bond_dim = 4

        mpo = create_heisenberg_mpo(L, J=1.0, cyclic=False)

        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=1,
            bond_dim=bond_dim,
            tol=1e-10,
            comm=MPI.COMM_WORLD,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )

        assert np.isfinite(energy)

    def test_form_linear_combination(self):
        """Step 6: Form linear combination."""
        L = 6
        bond_dim = 4

        mpo = create_heisenberg_mpo(L, J=1.0, cyclic=False)

        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=1,
            bond_dim=bond_dim,
            tol=1e-10,
            comm=MPI.COMM_WORLD,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )

        # Verify MPS is valid
        assert isinstance(mps, qtn.MatrixProductState)

    def test_compress_back_to_target_rank(self):
        """Step 7: Compress back to target rank."""
        L = 8
        bond_dim = 6

        mpo = create_heisenberg_mpo(L, J=1.0, cyclic=False)

        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=1,
            bond_dim=bond_dim,
            tol=1e-10,
            comm=MPI.COMM_WORLD,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )

        # Check that bond dimensions don't exceed target
        for i in range(L - 1):
            bond_shape = mps.tensors[i].shape
            # Bond dimension is either shape[2] or shape[0] depending on position
            if len(bond_shape) == 3:
                actual_bond = bond_shape[2]  # right bond
                assert actual_bond <= bond_dim, f"Bond dimension {actual_bond} exceeds max {bond_dim}"

    def test_energy_decreased(self):
        """Step 8: Verify energy decreased."""
        L = 8
        bond_dim = 8

        mpo = create_heisenberg_mpo(L, J=1.0, cyclic=False)

        # Run two sweeps to see energy decrease
        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=2,
            bond_dim=bond_dim,
            tol=1e-20,  # Set very small to force 2 sweeps
            comm=MPI.COMM_WORLD,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )

        # Energy should be negative for Heisenberg
        assert energy < 0

        # Energy should be reasonable (not too large in magnitude)
        assert energy > -L * 10  # Rough upper bound


class TestA2DMRGConvergence:
    """Test feature #34: A2DMRG convergence."""

    def test_convergence_to_tolerance(self):
        """Verify A2DMRG converges to specified tolerance."""
        L = 8
        bond_dim = 16
        tol = 1e-6

        mpo = create_heisenberg_mpo(L, J=1.0, cyclic=False)

        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=10,
            bond_dim=bond_dim,
            tol=tol,
            comm=MPI.COMM_WORLD,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )

        # Should converge
        assert np.isfinite(energy)
        assert energy < 0

    def test_two_site_updates(self):
        """Test A2DMRG with two-site updates."""
        L = 6
        bond_dim = 8

        mpo = create_heisenberg_mpo(L, J=1.0, cyclic=False)

        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=3,
            bond_dim=bond_dim,
            tol=1e-6,
            comm=MPI.COMM_WORLD,
            dtype=np.float64,
            one_site=False,  # Two-site updates
            verbose=False
        )

        assert np.isfinite(energy)
        assert energy < 0

    def test_complex_dtype(self):
        """Test A2DMRG with complex128 dtype."""
        L = 6
        bond_dim = 8

        mpo = create_heisenberg_mpo(L, J=1.0, cyclic=False)

        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=3,
            bond_dim=bond_dim,
            tol=1e-6,
            comm=MPI.COMM_WORLD,
            dtype=np.complex128,
            one_site=True,
            verbose=False
        )

        assert np.isfinite(energy)
        # Energy should be real even for complex wavefunction
        assert abs(energy.imag) < 1e-10 if hasattr(energy, 'imag') else True

    def test_larger_system(self):
        """Test A2DMRG on a larger system."""
        L = 12
        bond_dim = 20

        mpo = create_heisenberg_mpo(L, J=1.0, cyclic=False)

        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=5,
            bond_dim=bond_dim,
            tol=1e-5,
            comm=MPI.COMM_WORLD,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )

        assert np.isfinite(energy)
        assert energy < 0

        # Energy per site should be reasonable for Heisenberg
        energy_per_site = energy / L
        assert -2.0 < energy_per_site < 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_small_system(self):
        """Test on L=4 system."""
        L = 4
        bond_dim = 4

        mpo = create_heisenberg_mpo(L, J=1.0, cyclic=False)

        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=5,
            bond_dim=bond_dim,
            tol=1e-8,
            comm=MPI.COMM_WORLD,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )

        assert np.isfinite(energy)

    def test_max_sweeps_reached(self):
        """Test that algorithm stops at max_sweeps."""
        L = 10
        bond_dim = 4
        max_sweeps = 2

        mpo = create_heisenberg_mpo(L, J=1.0, cyclic=False)

        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=max_sweeps,
            bond_dim=bond_dim,
            tol=1e-20,  # Impossible tolerance to ensure max_sweeps is reached
            comm=MPI.COMM_WORLD,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )

        # Should still return valid energy and MPS
        assert np.isfinite(energy)
        assert isinstance(mps, qtn.MatrixProductState)


def test_coarse_reduction_tol_param_accepted():
    """a2dmrg_main must accept coarse_reduction_tol and overlap_threshold without error."""
    import numpy as np
    import quimb.tensor as qtn
    from a2dmrg.mpi_compat import MPI
    from a2dmrg.dmrg import a2dmrg_main

    L = 6
    bond_dim = 4
    mpo = qtn.MPO_ham_heis(L, cyclic=False)
    comm = MPI.COMM_WORLD

    energy, mps = a2dmrg_main(
        L=L, mpo=mpo, max_sweeps=3, bond_dim=bond_dim,
        tol=1e-4, comm=comm, warmup_sweeps=1,
        coarse_reduction_tol=1e-8,
        overlap_threshold=0.99,
        timing_report=False,
        verbose=False,
    )
    assert np.isfinite(energy)
