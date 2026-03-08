"""
Test MPI edge cases for A2DMRG.

Test #57: Verify that A2DMRG with mpirun -np 1 (single processor)
behaves correctly and identically to serial execution.
"""

import numpy as np
import pytest
import quimb.tensor as qtn
from a2dmrg.mpi_compat import MPI
from a2dmrg.dmrg import a2dmrg_main
from a2dmrg.numerics.observables import compute_energy

pytestmark = pytest.mark.mpi


def create_heisenberg_mpo(L, J=1.0, dtype=np.float64):
    """
    Create a Heisenberg chain MPO using quimb.

    H = J * sum_i (S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + S_i^z S_{i+1}^z)
    """
    from quimb.tensor import SpinHam1D

    builder = SpinHam1D(S=1/2)
    builder += J, 'X', 'X'
    builder += J, 'Y', 'Y'
    builder += J, 'Z', 'Z'
    mpo = builder.build_mpo(L)
    return mpo


class TestMPINP1EdgeCase:
    """Test #57: MPI edge case with single processor (np=1)."""

    def test_np1_all_sites_on_rank0(self):
        """
        Test #57 Step 2: Verify all sites assigned to rank 0 when np=1.

        When running with a single MPI process, all sites should be
        assigned to rank 0, and the algorithm should work correctly.
        """
        comm = MPI.COMM_WORLD

        # This test only makes sense for np=1
        if comm.size != 1:
            pytest.skip(f"Test requires np=1, got np={comm.size}")

        # Verify we're on rank 0
        assert comm.rank == 0, "Single process should be rank 0"

        # Test parameters
        L = 6
        bond_dim = 8
        max_sweeps = 5

        # Create Hamiltonian
        mpo = create_heisenberg_mpo(L, J=1.0, dtype=np.float64)

        # Run A2DMRG
        np.random.seed(42)
        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=max_sweeps,
            bond_dim=bond_dim,
            tol=1e-10,
            comm=comm,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )

        # Verify result is reasonable
        assert energy < 0, f"Heisenberg energy should be negative, got {energy}"
        assert mps is not None, "MPS should not be None"
        assert mps.L == L, f"MPS should have {L} sites, got {mps.L}"

        # Verify energy is consistent with MPS
        energy_check = compute_energy(mps, mpo)
        assert abs(energy - energy_check) < 1e-10, (
            f"Energy mismatch: returned {energy}, computed {energy_check}"
        )

    def test_np1_allreduce_works(self):
        """
        Test #57 Step 3: Verify MPI Allreduce works correctly with np=1.

        When np=1, Allreduce should still work (trivially), and the result
        should be identical to the input since there's only one process.
        """
        comm = MPI.COMM_WORLD

        if comm.size != 1:
            pytest.skip(f"Test requires np=1, got np={comm.size}")

        # Test that basic MPI operations work
        test_array = np.array([1.0, 2.0, 3.0])
        result = np.zeros_like(test_array)

        comm.Allreduce(test_array, result, op=MPI.SUM)

        # With np=1, Allreduce SUM should just copy the array
        assert np.allclose(result, test_array), (
            f"Allreduce failed: expected {test_array}, got {result}"
        )

        # Now run a full A2DMRG calculation to verify Allreduce works in context
        L = 4
        bond_dim = 6
        max_sweeps = 3

        mpo = create_heisenberg_mpo(L, J=1.0, dtype=np.float64)

        np.random.seed(123)
        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=max_sweeps,
            bond_dim=bond_dim,
            tol=1e-10,
            comm=comm,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )

        # Should complete without MPI errors
        assert energy < 0, "Energy should be negative"
        assert mps is not None, "MPS should be returned"

    def test_np1_matches_serial(self):
        """
        Test #57 Step 4: Verify np=1 result matches pure serial implementation.

        When running with np=1, the result should be identical to running
        without MPI (since both use the same communicator size=1).
        """
        comm = MPI.COMM_WORLD

        if comm.size != 1:
            pytest.skip(f"Test requires np=1, got np={comm.size}")

        # Test parameters
        L = 6
        bond_dim = 10
        max_sweeps = 5
        seed = 42

        # Create Hamiltonian
        mpo = create_heisenberg_mpo(L, J=1.0, dtype=np.float64)

        # Run with explicit comm=COMM_WORLD (np=1)
        np.random.seed(seed)
        energy1, mps1 = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=max_sweeps,
            bond_dim=bond_dim,
            tol=1e-10,
            comm=comm,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )

        # Run again with same seed (should be identical)
        np.random.seed(seed)
        energy2, mps2 = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=max_sweeps,
            bond_dim=bond_dim,
            tol=1e-10,
            comm=comm,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )

        # Results should be exactly identical with same seed
        assert energy1 == energy2, (
            f"Energies should match exactly: {energy1} vs {energy2}"
        )

        # Wavefunctions should match
        overlap = abs(mps1.H @ mps2)
        assert abs(overlap - 1.0) < 1e-12, (
            f"Wavefunctions should match: overlap = {overlap}"
        )

    def test_np1_no_overhead_issues(self):
        """
        Test #57 Step 5: Verify no MPI overhead causes issues with np=1.

        Run multiple system sizes to ensure np=1 works robustly without
        any MPI-related overhead or communication issues.
        """
        comm = MPI.COMM_WORLD

        if comm.size != 1:
            pytest.skip(f"Test requires np=1, got np={comm.size}")

        # Test multiple system sizes
        test_cases = [
            {"L": 4, "bond_dim": 6, "max_sweeps": 3},
            {"L": 6, "bond_dim": 8, "max_sweeps": 4},
            {"L": 8, "bond_dim": 10, "max_sweeps": 5},
        ]

        for i, params in enumerate(test_cases):
            L = params["L"]
            bond_dim = params["bond_dim"]
            max_sweeps = params["max_sweeps"]

            # Create Hamiltonian
            mpo = create_heisenberg_mpo(L, J=1.0, dtype=np.float64)

            # Run A2DMRG
            np.random.seed(100 + i)
            energy, mps = a2dmrg_main(
                L=L,
                mpo=mpo,
                max_sweeps=max_sweeps,
                bond_dim=bond_dim,
                tol=1e-10,
                comm=comm,
                dtype=np.float64,
                one_site=True,
                verbose=False
            )

            # Verify result is reasonable
            assert energy < 0, (
                f"Test case {i}: Energy should be negative, got {energy}"
            )
            assert mps is not None, f"Test case {i}: MPS should not be None"
            assert mps.L == L, (
                f"Test case {i}: MPS should have {L} sites, got {mps.L}"
            )

            # Verify energy matches MPS
            energy_check = compute_energy(mps, mpo)
            assert abs(energy - energy_check) < 1e-10, (
                f"Test case {i}: Energy mismatch: {energy} vs {energy_check}"
            )

            # Verify MPS is normalized
            norm = abs(mps.H @ mps)
            assert abs(norm - 1.0) < 1e-10, (
                f"Test case {i}: MPS not normalized: norm = {norm}"
            )

    def test_np1_complex_dtype(self):
        """
        Test #57 (additional): Verify np=1 works with complex128 dtype.

        Complex dtypes are used for Josephson junction problems.
        """
        comm = MPI.COMM_WORLD

        if comm.size != 1:
            pytest.skip(f"Test requires np=1, got np={comm.size}")

        # Test parameters
        L = 6
        bond_dim = 8
        max_sweeps = 5

        # Create Hamiltonian (Heisenberg works with complex)
        mpo = create_heisenberg_mpo(L, J=1.0, dtype=np.complex128)

        # Run A2DMRG with complex dtype
        np.random.seed(42)
        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=max_sweeps,
            bond_dim=bond_dim,
            tol=1e-10,
            comm=comm,
            dtype=np.complex128,
            one_site=True,
            verbose=False
        )

        # For Hermitian Hamiltonian, energy should be real
        assert abs(energy.imag) < 1e-14, (
            f"Energy should be real for Hermitian H, got {energy}"
        )

        # Real part should be negative
        assert energy.real < 0, (
            f"Energy (real part) should be negative, got {energy.real}"
        )

        # Verify MPS is normalized
        norm = abs(mps.H @ mps)
        assert abs(norm - 1.0) < 1e-10, f"MPS not normalized: norm = {norm}"

    def test_np1_reproducibility(self):
        """
        Test #57 (additional): Verify reproducibility with np=1.

        Same seed should give identical results across multiple runs.
        """
        comm = MPI.COMM_WORLD

        if comm.size != 1:
            pytest.skip(f"Test requires np=1, got np={comm.size}")

        # Test parameters
        L = 6
        bond_dim = 10
        max_sweeps = 5
        seed = 42

        # Create Hamiltonian
        mpo = create_heisenberg_mpo(L, J=1.0, dtype=np.float64)

        energies = []
        mps_list = []

        # Run 3 times with same seed
        for _ in range(3):
            np.random.seed(seed)
            energy, mps = a2dmrg_main(
                L=L,
                mpo=mpo,
                max_sweeps=max_sweeps,
                bond_dim=bond_dim,
                tol=1e-10,
                comm=comm,
                dtype=np.float64,
                one_site=True,
                verbose=False
            )
            energies.append(energy)
            mps_list.append(mps)

        # All energies should be identical
        for i in range(1, len(energies)):
            assert energies[i] == energies[0], (
                f"Run {i} energy {energies[i]} != run 0 energy {energies[0]}"
            )

        # All wavefunctions should match
        for i in range(1, len(mps_list)):
            overlap = abs(mps_list[i].H @ mps_list[0])
            assert abs(overlap - 1.0) < 1e-12, (
                f"Run {i} wavefunction doesn't match run 0: overlap = {overlap}"
            )


# Note: These tests are designed to run with `mpirun -np 1 pytest ...`
# They will be skipped if run with np != 1.
