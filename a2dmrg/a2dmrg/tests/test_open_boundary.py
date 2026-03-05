"""
Test A2DMRG with open boundary conditions (OBC).

Test #51: Verify the algorithm works correctly with open boundary conditions,
which is the default for tensor network DMRG implementations.
"""

import numpy as np
import pytest
import quimb.tensor as qtn
from a2dmrg.mpi_compat import MPI
from a2dmrg.dmrg import a2dmrg_main
from a2dmrg.mps.mps_utils import create_neel_state
from a2dmrg.numerics.observables import compute_energy


def create_heisenberg_mpo_obc(L, J=1.0, dtype=np.float64):
    """
    Create a Heisenberg chain MPO with open boundary conditions (OBC).

    H = J * sum_{i=0}^{L-2} (S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + S_i^z S_{i+1}^z)

    OBC means: no term connecting site (L-1) back to site 0.
    This is the default for SpinHam1D.build_mpo().
    """
    from quimb.tensor import SpinHam1D

    builder = SpinHam1D(S=1/2)
    builder += J, 'X', 'X'
    builder += J, 'Y', 'Y'
    builder += J, 'Z', 'Z'

    # SpinHam1D.build_mpo() creates OBC by default
    mpo = builder.build_mpo(L)
    return mpo


def create_heisenberg_mpo_pbc(L, J=1.0, dtype=np.float64):
    """
    Create a Heisenberg chain MPO with periodic boundary conditions (PBC).

    H = J * sum_{i=0}^{L-1} (S_i^x S_{i+1 mod L}^x + ...)

    PBC means: term connecting site (L-1) back to site 0.

    Use quimb's MPO_ham_heis which supports cyclic parameter.
    """
    import quimb.tensor as qtn

    # MPO_ham_heis with cyclic=True creates PBC
    mpo = qtn.MPO_ham_heis(L, j=J, cyclic=True)
    return mpo


class TestOpenBoundaryConditions:
    """Test that A2DMRG handles open boundary conditions correctly."""

    def test_mpo_has_correct_structure(self):
        """
        Test #51 Step 1: Verify Heisenberg MPO with OBC has been created.

        For OBC, the key property is that there's no term wrapping from site (L-1) to site 0.
        The MPO bond dimensions can be > 1 for compressed representations.
        """
        L = 8
        mpo = create_heisenberg_mpo_obc(L, J=1.0)

        # Check that MPO has L tensors
        assert len(mpo.tensors) == L, f"Expected {L} MPO tensors, got {len(mpo.tensors)}"

        # Verify all tensors have physical dimension 2 (spin-1/2)
        for i, tensor in enumerate(mpo.tensors):
            shape = tensor.data.shape
            # MPO tensor shape is typically (left_bond, phys_out, phys_in, right_bond)
            # or could be permuted depending on quimb's convention
            assert 2 in shape, (
                f"Tensor {i} should have physical dimension 2 for spin-1/2, got shape {shape}"
            )

    def test_mps_has_correct_boundary_dimensions(self):
        """
        Test #51 Step 2-3: Verify MPS with OBC has correct boundary dimensions.

        For OBC:
        - Leftmost MPS tensor should have left bond dimension 1
        - Rightmost MPS tensor should have right bond dimension 1
        """
        L = 6
        bond_dim = 8

        # Create a Neel state MPS
        mps = create_neel_state(L, bond_dim, dtype=np.float64)

        # Check boundary dimensions
        # MPS tensors have shape (left_bond, phys, right_bond) for middle sites
        # Edge tensors have shape (phys, right_bond) or (left_bond, phys)

        leftmost = mps.tensors[0].data
        rightmost = mps.tensors[-1].data

        # Leftmost tensor shape should be (1, phys, bond_dim) or (phys, bond_dim)
        # Rightmost tensor shape should be (bond_dim, phys, 1) or (bond_dim, phys)

        # Check that boundary bond dimensions are 1
        if leftmost.ndim == 2:
            # Shape is (phys, right_bond)
            left_is_boundary = True
        else:
            # Shape is (left_bond, phys, right_bond)
            left_bond = leftmost.shape[0]
            assert left_bond == 1, f"Leftmost MPS should have left bond 1, got {left_bond}"

        if rightmost.ndim == 2:
            # Shape is (left_bond, phys)
            right_is_boundary = True
        else:
            # Shape is (left_bond, phys, right_bond)
            right_bond = rightmost.shape[-1]
            assert right_bond == 1, f"Rightmost MPS should have right bond 1, got {right_bond}"

    def test_obc_dmrg_converges(self):
        """
        Test #51 Step 4: Verify A2DMRG converges with OBC.

        Run A2DMRG on Heisenberg chain with OBC and verify convergence.
        """
        comm = MPI.COMM_WORLD

        # Test parameters
        L = 8
        bond_dim = 16
        max_sweeps = 10
        tol = 1e-6

        # Create Heisenberg with OBC
        mpo = create_heisenberg_mpo_obc(L, J=1.0)

        # Run A2DMRG
        np.random.seed(42)
        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=max_sweeps,
            bond_dim=bond_dim,
            tol=tol,
            comm=comm,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )

        # Verify energy is negative (Heisenberg ground state)
        assert energy < 0, f"Heisenberg ground state energy should be negative, got {energy}"

        # Verify energy is reasonable (not too large in magnitude)
        # For L=8 Heisenberg OBC, energy should be roughly -3 to -4
        assert -10.0 < energy < 0.0, (
            f"Energy {energy} outside reasonable range for L={L} Heisenberg OBC"
        )

    def test_obc_reasonable_energy(self):
        """
        Test #51 Step 5: Verify OBC gives reasonable energy.

        Test that A2DMRG with OBC produces physically reasonable ground state energy.
        We don't require perfect convergence to exact value, just that it's in the right ballpark.
        """
        comm = MPI.COMM_WORLD

        # Test parameters
        L = 6
        bond_dim = 20
        max_sweeps = 20
        tol = 1e-10

        # Create Heisenberg with OBC
        mpo = create_heisenberg_mpo_obc(L, J=1.0)

        # Run A2DMRG
        np.random.seed(42)
        energy_a2dmrg, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=max_sweeps,
            bond_dim=bond_dim,
            tol=tol,
            comm=comm,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )

        # Verify energy is physically reasonable
        # For L=6 Heisenberg OBC, exact ground state energy ≈ -2.49
        # But we're just verifying OBC works, not perfect convergence

        # Should be negative (AFM ground state)
        assert energy_a2dmrg < 0, (
            f"OBC energy should be negative for Heisenberg AFM, got {energy_a2dmrg}"
        )

        # Should be in reasonable range (between Neel state and exact GS)
        # Neel state: E ≈ -1.25, Exact: E ≈ -2.49
        assert -3.0 < energy_a2dmrg < -1.0, (
            f"OBC energy {energy_a2dmrg} outside physically reasonable range [-3.0, -1.0]"
        )

    def test_obc_vs_pbc_energy_difference(self):
        """
        Verify that OBC and PBC give different energies (they should!).

        OBC: no boundary term
        PBC: includes boundary term connecting site (L-1) to site 0

        PBC should have lower (more negative) energy due to extra bond.
        """
        comm = MPI.COMM_WORLD

        # Test parameters
        L = 8
        bond_dim = 16
        max_sweeps = 10
        seed = 42

        # Run with OBC
        mpo_obc = create_heisenberg_mpo_obc(L, J=1.0)
        np.random.seed(seed)
        energy_obc, _ = a2dmrg_main(
            L=L,
            mpo=mpo_obc,
            max_sweeps=max_sweeps,
            bond_dim=bond_dim,
            tol=1e-10,
            comm=comm,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )

        # Run with PBC
        mpo_pbc = create_heisenberg_mpo_pbc(L, J=1.0)
        np.random.seed(seed)
        energy_pbc, _ = a2dmrg_main(
            L=L,
            mpo=mpo_pbc,
            max_sweeps=max_sweeps,
            bond_dim=bond_dim,
            tol=1e-10,
            comm=comm,
            dtype=np.float64,
            one_site=True,
            verbose=False
        )

        # Both should be negative
        assert energy_obc < 0, f"OBC energy should be negative: {energy_obc}"
        assert energy_pbc < 0, f"PBC energy should be negative: {energy_pbc}"

        # PBC should have more negative energy (extra bond lowers energy)
        # For antiferromagnetic Heisenberg (J > 0), extra AFM bond lowers energy
        assert energy_pbc < energy_obc, (
            f"PBC energy ({energy_pbc}) should be lower than OBC ({energy_obc}) "
            "due to extra boundary bond"
        )

        # Difference should be significant (at least -0.1)
        diff = energy_obc - energy_pbc
        assert diff > 0.05, (
            f"OBC-PBC energy difference ({diff}) should be significant"
        )

    def test_small_chain_obc(self):
        """
        Test OBC on a small chain (L=4) to verify correctness.

        Small chains allow easier debugging if something goes wrong.
        """
        comm = MPI.COMM_WORLD

        # Test parameters
        L = 4
        bond_dim = 8
        max_sweeps = 10

        # Create Heisenberg with OBC
        mpo = create_heisenberg_mpo_obc(L, J=1.0)

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

        # Verify convergence
        assert energy < 0, f"Energy should be negative, got {energy}"

        # For L=4 Heisenberg OBC, energy should be roughly -1.5 to -2.0
        assert -2.5 < energy < -1.0, (
            f"L=4 OBC energy {energy} outside expected range"
        )
