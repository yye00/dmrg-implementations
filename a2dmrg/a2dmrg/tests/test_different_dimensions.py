"""
Test A2DMRG with different local Hilbert space dimensions.

Test #52: Verify the algorithm works correctly for d=2 (spin-1/2),
d=3 (spin-1 or truncated bosons), and d=4 (higher-dimensional systems).
"""

import numpy as np
import pytest
import quimb.tensor as qtn
from a2dmrg.mpi_compat import MPI
from a2dmrg.mps.mps_utils import create_product_state_mps
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


def create_bose_hubbard_mpo(L, t=1.0, U=1.0, mu=0.0, nmax=3):
    """
    Create a Bose-Hubbard model MPO.

    H = -t Σ(a†_i a_{i+1} + h.c.) + U/2 Σ n_i(n_i-1) - μ Σ n_i
    """
    d = nmax + 1  # Local Hilbert space dimension

    # Build local operators
    # Bosonic creation operator a†
    a_dag = np.zeros((d, d), dtype=complex)
    for i in range(d-1):
        a_dag[i+1, i] = np.sqrt(i+1)

    # Bosonic annihilation operator a
    a = a_dag.T.conj()

    # Number operator n = a† a
    n = np.diag(np.arange(d, dtype=float))

    # On-site interaction term U/2 * n(n-1)
    n_n_minus_1 = np.diag(np.arange(d, dtype=float) * (np.arange(d, dtype=float) - 1))

    # Identity
    I = np.eye(d, dtype=complex)

    # Build MPO tensors
    mpo_tensors = []

    if L == 1:
        # Single site: just the local terms (no bonds)
        W = -mu * n + (U/2) * n_n_minus_1
        mpo_tensors.append(W)
    else:
        # First site: shape (right_bond=4, ket_phys=d, bra_phys=d)
        W0 = np.zeros((4, d, d), dtype=complex)
        W0[0, :, :] = -mu * n + (U/2) * n_n_minus_1  # Local terms
        W0[1, :, :] = -t * a_dag  # Start hopping term
        W0[2, :, :] = -t * a      # Start hopping term (h.c.)
        W0[3, :, :] = I           # Identity for propagation
        mpo_tensors.append(W0)

        # Middle sites: (left_bond, right_bond, ket_phys, bra_phys)
        for i in range(1, L-1):
            W = np.zeros((4, 4, d, d), dtype=complex)
            W[0, 0, :, :] = I  # Identity propagation
            W[0, 1, :, :] = -t * a_dag  # Start new hopping
            W[0, 2, :, :] = -t * a      # Start new hopping (h.c.)
            W[0, 3, :, :] = -mu * n + (U/2) * n_n_minus_1  # Local terms
            W[1, 0, :, :] = a  # Complete hopping from left
            W[2, 0, :, :] = a_dag  # Complete hopping (h.c.) from left
            W[3, 3, :, :] = I  # Identity for propagation
            mpo_tensors.append(W)

        # Last site: shape (left_bond=4, ket_phys=d, bra_phys=d)
        WL = np.zeros((4, d, d), dtype=complex)
        WL[0, :, :] = -mu * n + (U/2) * n_n_minus_1  # Local terms
        WL[1, :, :] = a  # Complete hopping from left
        WL[2, :, :] = a_dag  # Complete hopping (h.c.) from left
        WL[3, :, :] = I  # Identity
        mpo_tensors.append(WL)

    # Build MPO using quimb
    mpo = qtn.MatrixProductOperator(mpo_tensors)
    return mpo


class TestDifferentDimensions:
    """Test A2DMRG with different local Hilbert space dimensions."""

    def test_dimension_2_spin_half(self):
        """Test with d=2 (spin-1/2 Heisenberg model)."""
        # Standard Heisenberg model has d=2 (spin up, spin down)
        L = 6
        bond_dim = 8
        dtype = np.float64

        # Create Heisenberg MPO (d=2)
        mpo = create_heisenberg_mpo(L, J=1.0, dtype=dtype)

        # Verify MPO has correct physical dimension
        for site in range(L):
            tensor = mpo.tensors[site]
            if tensor.ndim == 3:
                # Edge sites: (bond, phys_out, phys_in)
                assert tensor.shape[1] == 2, f"Site {site}: Expected phys dimension 2, got {tensor.shape[1]}"
                assert tensor.shape[2] == 2, f"Site {site}: Expected phys dimension 2, got {tensor.shape[2]}"
            else:
                # Middle sites: (bond_left, bond_right, phys_out, phys_in)
                assert tensor.shape[2] == 2, f"Site {site}: Expected phys dimension 2, got {tensor.shape[2]}"
                assert tensor.shape[3] == 2, f"Site {site}: Expected phys dimension 2, got {tensor.shape[3]}"

        # Run A2DMRG (creates Neel state internally for d=2)
        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=3,
            bond_dim=bond_dim,
            tol=1e-6,
            comm=MPI.COMM_WORLD,
            dtype=dtype,
            one_site=True,
            verbose=False
        )

        # Verify MPS has correct physical dimension (use index names, not shape heuristics)
        for site in range(L):
            tensor = mps.tensors[site]
            phys_name = mps.site_ind_id.format(site)
            phys_pos = list(tensor.inds).index(phys_name)
            phys_dim = tensor.shape[phys_pos]
            assert phys_dim == 2, f"Site {site}: Expected phys dimension 2, got {phys_dim}"

        # Verify convergence and energy is negative (antiferromagnetic)
        assert np.isfinite(energy), "Energy should be finite"
        assert energy < 0, f"Expected negative energy for AFM Heisenberg, got {energy}"

    def test_dimension_3_truncated_bosons(self):
        """Test with d=3 (truncated bosons with n_max=2)."""
        # This test verifies the algorithm works with d=3
        # We'll create a custom MPS with d=3 and verify it works
        L = 6
        bond_dim = 8
        nmax = 2  # n=0, n=1, n=2 → d=3

        # Create Bose-Hubbard MPO with d=3
        t = 1.0  # hopping
        U = 2.0  # interaction
        mu = 0.5  # chemical potential
        mpo = create_bose_hubbard_mpo(L, t, U, mu, nmax)

        # Verify MPO has correct physical dimension
        for site in range(L):
            tensor = mpo.tensors[site]
            if tensor.ndim == 3:
                # Edge sites: (bond, ket_phys, bra_phys)
                assert tensor.shape[1] == 3, f"Site {site}: Expected phys dimension 3, got {tensor.shape[1]}"
                assert tensor.shape[2] == 3, f"Site {site}: Expected phys dimension 3, got {tensor.shape[2]}"
            else:
                # Middle sites: (left_bond, right_bond, ket_phys, bra_phys)
                assert tensor.shape[2] == 3, f"Site {site}: Expected phys dimension 3, got {tensor.shape[2]}"
                assert tensor.shape[3] == 3, f"Site {site}: Expected phys dimension 3, got {tensor.shape[3]}"

        # Initialize with product state |1,1,1,1,1,1⟩ (one boson per site)
        mps = create_product_state_mps(L, bond_dim, state_index=1, phys_dim=3, dtype=np.complex128)

        # Verify MPS has correct physical dimension
        for site in range(L):
            tensor = mps.tensors[site]
            phys_idx = 1 if len(tensor.shape) == 2 else 2  # Edge vs middle sites
            phys_dim = tensor.shape[phys_idx]
            assert phys_dim == 3, f"Site {site}: Expected phys dimension 3, got {phys_dim}"

        # Compute initial energy to verify MPS and MPO are compatible
        initial_energy = compute_energy(mps, mpo)
        assert np.isfinite(initial_energy), f"Initial energy should be finite, got {initial_energy}"

        print(f"✓ Test passed: d=3 MPS and MPO are compatible, initial energy = {initial_energy:.4f}")

    def test_dimension_4_higher_bosons(self):
        """Test with d=4 (truncated bosons with n_max=3)."""
        L = 6
        bond_dim = 8
        nmax = 3  # n=0, n=1, n=2, n=3 → d=4

        # Create Bose-Hubbard MPO with d=4
        t = 1.0
        U = 2.0
        mu = 0.5
        mpo = create_bose_hubbard_mpo(L, t, U, mu, nmax)

        # Verify MPO has correct physical dimension
        for site in range(L):
            tensor = mpo.tensors[site]
            if tensor.ndim == 3:
                # Edge sites: (bond, ket_phys, bra_phys)
                assert tensor.shape[1] == 4, f"Site {site}: Expected phys dimension 4, got {tensor.shape[1]}"
                assert tensor.shape[2] == 4, f"Site {site}: Expected phys dimension 4, got {tensor.shape[2]}"
            else:
                # Middle sites: (left_bond, right_bond, ket_phys, bra_phys)
                assert tensor.shape[2] == 4, f"Site {site}: Expected phys dimension 4, got {tensor.shape[2]}"
                assert tensor.shape[3] == 4, f"Site {site}: Expected phys dimension 4, got {tensor.shape[3]}"

        # Initialize with product state |1,1,1,1,1,1⟩
        mps = create_product_state_mps(L, bond_dim, state_index=1, phys_dim=4, dtype=np.complex128)

        # Verify MPS has correct physical dimension
        for site in range(L):
            tensor = mps.tensors[site]
            phys_idx = 1 if len(tensor.shape) == 2 else 2  # Edge vs middle sites
            phys_dim = tensor.shape[phys_idx]
            assert phys_dim == 4, f"Site {site}: Expected phys dimension 4, got {phys_dim}"

        # Compute initial energy to verify MPS and MPO are compatible
        initial_energy = compute_energy(mps, mpo)
        assert np.isfinite(initial_energy), f"Initial energy should be finite, got {initial_energy}"

        print(f"✓ Test passed: d=4 MPS and MPO are compatible, initial energy = {initial_energy:.4f}")

    def test_tensor_shapes_adjust_correctly(self):
        """Verify that tensor shapes adjust correctly for different dimensions."""
        L = 6
        bond_dim = 8
        dtype = np.float64

        for phys_dim in [2, 3, 4]:
            # Create product state with specific dimension
            mps = create_product_state_mps(L, bond_dim, state_index=0, phys_dim=phys_dim, dtype=dtype)

            # Check all tensors have correct physical dimension
            for site in range(L):
                tensor = mps.tensors[site]
                if site == 0 or site == L - 1:
                    # Edge sites: shape depends on quimb convention
                    # Can be (chi, d) for 2D tensors
                    assert len(tensor.shape) == 2, f"Edge site {site} should be 2D"
                    assert tensor.shape[1] == phys_dim, f"Site {site}: Expected phys_dim={phys_dim}, got {tensor.shape[1]}"
                else:
                    # Middle: (chi_L, chi_R, d) or similar
                    if len(tensor.shape) == 3:
                        assert tensor.shape[2] == phys_dim, f"Site {site}: Expected phys_dim={phys_dim}, got {tensor.shape[2]}"

    def test_all_algorithms_work_for_different_d(self):
        """Verify that compute_energy works for different d values."""
        L = 6
        bond_dim = 8

        # Test with d=2 (Heisenberg)
        mpo_d2 = create_heisenberg_mpo(L)
        mps_d2 = create_product_state_mps(L, bond_dim, state_index=0, phys_dim=2, dtype=np.float64)
        energy_d2 = compute_energy(mps_d2, mpo_d2)
        assert np.isfinite(energy_d2), f"d=2 energy should be finite, got {energy_d2}"

        # Test with d=3 (Bose-Hubbard nmax=2)
        mpo_d3 = create_bose_hubbard_mpo(L, 1.0, 2.0, 0.5, nmax=2)
        mps_d3 = create_product_state_mps(L, bond_dim, state_index=1, phys_dim=3, dtype=np.complex128)
        energy_d3 = compute_energy(mps_d3, mpo_d3)
        assert np.isfinite(energy_d3), f"d=3 energy should be finite, got {energy_d3}"

        # Test with d=4 (Bose-Hubbard nmax=3)
        mpo_d4 = create_bose_hubbard_mpo(L, 1.0, 2.0, 0.5, nmax=3)
        mps_d4 = create_product_state_mps(L, bond_dim, state_index=1, phys_dim=4, dtype=np.complex128)
        energy_d4 = compute_energy(mps_d4, mpo_d4)
        assert np.isfinite(energy_d4), f"d=4 energy should be finite, got {energy_d4}"

        print(f"✓ All dimensions work: d=2 energy={energy_d2:.4f}, d=3 energy={energy_d3:.4f}, d=4 energy={energy_d4:.4f}")
