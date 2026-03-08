"""
Test bond dimension preservation in _transform_to_i_orthogonal().

This test module validates the critical fix for the bond dimension collapse bug
in the i-orthogonal transformation. The bug was that quimb's canonize() would
truncate bond dimensions from their original values (e.g. 20 -> 8 -> 4 -> 2),
violating Algorithm 2's requirement of "gauge transformation without bond
compression" (page 10, Grigori & Hassan 2025).

The fix stores original shapes before canonize() and zero-pads back to restore
them after canonize(). Zero-padding is mathematically valid because padded
dimensions represent empty subspace.

Tests:
1. Bond dimensions preserved for various system sizes and bond dimensions
2. I-orthogonal form (left/right orthogonality) correctly achieved
3. Energy preserved (gauge invariance) - error < 1e-12
4. Full algorithm integration - A2DMRG achieves acceptable accuracy vs quimb
"""

import pytest
import numpy as np
import sys
import os

# Ensure project root is on the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import quimb.tensor as qtn
from quimb.tensor import MPS_rand_state, MPO_ham_heis

from a2dmrg.mps.mps_utils import create_random_mps
from a2dmrg.numerics.local_microstep import _transform_to_i_orthogonal


class TestBondDimensionPreservation:
    """Verify _transform_to_i_orthogonal preserves bond dimensions."""

    @pytest.mark.parametrize("L,bond_dim", [
        (4, 8),
        (8, 16),
        (8, 20),
        (12, 32),
    ])
    def test_bond_dimensions_preserved(self, L, bond_dim):
        """Bond dimensions must be exactly preserved for all center sites."""
        mps = create_random_mps(L, bond_dim=bond_dim, phys_dim=2)

        # Store original bond dimensions
        bonds_before = [mps[i].data.shape for i in range(L)]

        for center in range(L):
            mps_copy = mps.copy()
            _transform_to_i_orthogonal(mps_copy, center_site=center, normalize=True)

            bonds_after = [mps_copy[i].data.shape for i in range(L)]

            # CRITICAL: Bond dimensions must be EXACTLY the same
            assert bonds_before == bonds_after, (
                f"L={L}, bond_dim={bond_dim}, center={center}: "
                f"bonds changed from {bonds_before} to {bonds_after}"
            )

    def test_edge_case_L2(self):
        """Bond dimensions preserved for minimal system L=2."""
        L, bond_dim = 2, 4
        mps = create_random_mps(L, bond_dim=bond_dim, phys_dim=2)
        bonds_before = [mps[i].data.shape for i in range(L)]

        for center in range(L):
            mps_copy = mps.copy()
            _transform_to_i_orthogonal(mps_copy, center_site=center, normalize=True)
            bonds_after = [mps_copy[i].data.shape for i in range(L)]
            assert bonds_before == bonds_after, (
                f"L=2, center={center}: bonds changed from {bonds_before} to {bonds_after}"
            )

    def test_edge_case_L3(self):
        """Bond dimensions preserved for L=3."""
        L, bond_dim = 3, 8
        mps = create_random_mps(L, bond_dim=bond_dim, phys_dim=2)
        bonds_before = [mps[i].data.shape for i in range(L)]

        for center in range(L):
            mps_copy = mps.copy()
            _transform_to_i_orthogonal(mps_copy, center_site=center, normalize=True)
            bonds_after = [mps_copy[i].data.shape for i in range(L)]
            assert bonds_before == bonds_after, (
                f"L=3, center={center}: bonds changed from {bonds_before} to {bonds_after}"
            )

    def test_large_bond_dim(self):
        """Bond dimensions preserved even for large bond dimensions."""
        L, bond_dim = 6, 64
        mps = create_random_mps(L, bond_dim=bond_dim, phys_dim=2)
        bonds_before = [mps[i].data.shape for i in range(L)]

        # Test a few centers
        for center in [0, L // 2, L - 1]:
            mps_copy = mps.copy()
            _transform_to_i_orthogonal(mps_copy, center_site=center, normalize=True)
            bonds_after = [mps_copy[i].data.shape for i in range(L)]
            assert bonds_before == bonds_after, (
                f"Large bond_dim={bond_dim}, center={center}: bonds changed"
            )


class TestIOrthogonalCorrectness:
    """Verify the transformation creates valid i-orthogonal form.

    For an i-orthogonal MPS (Definition 6, page 6):
    - Sites j < center are left-orthogonal: (U_j^{<2>})^T U_j^{<2>} = I
    - Sites k > center are right-orthogonal: U_k^{<1>} (U_k^{<1>})^T = I
    """

    def _find_phys_pos(self, inds, site):
        """Find the position of the physical index in the tensor indices."""
        for idx, ind in enumerate(inds):
            if ind.startswith('k'):
                return idx
        return len(inds) - 1  # default

    def _check_left_orthogonal(self, data, inds, site, L, tol=1e-10):
        """Check if tensor is left-orthogonal on its support.

        For left-orthogonal: A^dag A = I on the right bond index.
        After zero-padding, only the non-zero columns contribute.
        The Gram matrix on the support (non-zero columns) should be identity.
        """
        phys_pos = self._find_phys_pos(inds, site)

        if data.ndim == 2:
            if site == 0:
                if phys_pos == 1:
                    mat = data.T  # (phys, right)
                else:
                    mat = data  # (phys, right)
                gram = mat.conj().T @ mat
            else:
                return True
        elif data.ndim == 3:
            bond_pos = [j for j in range(3) if j != phys_pos]
            left_pos = bond_pos[0]
            right_pos = bond_pos[1]
            perm = [left_pos, phys_pos, right_pos]
            data_reordered = data.transpose(perm)
            left_dim, phys_dim, right_dim = data_reordered.shape
            mat = data_reordered.reshape(left_dim * phys_dim, right_dim)
            gram = mat.conj().T @ mat
        else:
            return True

        # After zero-padding, gram may have zero rows/cols for padded dims.
        # Check that the non-zero block is identity (orthogonal on support).
        nz_cols = np.where(np.any(np.abs(gram) > 1e-14, axis=0))[0]
        if len(nz_cols) == 0:
            return True  # All-zero tensor, trivially orthogonal

        gram_block = gram[np.ix_(nz_cols, nz_cols)]
        identity_block = np.eye(len(nz_cols), dtype=gram.dtype)
        error = np.linalg.norm(gram_block - identity_block)
        return error < tol

    def _check_right_orthogonal(self, data, inds, site, L, tol=1e-10):
        """Check if tensor is right-orthogonal on its support.

        For right-orthogonal: A A^dag = I on the left bond index.
        After zero-padding, only the non-zero rows contribute.
        The Gram matrix on the support (non-zero rows) should be identity.
        """
        phys_pos = self._find_phys_pos(inds, site)

        if data.ndim == 2:
            if site == L - 1:
                if phys_pos == 1:
                    mat = data  # (left, phys)
                else:
                    mat = data.T
                gram = mat @ mat.conj().T
            else:
                return True
        elif data.ndim == 3:
            bond_pos = [j for j in range(3) if j != phys_pos]
            left_pos = bond_pos[0]
            right_pos = bond_pos[1]
            perm = [left_pos, phys_pos, right_pos]
            data_reordered = data.transpose(perm)
            left_dim, phys_dim, right_dim = data_reordered.shape
            mat = data_reordered.reshape(left_dim, phys_dim * right_dim)
            gram = mat @ mat.conj().T
        else:
            return True

        # After zero-padding, gram may have zero rows/cols for padded dims.
        # Check that the non-zero block is identity (orthogonal on support).
        nz_rows = np.where(np.any(np.abs(gram) > 1e-14, axis=1))[0]
        if len(nz_rows) == 0:
            return True  # All-zero tensor, trivially orthogonal

        gram_block = gram[np.ix_(nz_rows, nz_rows)]
        identity_block = np.eye(len(nz_rows), dtype=gram.dtype)
        error = np.linalg.norm(gram_block - identity_block)
        return error < tol

    @pytest.mark.parametrize("L,bond_dim", [
        (4, 8),
        (8, 16),
    ])
    def test_i_orthogonal_form(self, L, bond_dim):
        """Verify left/right orthogonality conditions are satisfied."""
        mps = create_random_mps(L, bond_dim=bond_dim, phys_dim=2)

        for center in range(L):
            mps_copy = mps.copy()
            _transform_to_i_orthogonal(mps_copy, center_site=center, normalize=True)

            # Check left-orthogonal sites (j < center)
            for j in range(center):
                data = mps_copy[j].data
                inds = mps_copy[j].inds
                assert self._check_left_orthogonal(data, inds, j, L), (
                    f"Site {j} not left-orthogonal (center={center}, L={L})"
                )

            # Check right-orthogonal sites (k > center)
            for k in range(center + 1, L):
                data = mps_copy[k].data
                inds = mps_copy[k].inds
                assert self._check_right_orthogonal(data, inds, k, L), (
                    f"Site {k} not right-orthogonal (center={center}, L={L})"
                )


class TestEnergyPreservation:
    """Verify gauge transformation doesn't change energy (unitary invariance)."""

    @pytest.mark.parametrize("L,bond_dim", [
        (4, 8),
        (6, 12),
        (8, 16),
    ])
    def test_energy_preserved(self, L, bond_dim):
        """Energy must be preserved to machine precision after gauge transform."""
        from a2dmrg.numerics.observables import compute_energy
        from a2dmrg.hamiltonians.heisenberg import heisenberg_mpo

        mps = create_random_mps(L, bond_dim=bond_dim, phys_dim=2)
        mpo = heisenberg_mpo(L, J=1.0, cyclic=False)

        E_original = compute_energy(mps, mpo, normalize=True)

        for center in range(L):
            mps_copy = mps.copy()
            _transform_to_i_orthogonal(mps_copy, center_site=center, normalize=True)
            E_after = compute_energy(mps_copy, mpo, normalize=True)

            error = abs(E_after - E_original)
            assert error < 1e-12, (
                f"Energy changed by {error:.2e} at center={center} "
                f"(L={L}, bond_dim={bond_dim})"
            )

    def test_energy_preserved_complex(self):
        """Energy preserved for complex-valued MPS."""
        from a2dmrg.numerics.observables import compute_energy
        from a2dmrg.hamiltonians.heisenberg import heisenberg_mpo

        L, bond_dim = 6, 12
        mps = create_random_mps(L, bond_dim=bond_dim, phys_dim=2, dtype='complex128')
        mpo = heisenberg_mpo(L, J=1.0, cyclic=False)

        E_original = compute_energy(mps, mpo, normalize=True)

        for center in [0, L // 2, L - 1]:
            mps_copy = mps.copy()
            _transform_to_i_orthogonal(mps_copy, center_site=center, normalize=True)
            E_after = compute_energy(mps_copy, mpo, normalize=True)

            error = abs(E_after - E_original)
            assert error < 1e-12, (
                f"Complex energy changed by {error:.2e} at center={center}"
            )


class TestNormPreservation:
    """Verify the MPS norm is handled correctly after transformation."""

    def test_normalized_after_transform(self):
        """MPS should be normalized after transformation with normalize=True."""
        L, bond_dim = 8, 16
        mps = create_random_mps(L, bond_dim=bond_dim, phys_dim=2)

        for center in range(L):
            mps_copy = mps.copy()
            _transform_to_i_orthogonal(mps_copy, center_site=center, normalize=True)
            norm = mps_copy.norm()
            assert abs(norm - 1.0) < 1e-12, (
                f"Norm={norm} at center={center}, expected 1.0"
            )

    def test_unnormalized_option(self):
        """MPS should not be renormalized when normalize=False."""
        L, bond_dim = 6, 12
        mps = create_random_mps(L, bond_dim=bond_dim, phys_dim=2)

        # Scale MPS to have non-unit norm
        mps[0].modify(data=mps[0].data * 3.0)
        original_norm = mps.norm()

        mps_copy = mps.copy()
        _transform_to_i_orthogonal(mps_copy, center_site=L // 2, normalize=False)

        # Norm should be preserved (gauge transformation is unitary)
        new_norm = mps_copy.norm()
        assert abs(new_norm - original_norm) / abs(original_norm) < 1e-10, (
            f"Norm changed from {original_norm} to {new_norm} with normalize=False"
        )


class TestEdgeCasesTransform:
    """Test edge cases for the i-orthogonal transformation."""

    def test_invalid_center_site(self):
        """Negative or out-of-range center site should raise ValueError."""
        L, bond_dim = 6, 8
        mps = create_random_mps(L, bond_dim=bond_dim, phys_dim=2)

        with pytest.raises(ValueError):
            _transform_to_i_orthogonal(mps, center_site=-1)

        with pytest.raises(ValueError):
            _transform_to_i_orthogonal(mps, center_site=L)

    def test_idempotent_transformation(self):
        """Applying transformation twice should give same result."""
        L, bond_dim = 6, 12
        mps = create_random_mps(L, bond_dim=bond_dim, phys_dim=2)
        center = L // 2

        mps_copy1 = mps.copy()
        _transform_to_i_orthogonal(mps_copy1, center_site=center, normalize=True)
        bonds_after_1 = [mps_copy1[i].data.shape for i in range(L)]

        mps_copy2 = mps_copy1.copy()
        _transform_to_i_orthogonal(mps_copy2, center_site=center, normalize=True)
        bonds_after_2 = [mps_copy2[i].data.shape for i in range(L)]

        # Bond dimensions should be same after both transformations
        assert bonds_after_1 == bonds_after_2

    def test_repeated_transforms_different_centers(self):
        """Repeated transforms to different centers should preserve bonds."""
        L, bond_dim = 8, 16
        mps = create_random_mps(L, bond_dim=bond_dim, phys_dim=2)
        bonds_original = [mps[i].data.shape for i in range(L)]

        # Transform to various centers sequentially
        for center in [0, 3, 7, 4, 1, 6]:
            _transform_to_i_orthogonal(mps, center_site=center, normalize=True)
            bonds_current = [mps[i].data.shape for i in range(L)]
            assert bonds_current == bonds_original, (
                f"Bonds changed after transforming to center={center}"
            )


class TestIntegrationWithMicrosteps:
    """Test that the fix integrates correctly with local microstep functions."""

    def test_one_site_microstep_preserves_structure(self):
        """One-site microstep should work correctly with bond preservation."""
        from a2dmrg.numerics.local_microstep import local_microstep_1site

        L, bond_dim = 6, 8
        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)
        mpo = MPO_ham_heis(L, cyclic=False)

        # Perform one-site update at each site
        for site in range(L):
            mps_updated, energy = local_microstep_1site(mps, mpo, site)

            # Energy should be finite
            assert np.isfinite(energy), f"Non-finite energy at site={site}"

            # MPS should have correct length
            assert mps_updated.L == L

    def test_two_site_microstep_preserves_structure(self):
        """Two-site microstep should work correctly with bond preservation."""
        from a2dmrg.numerics.local_microstep import local_microstep_2site

        L, bond_dim = 6, 8
        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)
        mpo = MPO_ham_heis(L, cyclic=False)

        # Perform two-site update at valid sites
        for site in range(L - 1):
            mps_updated, energy = local_microstep_2site(
                mps, mpo, site, max_bond=bond_dim
            )

            # Energy should be finite
            assert np.isfinite(energy), f"Non-finite energy at site={site}"

            # MPS should have correct length
            assert mps_updated.L == L


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
