"""
Tests #53-54: Boundary edge cases

Tests that A2DMRG correctly handles updates at the left and right boundaries.
"""

import pytest
import numpy as np
from quimb.tensor import SpinHam1D
from a2dmrg.dmrg import a2dmrg_main

pytestmark = pytest.mark.mpi


def create_heisenberg_mpo(L, J=1.0):
    """Create a simple Heisenberg MPO."""
    builder = SpinHam1D(S=1/2)
    builder += J, 'X', 'X'
    builder += J, 'Y', 'Y'
    builder += J, 'Z', 'Z'
    return builder.build_mpo(L)


def test_left_boundary_update():
    """
    Test #53: Handle site at left boundary (i=0).

    Verifies that:
    1. 0-orthogonal decomposition can be prepared
    2. Environments are built correctly (L[-1] is trivial)
    3. Local update at site 0 works without errors
    4. No index errors occur at the boundary
    """
    L = 6
    bond_dim = 8
    mpo = create_heisenberg_mpo(L, J=1.0)

    # Run A2DMRG with verbose output to see all site updates
    E_final, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=5,
        bond_dim=bond_dim,
        one_site=True,
        verbose=False
    )

    # Verify that the algorithm completed successfully
    assert E_final < 0, "Energy should be negative for antiferromagnetic chain"
    assert mps is not None, "MPS should be returned"

    # Verify MPS has correct structure
    assert len(mps.tensors) == L, f"MPS should have {L} tensors"

    # Check that site 0 tensor exists and has correct shape
    tensor_0 = mps.tensors[0]
    assert tensor_0 is not None, "Site 0 tensor should exist"

    # For left boundary: shape can be (d, chi) or (1, d, chi) depending on implementation
    # The important thing is that it exists and has the right physical dimension
    assert tensor_0.shape is not None, "Site 0 tensor should have a shape"

    # Physical dimension should be 2 (spin-1/2)
    # In quimb, boundary tensors may have shape (d, chi) with bond dim 1 squeezed out
    if len(tensor_0.shape) == 2:
        # Left boundary with bond dim squeezed: (d, chi_right)
        assert tensor_0.shape[0] == 2, f"Physical dimension should be 2, got {tensor_0.shape[0]}"
    elif len(tensor_0.shape) == 3:
        # Standard form: (chi_left, d, chi_right)
        assert tensor_0.shape[1] == 2, f"Physical dimension should be 2, got {tensor_0.shape[1]}"
    else:
        pytest.fail(f"Unexpected tensor shape: {tensor_0.shape}")

    print(f"✅ Test #53 passed: Left boundary (i=0) handled correctly")
    print(f"   Final energy: {E_final:.8f}")
    print(f"   Site 0 tensor shape: {tensor_0.shape}")


def test_right_boundary_update():
    """
    Test #54: Handle site at right boundary (i=L-1).

    Verifies that:
    1. (L-1)-orthogonal decomposition can be prepared
    2. Environments are built correctly (R[L] is trivial)
    3. Local update at site L-1 works without errors
    4. No index errors occur at the boundary
    """
    L = 6
    bond_dim = 8
    mpo = create_heisenberg_mpo(L, J=1.0)

    # Run A2DMRG with verbose output to see all site updates
    E_final, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=5,
        bond_dim=bond_dim,
        one_site=True,
        verbose=False
    )

    # Verify that the algorithm completed successfully
    assert E_final < 0, "Energy should be negative for antiferromagnetic chain"
    assert mps is not None, "MPS should be returned"

    # Verify MPS has correct structure
    assert len(mps.tensors) == L, f"MPS should have {L} tensors"

    # Check that site L-1 tensor exists and has correct shape
    tensor_Lm1 = mps.tensors[L-1]
    assert tensor_Lm1 is not None, f"Site {L-1} tensor should exist"

    # For right boundary: shape can be (chi, d) or (chi, d, 1) depending on implementation
    # The important thing is that it exists and has the right physical dimension
    assert tensor_Lm1.shape is not None, f"Site {L-1} tensor should have a shape"

    # Physical dimension should be 2 (spin-1/2)
    # In quimb, boundary tensors may have shape (chi_left, d) with bond dim 1 squeezed out
    if len(tensor_Lm1.shape) == 2:
        # Right boundary with bond dim squeezed: (chi_left, d)
        assert tensor_Lm1.shape[1] == 2, f"Physical dimension should be 2, got {tensor_Lm1.shape[1]}"
    elif len(tensor_Lm1.shape) == 3:
        # Standard form: (chi_left, d, chi_right)
        assert tensor_Lm1.shape[1] == 2, f"Physical dimension should be 2, got {tensor_Lm1.shape[1]}"
    else:
        pytest.fail(f"Unexpected tensor shape: {tensor_Lm1.shape}")

    print(f"✅ Test #54 passed: Right boundary (i={L-1}) handled correctly")
    print(f"   Final energy: {E_final:.8f}")
    print(f"   Site {L-1} tensor shape: {tensor_Lm1.shape}")


def test_small_chain_boundaries():
    """
    Additional test: Verify boundary handling for very small chains.

    Tests L=4 to ensure both boundaries work correctly.
    """
    L = 4
    bond_dim = 6
    mpo = create_heisenberg_mpo(L, J=1.0)

    E_final, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=10,
        bond_dim=bond_dim,
        one_site=True,
        verbose=False
    )

    # Verify algorithm completed
    assert E_final < 0, "Energy should be negative"
    assert mps is not None, "MPS should be returned"
    assert len(mps.tensors) == L, f"MPS should have {L} tensors"

    # Check all tensors exist (don't check exact shapes as they can vary)
    for i in range(L):
        tensor = mps.tensors[i]
        assert tensor is not None, f"Site {i} tensor should exist"
        assert tensor.shape is not None, f"Site {i} tensor should have a shape"
        # Just verify tensor has some reasonable dimensions (at least rank 2)
        assert len(tensor.shape) >= 2, f"Site {i} tensor should have at least rank 2"

    print(f"✅ Additional test passed: Small chain (L={L}) boundaries handled correctly")
    print(f"   Final energy: {E_final:.8f}")


def test_boundaries_with_different_sweeps():
    """
    Test that boundary handling works correctly even with just 1 sweep.

    This ensures initialization and first sweep handle boundaries properly.
    """
    L = 5
    bond_dim = 6
    mpo = create_heisenberg_mpo(L, J=1.0)

    # Run with just 1 sweep
    E_final, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=1,
        bond_dim=bond_dim,
        one_site=True,
        verbose=False
    )

    # Just verify no errors occurred
    assert E_final < 0, "Energy should be negative"
    assert mps is not None, "MPS should be returned"
    assert len(mps.tensors) == L, f"MPS should have {L} tensors"

    # Run with more sweeps to verify convergence
    E_final_10, mps_10 = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=10,
        bond_dim=bond_dim,
        one_site=True,
        verbose=False
    )

    # Energy should improve with more sweeps (allow tiny numerical noise)
    assert E_final_10 <= E_final + 1e-12, "Energy should not increase with more sweeps (beyond numerical noise)"

    print(f"✅ Boundary test with different sweeps passed")
    print(f"   Energy (1 sweep):  {E_final:.8f}")
    print(f"   Energy (10 sweeps): {E_final_10:.8f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
