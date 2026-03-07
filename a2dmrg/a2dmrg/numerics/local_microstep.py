"""
Local DMRG micro-steps for A2DMRG.

This module implements the local micro-step operations for Phase 2 of the
Additive Two-Level DMRG (A2DMRG) algorithm:
- One-site DMRG update (Definition 9, Eq. 11 in Grigori & Hassan)
- Two-site DMRG update with SVD splitting (Definition 9, Eq. 12)

The critical requirement is that each micro-step operates on an
**i-orthogonal** tensor train decomposition (Definition 6, page 6),
where the orthogonality center is at the site(s) being optimized.
This ensures the effective Hamiltonian is well-conditioned (Lemma 10,
Remark 11) and the retraction operators are orthogonal projections
(Lemma 8).

Reference: Grigori & Hassan, "An Additive Two-Level Parallel Variant
of the DMRG Algorithm with Coarse-Space Correction",
arXiv:2505.23429v2 (2025).
"""

import numpy as np
import quimb.tensor as qtn
from typing import Tuple

from .effective_ham import build_effective_hamiltonian_1site, build_effective_hamiltonian_2site
from .eigensolver import solve_effective_hamiltonian
from .truncated_svd import truncated_svd
from ..environments.environment import build_left_environments, build_right_environments


def _transform_to_i_orthogonal(mps, center_site, normalize=True):
    """
    Transform MPS to i-orthogonal canonical form with orthogonality center
    at the specified site, WITHOUT changing bond dimensions.

    This implements the gauge transformation described in Definition 6
    (page 6) of Grigori & Hassan (2025):

        A TT decomposition U = (U_1, ..., U_d) is i-orthogonal if:
        - (U_j^{<2>})^T U_j^{<2>} = I  for j in {1, ..., i-1}  (left-orthogonal)
        - U_k^{<1>} (U_k^{<1>})^T = I  for k in {i+1, ..., d}  (right-orthogonal)

    The transformation is performed by:
    1. Sweeping LEFT (site 0 to center-1): QR decomposition makes each
       site left-orthogonal, contracting R into the next site.
    2. Sweeping RIGHT (site L-1 to center+1): LQ decomposition makes each
       site right-orthogonal, contracting L into the next site.

    This is a pure gauge transformation that preserves the physical state
    exactly (no truncation or compression). Bond dimensions are unchanged.

    Parameters
    ----------
    mps : quimb.tensor.MatrixProductState
        MPS to transform (modified in-place).
    center_site : int
        Site index for the orthogonality center (0-indexed).
    normalize : bool, optional
        Whether to normalize the MPS after transformation (default: True).

    Returns
    -------
    mps : quimb.tensor.MatrixProductState
        The same MPS object, now in i-orthogonal form.

    Notes
    -----
    This function uses quimb's built-in canonize() method, which performs
    exact QR/LQ sweeps without bond truncation. The bond dimensions are
    guaranteed to be preserved.

    See Also
    --------
    Algorithm 2, Step 1 (page 10): The orthogonalization sweep that
    creates i-orthogonal forms U^{(n),i} for each site i.
    """
    L = mps.L
    if not (0 <= center_site < L):
        raise ValueError(
            f"center_site={center_site} out of range for MPS with L={L}"
        )

    # quimb's canonize(where=center_site) performs exact QR sweep from left
    # and LQ sweep from right, placing the orthogonality center at the
    # specified site. This does NOT truncate bonds.
    mps.canonize(where=center_site)

    # Normalize if requested (the norm accumulates at the center site)
    if normalize:
        norm = mps.norm()
        if abs(norm) > 1e-15:
            mps /= norm

    return mps


def local_microstep_1site(
    mps: qtn.MatrixProductState,
    mpo,
    site: int,
    tol: float = 1e-10,
    L_env: np.ndarray = None,
    R_env: np.ndarray = None,
) -> Tuple[qtn.MatrixProductState, float]:
    """
    Perform one-site DMRG local micro-step at a single site.

    Implements the one-site DMRG micro-iteration S_j (Definition 9, Eq. 11)
    from Grigori & Hassan (2025):

        S_j(U) = argmin_{W_j} g o P_{U,j,1}(W_j)

    where P_{U,j,1} is the one-site retraction operator (Definition 7) and
    U must be in j-orthogonal form (Definition 6).

    Algorithm steps:
    1. Copy MPS and transform to i-orthogonal form centered at `site`
       (Algorithm 2, Step 1; implements the gauge transformation without
       bond compression)
    2. Build left L[site] and right R[site+1] environments from the
       i-orthogonal MPS (ensures consistency per Lemma 8)
    3. Construct effective Hamiltonian H_eff and solve eigenvalue problem
       (Lemma 10: reduces to standard eigenvalue problem P*AP W = lambda W)
    4. Update MPS[site] with the eigenvector (the optimized TT core V_i)
    5. Compute and return total energy

    Parameters
    ----------
    mps : quimb.tensor.MatrixProductState
        Input MPS (will be copied, not modified in-place).
    mpo : quimb MPO
        Matrix Product Operator (Hamiltonian).
    site : int
        Site index to update (0-indexed).
    tol : float, optional
        Tolerance for eigensolver (default: 1e-10).
    L_env : np.ndarray, optional
        Precomputed left environment L[site]. If None, built internally
        from the i-orthogonal MPS.
    R_env : np.ndarray, optional
        Precomputed right environment R[site+1]. If None, built internally
        from the i-orthogonal MPS.

    Returns
    -------
    mps_updated : quimb.tensor.MatrixProductState
        Updated MPS with site i optimized.
    energy : float
        Ground state energy from local optimization.

    Notes
    -----
    Per Remark 11 (page 8), if U is NOT i-orthogonal, the DMRG
    micro-iteration consists of solving a badly-conditioned generalized
    eigenvalue problem. The i-orthogonal transformation is therefore
    CRITICAL for numerical stability.
    """
    from ..environments.environment import _reshape_mps_tensor_from_quimb

    # Step 1: Copy MPS and transform to i-orthogonal form
    # (Algorithm 2, Step 1: orthogonalization sweep)
    mps_updated = mps.copy()
    _transform_to_i_orthogonal(mps_updated, center_site=site, normalize=True)

    L = mps_updated.L

    # Step 2: Build environments from the i-orthogonal MPS
    # CRITICAL: Environments must be built from the canonicalized MPS,
    # not the original, to ensure consistency with the gauge choice.
    # Pre-computed environments from the non-canonical MPS are invalid
    # after gauge transformation, so we always rebuild.
    L_envs = build_left_environments(mps_updated, mpo)
    R_envs = build_right_environments(mps_updated, mpo)
    L_env_local = L_envs[site]
    R_env_local = R_envs[site + 1]

    # Extract site tensor in standard (left_bond, phys, right_bond) format
    # This is the format expected by the effective Hamiltonian
    mps_tensor_std = _reshape_mps_tensor_from_quimb(mps_updated[site], site, L)
    chi_L, d, chi_R = mps_tensor_std.shape

    # Get MPO tensor at site i
    W_i = mpo[site].data

    # Step 3: Build effective Hamiltonian and solve
    # H_eff operates on vectors of shape (chi_L, d, chi_R) flattened
    # (see build_effective_hamiltonian_1site: psi = v.reshape(ket_L, phys, ket_R))
    H_eff = build_effective_hamiltonian_1site(
        L_env_local, W_i, R_env_local, site, L
    )

    # Initial guess: current MPS tensor in (chi_L, d, chi_R) format, flattened
    v0 = mps_tensor_std.ravel()

    # Solve for ground state (Lemma 10: eigenvalue problem)
    energy, eigvec = solve_effective_hamiltonian(H_eff, v0=v0, tol=tol)

    # Step 4: Update MPS[site] with the optimized eigenvector
    # eigvec is in (chi_L, d, chi_R) format (matching H_eff convention)
    new_tensor_std = eigvec.reshape(chi_L, d, chi_R)

    # Convert back to quimb's storage format for this site
    # Detect quimb's index ordering from the original tensor
    orig_inds = mps_updated[site].inds
    phys_name = f'k{site}'
    if phys_name in orig_inds:
        phys_pos = list(orig_inds).index(phys_name)
    else:
        phys_pos = 1  # default middle-site ordering

    orig_data = mps_updated[site].data
    if orig_data.ndim == 2:
        # Edge site: 2D tensor
        if site == 0:
            # Standard: (chi_L=1, d, chi_R) -> quimb first site
            if phys_pos == 0:
                new_tensor = new_tensor_std[0, :, :]  # (d, chi_R)
            else:
                new_tensor = new_tensor_std[0, :, :].T  # (chi_R, d)
        else:
            # Last site: (chi_L, d, chi_R=1) -> quimb last site
            if phys_pos == 1:
                new_tensor = new_tensor_std[:, :, 0]  # (chi_L, d)
            else:
                new_tensor = new_tensor_std[:, :, 0].T  # (d, chi_L)
    else:
        # Middle site: 3D tensor
        if phys_pos == 1:
            new_tensor = new_tensor_std  # already (chi_L, d, chi_R)
        elif phys_pos == 2:
            new_tensor = new_tensor_std.transpose(0, 2, 1)  # (chi_L, chi_R, d)
        else:
            new_tensor = new_tensor_std.transpose(1, 0, 2)  # (d, chi_L, chi_R)

    mps_updated[site].modify(data=new_tensor)

    # Step 5: Compute actual total energy
    # The eigenvalue from H_eff is the Rayleigh quotient at the orthogonality
    # center. For an i-orthogonal MPS, this equals the total energy
    # (since left/right parts are orthonormal). However, after replacing the
    # center tensor the MPS is no longer normalized, so we compute the full
    # energy for robustness.
    from .observables import compute_energy
    actual_energy = compute_energy(mps_updated, mpo, normalize=True)

    return mps_updated, float(np.real(actual_energy))


def local_microstep_2site(
    mps: qtn.MatrixProductState,
    mpo,
    site: int,
    max_bond: int = None,
    cutoff: float = 1e-10,
    tol: float = 1e-10,
    L_env: np.ndarray = None,
    R_env: np.ndarray = None,
) -> Tuple[qtn.MatrixProductState, float]:
    """
    Perform two-site DMRG local micro-step at sites i and i+1.

    Implements the two-site DMRG micro-iteration S_{k,k+1} (Definition 9,
    Eq. 12) from Grigori & Hassan (2025):

        S_{k,k+1}(V) = argmin_{W_{k,k+1}} g o P_{V,k,2}(W_{k,k+1})

    where P_{V,k,2} is the two-site retraction operator (Definition 7) and
    V must be in k-orthogonal form (Definition 6).

    Algorithm steps:
    1. Copy MPS and transform to i-orthogonal form centered at `site`
       (Algorithm 2, Step 1)
    2. Build left L[site] and right R[site+2] environments from the
       i-orthogonal MPS
    3. Construct two-site H_eff and solve eigenvalue problem (Lemma 10)
    4. Reshape eigenvector to (chi_L, d1, d2, chi_R)
    5. Apply truncated SVD to split back to two tensors
       (Algorithm 2, Step 2: W_{i,i+1} = P_i S_i Q_i)
    6. Return updated MPS and energy

    Parameters
    ----------
    mps : quimb.tensor.MatrixProductState
        Input MPS (will be copied, not modified in-place).
    mpo : quimb MPO
        Matrix Product Operator (Hamiltonian).
    site : int
        Left site index (will update sites i and i+1).
    max_bond : int, optional
        Maximum bond dimension after SVD (default: None, no limit).
    cutoff : float, optional
        SVD truncation tolerance (default: 1e-10).
    tol : float, optional
        Tolerance for eigensolver (default: 1e-10).
    L_env : np.ndarray, optional
        Precomputed left environment L[site]. If None, built internally
        from the i-orthogonal MPS.
    R_env : np.ndarray, optional
        Precomputed right environment R[site+2]. If None, built internally
        from the i-orthogonal MPS.

    Returns
    -------
    mps_updated : quimb.tensor.MatrixProductState
        Updated MPS with sites i and i+1 optimized.
    energy : float
        Ground state energy from local optimization.

    Notes
    -----
    The SVD splitting in step 5 decomposes the two-site tensor as:
        theta = U @ diag(S) @ V^H
    The left tensor becomes U (left-isometric) and the right tensor
    absorbs S @ V^H. This corresponds to the factorization
    W_{i,i+1} = P_i^{<2>} S_i Q_i^{<i>} in Algorithm 2.
    """
    from ..environments.environment import _reshape_mps_tensor_from_quimb

    L = mps.L
    if site >= L - 1:
        raise ValueError(f"Two-site update requires site < L-1, got site={site}, L={L}")

    # Step 1: Copy MPS and transform to i-orthogonal form
    # For two-site update, we place the orthogonality center at `site`.
    # Both site and site+1 then participate in the two-site block
    # containing the gauge freedom.
    mps_updated = mps.copy()
    _transform_to_i_orthogonal(mps_updated, center_site=site, normalize=True)

    # Step 2: Build environments from the i-orthogonal MPS
    # CRITICAL: Must rebuild from canonicalized MPS for consistency
    L_envs = build_left_environments(mps_updated, mpo)
    R_envs = build_right_environments(mps_updated, mpo)
    L_env_local = L_envs[site]
    R_env_local = R_envs[site + 2]

    # Extract tensors in standard (left_bond, phys, right_bond) format
    tensor_i = _reshape_mps_tensor_from_quimb(mps_updated[site], site, L)
    tensor_ip1 = _reshape_mps_tensor_from_quimb(mps_updated[site + 1], site + 1, L)

    chi_L, d1, chi_M1 = tensor_i.shape
    chi_M2, d2, chi_R = tensor_ip1.shape

    # Store original quimb index info for write-back
    orig_inds_i = mps_updated[site].inds
    orig_inds_ip1 = mps_updated[site + 1].inds

    phys_name_i = f'k{site}'
    phys_name_ip1 = f'k{site + 1}'
    _phys_pos_i = list(orig_inds_i).index(phys_name_i) if phys_name_i in orig_inds_i else 1
    _phys_pos_ip1 = list(orig_inds_ip1).index(phys_name_ip1) if phys_name_ip1 in orig_inds_ip1 else 1

    # Middle bond dimensions should match
    assert chi_M1 == chi_M2, f"Bond dimension mismatch: {chi_M1} != {chi_M2}"

    # Get MPO tensors
    W_i = mpo[site].data
    W_ip1 = mpo[site + 1].data

    # Step 3: Build two-site effective Hamiltonian and solve
    # H_eff operates on theta of shape (chi_L, d1, d2, chi_R) flattened
    H_eff = build_effective_hamiltonian_2site(
        L_env_local, W_i, W_ip1, R_env_local,
        site, L
    )

    # Initial guess: contract two-site tensor
    # theta[l, p1, p2, r] = sum_m tensor_i[l, p1, m] * tensor_ip1[m, p2, r]
    theta = np.einsum('lpm,mqr->lpqr', tensor_i, tensor_ip1)
    v0 = theta.ravel()

    # Solve for ground state
    energy, eigvec = solve_effective_hamiltonian(H_eff, v0=v0, tol=tol)

    # Step 4: Reshape to two-site tensor
    theta_new = eigvec.reshape(chi_L, d1, d2, chi_R)

    # Step 5: Apply truncated SVD to split back to two tensors
    # Reshape to matrix: (chi_L * d1) x (d2 * chi_R)
    theta_mat = theta_new.reshape(chi_L * d1, d2 * chi_R)

    U, S, Vh = truncated_svd(theta_mat, max_rank=max_bond, tol=cutoff)
    chi_new = len(S)

    # Absorb singular values into right tensor (left-canonical split)
    SV = np.diag(S) @ Vh

    # Convert U back to quimb format for site i
    if site == 0:
        # First site: 2D tensor
        u_2d = U.reshape(d1, chi_new)  # (phys, right_bond)
        if _phys_pos_i == 0:
            tensor_i_new = u_2d
        else:
            tensor_i_new = u_2d.T
    else:
        # Middle site: 3D tensor
        u_3d = U.reshape(chi_L, d1, chi_new)  # (left, phys, right)
        if _phys_pos_i == 1:
            tensor_i_new = u_3d
        elif _phys_pos_i == 2:
            tensor_i_new = u_3d.transpose(0, 2, 1)
        else:
            tensor_i_new = u_3d.transpose(1, 0, 2)

    # Convert SV back to quimb format for site i+1
    if site + 1 == L - 1:
        # Last site: 2D tensor
        sv_2d = SV.reshape(chi_new, d2)  # (left_bond, phys)
        if _phys_pos_ip1 == 1:
            tensor_ip1_new = sv_2d
        else:
            tensor_ip1_new = sv_2d.T
    else:
        # Middle site: 3D tensor
        sv_3d = SV.reshape(chi_new, d2, chi_R)  # (left, phys, right)
        if _phys_pos_ip1 == 1:
            tensor_ip1_new = sv_3d
        elif _phys_pos_ip1 == 2:
            tensor_ip1_new = sv_3d.transpose(0, 2, 1)
        else:
            tensor_ip1_new = sv_3d.transpose(1, 0, 2)

    # Step 6: Update MPS tensors
    mps_updated[site].modify(data=tensor_i_new)
    mps_updated[site + 1].modify(data=tensor_ip1_new)

    # Compute actual total energy
    from .observables import compute_energy
    actual_energy = compute_energy(mps_updated, mpo, normalize=True)

    return mps_updated, float(np.real(actual_energy))
