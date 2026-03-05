#!/usr/bin/env python3
"""
Verify the H_eff contraction and MPO logic used in pdmrg_gpu.cpp.

This script constructs a small Josephson junction system (L=4, n_max=1, d=3)
and tests:
1. The GPU-style MPO produces the correct Hamiltonian
2. The 4-step H_eff contraction is mathematically correct
3. The ground state energy matches exact diagonalization

NO dependency on quimb - uses only numpy.
"""

import numpy as np
from scipy.sparse.linalg import eigsh


def build_josephson_mpo_gpu_style(L, E_J=1.0, E_C=0.5, mu=0.0, n_max=1, phi_ext=np.pi/4):
    """Build Josephson MPO tensors in the same format as pdmrg_gpu.cpp.

    Returns list of W[wl, s, sp, wr] tensors.
    """
    d = 2 * n_max + 1
    D_mpo = 4

    # Build operators
    eye = np.eye(d, dtype='complex128')
    exp_iphi = np.zeros((d, d), dtype='complex128')
    exp_miphi = np.zeros((d, d), dtype='complex128')
    H_onsite = np.zeros((d, d), dtype='complex128')

    for i in range(d):
        charge = float(i - n_max)
        H_onsite[i, i] = E_C * charge * charge - mu * charge

    for i in range(d - 1):
        exp_iphi[i + 1, i] = 1.0 + 0j  # e^{i*phi}|n> = |n+1>
        exp_miphi[i, i + 1] = 1.0 + 0j  # e^{-i*phi}|n> = |n-1>

    cos_p = np.cos(phi_ext)
    sin_p = np.sin(phi_ext)
    alpha_coup = complex(-E_J/2.0 * cos_p, -E_J/2.0 * sin_p)  # -E_J/2 * e^{i*phi_ext}
    alpha_conj = complex(-E_J/2.0 * cos_p,  E_J/2.0 * sin_p)  # -E_J/2 * e^{-i*phi_ext}

    mpo_tensors = []

    for site in range(L):
        if site == 0:
            D_L = 1
        else:
            D_L = D_mpo
        if site == L - 1:
            D_R = 1
        else:
            D_R = D_mpo

        W = np.zeros((D_L, d, d, D_R), dtype='complex128')

        if site == 0:
            # Left boundary: row vector = row 3 of bulk
            W[0, :, :, 0] = H_onsite
            W[0, :, :, 1] = alpha_coup * exp_iphi
            W[0, :, :, 2] = alpha_conj * exp_miphi
            W[0, :, :, 3] = eye
        elif site == L - 1:
            # Right boundary: column vector = column 0 of bulk
            W[0, :, :, 0] = eye
            W[1, :, :, 0] = exp_miphi
            W[2, :, :, 0] = exp_iphi
            W[3, :, :, 0] = H_onsite
        else:
            # Bulk
            W[0, :, :, 0] = eye
            W[1, :, :, 0] = exp_miphi
            W[2, :, :, 0] = exp_iphi
            W[3, :, :, 0] = H_onsite
            W[3, :, :, 1] = alpha_coup * exp_iphi
            W[3, :, :, 2] = alpha_conj * exp_miphi
            W[3, :, :, 3] = eye

        mpo_tensors.append(W)

    return mpo_tensors


def build_josephson_hamiltonian_direct(L, E_J=1.0, E_C=0.5, mu=0.0, n_max=1, phi_ext=np.pi/4):
    """Build full Josephson Hamiltonian directly (for verification).

    H = -E_J/2 * sum_<ij> (e^{i*phi_ext} exp_iphi_i exp_miphi_j + h.c.)
        + E_C * sum_i n_i^2 - mu * sum_i n_i
    """
    d = 2 * n_max + 1
    D = d ** L

    # Build operators
    eye = np.eye(d, dtype='complex128')
    exp_iphi = np.zeros((d, d), dtype='complex128')
    exp_miphi = np.zeros((d, d), dtype='complex128')
    n_op = np.zeros((d, d), dtype='complex128')

    for i in range(d):
        charge = float(i - n_max)
        n_op[i, i] = charge

    for i in range(d - 1):
        exp_iphi[i + 1, i] = 1.0 + 0j
        exp_miphi[i, i + 1] = 1.0 + 0j

    flux_phase = np.exp(1j * phi_ext)

    H = np.zeros((D, D), dtype='complex128')

    # Helper to embed a single-site operator
    def embed_1site(op, site):
        result = np.eye(1, dtype='complex128')
        for i in range(L):
            if i == site:
                result = np.kron(result, op)
            else:
                result = np.kron(result, eye)
        return result

    # Helper to embed a two-site operator
    def embed_2site(op1, site1, op2, site2):
        result = np.eye(1, dtype='complex128')
        for i in range(L):
            if i == site1:
                result = np.kron(result, op1)
            elif i == site2:
                result = np.kron(result, op2)
            else:
                result = np.kron(result, eye)
        return result

    # On-site terms
    n2 = n_op @ n_op
    for i in range(L):
        H += E_C * embed_1site(n2, i)
        if mu != 0:
            H -= mu * embed_1site(n_op, i)

    # Coupling terms (nearest-neighbor)
    for i in range(L - 1):
        H += -E_J/2 * flux_phase * embed_2site(exp_iphi, i, exp_miphi, i+1)
        H += -E_J/2 * np.conj(flux_phase) * embed_2site(exp_miphi, i, exp_iphi, i+1)

    return H


def mpo_to_full_hamiltonian(mpo_tensors, d):
    """Contract all MPO tensors to build the full Hamiltonian matrix."""
    L = len(mpo_tensors)
    D = d ** L

    # Contract the MPO chain
    # W[wl, s, sp, wr] - indices: (left_mpo, phys_ket, phys_bra, right_mpo)
    result = mpo_tensors[0]  # (1, d, d, D_R0)

    for site in range(1, L):
        W = mpo_tensors[site]  # (D_L, d, d, D_R)
        # Contract: result[..., wr] with W[wl=wr, s, sp, wr_new]
        result = np.einsum('...w,wjkl->...jkl', result, W)

    # Squeeze boundary MPO dims
    result = result.squeeze()  # shape: (d, d, d, d, ...) with 2L indices

    # Indices: (s0_ket, s0_bra, s1_ket, s1_bra, ...)
    ket_indices = list(range(0, 2 * L, 2))
    bra_indices = list(range(1, 2 * L, 2))
    new_order = ket_indices + bra_indices
    result = result.transpose(new_order)
    result = result.reshape(D, D)

    return result


def apply_heff_4step(L_env, W1, W2, R_env, theta, D_L, D_R, D_mpo_L, D_mpo_M, D_mpo_R, d):
    """Implement the exact 4-step H_eff contraction from pdmrg_gpu.cpp apply_H_eff_cpu.

    theta[a, s1, s2, b] with shape (D_L, d, d, D_R)
    L_env[a, w, ap] with shape (D_L, D_mpo_L, D_L)
    W1[w, s1, s1p, wm] with shape (D_mpo_L, d, d, D_mpo_M)
    W2[wm, s2, s2p, wr] with shape (D_mpo_M, d, d, D_mpo_R)
    R_env[b, wr, bp] with shape (D_R, D_mpo_R, D_R)
    """
    # Step 1: T1[w, ap, s1, s2, b] = sum_a L[a, w, ap] * theta[a, s1, s2, b]
    T1 = np.zeros((D_mpo_L, D_L, d, d, D_R), dtype='complex128')
    for w in range(D_mpo_L):
        for ap in range(D_L):
            for s1 in range(d):
                for s2 in range(d):
                    for b in range(D_R):
                        s = 0.0j
                        for a in range(D_L):
                            s += L_env[a, w, ap] * theta[a, s1, s2, b]
                        T1[w, ap, s1, s2, b] = s

    # Step 2: T2[wm, ap, s1p, s2, b] = sum_{w, s1} W1[w, s1, s1p, wm] * T1[w, ap, s1, s2, b]
    T2 = np.zeros((D_mpo_M, D_L, d, d, D_R), dtype='complex128')
    for wm in range(D_mpo_M):
        for ap in range(D_L):
            for s1p in range(d):
                for s2 in range(d):
                    for b in range(D_R):
                        s = 0.0j
                        for w in range(D_mpo_L):
                            for s1 in range(d):
                                s += W1[w, s1, s1p, wm] * T1[w, ap, s1, s2, b]
                        T2[wm, ap, s1p, s2, b] = s

    # Step 3: T3[ap, s1p, s2p, wr, b] = sum_{wm, s2} W2[wm, s2, s2p, wr] * T2[wm, ap, s1p, s2, b]
    T3 = np.zeros((D_L, d, d, D_mpo_R, D_R), dtype='complex128')
    for ap in range(D_L):
        for s1p in range(d):
            for s2p in range(d):
                for wr in range(D_mpo_R):
                    for b in range(D_R):
                        s = 0.0j
                        for wm in range(D_mpo_M):
                            for s2 in range(d):
                                s += W2[wm, s2, s2p, wr] * T2[wm, ap, s1p, s2, b]
                        T3[ap, s1p, s2p, wr, b] = s

    # Step 4: result[ap, s1p, s2p, bp] = sum_{b, wr} R[b, wr, bp] * T3[ap, s1p, s2p, wr, b]
    result = np.zeros((D_L, d, d, D_R), dtype='complex128')
    for ap in range(D_L):
        for s1p in range(d):
            for s2p in range(d):
                for bp in range(D_R):
                    s = 0.0j
                    for b in range(D_R):
                        for wr in range(D_mpo_R):
                            s += R_env[b, wr, bp] * T3[ap, s1p, s2p, wr, b]
                    result[ap, s1p, s2p, bp] = s

    return result


def build_left_env(mps_tensors, mpo_tensors, d, up_to_site):
    """Build left environment up to (but not including) up_to_site."""
    L_env = np.ones((1, 1, 1), dtype='complex128')

    for site in range(up_to_site):
        A = mps_tensors[site]
        W = mpo_tensors[site]
        Da = A.shape[0]
        Db = A.shape[2]
        Dw = W.shape[0]
        Dwp = W.shape[3]

        L_new = np.zeros((Db, Dwp, Db), dtype='complex128')
        for b in range(Db):
            for wp in range(Dwp):
                for bstar in range(Db):
                    s = 0.0j
                    for a in range(Da):
                        for astar in range(Da):
                            for w in range(Dw):
                                for si in range(d):
                                    for sp in range(d):
                                        Lv = L_env[a, w, astar]
                                        Av = A[a, si, b]
                                        Wv = W[w, si, sp, wp]
                                        Ac = np.conj(A[astar, sp, bstar])
                                        s += Lv * Av * Wv * Ac
                    L_new[b, wp, bstar] = s
        L_env = L_new

    return L_env


def build_right_env(mps_tensors, mpo_tensors, d, from_site):
    """Build right environment from from_site to end."""
    L = len(mps_tensors)
    R_env = np.ones((1, 1, 1), dtype='complex128')

    for site in range(L - 1, from_site - 1, -1):
        A = mps_tensors[site]
        W = mpo_tensors[site]
        Da = A.shape[0]
        Db = A.shape[2]
        Dw = W.shape[0]
        Dwp = W.shape[3]

        R_new = np.zeros((Da, Dw, Da), dtype='complex128')
        for a in range(Da):
            for w in range(Dw):
                for astar in range(Da):
                    s = 0.0j
                    for b in range(Db):
                        for si in range(d):
                            Av = A[a, si, b]
                            for sp in range(d):
                                for wp in range(Dwp):
                                    Wv = W[w, si, sp, wp]
                                    for bstar in range(Db):
                                        Rv = R_env[b, wp, bstar]
                                        Ac = np.conj(A[astar, sp, bstar])
                                        s += Av * Wv * Rv * Ac
                    R_new[a, w, astar] = s
        R_env = R_new

    return R_env


def test_mpo_hamiltonians():
    """Test that GPU-style MPO produces the same Hamiltonian as direct construction."""
    print("="*60)
    print("TEST 1: MPO vs Direct Hamiltonian")
    print("="*60)

    for L in [2, 3, 4]:
        for n_max in [1, 2]:
            d = 2 * n_max + 1
            D = d ** L
            if D > 10000:
                continue

            print(f"\n  L={L}, n_max={n_max}, d={d}, D={D}")

            H_direct = build_josephson_hamiltonian_direct(L, n_max=n_max)
            mpo = build_josephson_mpo_gpu_style(L, n_max=n_max)
            H_mpo = mpo_to_full_hamiltonian(mpo, d)

            diff = np.max(np.abs(H_direct - H_mpo))
            print(f"    |H_direct - H_mpo| = {diff:.2e}")

            herm_err = np.max(np.abs(H_mpo - H_mpo.conj().T))
            print(f"    Hermiticity error   = {herm_err:.2e}")

            evals_direct = np.linalg.eigvalsh(H_direct)
            evals_mpo = np.linalg.eigvalsh(H_mpo)

            print(f"    E0 (direct) = {evals_direct[0]:.12f}")
            print(f"    E0 (MPO)    = {evals_mpo[0]:.12f}")

            if diff > 1e-10:
                print(f"    *** MISMATCH! ***")
            else:
                print(f"    OK")


def test_heff_contraction():
    """Test the 4-step H_eff contraction."""
    print("\n" + "="*60)
    print("TEST 2: 4-step H_eff Contraction")
    print("="*60)

    L = 4
    n_max = 1
    d = 2 * n_max + 1

    mpo = build_josephson_mpo_gpu_style(L, n_max=n_max)

    # Create random MPS
    np.random.seed(42)
    bond_dims = [1]
    for i in range(1, L):
        bd = min(10, d ** i, d ** (L - i))
        bond_dims.append(bd)
    bond_dims.append(1)

    print(f"  L={L}, d={d}, bond_dims={bond_dims}")

    mps = []
    for i in range(L):
        Da = bond_dims[i]
        Db = bond_dims[i + 1]
        A = np.random.randn(Da, d, Db) + 1j * np.random.randn(Da, d, Db)
        # Normalize
        A /= np.linalg.norm(A)
        mps.append(A)

    # Test H_eff for each 2-site block
    for site in range(L - 1):
        print(f"\n  Testing 2-site block [{site}, {site+1}]...")

        D_L = bond_dims[site]
        D_R = bond_dims[site + 2]

        # Build left environment (sites 0..site-1)
        L_env = build_left_env(mps, mpo, d, site)

        # Build right environment (sites site+2..L-1)
        R_env = build_right_env(mps, mpo, d, site + 2)

        W1 = mpo[site]
        W2 = mpo[site + 1]

        D_mpo_L = W1.shape[0]
        D_mpo_M = W1.shape[3]
        D_mpo_R = W2.shape[3]

        print(f"    D_L={D_L}, D_R={D_R}")
        print(f"    D_mpo: L={D_mpo_L}, M={D_mpo_M}, R={D_mpo_R}")
        print(f"    L_env shape: {L_env.shape}")
        print(f"    R_env shape: {R_env.shape}")

        # Create random theta
        theta = np.random.randn(D_L, d, d, D_R) + 1j * np.random.randn(D_L, d, d, D_R)

        # Apply H_eff using 4-step contraction
        result_4step = apply_heff_4step(L_env, W1, W2, R_env, theta,
                                        D_L, D_R, D_mpo_L, D_mpo_M, D_mpo_R, d)

        # Apply H_eff using direct contraction (reference)
        # H_eff * theta = sum_{a,w,s1,s2,b,wm,wr} L[a,w,ap] W1[w,s1,s1p,wm] W2[wm,s2,s2p,wr] R[b,wr,bp] theta[a,s1,s2,b]
        result_direct = np.einsum('awp,wsqm,mjnr,bro,asdb->pqno',
                                  L_env, W1, W2, R_env, theta,
                                  optimize=True)

        diff = np.max(np.abs(result_4step - result_direct))
        print(f"    |4step - direct| = {diff:.2e}")

        if diff > 1e-10:
            print(f"    *** MISMATCH! ***")
        else:
            print(f"    OK")

        # Check Hermiticity of H_eff
        psi_size = D_L * d * d * D_R
        H_eff = np.zeros((psi_size, psi_size), dtype='complex128')
        for idx in range(psi_size):
            e_i = np.zeros(psi_size, dtype='complex128')
            e_i[idx] = 1.0
            e_i_tensor = e_i.reshape(D_L, d, d, D_R)
            result_i = apply_heff_4step(L_env, W1, W2, R_env, e_i_tensor,
                                        D_L, D_R, D_mpo_L, D_mpo_M, D_mpo_R, d)
            H_eff[:, idx] = result_i.flatten()

        herm_err = np.max(np.abs(H_eff - H_eff.conj().T))
        print(f"    H_eff Hermiticity error: {herm_err:.2e}")


def test_full_dmrg_small():
    """Test a complete mini-DMRG (2-site) on a small system."""
    print("\n" + "="*60)
    print("TEST 3: Mini-DMRG on L=8, n_max=2 (d=5)")
    print("="*60)

    L = 8
    n_max = 2
    d = 2 * n_max + 1
    max_D = 50

    mpo = build_josephson_mpo_gpu_style(L, n_max=n_max)

    # Initialize bond dims (same as GPU code)
    bond_dims = [1]
    for i in range(1, L):
        dim_left = d ** i
        dim_right = d ** (L - i)
        bond_dims.append(min(max_D, dim_left, dim_right))
    bond_dims.append(1)
    print(f"  Bond dims: {bond_dims}")

    # Initialize random complex MPS
    np.random.seed(42)
    mps = []
    for i in range(L):
        Da = bond_dims[i]
        Db = bond_dims[i + 1]
        A = np.random.randn(Da, d, Db) + 1j * np.random.randn(Da, d, Db)
        mps.append(A)

    # Right-canonicalize (proper SVD-based)
    for site in range(L - 1, 0, -1):
        Da = bond_dims[site]
        Db = bond_dims[site + 1]
        A = mps[site].reshape(Da, d * Db)
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        k = min(Da, d * Db)
        # A[site] = Vt (right-canonical)
        mps[site] = Vt.reshape(k, d, Db)
        # A[site-1] = A[site-1] * U * S
        mps[site - 1] = np.einsum('ijk,kl,l->ijl', mps[site - 1], U, S)
        bond_dims[site] = k

    print(f"  After right-canon bond dims: {bond_dims}")

    # Build right environments
    right_envs = [None] * (L + 1)
    right_envs[L] = np.ones((1, 1, 1), dtype='complex128')
    for site in range(L - 1, 0, -1):
        right_envs[site] = build_right_env_single(mps[site], mpo[site], right_envs[site + 1], d)

    left_envs = [None] * (L + 1)
    left_envs[0] = np.ones((1, 1, 1), dtype='complex128')

    n_sweeps = 10
    for sweep in range(n_sweeps):
        # Left-to-right
        for site in range(L - 1):
            D_L = bond_dims[site]
            D_R = bond_dims[site + 2]
            W1 = mpo[site]
            W2 = mpo[site + 1]
            D_mpo_L = W1.shape[0]
            D_mpo_M = W1.shape[3]
            D_mpo_R = W2.shape[3]

            # Form theta
            theta = np.einsum('ijk,klm->ijlm', mps[site], mps[site + 1])

            # Lanczos
            psi_size = D_L * d * d * D_R

            def matvec(v):
                vt = v.reshape(D_L, d, d, D_R)
                result = apply_heff_4step(left_envs[site], W1, W2, right_envs[site + 2],
                                         vt, D_L, D_R, D_mpo_L, D_mpo_M, D_mpo_R, d)
                return result.flatten()

            from scipy.sparse.linalg import LinearOperator
            H_op = LinearOperator((psi_size, psi_size), matvec=matvec, dtype='complex128')

            theta_flat = theta.flatten()
            if np.linalg.norm(theta_flat) < 1e-15:
                theta_flat = np.random.randn(psi_size) + 1j * np.random.randn(psi_size)
            theta_flat /= np.linalg.norm(theta_flat)

            try:
                evals, evecs = eigsh(H_op, k=1, which='SA', v0=theta_flat, maxiter=100)
                E = evals[0]
                theta_opt = evecs[:, 0].reshape(D_L, d, d, D_R)
            except Exception as e:
                print(f"    eigsh failed at site {site}: {e}")
                continue

            # SVD split
            theta_mat = theta_opt.reshape(D_L * d, d * D_R)
            U, S, Vt = np.linalg.svd(theta_mat, full_matrices=False)
            D_new = min(max_D, len(S))

            U = U[:, :D_new]
            S = S[:D_new]
            Vt = Vt[:D_new, :]

            mps[site] = U.reshape(D_L, d, D_new)
            mps[site + 1] = (np.diag(S) @ Vt).reshape(D_new, d, D_R)
            bond_dims[site + 1] = D_new

            # Update left env
            if site < L - 2:
                left_envs[site + 1] = build_left_env_single(mps[site], mpo[site], left_envs[site], d)

        # Right-to-left
        for site in range(L - 2, -1, -1):
            D_L = bond_dims[site]
            D_R = bond_dims[site + 2]
            W1 = mpo[site]
            W2 = mpo[site + 1]
            D_mpo_L = W1.shape[0]
            D_mpo_M = W1.shape[3]
            D_mpo_R = W2.shape[3]

            theta = np.einsum('ijk,klm->ijlm', mps[site], mps[site + 1])

            psi_size = D_L * d * d * D_R

            def matvec(v):
                vt = v.reshape(D_L, d, d, D_R)
                result = apply_heff_4step(left_envs[site], W1, W2, right_envs[site + 2],
                                         vt, D_L, D_R, D_mpo_L, D_mpo_M, D_mpo_R, d)
                return result.flatten()

            H_op = LinearOperator((psi_size, psi_size), matvec=matvec, dtype='complex128')

            theta_flat = theta.flatten()
            if np.linalg.norm(theta_flat) < 1e-15:
                theta_flat = np.random.randn(psi_size) + 1j * np.random.randn(psi_size)
            theta_flat /= np.linalg.norm(theta_flat)

            try:
                evals, evecs = eigsh(H_op, k=1, which='SA', v0=theta_flat, maxiter=100)
                E = evals[0]
                theta_opt = evecs[:, 0].reshape(D_L, d, d, D_R)
            except Exception as e:
                print(f"    eigsh failed at site {site}: {e}")
                continue

            # SVD split
            theta_mat = theta_opt.reshape(D_L * d, d * D_R)
            U, S, Vt = np.linalg.svd(theta_mat, full_matrices=False)
            D_new = min(max_D, len(S))

            U = U[:, :D_new]
            S = S[:D_new]
            Vt = Vt[:D_new, :]

            mps[site] = (U @ np.diag(S)).reshape(D_L, d, D_new)
            mps[site + 1] = Vt.reshape(D_new, d, D_R)
            bond_dims[site + 1] = D_new

            # Update right env
            if site > 0:
                right_envs[site + 1] = build_right_env_single(mps[site + 1], mpo[site + 1], right_envs[site + 2], d)

        # Compute energy
        E_full = compute_energy(mps, mpo, d, L, bond_dims)
        print(f"  Sweep {sweep:2d}: E = {E_full:.12f}")


def build_left_env_single(A, W, L_env, d):
    """Build L_new from L_env, A, W."""
    Da = A.shape[0]
    Db = A.shape[2]
    Dw = W.shape[0]
    Dwp = W.shape[3]

    L_new = np.zeros((Db, Dwp, Db), dtype='complex128')
    for b in range(Db):
        for wp in range(Dwp):
            for bstar in range(Db):
                s = 0.0j
                for a in range(Da):
                    for w in range(Dw):
                        for astar in range(Da):
                            Lv = L_env[a, w, astar]
                            for si in range(d):
                                Av = A[a, si, b]
                                for sp in range(d):
                                    Wv = W[w, si, sp, wp]
                                    Ac = np.conj(A[astar, sp, bstar])
                                    s += Lv * Av * Wv * Ac
                L_new[b, wp, bstar] = s
    return L_new


def build_right_env_single(A, W, R_env, d):
    """Build R_new from R_env, A, W."""
    Da = A.shape[0]
    Db = A.shape[2]
    Dw = W.shape[0]
    Dwp = W.shape[3]

    R_new = np.zeros((Da, Dw, Da), dtype='complex128')
    for a in range(Da):
        for w in range(Dw):
            for astar in range(Da):
                s = 0.0j
                for b in range(Db):
                    for si in range(d):
                        Av = A[a, si, b]
                        for sp in range(d):
                            for wp in range(Dwp):
                                Wv = W[w, si, sp, wp]
                                for bstar in range(Db):
                                    Rv = R_env[b, wp, bstar]
                                    Ac = np.conj(A[astar, sp, bstar])
                                    s += Av * Wv * Rv * Ac
                R_new[a, w, astar] = s
    return R_new


def compute_energy(mps, mpo, d, L, bond_dims):
    """Compute <psi|H|psi> / <psi|psi> via transfer matrix."""
    # Build left environment all the way through
    L_env = np.ones((1, 1, 1), dtype='complex128')
    for site in range(L):
        L_env = build_left_env_single(mps[site], mpo[site], L_env, d)

    energy = L_env[0, 0, 0].real

    # Compute norm
    N_env = np.ones((1, 1), dtype='complex128')
    for site in range(L):
        A = mps[site]
        Da = A.shape[0]
        Db = A.shape[2]

        N_new = np.zeros((Db, Db), dtype='complex128')
        for b in range(Db):
            for bstar in range(Db):
                s = 0.0j
                for a in range(Da):
                    for astar in range(Da):
                        Nv = N_env[a, astar]
                        for si in range(d):
                            Av = A[a, si, b]
                            Ac = np.conj(A[astar, si, bstar])
                            s += Nv * Av * Ac
                N_new[b, bstar] = s
        N_env = N_new

    norm = N_env[0, 0].real
    if abs(norm) > 1e-15:
        energy /= norm
    return energy


if __name__ == '__main__':
    print("=" * 60)
    print("H_eff Verification Test for Josephson Junction MPO")
    print("=" * 60)

    # Test 1: MPO vs Direct Hamiltonian
    test_mpo_hamiltonians()

    # Test 2: 4-step H_eff contraction
    test_heff_contraction()

    # Test 3: Full mini-DMRG
    test_full_dmrg_small()
