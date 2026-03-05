#!/usr/bin/env python3
"""Test whether the GPU environment convention gives correct energy."""

import numpy as np


def build_josephson_mpo_gpu_style(L, n_max=1, phi_ext=np.pi/4):
    d = 2 * n_max + 1
    D_mpo = 4
    E_J, E_C, mu = 1.0, 0.5, 0.0
    eye = np.eye(d, dtype='complex128')
    exp_iphi = np.zeros((d, d), dtype='complex128')
    exp_miphi = np.zeros((d, d), dtype='complex128')
    H_onsite = np.zeros((d, d), dtype='complex128')
    for i in range(d):
        charge = float(i - n_max)
        H_onsite[i, i] = E_C * charge * charge - mu * charge
    for i in range(d - 1):
        exp_iphi[i + 1, i] = 1.0 + 0j
        exp_miphi[i, i + 1] = 1.0 + 0j
    cos_p = np.cos(phi_ext)
    sin_p = np.sin(phi_ext)
    alpha_coup = complex(-E_J/2.0 * cos_p, -E_J/2.0 * sin_p)
    alpha_conj = complex(-E_J/2.0 * cos_p,  E_J/2.0 * sin_p)
    mpo_tensors = []
    for site in range(L):
        D_L = 1 if site == 0 else D_mpo
        D_R = 1 if site == L - 1 else D_mpo
        W = np.zeros((D_L, d, d, D_R), dtype='complex128')
        if site == 0:
            W[0, :, :, 0] = H_onsite
            W[0, :, :, 1] = alpha_coup * exp_iphi
            W[0, :, :, 2] = alpha_conj * exp_miphi
            W[0, :, :, 3] = eye
        elif site == L - 1:
            W[0, :, :, 0] = eye
            W[1, :, :, 0] = exp_miphi
            W[2, :, :, 0] = exp_iphi
            W[3, :, :, 0] = H_onsite
        else:
            W[0, :, :, 0] = eye
            W[1, :, :, 0] = exp_miphi
            W[2, :, :, 0] = exp_iphi
            W[3, :, :, 0] = H_onsite
            W[3, :, :, 1] = alpha_coup * exp_iphi
            W[3, :, :, 2] = alpha_conj * exp_miphi
            W[3, :, :, 3] = eye
        mpo_tensors.append(W)
    return mpo_tensors


def build_H_direct(L, n_max=1, phi_ext=np.pi/4):
    d = 2 * n_max + 1
    D = d ** L
    E_J, E_C = 1.0, 0.5
    eye = np.eye(d, dtype='complex128')
    exp_iphi = np.zeros((d, d), dtype='complex128')
    exp_miphi = np.zeros((d, d), dtype='complex128')
    n_op = np.zeros((d, d), dtype='complex128')
    for i in range(d):
        n_op[i, i] = float(i - n_max)
    for i in range(d - 1):
        exp_iphi[i + 1, i] = 1.0 + 0j
        exp_miphi[i, i + 1] = 1.0 + 0j
    flux_phase = np.exp(1j * phi_ext)
    H = np.zeros((D, D), dtype='complex128')
    def embed(ops):
        result = np.eye(1, dtype='complex128')
        for op in ops:
            result = np.kron(result, op)
        return result
    n2 = n_op @ n_op
    for i in range(L):
        ops = [eye]*L; ops[i] = n2
        H += E_C * embed(ops)
    for i in range(L-1):
        ops1 = [eye]*L; ops1[i] = exp_iphi; ops1[i+1] = exp_miphi
        ops2 = [eye]*L; ops2[i] = exp_miphi; ops2[i+1] = exp_iphi
        H += -E_J/2 * flux_phase * embed(ops1)
        H += -E_J/2 * np.conj(flux_phase) * embed(ops2)
    return H


def gpu_left_env_update(L_in, A, W, d):
    """Replicate GPU's kernel_update_left_env."""
    Da = A.shape[0]
    Db = A.shape[2]
    Dw = W.shape[0]
    Dwp = W.shape[3]

    L_out = np.zeros((Db, Dwp, Db), dtype='complex128')
    for b in range(Db):
        for wp in range(Dwp):
            for bstar in range(Db):
                s = 0.0j
                for a in range(Da):
                    for si in range(d):
                        Av = A[a, si, b]
                        for w in range(Dw):
                            for sp in range(d):
                                Wv = W[w, si, sp, wp]
                                aw = Av * Wv
                                for astar in range(Da):
                                    Lv = L_in[a, w, astar]
                                    Ac = np.conj(A[astar, sp, bstar])
                                    s += Lv * aw * Ac
                L_out[b, wp, bstar] = s
    return L_out


def compute_energy_gpu_style(mps, mpo, d, L):
    """Compute energy using GPU's approach: propagate L env through all sites."""
    L_env = np.ones((1, 1, 1), dtype='complex128')
    for site in range(L):
        L_env = gpu_left_env_update(L_env, mps[site], mpo[site], d)
    energy = L_env.squeeze().real

    # Compute norm
    N = np.ones((1, 1), dtype='complex128')
    for site in range(L):
        A = mps[site]
        Da, Db = A.shape[0], A.shape[2]
        N_new = np.zeros((Db, Db), dtype='complex128')
        for b in range(Db):
            for bstar in range(Db):
                s = 0.0j
                for a in range(Da):
                    for astar in range(Da):
                        for si in range(d):
                            s += N[a, astar] * A[a, si, b] * np.conj(A[astar, si, bstar])
                N_new[b, bstar] = s
        N = N_new
    norm = N.squeeze().real
    return energy / norm


def compute_energy_exact(mps, H_full, d, L):
    """Compute <psi|H|psi> / <psi|psi> using full H matrix."""
    # Contract MPS to get full state vector
    psi = mps[0]  # (1, d, D1)
    for i in range(1, L):
        psi = np.einsum('...j,jkl->...kl', psi, mps[i])
    psi = psi.squeeze()  # (d, d, ..., d) with L indices
    psi = psi.flatten()  # d^L vector

    E = (np.conj(psi) @ H_full @ psi) / (np.conj(psi) @ psi)
    return E.real


if __name__ == '__main__':
    L = 4
    n_max = 1
    d = 2 * n_max + 1

    print(f"L={L}, d={d}")

    mpo = build_josephson_mpo_gpu_style(L, n_max=n_max)
    H_full = build_H_direct(L, n_max=n_max)

    np.random.seed(42)
    bond_dims = [1, 3, 9, 3, 1]
    mps = []
    for i in range(L):
        Da, Db = bond_dims[i], bond_dims[i+1]
        A = np.random.randn(Da, d, Db) + 1j * np.random.randn(Da, d, Db)
        mps.append(A)

    E_env = compute_energy_gpu_style(mps, mpo, d, L)
    E_exact = compute_energy_exact(mps, H_full, d, L)

    print(f"Energy (GPU env style):  {E_env:.12f}")
    print(f"Energy (exact <psi|H|psi>): {E_exact:.12f}")
    print(f"Difference: {abs(E_env - E_exact):.2e}")

    if abs(E_env - E_exact) > 1e-10:
        print("*** ENVIRONMENT CONVENTION BUG: energy from envs != exact ***")
    else:
        print("OK: environment computes correct energy")

    # Now test: does H_eff with current convention give the right result?
    print("\n--- H_eff test for site 0 ---")

    # Build left env for site 0
    L_env = np.ones((1, 1, 1), dtype='complex128')
    # Build right env for site 2..L-1 using GPU convention
    R_env = np.ones((1, 1, 1), dtype='complex128')
    for site in range(L-1, 1, -1):
        A = mps[site]
        W = mpo[site]
        Da, Db = A.shape[0], A.shape[2]
        Dw, Dwp = W.shape[0], W.shape[3]
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

    W1 = mpo[0]
    W2 = mpo[1]
    D_L = bond_dims[0]
    D_R = bond_dims[2]
    D_mpo_L = W1.shape[0]
    D_mpo_M = W1.shape[3]
    D_mpo_R = W2.shape[3]

    theta = np.einsum('ijk,klm->ijlm', mps[0], mps[1])

    # Apply H_eff (GPU convention)
    T1 = np.einsum('awp,asdb->wpsdb', L_env, theta)
    T2 = np.einsum('wsqm,wpsdb->mpqdb', W1, T1)
    T3 = np.einsum('mdnr,mpqdb->pqnrb', W2, T2)
    Htheta_gpu = np.einsum('brc,pqnrb->pqnc', R_env, T3)

    # Compute <theta|H_eff|theta> / <theta|theta>
    numerator = np.sum(np.conj(theta) * Htheta_gpu)
    denominator = np.sum(np.conj(theta) * theta)
    E_heff = (numerator / denominator).real
    print(f"E from H_eff (GPU conv): {E_heff:.12f}")

    # Now compute exact <theta|H|theta> using the full Hamiltonian
    # theta[a, s1, s2, b] represents the 2-site block
    # For the full overlap, we need to contract with mps on the rest of the chain
    # Actually, let's compute <psi|H|psi> where psi has theta at sites 0,1
    # and mps[2], mps[3] at sites 2,3
    psi_01 = theta  # (D_L, d, d, D_R) = (1, 3, 3, 9)
    psi = psi_01
    for i in range(2, L):
        psi = np.einsum('...j,jkl->...kl', psi, mps[i])
    psi = psi.squeeze().flatten()

    E_exact_theta = ((np.conj(psi) @ H_full @ psi) / (np.conj(psi) @ psi)).real
    print(f"E from exact H (same theta): {E_exact_theta:.12f}")
    print(f"Difference: {abs(E_heff - E_exact_theta):.2e}")

    if abs(E_heff - E_exact_theta) > 1e-10:
        print("*** H_EFF BUG CONFIRMED! ***")

        # Try with swapped convention
        print("\n--- Testing with SWAPPED physical indices in W ---")
        # Swap s and sp: use W[w, sp, s, wp] instead of W[w, s, sp, wp]
        T1 = np.einsum('awp,asdb->wpsdb', L_env, theta)
        T2 = np.einsum('wqsm,wpsdb->mpqdb', W1, T1)  # swap s,q in W1 indices
        T3 = np.einsum('mndr,mpqdb->pqnrb', W2, T2)  # swap d,n in W2 indices
        Htheta_swapped = np.einsum('brc,pqnrb->pqnc', R_env, T3)

        numerator2 = np.sum(np.conj(theta) * Htheta_swapped)
        E_heff2 = (numerator2 / denominator).real
        print(f"E from H_eff (swapped): {E_heff2:.12f}")
        print(f"Difference from exact: {abs(E_heff2 - E_exact_theta):.2e}")

        if abs(E_heff2 - E_exact_theta) < 1e-10:
            print("==> SWAPPING fixes the H_eff!")
        else:
            print("==> Swapping alone doesn't fix it. Need different approach.")
    else:
        print("OK: H_eff is correct with current convention")
