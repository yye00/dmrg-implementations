#!/usr/bin/env python3
"""Debug H_eff contraction: find out which method is correct."""

import numpy as np


def build_josephson_mpo_gpu_style(L, E_J=1.0, E_C=0.5, mu=0.0, n_max=1, phi_ext=np.pi/4):
    d = 2 * n_max + 1
    D_mpo = 4
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


def build_josephson_hamiltonian_direct(L, E_J=1.0, E_C=0.5, mu=0.0, n_max=1, phi_ext=np.pi/4):
    d = 2 * n_max + 1
    D = d ** L
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
    def embed_1site(op, site):
        result = np.eye(1, dtype='complex128')
        for i in range(L):
            result = np.kron(result, op if i == site else eye)
        return result
    def embed_2site(op1, site1, op2, site2):
        result = np.eye(1, dtype='complex128')
        for i in range(L):
            if i == site1: result = np.kron(result, op1)
            elif i == site2: result = np.kron(result, op2)
            else: result = np.kron(result, eye)
        return result
    n2 = n_op @ n_op
    for i in range(L):
        H += E_C * embed_1site(n2, i)
    for i in range(L - 1):
        H += -E_J/2 * flux_phase * embed_2site(exp_iphi, i, exp_miphi, i+1)
        H += -E_J/2 * np.conj(flux_phase) * embed_2site(exp_miphi, i, exp_iphi, i+1)
    return H


def apply_heff_4step(L_env, W1, W2, R_env, theta, D_L, D_R, D_mpo_L, D_mpo_M, D_mpo_R, d):
    """4-step contraction matching pdmrg_gpu.cpp apply_H_eff_cpu."""
    T1 = np.einsum('awp,asdb->wpsdb', L_env, theta)
    T2 = np.einsum('wsqm,wpsdb->mpqdb', W1, T1)
    T3 = np.einsum('mdnr,mpqdb->pqnrb', W2, T2)
    result = np.einsum('brc,pqnrb->pqnc', R_env, T3)
    return result


def apply_heff_direct(L_env, W1, W2, R_env, theta, D_L, D_R, D_mpo_L, D_mpo_M, D_mpo_R, d):
    """Direct single einsum for H_eff."""
    # H_eff * theta[ap, s1p, s2p, bp] =
    #   sum_{a,w,s1,s2,b,wm,wr} L[a,w,ap] * theta[a,s1,s2,b] * W1[w,s1,s1p,wm] * W2[wm,s2,s2p,wr] * R[b,wr,bp]
    return np.einsum('awp,asdb,wsqm,mdnr,brc->pqnc',
                     L_env, theta, W1, W2, R_env, optimize=True)


if __name__ == '__main__':
    print("=" * 60)
    print("DEBUG: H_eff contraction for L=2 (full comparison)")
    print("=" * 60)

    L = 2
    n_max = 1
    d = 2 * n_max + 1
    D = d ** L

    # Build full Hamiltonian
    H_full = build_josephson_hamiltonian_direct(L, n_max=n_max)
    evals_full = np.linalg.eigvalsh(H_full)
    print(f"Full H ground state: {evals_full[0]:.12f}")
    print(f"Full H shape: {H_full.shape}")

    # Build MPO
    mpo = build_josephson_mpo_gpu_style(L, n_max=n_max)
    W1 = mpo[0]  # (1, d, d, 4) for L=2 left boundary
    W2 = mpo[1]  # (4, d, d, 1) for L=2 right boundary

    print(f"\nW1 shape: {W1.shape}")
    print(f"W2 shape: {W2.shape}")

    # For L=2, entire system is one 2-site block
    # L_env = delta(a, ap) = 1x1x1 identity
    # R_env = delta(b, bp) = 1x1x1 identity
    L_env = np.ones((1, 1, 1), dtype='complex128')
    R_env = np.ones((1, 1, 1), dtype='complex128')

    D_L = 1
    D_R = 1
    D_mpo_L = W1.shape[0]  # = 1
    D_mpo_M = W1.shape[3]  # = 4
    D_mpo_R = W2.shape[3]  # = 1

    print(f"\nD_mpo: L={D_mpo_L}, M={D_mpo_M}, R={D_mpo_R}")
    print(f"D_L={D_L}, D_R={D_R}")

    psi_size = D_L * d * d * D_R
    print(f"psi_size = {psi_size}, D = {D}")
    assert psi_size == D, "For L=2 with D_L=D_R=1, psi_size should equal full Hilbert dim"

    # Build H_eff matrix using 4-step method
    H_eff_4step = np.zeros((psi_size, psi_size), dtype='complex128')
    for idx in range(psi_size):
        e_i = np.zeros(psi_size, dtype='complex128')
        e_i[idx] = 1.0
        e_i_tensor = e_i.reshape(D_L, d, d, D_R)
        result_i = apply_heff_4step(L_env, W1, W2, R_env, e_i_tensor,
                                    D_L, D_R, D_mpo_L, D_mpo_M, D_mpo_R, d)
        H_eff_4step[:, idx] = result_i.flatten()

    # Build H_eff matrix using direct einsum
    H_eff_direct = np.zeros((psi_size, psi_size), dtype='complex128')
    for idx in range(psi_size):
        e_i = np.zeros(psi_size, dtype='complex128')
        e_i[idx] = 1.0
        e_i_tensor = e_i.reshape(D_L, d, d, D_R)
        result_i = apply_heff_direct(L_env, W1, W2, R_env, e_i_tensor,
                                     D_L, D_R, D_mpo_L, D_mpo_M, D_mpo_R, d)
        H_eff_direct[:, idx] = result_i.flatten()

    print("\n4-step H_eff:")
    evals_4step = np.linalg.eigvalsh(H_eff_4step)
    print(f"  Ground state: {evals_4step[0]:.12f}")
    print(f"  Hermiticity: {np.max(np.abs(H_eff_4step - H_eff_4step.conj().T)):.2e}")

    print("\nDirect einsum H_eff:")
    evals_direct = np.linalg.eigvalsh(H_eff_direct)
    print(f"  Ground state: {evals_direct[0]:.12f}")
    print(f"  Hermiticity: {np.max(np.abs(H_eff_direct - H_eff_direct.conj().T)):.2e}")

    print("\nFull Hamiltonian:")
    print(f"  Ground state: {evals_full[0]:.12f}")

    # Compare all three
    diff_4step_full = np.max(np.abs(H_eff_4step - H_full))
    diff_direct_full = np.max(np.abs(H_eff_direct - H_full))
    diff_4step_direct = np.max(np.abs(H_eff_4step - H_eff_direct))

    print(f"\n|H_eff_4step - H_full|:    {diff_4step_full:.2e}")
    print(f"|H_eff_direct - H_full|:   {diff_direct_full:.2e}")
    print(f"|H_eff_4step - H_direct|:  {diff_4step_direct:.2e}")

    if diff_4step_full > 1e-10:
        print("\n*** 4-step differs from full H! ***")
        # Print where they differ
        diff = np.abs(H_eff_4step - H_full)
        for i in range(psi_size):
            for j in range(psi_size):
                if diff[i, j] > 1e-10:
                    print(f"  [{i},{j}]: 4step={H_eff_4step[i,j]:.6f}, full={H_full[i,j]:.6f}, "
                          f"diff={diff[i,j]:.6f}")

    if diff_direct_full > 1e-10:
        print("\n*** Direct einsum differs from full H! ***")
        diff = np.abs(H_eff_direct - H_full)
        for i in range(psi_size):
            for j in range(psi_size):
                if diff[i, j] > 1e-10:
                    print(f"  [{i},{j}]: direct={H_eff_direct[i,j]:.6f}, full={H_full[i,j]:.6f}, "
                          f"diff={diff[i,j]:.6f}")
