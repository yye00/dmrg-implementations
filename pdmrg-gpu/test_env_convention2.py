#!/usr/bin/env python3
"""
Determine the correct convention for MPO + environments + H_eff.

The GPU code stores W[wl, s, sp, wr] where op[s,sp] = <s|op|sp>, meaning s=bra, sp=ket.

The standard DMRG convention for the left environment update is:
  L_new[b, wp, b*] = sum_{a,a*,w,sigma,sigma'} L[a,w,a*] * A[a,sigma,b] * W[w,sigma,sigma',wp] * conj(A[a*,sigma',b*])

where sigma is ket, sigma' is bra in W.

But the GPU MPO has W[w, bra, ket, wp], so to use the standard formula
we need to SWAP the physical indices when indexing W.

Let's test this systematically.
"""

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
        H_onsite[i, i] = E_C * charge * charge
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


def test_convention(name, left_env_fn, right_env_fn, heff_fn, mps, mpo, d, L, H_full):
    """Test a specific convention for environments + H_eff."""
    bond_dims = [mps[i].shape[0] for i in range(L)] + [mps[L-1].shape[2]]

    # Compute energy from propagating left environment
    L_env = np.ones((1, 1, 1), dtype='complex128')
    for site in range(L):
        L_env = left_env_fn(L_env, mps[site], mpo[site], d)
    E_env = L_env.squeeze().real

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
    E_env /= norm

    # Compute exact energy
    psi = mps[0]
    for i in range(1, L):
        psi = np.einsum('...j,jkl->...kl', psi, mps[i])
    psi_flat = psi.squeeze().flatten()
    E_exact = ((np.conj(psi_flat) @ H_full @ psi_flat) / (np.conj(psi_flat) @ psi_flat)).real

    env_ok = abs(E_env - E_exact) < 1e-10

    # Test H_eff at site 0
    # Build right env from sites 2..L-1
    R_env = np.ones((1, 1, 1), dtype='complex128')
    for site in range(L-1, 1, -1):
        R_env = right_env_fn(R_env, mps[site], mpo[site], d)

    L_env_0 = np.ones((1, 1, 1), dtype='complex128')
    theta = np.einsum('ijk,klm->ijlm', mps[0], mps[1])
    Htheta = heff_fn(L_env_0, mpo[0], mpo[1], R_env, theta, d)

    numer = np.sum(np.conj(theta) * Htheta)
    denom = np.sum(np.conj(theta) * theta)

    # Exact <psi|H|psi> with this theta
    psi_test = theta
    for i in range(2, L):
        psi_test = np.einsum('...j,jkl->...kl', psi_test, mps[i])
    psi_test_flat = psi_test.squeeze().flatten()
    E_heff_exact = ((np.conj(psi_test_flat) @ H_full @ psi_test_flat) / (np.conj(psi_test_flat) @ psi_test_flat)).real

    # The H_eff Rayleigh quotient should equal the full energy with the rest of the chain
    # contracted out via environments
    E_heff = (numer / denom).real

    # Actually, E_heff won't equal E_heff_exact unless environments contract the rest perfectly.
    # The environments contract the chain parts 2..L-1 using conj(A) on the bra side.
    # For a general (non-canonical) MPS, this DOES depend on the overlap matrix.
    # So let's instead check: does H_eff give a Hermitian operator?
    D_L = bond_dims[0]
    D_R = bond_dims[2]
    psi_size = D_L * d * d * D_R
    H_eff_mat = np.zeros((psi_size, psi_size), dtype='complex128')
    for idx in range(psi_size):
        e_i = np.zeros(psi_size, dtype='complex128')
        e_i[idx] = 1.0
        e_i_tensor = e_i.reshape(D_L, d, d, D_R)
        result_i = heff_fn(L_env_0, mpo[0], mpo[1], R_env, e_i_tensor, d)
        H_eff_mat[:, idx] = result_i.flatten()

    herm_err = np.max(np.abs(H_eff_mat - H_eff_mat.conj().T))
    heff_ok = herm_err < 1e-10

    print(f"  {name}:")
    print(f"    E_env = {E_env:.10f}, E_exact = {E_exact:.10f}, diff = {abs(E_env - E_exact):.2e} {'OK' if env_ok else 'FAIL'}")
    print(f"    H_eff herm_err = {herm_err:.2e} {'OK' if heff_ok else 'FAIL'}")
    print(f"    E_heff = {E_heff:.10f}, E_exact_theta = {E_heff_exact:.10f}")

    return env_ok, heff_ok


# Convention A: GPU current (W[w, s_bra, sp_ket, wp])
# Env: L_new = A[s_bra] * W[w, s_bra, sp_ket, wp] * conj(A[sp_ket])
# H_eff: contract W's first phys idx with theta's phys idx
def left_env_A(L_in, A, W, d):
    """Current GPU convention."""
    return np.einsum('awe,asb,wspr,epc->brc', L_in, A, W, np.conj(A))

def right_env_A(R_in, A, W, d):
    return np.einsum('asb,wspr,brc,epc->awe', A, W, R_in, np.conj(A))

def heff_A(L_env, W1, W2, R_env, theta, d):
    T1 = np.einsum('awp,asdb->wpsdb', L_env, theta)
    T2 = np.einsum('wsqm,wpsdb->mpqdb', W1, T1)
    T3 = np.einsum('mdnr,mpqdb->pqnrb', W2, T2)
    return np.einsum('brc,pqnrb->pqnc', R_env, T3)


# Convention B: Standard DMRG (W[w, s_ket, sp_bra, wp])
# Fix: swap s and sp in the W tensor BEFORE building envs and H_eff
# Since our W stores [w, bra, ket, wp], we access W[w, sp_ket, s_bra, wp]
# which is equivalent to using W_fixed[w, s_ket, s_bra, wp] = W_orig[w, s_bra, s_ket, wp].T in phys

def left_env_B(L_in, A, W, d):
    """Correct: swap W's physical indices to match standard DMRG."""
    # Use W[w, sp, s, wp] instead of W[w, s, sp, wp]
    return np.einsum('awe,asb,wpsr,epc->brc', L_in, A, W, np.conj(A))

def right_env_B(R_in, A, W, d):
    return np.einsum('asb,wpsr,brc,epc->awe', A, W, R_in, np.conj(A))

def heff_B(L_env, W1, W2, R_env, theta, d):
    # Swap: use W[w, s1p_bra, s1_ket, wm] -> theta contracts with s1_ket
    T1 = np.einsum('awp,asdb->wpsdb', L_env, theta)
    T2 = np.einsum('wqsm,wpsdb->mpqdb', W1, T1)  # W1[w, q=s1p, s=s1, m]
    T3 = np.einsum('mndr,mpqdb->pqnrb', W2, T2)   # W2[m, n=s2p, d=s2, r]
    return np.einsum('brc,pqnrb->pqnc', R_env, T3)


# Convention C: env computes with W[w,s,sp,wp] directly (current GPU env)
# but H_eff should use the SWAPPED W
# This tests if the envs are correct but H_eff is wrong
def heff_C(L_env, W1, W2, R_env, theta, d):
    """Env from convention A, H_eff with swapped W."""
    T1 = np.einsum('awp,asdb->wpsdb', L_env, theta)
    T2 = np.einsum('wqsm,wpsdb->mpqdb', W1, T1)
    T3 = np.einsum('mndr,mpqdb->pqnrb', W2, T2)
    return np.einsum('brc,pqnrb->pqnc', R_env, T3)


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

    print("\nConvention A: Current GPU (all consistent, but wrong W interpretation)")
    test_convention("A", left_env_A, right_env_A, heff_A, mps, mpo, d, L, H_full)

    print("\nConvention B: Standard DMRG (swap phys indices in W everywhere)")
    test_convention("B", left_env_B, right_env_B, heff_B, mps, mpo, d, L, H_full)

    print("\nConvention C: Current env (A) + swapped H_eff")
    test_convention("C-env", left_env_A, right_env_A, heff_C, mps, mpo, d, L, H_full)
