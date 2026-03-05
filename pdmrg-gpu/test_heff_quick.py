#!/usr/bin/env python3
"""Quick test: 4-step H_eff contraction vs einsum reference."""

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


def apply_heff_4step(L_env, W1, W2, R_env, theta, D_L, D_R, D_mpo_L, D_mpo_M, D_mpo_R, d):
    """Exact replica of apply_H_eff_cpu from pdmrg_gpu.cpp."""
    # Use distinct labels for all indices to avoid confusion:
    # a=ket left bond, p=bra left bond (ap)
    # s=ket phys1 (s1), q=bra phys1 (s1p)
    # d=ket phys2 (s2), n=bra phys2 (s2p)
    # b=ket right bond, c=bra right bond (bp)
    # w=left MPO bond, m=mid MPO bond (wm), r=right MPO bond (wr)

    # Step 1: T1[w, ap, s1, s2, b] = sum_a L[a, w, ap] * theta[a, s1, s2, b]
    # L_env[a, w, p], theta[a, s, d, b] -> T1[w, p, s, d, b]
    T1 = np.einsum('awp,asdb->wpsdb', L_env, theta)

    # Step 2: T2[wm, ap, s1p, s2, b] = sum_{w, s1} W1[w, s1, s1p, wm] * T1[w, ap, s1, s2, b]
    # W1[w, s, q, m], T1[w, p, s, d, b] -> T2[m, p, q, d, b]
    T2 = np.einsum('wsqm,wpsdb->mpqdb', W1, T1)

    # Step 3: T3[ap, s1p, s2p, wr, b] = sum_{wm, s2} W2[wm, s2, s2p, wr] * T2[wm, ap, s1p, s2, b]
    # W2[m, d, n, r], T2[m, p, q, d, b] -> T3[p, q, n, r, b]
    T3 = np.einsum('mdnr,mpqdb->pqnrb', W2, T2)

    # Step 4: result[ap, s1p, s2p, bp] = sum_{b, wr} R[b, wr, bp] * T3[ap, s1p, s2p, wr, b]
    # R[b, r, c], T3[p, q, n, r, b] -> result[p, q, n, c]
    result = np.einsum('brc,pqnrb->pqnc', R_env, T3)

    return result


def apply_heff_direct(L_env, W1, W2, R_env, theta, D_L, D_R, D_mpo_L, D_mpo_M, D_mpo_R, d):
    """Direct einsum for H_eff (reference)."""
    # H_eff * theta = L[a,w,ap] * theta[a,s1,s2,b] * W1[w,s1,s1p,wm] * W2[wm,s2,s2p,wr] * R[b,wr,bp]
    return np.einsum('awp,asdb,wsqm,mjnr,bro->pqno',
                     L_env, theta, W1, W2, R_env, optimize=True)


def build_left_env(mps, mpo, d, up_to_site):
    """Build left environment."""
    L_env = np.ones((1, 1, 1), dtype='complex128')
    for site in range(up_to_site):
        A = mps[site]
        W = mpo[site]
        # L_new[b, wp, bstar] = sum_{a,astar,w,s,sp} L[a,w,astar] * A[a,s,b] * W[w,s,sp,wp] * conj(A[astar,sp,bstar])
        L_new = np.einsum('awe,asb,wspr,epc->brc', L_env, A, W, np.conj(A))
        L_env = L_new
    return L_env


def build_right_env(mps, mpo, d, from_site, L):
    """Build right environment from from_site to L-1."""
    R_env = np.ones((1, 1, 1), dtype='complex128')
    for site in range(L - 1, from_site - 1, -1):
        A = mps[site]
        W = mpo[site]
        # R_new[a, w, astar] = sum_{b,bstar,wp,s,sp} A[a,s,b] * W[w,s,sp,wp] * R[b,wp,bstar] * conj(A[astar,sp,bstar])
        R_new = np.einsum('asb,wspr,brc,epc->awe', A, W, R_env, np.conj(A))
        R_env = R_new
    return R_env


if __name__ == '__main__':
    print("H_eff 4-step Contraction Tests")
    print("=" * 60)

    L = 4
    n_max = 1
    d = 2 * n_max + 1

    mpo = build_josephson_mpo_gpu_style(L, n_max=n_max)

    np.random.seed(42)
    bond_dims = [1]
    for i in range(1, L):
        bd = min(10, d ** i, d ** (L - i))
        bond_dims.append(bd)
    bond_dims.append(1)
    print(f"L={L}, d={d}, bond_dims={bond_dims}")

    mps = []
    for i in range(L):
        Da = bond_dims[i]
        Db = bond_dims[i + 1]
        A = np.random.randn(Da, d, Db) + 1j * np.random.randn(Da, d, Db)
        A /= np.linalg.norm(A)
        mps.append(A)

    for site in range(L - 1):
        D_L = bond_dims[site]
        D_R = bond_dims[site + 2]
        W1 = mpo[site]
        W2 = mpo[site + 1]
        D_mpo_L = W1.shape[0]
        D_mpo_M = W1.shape[3]
        D_mpo_R = W2.shape[3]

        L_env = build_left_env(mps, mpo, d, site)
        R_env = build_right_env(mps, mpo, d, site + 2, L)

        theta = np.random.randn(D_L, d, d, D_R) + 1j * np.random.randn(D_L, d, d, D_R)

        result_4step = apply_heff_4step(L_env, W1, W2, R_env, theta,
                                        D_L, D_R, D_mpo_L, D_mpo_M, D_mpo_R, d)
        result_direct = apply_heff_direct(L_env, W1, W2, R_env, theta,
                                          D_L, D_R, D_mpo_L, D_mpo_M, D_mpo_R, d)

        diff = np.max(np.abs(result_4step - result_direct))
        status = "OK" if diff < 1e-10 else "FAIL"
        print(f"  site {site}: D_L={D_L}, D_R={D_R}, D_mpo=[{D_mpo_L},{D_mpo_M},{D_mpo_R}] "
              f"| diff={diff:.2e} [{status}]")

    # Now test Hermiticity of H_eff
    print("\nH_eff Hermiticity tests:")
    for site in range(L - 1):
        D_L = bond_dims[site]
        D_R = bond_dims[site + 2]
        W1 = mpo[site]
        W2 = mpo[site + 1]
        D_mpo_L = W1.shape[0]
        D_mpo_M = W1.shape[3]
        D_mpo_R = W2.shape[3]

        L_env = build_left_env(mps, mpo, d, site)
        R_env = build_right_env(mps, mpo, d, site + 2, L)

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
        evals = np.linalg.eigvalsh(H_eff)
        status = "OK" if herm_err < 1e-10 else "FAIL"
        print(f"  site {site}: psi_size={psi_size}, herm_err={herm_err:.2e}, "
              f"E0={evals[0]:.10f} [{status}]")
