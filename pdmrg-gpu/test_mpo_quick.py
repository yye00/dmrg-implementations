#!/usr/bin/env python3
"""Quick test: GPU-style MPO vs direct Hamiltonian construction."""

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
        charge = float(i - n_max)
        n_op[i, i] = charge
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
        if mu != 0:
            H -= mu * embed_1site(n_op, i)
    for i in range(L - 1):
        H += -E_J/2 * flux_phase * embed_2site(exp_iphi, i, exp_miphi, i+1)
        H += -E_J/2 * np.conj(flux_phase) * embed_2site(exp_miphi, i, exp_iphi, i+1)
    return H


def mpo_to_full_hamiltonian(mpo_tensors, d):
    L = len(mpo_tensors)
    D = d ** L
    result = mpo_tensors[0]
    for site in range(1, L):
        W = mpo_tensors[site]
        result = np.einsum('...w,wjkl->...jkl', result, W)
    result = result.squeeze()
    ket_indices = list(range(0, 2 * L, 2))
    bra_indices = list(range(1, 2 * L, 2))
    new_order = ket_indices + bra_indices
    result = result.transpose(new_order)
    result = result.reshape(D, D)
    return result


if __name__ == '__main__':
    print("MPO Verification Tests")
    print("=" * 60)

    for L in [2, 3, 4]:
        for n_max in [1, 2]:
            d = 2 * n_max + 1
            D = d ** L
            if D > 10000:
                continue

            H_direct = build_josephson_hamiltonian_direct(L, n_max=n_max)
            mpo = build_josephson_mpo_gpu_style(L, n_max=n_max)
            H_mpo = mpo_to_full_hamiltonian(mpo, d)

            diff = np.max(np.abs(H_direct - H_mpo))
            herm_err = np.max(np.abs(H_mpo - H_mpo.conj().T))
            evals = np.linalg.eigvalsh(H_direct)

            status = "OK" if diff < 1e-10 else "FAIL"
            print(f"  L={L}, n_max={n_max} (d={d}, D={D}): "
                  f"|H_diff|={diff:.2e}, herm={herm_err:.2e}, "
                  f"E0={evals[0]:.10f} [{status}]")

            if diff > 1e-10:
                evals_mpo = np.linalg.eigvalsh(H_mpo)
                print(f"    E0_direct={evals[0]:.10f}, E0_mpo={evals_mpo[0]:.10f}")
