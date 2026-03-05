#!/usr/bin/env python3
"""Debug: find the exact index convention bug in H_eff."""

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


def mpo_to_full_hamiltonian(mpo_tensors, d):
    """Contract all MPO tensors to build the full Hamiltonian matrix.
    W[wl, s, sp, wr] where s=ket, sp=bra."""
    L = len(mpo_tensors)
    D = d ** L
    result = mpo_tensors[0]
    for site in range(1, L):
        W = mpo_tensors[site]
        result = np.einsum('...w,wjkl->...jkl', result, W)
    result = result.squeeze()
    # result has 2L indices: (s0, s0', s1, s1', ...)
    # where first index of each pair is ket, second is bra
    ket_indices = list(range(0, 2 * L, 2))
    bra_indices = list(range(1, 2 * L, 2))
    new_order = ket_indices + bra_indices
    result = result.transpose(new_order)
    result = result.reshape(D, D)
    return result


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


if __name__ == '__main__':
    L = 2
    n_max = 1
    d = 2 * n_max + 1

    H_direct = build_josephson_hamiltonian_direct(L, n_max=n_max)
    mpo = build_josephson_mpo_gpu_style(L, n_max=n_max)

    # First verify the MPO -> full H contraction
    H_mpo = mpo_to_full_hamiltonian(mpo, d)
    print(f"|H_mpo - H_direct| = {np.max(np.abs(H_mpo - H_direct)):.2e}")

    W1 = mpo[0]  # (1, d, d, 4)
    W2 = mpo[1]  # (4, d, d, 1)

    print("\nChecking MPO convention:")
    print(f"W1 shape: {W1.shape}  =  (D_L, s, sp, D_R)")
    print(f"W2 shape: {W2.shape}  =  (D_L, s, sp, D_R)")

    # Let's check: is s=ket and sp=bra, or vice versa?
    # The MPO acts as: H|psi> where |psi> = sum_s psi_s |s>
    # So H[s', s] = <s'|H|s>, where s is ket (input) and s' is bra (output)
    # In the MPO: W[wl, ?, ?, wr] -> the two ? are phys indices
    # For H[s',s] = sum_w W1[1,?,?,w] * W2[w,?,?,1]

    # Let's manually check element H[3,1] (row 3, col 1) from the direct H:
    print(f"\nDirect H[3,1] = {H_direct[3,1]:.6f}")  # expect -0.353553-0.353553j (from earlier output but let me check)
    print(f"Direct H[1,3] = {H_direct[1,3]:.6f}")

    # For L=2, d=3: index 1 = (s0=0, s1=1), index 3 = (s0=1, s1=0)
    # So H[3,1] = H[(s0'=1,s1'=0), (s0=0,s1=1)] = <1,0|H|0,1>
    # This involves the hopping term: exp_iphi on site 0, exp_miphi on site 1
    # exp_iphi[1,0] = 1 (raises charge from 0 to 1)
    # exp_miphi[0,1] = 1 (lowers charge from 1 to 0)
    # Coupling: -E_J/2 * flux_phase = -0.5 * exp(i*pi/4) = -0.5*(cos(pi/4)+i*sin(pi/4))
    # = -0.5*(0.7071+0.7071i) = -0.35355-0.35355i
    print(f"\nExpected H[3,1] = -E_J/2 * exp(i*pi/4) = {-0.5 * np.exp(1j * np.pi/4):.6f}")

    # Now let's compute from MPO using the H_eff formula:
    # H_eff[s1',s2', s1,s2] = sum_w W1[1, s1, s1', w] * W2[w, s2, s2', 1]
    # where W[wl, s, sp, wr] and s=ket, sp=bra
    # So W1[1, s1=0, s1'=1, w] * W2[w, s2=1, s2'=0, 1]
    # = sum_w W1[0, 0, 1, w] * W2[w, 1, 0, 0]

    print("\nMPO H_eff computation (W[wl,s_ket,s_bra,wr]):")
    val = 0.0j
    for w in range(4):
        v1 = W1[0, 0, 1, w]
        v2 = W2[w, 1, 0, 0]
        if abs(v1) > 1e-10 and abs(v2) > 1e-10:
            print(f"  w={w}: W1[0,0,1,{w}]={v1:.6f} * W2[{w},1,0,0]={v2:.6f} = {v1*v2:.6f}")
        val += v1 * v2
    print(f"  Total: {val:.6f}")
    print(f"  Expected: {H_direct[3,1]:.6f}")

    if abs(val - H_direct[3, 1]) > 1e-10:
        print(f"  MISMATCH! diff = {abs(val - H_direct[3,1]):.6f}")
        # Try swapped convention: s=bra, sp=ket
        print("\nTrying swapped convention (W[wl,s_bra,s_ket,wr]):")
        val2 = 0.0j
        for w in range(4):
            v1 = W1[0, 1, 0, w]  # W1[1, s1'=1, s1=0, w]
            v2 = W2[w, 0, 1, 0]  # W2[w, s2'=0, s2=1, 1]
            if abs(v1) > 1e-10 and abs(v2) > 1e-10:
                print(f"  w={w}: W1[0,1,0,{w}]={v1:.6f} * W2[{w},0,1,0]={v2:.6f} = {v1*v2:.6f}")
            val2 += v1 * v2
        print(f"  Total: {val2:.6f}")
        print(f"  Expected: {H_direct[3,1]:.6f}")
        if abs(val2 - H_direct[3, 1]) < 1e-10:
            print("  ==> s and sp are SWAPPED in the MPO! s=bra, sp=ket (NOT s=ket, sp=bra)")

    # Now the key question: in the H_eff contraction,
    # the GPU code does: W[w, s, sp, wp] where it treats s as ket and sp as bra
    # But from the MPO construction, the convention in set_op is:
    # idx = wl * d * d * D_R + s * d * D_R + sp * D_R + wr
    # And op[s * d + sp] accesses the operator
    # For exp_iphi[(i+1)*d + i] = 1, this means exp_iphi[s,sp] where s=i+1, sp=i
    # exp_iphi matrix: row=s, col=sp -> exp_iphi[1,0] = 1 means |1><0| or raise
    # So in the GPU MPO: W[wl, s, sp, wr] where op[s,sp] = <s|op|sp>
    # This means s=bra and sp=ket!

    print("\n\n=== KEY FINDING ===")
    print("In the GPU MPO construction (set_op), op[s*d + sp] is stored in W[wl, s, sp, wr]")
    print("The operator matrix op[i,j] = <i|op|j>, so op[s,sp] = <s|op|sp>")
    print("Therefore: s = bra index, sp = ket index")
    print("")
    print("But in the H_eff contraction (apply_H_eff_cpu), the code contracts:")
    print("  W1[w, s1, s1p, wm] with theta[a, s1, s2, b]")
    print("  where s1 in theta is a KET index")
    print("  and s1 in W is a BRA index")
    print("")
    print("This means the H_eff is contracting the WRONG physical index of W with theta!")
    print("It should contract W's sp (ket) index with theta's s (ket), not W's s (bra) index.")
    print("")
    print("The fix: swap s and sp in the W contraction, i.e., use W[w, s1p, s1, wm]")
    print("Or equivalently, transpose the physical indices of the MPO.")
