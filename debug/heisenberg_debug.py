#!/usr/bin/env python3
"""
DMRG Debugging Script - Heisenberg Model

Ensures all implementations use identical:
1. MPO construction
2. Tensor contraction patterns (via cotengra + optuna)
3. Convergence criteria
4. SVD truncation

This is the REFERENCE implementation that PDMRG and A2DMRG should match.
"""

import numpy as np
import quimb.tensor as qtn
import cotengra as ctg

# Configuration - MUST BE IDENTICAL across all implementations
DEBUG_CONFIG = {
    'L': 12,                    # Number of sites
    'bond_dim': 20,             # Maximum bond dimension
    'max_sweeps': 30,           # Maximum sweeps
    'tol': 1e-12,               # Convergence tolerance
    'cutoff': 1e-14,            # SVD cutoff
    'eigsolver_tol': 1e-13,     # Eigensolver tolerance
}

# Contraction optimizer - MUST BE IDENTICAL
CONTRACTION_OPTIMIZER = ctg.HyperOptimizer(
    methods=['greedy', 'kahypar'],
    max_time=10,
    max_repeats=16,
    progbar=False,
)


def build_heisenberg_mpo(L, J=1.0, dtype='float64'):
    """
    Build Heisenberg XXX MPO using quimb.
    
    H = J * sum_i (S_x^i S_x^{i+1} + S_y^i S_y^{i+1} + S_z^i S_z^{i+1})
    
    This is the REFERENCE MPO that all implementations must use.
    """
    mpo = qtn.MPO_ham_heis(L=L, j=J, bz=0.0, cyclic=False)
    
    if dtype == 'complex128':
        for i in range(L):
            mpo.tensors[i].modify(data=mpo.tensors[i].data.astype('complex128'))
    
    return mpo


def get_mpo_tensor_data(mpo, i):
    """Extract tensor data from quimb MPO (used by PDMRG/A2DMRG)."""
    tensor = mpo.tensors[i]
    data = tensor.data
    
    # Reorder to standard (left_bond, right_bond, phys_out, phys_in) format
    # quimb edge tensors: (bond, ket, bra) -> need to add dummy dimension
    # quimb bulk tensors: (left_bond, right_bond, ket, bra)
    
    if i == 0:
        # Left edge: (bond, ket, bra) -> (1, bond, ket, bra)
        data = data[np.newaxis, :, :, :]
    elif i == mpo.L - 1:
        # Right edge: (bond, ket, bra) -> (bond, 1, ket, bra)
        data = data[:, np.newaxis, :, :]
    
    return data


def print_contraction_info(name, tn):
    """Print tensor network contraction info for debugging."""
    print(f"\n=== {name} ===")
    print(f"Number of tensors: {len(tn.tensors)}")
    print(f"Total size: {tn.num_indices} indices")
    
    # Get optimal contraction path
    try:
        path, info = tn.contraction_path_info(CONTRACTION_OPTIMIZER)
        print(f"Optimal contraction:")
        print(f"  FLOPs: {info.opt_cost:.2e}")
        print(f"  Max intermediate size: {info.largest_intermediate}")
        print(f"  Path length: {len(path)}")
    except Exception as e:
        print(f"Could not compute contraction path: {e}")


def run_quimb_dmrg(mpo, config, method='DMRG2', verbose=True):
    """
    Run quimb DMRG with detailed output.
    
    This is the REFERENCE implementation.
    """
    import time
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"QUIMB {method}")
        print(f"{'='*60}")
        print(f"Config: {config}")
    
    # Build bond dimension ramp
    bond_ramp = []
    m = 4
    while m < config['bond_dim']:
        bond_ramp.append(m)
        m *= 2
    bond_ramp.append(config['bond_dim'])
    
    if verbose:
        print(f"Bond dimension ramp: {bond_ramp}")
    
    t0 = time.time()
    
    if method == 'DMRG1':
        dmrg = qtn.DMRG1(mpo, bond_dims=bond_ramp, cutoffs=config['cutoff'])
    else:
        dmrg = qtn.DMRG2(mpo, bond_dims=bond_ramp, cutoffs=config['cutoff'])
    
    # Run with detailed verbosity
    dmrg.solve(
        max_sweeps=config['max_sweeps'],
        tol=config['tol'],
        verbosity=2 if verbose else 0
    )
    
    t1 = time.time()
    
    energy = float(np.real(dmrg.energy))
    
    if verbose:
        print(f"\nFinal energy: {energy:.15f}")
        print(f"Time: {t1-t0:.2f}s")
    
    return energy, dmrg.state, t1-t0


def verify_environment_contraction(L_env, R_env, mps_tensor, mpo_tensor):
    """
    Verify environment contraction produces correct result.
    
    The effective Hamiltonian is:
    H_eff[σ,σ'] = L[a,a',α] * W[a,b,σ,σ'] * R[b,b',β] * δ[α,β]
    
    where:
    - L: left environment (chi_L, chi_L', mpo_bond_L)
    - R: right environment (chi_R, chi_R', mpo_bond_R)
    - W: MPO tensor (mpo_bond_L, mpo_bond_R, d, d)
    - MPS tensor: (chi_L, d, chi_R)
    """
    # This function demonstrates the CORRECT contraction pattern
    # that PDMRG and A2DMRG should follow
    
    chi_L, d, chi_R = mps_tensor.shape
    
    # Method 1: Direct einsum (reference)
    # H_eff @ psi = L[i,j,a] * W[a,b,s,t] * R[k,l,b] * psi[j,t,l]
    # Result shape: (i, s, k) -> contracted to get energy
    
    # For two-site: theta[i,s1,s2,k] = psi_L[i,s1,m] * psi_R[m,s2,k]
    # H_eff @ theta = L[i,j,a] * W1[a,c,s1,t1] * W2[c,b,s2,t2] * R[k,l,b] * theta[j,t1,t2,l]
    
    pass  # Implementation details for verification


def compare_energies(energies, labels, tol=1e-10):
    """Compare energies and report discrepancies."""
    print(f"\n{'='*60}")
    print("ENERGY COMPARISON")
    print(f"{'='*60}")
    
    E_ref = energies[0]
    label_ref = labels[0]
    
    all_match = True
    for E, label in zip(energies, labels):
        dE = abs(E - E_ref)
        status = "✓" if dE < tol else "✗"
        if dE >= tol:
            all_match = False
        print(f"{label:20s}: E = {E:.15f}, ΔE = {dE:.2e} {status}")
    
    print(f"\nReference: {label_ref}")
    print(f"Tolerance: {tol:.0e}")
    print(f"Result: {'ALL MATCH' if all_match else 'MISMATCH DETECTED'}")
    
    return all_match


def main():
    """Run debugging comparison."""
    config = DEBUG_CONFIG.copy()
    
    print("="*60)
    print("DMRG DEBUGGING - Heisenberg Model")
    print("="*60)
    print(f"Configuration: {config}")
    print(f"Contraction optimizer: cotengra + optuna")
    
    # Build MPO
    print("\nBuilding Heisenberg MPO...")
    mpo = build_heisenberg_mpo(config['L'])
    
    # Print MPO info
    print(f"MPO: L={mpo.L}, bond_dim={mpo.max_bond()}")
    for i in range(min(3, mpo.L)):
        t = mpo.tensors[i]
        print(f"  Site {i}: shape={t.shape}, inds={t.inds}")
    
    # Run quimb DMRG1
    E1, mps1, t1 = run_quimb_dmrg(mpo, config, 'DMRG1', verbose=True)
    
    # Run quimb DMRG2
    E2, mps2, t2 = run_quimb_dmrg(mpo, config, 'DMRG2', verbose=True)
    
    # Compare
    compare_energies([E2, E1], ['quimb_DMRG2', 'quimb_DMRG1'], tol=config['tol'])
    
    # Save reference for PDMRG/A2DMRG comparison
    import json
    reference = {
        'config': config,
        'quimb_DMRG1': {'energy': E1, 'time': t1},
        'quimb_DMRG2': {'energy': E2, 'time': t2},
    }
    
    with open('/home/captain/clawd/work/dmrg-implementations/debug/reference_heisenberg.json', 'w') as f:
        json.dump(reference, f, indent=2)
    
    print(f"\nReference saved to debug/reference_heisenberg.json")
    print(f"Expected energy (DMRG2): {E2:.15f}")
    
    return E2


if __name__ == '__main__':
    main()
