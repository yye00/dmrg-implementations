#!/usr/bin/env python3
"""
Josephson Junction Array Benchmark

A physically realistic model for superconducting quantum computing circuits.
This model requires complex128 consistently and represents transmon/fluxonium physics.

Hamiltonian:
    H = -E_J Σ_<ij> cos(φ_i - φ_j) + E_C Σ_i n_i² - μ Σ_i n_i

Where:
    - E_J = Josephson coupling energy
    - E_C = Charging energy (capacitive)
    - n_i = charge number operator (integer valued)
    - φ_i = superconducting phase (conjugate to n)

In the charge basis, cos(φ_i - φ_j) = (e^{i(φ_i-φ_j)} + h.c.)/2
where e^{iφ}|n⟩ = |n+1⟩ (phase shift raises charge by 1)
"""

import numpy as np
import quimb.tensor as qtn


def build_josephson_operators(n_max, dtype='complex128'):
    """Build charge and phase operators for truncated charge basis.
    
    Parameters
    ----------
    n_max : int
        Maximum charge number. Charge states: n ∈ {-n_max, ..., +n_max}
    dtype : str
        Data type (should be complex128 for proper phase operators)
    
    Returns
    -------
    n_op : ndarray
        Charge number operator (diagonal)
    exp_iphi : ndarray
        Phase shift operator e^{iφ} (raises charge by 1)
    exp_miphi : ndarray
        Conjugate phase e^{-iφ} (lowers charge by 1)
    """
    d = 2 * n_max + 1  # Local Hilbert space dimension
    
    # Charge number operator: n|n⟩ = n|n⟩
    # States are indexed 0, 1, ..., d-1 corresponding to charges -n_max, ..., +n_max
    charges = np.arange(-n_max, n_max + 1)
    n_op = np.diag(charges.astype(dtype))
    
    # Phase shift operator: e^{iφ}|n⟩ = |n+1⟩
    # This is a raising operator in charge basis
    exp_iphi = np.zeros((d, d), dtype=dtype)
    for i in range(d - 1):
        exp_iphi[i + 1, i] = 1.0 + 0j
    
    # Conjugate: e^{-iφ}|n⟩ = |n-1⟩
    exp_miphi = exp_iphi.conj().T
    
    return n_op, exp_iphi, exp_miphi


def build_josephson_mpo(L, E_J=1.0, E_C=0.5, mu=0.0, n_max=2, 
                        dtype='complex128', with_flux=True):
    """Build Josephson Junction Array MPO.
    
    Parameters
    ----------
    L : int
        Number of junctions (sites)
    E_J : float
        Josephson energy (coupling strength)
    E_C : float
        Charging energy (on-site repulsion)
    mu : float
        Chemical potential (gate charge offset)
    n_max : int
        Maximum charge number per site
    dtype : str
        Data type (complex128 required for phase operators)
    with_flux : bool
        If True, add external flux threading (ensures complex128)
    
    Returns
    -------
    mpo : quimb MatrixProductOperator
    """
    d = 2 * n_max + 1
    n_op, exp_iphi, exp_miphi = build_josephson_operators(n_max, dtype)
    
    # Build with quimb's SpinHam1D
    # S = (d-1)/2 gives local dimension d
    S = (d - 1) / 2
    builder = qtn.SpinHam1D(S=S)
    
    # Add external flux threading to ensure complex128
    # This models flux through the SQUID loops: φ → φ + Φ_ext
    # Physical interpretation: external magnetic field
    if with_flux:
        # External flux per junction (in units of flux quantum)
        phi_ext = np.pi / 4  # π/4 flux quantum threading
        flux_phase = np.exp(1j * phi_ext)
        
        # Josephson coupling with flux: -E_J/2 * (e^{i(φ_i-φ_j+Φ)} + h.c.)
        builder.add_term(-E_J / 2 * flux_phase, exp_iphi, exp_miphi)
        builder.add_term(-E_J / 2 * np.conj(flux_phase), exp_miphi, exp_iphi)
    else:
        # No flux: symmetric hopping (may simplify to real)
        builder.add_term(-E_J / 2, exp_iphi, exp_miphi)
        builder.add_term(-E_J / 2, exp_miphi, exp_iphi)
    
    # Charging energy: E_C * n²
    n2 = n_op @ n_op
    builder.add_term(E_C, n2)
    
    # Chemical potential: -μ * n (gate voltage offset)
    if mu != 0:
        builder.add_term(-mu, n_op)
    
    return builder.build_mpo(L)


def build_transmon_mpo(L, E_J=20.0, E_C=0.3, ng=0.0, n_max=3,
                       dtype='complex128'):
    """Build transmon-like chain MPO.
    
    In the transmon regime: E_J >> E_C (typically E_J/E_C ~ 50-100)
    
    Parameters
    ----------
    L : int
        Number of transmons
    E_J : float
        Josephson energy (large in transmon)
    E_C : float
        Charging energy (small in transmon)
    ng : float
        Gate charge (offset)
    n_max : int
        Charge truncation
    dtype : str
        Data type
    
    Returns
    -------
    mpo : quimb MatrixProductOperator
    """
    return build_josephson_mpo(L, E_J=E_J, E_C=E_C, mu=ng, n_max=n_max, dtype=dtype)


def verify_complex_dtype(mpo):
    """Verify that MPO uses complex128 throughout."""
    for i, tensor in enumerate(mpo.tensors):
        if not np.iscomplexobj(tensor.data):
            return False, f"Site {i} is not complex"
    return True, "All tensors are complex128"


if __name__ == '__main__':
    import time
    
    # Test the Josephson junction model
    print("=" * 60)
    print("Josephson Junction Array Benchmark Test")
    print("=" * 60)
    
    L = 20
    bond_dim = 50
    n_max = 2  # d = 5 local dimension
    
    print(f"\nBuilding Josephson Junction MPO...")
    print(f"  L = {L} sites")
    print(f"  n_max = {n_max} (d = {2*n_max+1} local dim)")
    print(f"  E_J = 1.0, E_C = 0.5")
    
    mpo = build_josephson_mpo(L, E_J=1.0, E_C=0.5, n_max=n_max)
    
    # Verify complex dtype
    is_complex, msg = verify_complex_dtype(mpo)
    print(f"  Complex128: {msg}")
    
    print(f"  MPO shape at site 1: {mpo[1].shape}")
    print(f"  MPO dtype: {mpo[1].data.dtype}")
    
    # Run quimb DMRG2 as reference
    print(f"\nRunning quimb DMRG2 (serial reference)...")
    t0 = time.time()
    dmrg = qtn.DMRG2(mpo, bond_dims=bond_dim)
    dmrg.solve(max_sweeps=20, tol=1e-10, verbosity=0)
    t1 = time.time()
    
    E_ref = dmrg.energy
    print(f"  Energy: {E_ref:.12f}")
    print(f"  Time: {t1-t0:.2f}s")
    print(f"  MPS dtype: {dmrg._k[1].data.dtype}")
    
    print("\n" + "=" * 60)
    print("Ready for PDMRG vs A2DMRG comparison")
    print("=" * 60)
