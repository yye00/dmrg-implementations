#!/usr/bin/env python3
"""Quick test of Josephson d=5 implementation."""

import sys
import os
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'pdmrg'))

import quimb.tensor as qtn


def build_josephson_mpo(L, E_J=1.0, E_C=0.5, mu=0.0, n_max=2, dtype='complex128'):
    """Build Josephson junction array MPO in the charge basis."""
    d = 2 * n_max + 1

    charges = np.arange(-n_max, n_max + 1, dtype=np.float64)
    n_op = np.diag(charges.astype(dtype))

    exp_iphi = np.zeros((d, d), dtype=dtype)
    for i in range(d - 1):
        exp_iphi[i + 1, i] = 1.0 + 0j

    exp_miphi = exp_iphi.conj().T

    S = (d - 1) / 2
    builder = qtn.SpinHam1D(S=S)

    phi_ext = np.pi / 4
    flux_phase = np.exp(1j * phi_ext)

    builder.add_term(-E_J / 2 * flux_phase, exp_iphi, exp_miphi)
    builder.add_term(-E_J / 2 * np.conj(flux_phase), exp_miphi, exp_iphi)

    n2 = n_op @ n_op
    builder.add_term(E_C, n2)

    if mu != 0:
        builder.add_term(-mu, n_op)

    return builder.build_mpo(L)


print("Testing Josephson d=5 MPO construction...")
print(f"Building L=8, n_max=2 (d=5) MPO...")

t0 = time.time()
mpo = build_josephson_mpo(L=8, n_max=2)
t1 = time.time()

print(f"  MPO built in {t1-t0:.3f}s")
print(f"  MPO has {len(mpo.tensors)} tensors")
print(f"  Physical dim: {mpo.phys_dim()}")

print("\nRunning DMRG1 with very low bond_dim and max_sweeps for quick test...")
t0 = time.time()
dmrg = qtn.DMRG1(mpo, bond_dims=20, cutoffs=1e-14)
dmrg.solve(max_sweeps=5, tol=1e-6, verbosity=1)
t1 = time.time()

energy = float(np.real(dmrg.energy))
print(f"\nResult:")
print(f"  Energy: {energy:.12f}")
print(f"  Time: {t1-t0:.2f}s")
print(f"  Expected (L=8, D=50, full convergence): -2.843801043291333")
print(f"  This is a quick test with D=20, so energy will be higher")
