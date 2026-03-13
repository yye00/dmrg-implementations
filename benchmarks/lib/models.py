"""
Physics model builders for DMRG benchmarks.

Provides Hamiltonian MPO construction for supported models.
"""

import numpy as np


def build_josephson_mpo(L, E_J=1.0, E_C=0.5, mu=0.0, n_max=2):
    """Build Josephson junction array MPO using quimb SpinHam1D.

    Args:
        L: chain length
        E_J: Josephson coupling energy
        E_C: charging energy
        mu: chemical potential
        n_max: charge truncation (d = 2*n_max + 1)

    Returns:
        quimb MatrixProductOperator
    """
    import quimb.tensor as qtn

    d = 2 * n_max + 1
    charges = np.arange(-n_max, n_max + 1, dtype=np.float64)
    n_op = np.diag(charges.astype(np.complex128))

    exp_iphi = np.zeros((d, d), dtype=np.complex128)
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
