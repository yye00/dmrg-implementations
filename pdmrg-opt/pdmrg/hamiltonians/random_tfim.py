"""Random Transverse-Field Ising Model (RTFIM).

H = -Σᵢ Jᵢ ZᵢZᵢ₊₁ - Σᵢ hᵢ Xᵢ

This is a standard benchmark for quantum computing simulations:
- ZZ interactions mimic CZ gate entanglement
- Random couplings create disorder/complexity
- Requires significant bond dimension for accurate ground state
"""

import numpy as np
import quimb.tensor as qtn


def build_random_tfim_mpo(L, J_mean=1.0, J_std=0.5, h_mean=1.0, h_std=0.5,
                           seed=42, dtype='float64'):
    """Build Random Transverse-Field Ising Model MPO.

    H = -Σᵢ Jᵢ ZᵢZᵢ₊₁ - Σᵢ hᵢ Xᵢ

    Parameters
    ----------
    L : int
        Number of sites (qubits).
    J_mean, J_std : float
        Mean and std of ZZ coupling strengths.
    h_mean, h_std : float
        Mean and std of transverse field strengths.
    seed : int
        Random seed for reproducibility.
    dtype : str
        Data type.

    Returns
    -------
    mpo : quimb MPO
    couplings : dict
        Dictionary with 'J' and 'h' arrays for reference.
    """
    np.random.seed(seed)
    
    # Generate random couplings
    J = J_mean + J_std * np.random.randn(L - 1)
    h = h_mean + h_std * np.random.randn(L)
    
    # Build with quimb SpinHam1D
    builder = qtn.SpinHam1D(S=1/2)
    
    # ZZ interactions: -Jᵢ ZᵢZᵢ₊₁
    for i in range(L - 1):
        builder[i, i+1] += -J[i], 'Z', 'Z'
    
    # Transverse field: -hᵢ Xᵢ
    for i in range(L):
        builder[i] += -h[i], 'X'
    
    mpo = builder.build_mpo(L)
    
    # Convert dtype if needed
    if dtype == 'complex128':
        for i in range(L):
            mpo[i].modify(data=mpo[i].data.astype('complex128'))
    
    couplings = {'J': J, 'h': h}
    return mpo, couplings


def get_exact_energy_small(L, J, h):
    """Compute exact ground state energy for small systems (L <= 16).
    
    Uses exact diagonalization for verification.
    """
    if L > 16:
        return None
    
    from scipy.sparse import kron, eye
    from scipy.sparse.linalg import eigsh
    
    # Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def pauli_at(op, i, L):
        """Put Pauli operator at site i in L-site system."""
        result = 1
        for j in range(L):
            if j == i:
                result = kron(result, op, format='csr')
            else:
                result = kron(result, I, format='csr')
        return result
    
    # Build Hamiltonian
    H = 0
    for i in range(L - 1):
        H = H - J[i] * pauli_at(Z, i, L) @ pauli_at(Z, i+1, L)
    for i in range(L):
        H = H - h[i] * pauli_at(X, i, L)
    
    # Find ground state
    E0, _ = eigsh(H, k=1, which='SA')
    return E0[0]
