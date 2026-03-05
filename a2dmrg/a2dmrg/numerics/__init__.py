"""
Numerical methods for A2DMRG.

This module provides numerical algorithms including:
- Standard truncated SVD (NOT recursive accurate SVD)
- Eigensolver wrappers for effective Hamiltonians
- Effective Hamiltonian construction as LinearOperator
- Local DMRG micro-steps (one-site and two-site updates)
- Observable computations (energy and overlap)
- Coarse-space eigenvalue problem solver
"""

from .truncated_svd import truncated_svd, reconstruct_from_svd, truncation_error_bound
from .eigensolver import solve_effective_hamiltonian, solve_effective_hamiltonian_2site
from .effective_ham import build_effective_hamiltonian_1site, build_effective_hamiltonian_2site
from .local_microstep import local_microstep_1site, local_microstep_2site
from .observables import compute_energy, compute_overlap
from .coarse_eigenvalue import (
    solve_coarse_eigenvalue_problem,
    verify_solution,
    estimate_condition_number
)

__all__ = [
    'truncated_svd', 'reconstruct_from_svd', 'truncation_error_bound',
    'solve_effective_hamiltonian', 'solve_effective_hamiltonian_2site',
    'build_effective_hamiltonian_1site', 'build_effective_hamiltonian_2site',
    'local_microstep_1site', 'local_microstep_2site',
    'compute_energy', 'compute_overlap',
    'solve_coarse_eigenvalue_problem', 'verify_solution', 'estimate_condition_number'
]
