"""
MPI parallelization for A2DMRG.

This module handles all MPI communication patterns including:
- Site distribution across processors
- Coarse-space matrix computation and Allreduce
- Coefficient broadcasting
- Linear combination formation
"""

from .distribute import distribute_sites, verify_distribution
from .local_steps import (
    parallel_local_microsteps,
    gather_local_results,
    prepare_candidate_mps_list,
)
from .coarse_space import (
    build_coarse_matrices,
    verify_hermitian,
    verify_positive_semidefinite,
)
from .linear_combination import (
    form_linear_combination,
    verify_linear_combination_energy,
)
# from .communication import allreduce_matrices, broadcast_coefficients

__all__ = [
    'distribute_sites',
    'verify_distribution',
    'parallel_local_microsteps',
    'gather_local_results',
    'prepare_candidate_mps_list',
    'build_coarse_matrices',
    'verify_hermitian',
    'verify_positive_semidefinite',
    'form_linear_combination',
    'verify_linear_combination_energy',
]
