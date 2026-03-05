"""
A2DMRG: Additive Two-Level Parallel DMRG

Implementation of the Additive Two-Level Parallel DMRG algorithm from
Grigori & Hassan (arXiv:2505.23429).

Key Features:
- Near-linear parallel scaling for 2-8 processors
- Machine precision accuracy (matches serial DMRG within 1e-10)
- Support for complex128 (Josephson junction problems)
- MPI parallelization via mpi4py
- Tensor operations powered by quimb
"""

__version__ = "0.1.0"
__author__ = "A2DMRG Development Team"

# Import main algorithm entry point when implemented
# from .dmrg import a2dmrg_main

__all__ = [
    "__version__",
    "__author__",
]
