"""
Environment tensor management.

This module handles building and caching of left and right environment tensors
used in effective Hamiltonian construction.
"""

from .environment import build_environments_incremental

__all__ = ['build_environments_incremental']
