"""
MPS data structures and canonical forms.

This module handles Matrix Product State representations and transformations
between different canonical forms (left-orthogonal, right-orthogonal, i-orthogonal).
"""

from .mps_utils import (
    create_random_mps,
    verify_left_canonical,
    verify_right_canonical,
    get_mps_norm
)

from .canonical import (
    left_canonicalize,
    right_canonicalize,
    move_orthogonality_center,
    prepare_orthogonal_decompositions,
    verify_i_orthogonal,
    compress_mps
)

__all__ = [
    # MPS creation and utilities
    'create_random_mps',
    'verify_left_canonical',
    'verify_right_canonical',
    'get_mps_norm',
    # Canonical forms
    'left_canonicalize',
    'right_canonicalize',
    'move_orthogonality_center',
    'prepare_orthogonal_decompositions',
    'verify_i_orthogonal',
    'compress_mps',
]
