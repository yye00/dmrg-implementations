"""Tangent-vector arithmetic helpers for R-LBFGS on the TT manifold.

Tangent vectors on the TT manifold are stored as **lists of numpy
arrays** with the same shapes as the underlying MPS cores.  These
helpers treat the tangent space as a flat vector space (i.e., we use
the Euclidean / Frobenius inner product on the list-of-cores
representation).

Strictly speaking the manifold-induced metric at a right-canonical
gauge has non-trivial weights from ``L_norm[i]`` at sites ``i >= 1``;
however, the gauge-orthogonality of the tangent vectors (as enforced
by :func:`rlbfgs.projection.project_right_canonical`) makes the
Euclidean choice a faithful and, in practice, well-conditioned proxy.

All routines accept either real or complex dtype core lists.  Inner
products on complex cores use :func:`numpy.vdot` and we take the real
part when a real-valued scalar is needed (the imaginary part is zero
by Hermiticity of the underlying form for a holomorphic energy
functional, but we project to be safe).
"""

from __future__ import annotations

import numpy as np


def inner_real(V, W) -> float:
    """Real part of the Euclidean inner product of two tangent core lists."""
    s = 0.0
    for vi, wi in zip(V, W):
        s += float(np.vdot(vi, wi).real)
    return s


def norm_tangent(V) -> float:
    """Frobenius norm of a tangent vector stored as a core list."""
    return float(np.sqrt(max(inner_real(V, V), 0.0)))


def scale_cores(V, alpha):
    """Return ``[alpha * v for v in V]``."""
    return [alpha * vi for vi in V]


def add_cores(V, W, alpha: float = 1.0):
    """Return ``[v + alpha * w for v, w in zip(V, W)]``."""
    return [vi + alpha * wi for vi, wi in zip(V, W)]


def sub_cores(V, W):
    """Return ``[v - w for v, w in zip(V, W)]``."""
    return [vi - wi for vi, wi in zip(V, W)]


def copy_cores(V):
    """Return a deep numpy-level copy of the core list."""
    return [vi.copy() for vi in V]
