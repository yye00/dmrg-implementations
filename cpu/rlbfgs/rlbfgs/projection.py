"""Tangent-space projection and vector transport for TT manifold.

Assuming ``X`` is **right-canonical** with the orthogonality center at
site 0, the tangent space at ``X`` is parameterized by a core list
``{V_i}`` (same shapes as ``X``) subject to the gauge-orthogonality
condition

    for i = 1..L-1:  V_i . A_i^H = 0

viewing each ``V_i, A_i`` as a ``(chi_L, d*chi_R)`` matrix.  At site
``i = 0`` (the center) ``V_0`` is unrestricted.

The orthogonal projection onto the tangent space is applied site-wise::

    V_i <- V_i - (V_i . A_i^H) . A_i         (for i >= 1)
    V_0 <- V_0                                (unchanged)

Vector transport uses the simplest, most common choice: just re-apply
the new-site projection to the transported momentum core list.  This is
the identity map on the tangent space of the new point when the point
hasn't moved, and is a cheap, practical approximation otherwise.
"""

from __future__ import annotations

import numpy as np


def project_right_canonical(X, V):
    """Project the core-list ``V`` onto the tangent space of ``X``.

    ``X`` is assumed right-canonical with the orthogonality center at
    site 0, i.e., for every ``i >= 1`` the core ``X[i]`` (flattened as
    ``(chi_L, d*chi_R)``) has orthonormal rows.

    ``V`` is a list of numpy arrays with the same shape as ``X``.  A new
    core list of the same shapes is returned.
    """
    out = [V[0].copy()]
    for i in range(1, len(X)):
        A = X[i]
        Vi = V[i]
        chi_L, d, chi_R = A.shape
        A_mat = A.reshape(chi_L, d * chi_R)          # (chi_L, d*chi_R), right-isometric
        V_mat = Vi.reshape(chi_L, d * chi_R)
        # Component along the gauge direction: (V_mat A_mat^H) A_mat.
        gauge = (V_mat @ A_mat.conj().T) @ A_mat
        V_proj = V_mat - gauge
        out.append(V_proj.reshape(chi_L, d, chi_R))
    return out


def transport_momentum(X_new, M):
    """Vector transport of momentum ``M`` to the tangent space at ``X_new``.

    Implemented by re-projecting ``M`` using the projection operator of
    ``X_new``.
    """
    return project_right_canonical(X_new, M)
