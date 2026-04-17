"""Retraction step: apply a tangent update and re-canonicalize.

In the simultaneous R-Adam update used by this package, the step
direction is stored with the **same core-wise shape as the MPS** (bond
dim ``chi``), not as a formal rank-``2*chi`` TT tangent vector.  The
retraction is therefore::

    A_i <- A_i + Delta_i               (element-wise for every core)
    X   <- right_canonicalize(X)        (sweep of QRs)
    X[0] <- X[0] / sqrt(<X|X>)          (project off the radial mode)

The ``right_canonicalize`` sweep preserves the bond dimension (no SVD
truncation is necessary since the addition does not enlarge any bond)
and restores the right-canonical form with the orthogonality center at
site 0.

The final normalization step is essential for high-precision
optimization: the Rayleigh quotient ``<X|H|X>/<X|X>`` is invariant
under scaling, so the gradient at site 0 has no component in the
radial direction at the analytic level, but tangent-space updates
followed by retraction will accumulate numerical drift in ``<X|X>``
that re-scales the gradient and, more importantly, corrupts the
L-BFGS curvature pairs ``(s, y)`` across iterations.  Explicitly
fixing ``<X|X> = 1`` after every retraction removes that drift.

If you want to support an enlarging/truncating retraction (e.g., to
run with a dynamic bond dimension), the formal rank-``2*chi`` TT
addition followed by SVD compression would be plugged in here; it is
not used by the default driver.
"""

from __future__ import annotations

import numpy as np

from .mps import right_canonicalize


def retract_and_recanonicalize(X, delta, *, normalize: bool = True):
    """Return ``right_canonicalize([A + D for A, D in zip(X, delta)])``.

    Parameters
    ----------
    X : list of ndarray
        Current MPS (right-canonical with center at site 0).
    delta : list of ndarray
        Step direction with the same shapes as ``X``.  Expected to lie
        in the tangent space, but the retraction is well-defined
        regardless.
    normalize : bool
        If True (default) the returned MPS satisfies ``<X|X> = 1``.

    Returns
    -------
    list of ndarray
        New MPS in right-canonical form with center at site 0.
    """
    assert len(X) == len(delta)
    new_cores = [A + D for A, D in zip(X, delta)]
    out = right_canonicalize(new_cores)
    if normalize:
        # In right-canonical form with centre at site 0, <X|X> = ||X[0]||_F^2.
        n2 = float(np.vdot(out[0], out[0]).real)
        if n2 > 0.0:
            out[0] = out[0] / np.sqrt(n2)
    return out
