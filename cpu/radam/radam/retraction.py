"""Retraction step: apply a tangent update and re-canonicalize.

In the simultaneous R-Adam update used by this package, the step
direction is stored with the **same core-wise shape as the MPS** (bond
dim ``chi``), not as a formal rank-``2*chi`` TT tangent vector.  The
retraction is therefore::

    A_i <- A_i + Delta_i           (element-wise for every core)
    X   <- right_canonicalize(X)    (sweep of QRs)

The ``right_canonicalize`` sweep preserves the bond dimension (no SVD
truncation is necessary since the addition does not enlarge any bond)
and restores the right-canonical form with the orthogonality center at
site 0.

If you want to support an enlarging/truncating retraction (e.g., to
run with a dynamic bond dimension), the commented-out
``retract_via_tt_sum`` shows how the formal rank-``2*chi`` TT addition
followed by SVD compression would be implemented; it is not used by
the default driver.
"""

from __future__ import annotations

import numpy as np

from .mps import right_canonicalize


def retract_and_recanonicalize(X, delta):
    """Return ``right_canonicalize([A + D for A, D in zip(X, delta)])``.

    Parameters
    ----------
    X : list of ndarray
        Current MPS (right-canonical with center at site 0).
    delta : list of ndarray
        Step direction with the same shapes as ``X``.  Expected to lie
        in the tangent space, but the retraction is well-defined
        regardless.

    Returns
    -------
    list of ndarray
        New MPS in right-canonical form with center at site 0.
    """
    assert len(X) == len(delta)
    new_cores = [A + D for A, D in zip(X, delta)]
    return right_canonicalize(new_cores)
