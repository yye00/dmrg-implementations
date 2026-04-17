"""MPS / TT data structure helpers (pure numpy).

An MPS is a ``list`` of numpy arrays, one per site, each with shape
``(chi_L, d, chi_R)`` where ``d`` is the physical dimension.  Boundary
sites use ``chi_L = 1`` (site 0) and ``chi_R = 1`` (last site).

Two canonical forms are supported:

* **Right-canonical** (``right_canonicalize``):
  sites 1..L-1 satisfy ``sum_s A[s] A[s]^H = I`` when flattened as
  ``(chi_L, d*chi_R)`` matrices.  Site 0 absorbs the norm and is the
  "orthogonality center".  This is the form used by the R-Adam driver.

* **Left-canonical** (``left_canonicalize``):
  sites 0..L-2 satisfy ``sum_s A[s]^H A[s] = I`` when flattened as
  ``(chi_L*d, chi_R)`` matrices.  Site L-1 is the center.
"""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray


def _tapered_bond_dims(L: int, chi: int, d: int):
    """Return the bond dims ``chi_0, chi_1, ..., chi_L`` with edge tapering.

    The interior bonds sit at ``chi``, but near the edges we clip to
    ``d^min(i, L-i)`` so that the right-canonical QR sweep does not
    collapse the state to a sub-TT with zero tangent space at the
    tapered sites.  ``chi_0 = chi_L = 1`` always.
    """
    dims = [1] * (L + 1)
    for i in range(1, L):
        dims[i] = min(chi, d ** i, d ** (L - i))
    return dims


def random_mps(L: int, chi: int, d: int = 2, *, dtype=np.complex128, seed: int | None = None):
    """Build a random MPS with tapered bond structure and right-canonicalize.

    Parameters
    ----------
    L : int
        Number of physical sites.
    chi : int
        Maximum interior bond dimension.  Near the edges the bond dim
        is clipped to ``d^min(i, L-i)`` so that the right-canonical
        constraint ``A A^H = I`` is feasible without rank deficiency
        (otherwise the tangent space at those sites collapses to zero).
    d : int, optional
        Local Hilbert space dimension.
    dtype : numpy dtype, optional
        ``complex128`` by default.
    seed : int or None
        RNG seed.

    Returns
    -------
    list of ndarray
        Right-canonical MPS with site 0 holding the norm.
    """
    rng = np.random.default_rng(seed)
    dims = _tapered_bond_dims(L, chi, d)
    cores = []
    for i in range(L):
        chi_L, chi_R = dims[i], dims[i + 1]
        if np.issubdtype(dtype, np.complexfloating):
            real = rng.standard_normal((chi_L, d, chi_R))
            imag = rng.standard_normal((chi_L, d, chi_R))
            core = (real + 1j * imag).astype(dtype)
        else:
            core = rng.standard_normal((chi_L, d, chi_R)).astype(dtype)
        cores.append(core)
    cores = right_canonicalize(cores)
    # Normalize the state so that <X|X> = 1 (concentrated in X[0]).
    norm = np.sqrt(np.vdot(cores[0], cores[0]).real)
    if norm > 0:
        cores[0] = cores[0] / norm
    return cores


def zeros_like_mps(X):
    """Return a list of zero arrays with the same shapes/dtypes as ``X``."""
    return [np.zeros_like(A) for A in X]


def copy_mps(X):
    """Return a deep numpy-level copy of the MPS core list."""
    return [A.copy() for A in X]


def scale_mps(X, alpha):
    """Return a new core list with the left-most core multiplied by ``alpha``.

    This is equivalent to scalar-multiplying the full MPS.  Used by R-Adam
    to form ``-step_size * M_hat`` before combining with ``X``.
    """
    out = [A.copy() for A in X]
    out[0] = out[0] * alpha
    return out


def right_canonicalize(X):
    """Bring ``X`` to right-canonical form with orthogonality center at site 0.

    Uses a sweep of QR decompositions from right to left.  After the sweep,
    sites 1..L-1 satisfy ``sum_s A[s] A[s]^H = I`` (row-orthonormal when
    flattened as ``(chi_L, d*chi_R)`` matrices) and the norm of the full
    MPS equals ``||X[0]||_F``.

    Operates on a copy; the input is not modified.
    """
    L = len(X)
    out = [A.copy() for A in X]
    for i in range(L - 1, 0, -1):
        A = out[i]
        chi_L, d, chi_R = A.shape
        # Flatten so that the LEFT bond becomes the column space.
        # M has shape (d*chi_R, chi_L) so that Q is row-orthonormal
        # in the original (chi_L, d*chi_R) flattening.
        M = A.reshape(chi_L, d * chi_R).conj().T  # (d*chi_R, chi_L)
        Q, R = np.linalg.qr(M)  # Q: (d*chi_R, k), R: (k, chi_L), k = min(d*chi_R, chi_L)
        new_chi_L = Q.shape[1]
        # Put the isometric piece back at site i.
        # Original flat was (chi_L, d*chi_R); after conj-transpose + QR we
        # have A_flat = (Q R)^H = R^H Q^H, so the right-canonical core is
        # R^H acting on the left bond and Q^H as the isometric part.
        # Concretely: A_new = Q^H reshaped; contract R^H into neighbour.
        A_new = Q.conj().T.reshape(new_chi_L, d, chi_R)
        out[i] = A_new
        # Absorb R^H into the right bond of the left neighbour.
        out[i - 1] = np.einsum("lps,sk->lpk", out[i - 1], R.conj().T)
    return out


def left_canonicalize(X):
    """Bring ``X`` to left-canonical form with orthogonality center at site L-1."""
    L = len(X)
    out = [A.copy() for A in X]
    for i in range(L - 1):
        A = out[i]
        chi_L, d, chi_R = A.shape
        M = A.reshape(chi_L * d, chi_R)
        Q, R = np.linalg.qr(M)  # Q: (chi_L*d, k), R: (k, chi_R)
        new_chi_R = Q.shape[1]
        out[i] = Q.reshape(chi_L, d, new_chi_R)
        out[i + 1] = np.einsum("kr,rps->kps", R, out[i + 1])
    return out


def mps_inner(X, Y):
    """Return ``<X|Y>`` for two MPSs with matching shapes.

    Both ``X`` and ``Y`` are lists of cores ``(chi_L, d, chi_R)``.  The
    physical indices are contracted; bra ``<X|`` is complex-conjugated.
    """
    L = len(X)
    assert len(Y) == L
    # env has shape (chi_X_R, chi_Y_R) after absorbing sites 0..i-1.
    env = np.ones((1, 1), dtype=np.result_type(X[0].dtype, Y[0].dtype))
    for i in range(L):
        # contract: env[a,b] * X[i][a,p,c]* * Y[i][b,p,d] -> env_new[c,d]
        env = np.einsum("ab,apc,bpd->cd", env, X[i].conj(), Y[i])
    return env.reshape(-1)[0]


def mps_norm_squared(X):
    """Return ``<X|X>`` as a real scalar."""
    val = mps_inner(X, X)
    return float(np.real(val))


def mps_frob_norm_squared_cores(cores):
    """Return ``sum_i ||cores[i]||_F^2`` as a real scalar.

    This is the *Euclidean / flat* norm of the tangent-vector-like core
    list, used by R-Adam to track the global variance term ``v``.
    """
    return float(sum(np.vdot(C, C).real for C in cores))
