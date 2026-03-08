"""GEMM-optimized linear algebra utilities for PDMRG2.

Replaces memory-bandwidth-bound operations (Lanczos, QR) with
BLAS-3-heavy alternatives that map efficiently onto CPU cache hierarchies
and are structurally ready for GPU porting.

Exposed routines
----------------
block_davidson        Block-Davidson (LOBPCG-style) eigensolver
newton_schulz_polar   Iterative polar decomposition for gauge shifts

Top-level configuration
-----------------------
NS_TOL          : float  Newton-Schulz convergence tolerance       (default 1e-10)
"""

import numpy as np
from scipy.linalg import eigh

# ── Top-level configuration variables ─────────────────────────────────────────
NS_TOL = 1e-10       # Newton-Schulz convergence tolerance


# ── Block-Davidson (LOBPCG-style) eigensolver ──────────────────────────────────

def block_davidson(matvec, dim, x_init, b=4, max_iter=30, tol=1e-10):
    """Block-Davidson eigensolver for the lowest eigenvalue/vector.

    Uses subspace expansion with restart: starts from b trial vectors,
    expands by adding residual corrections, and restarts when the subspace
    reaches max_subspace = b * max_expand.  All heavy operations use the
    ``@`` operator (BLAS-3 gemm).

    On a GPU port, the batched matvecs and gemm projections map directly
    to hipBLAS / cuBLAS calls.

    Parameters
    ----------
    matvec : callable
        Maps 1-D ndarray of length *dim* → 1-D ndarray (H_eff application).
    dim : int
        Hilbert-space dimension (vector length).
    x_init : ndarray, shape (dim,)
        Initial guess; will be normalized internally.
    b : int
        Block size (number of new vectors per expansion).  Default 4.
    max_iter : int
        Maximum expansion iterations.
    tol : float
        Stop when residual norm < *tol* **and** |ΔE| < *tol*.

    Returns
    -------
    energy : float
        Lowest eigenvalue.
    v : ndarray, shape (dim,)
        Corresponding eigenvector (unit norm).
    """
    rng = np.random.default_rng(seed=42)
    dtype = x_init.dtype
    max_subspace = min(b * 8, dim)  # restart threshold

    # Build starting block: first column = normalized x_init, rest = random.
    V = np.empty((dim, b), dtype=dtype)
    nrm = np.linalg.norm(x_init)
    V[:, 0] = x_init / (nrm if nrm > 0 else 1.0)
    for i in range(1, b):
        v = rng.standard_normal(dim).astype(dtype)
        for j in range(i):
            v -= np.dot(V[:, j].conj(), v) * V[:, j]
        nrm_v = np.linalg.norm(v)
        V[:, i] = v / (nrm_v if nrm_v > 0 else 1.0)

    # AV stores H @ V columns (reused across iterations)
    AV = np.column_stack([matvec(V[:, j]) for j in range(b)])

    energy_prev = np.inf
    best_energy = np.inf
    best_vec = V[:, 0].copy()

    for iteration in range(max_iter):
        k = V.shape[1]  # current subspace size

        # Rayleigh-Ritz: project H into subspace — (k, k) BLAS-3 gemm
        H_proj = V.conj().T @ AV                          # (k, k)
        H_proj = 0.5 * (H_proj + H_proj.conj().T)         # symmetrize

        # Dense k×k eigenproblem (negligible cost for k << dim)
        eigvals, eigvecs = eigh(H_proj)                    # ascending order

        # Ritz vectors: X = V @ eigvecs — BLAS-3 gemm
        X = V @ eigvecs                                    # (dim, k)
        AX = AV @ eigvecs                                  # (dim, k)

        energy = float(eigvals[0].real)
        x0 = X[:, 0]

        # Track best result across restarts
        if energy < best_energy:
            best_energy = energy
            best_vec = x0.copy()

        # Residual of lowest eigenpair: r = H x0 - E x0
        r = AX[:, 0] - energy * x0
        res_norm = np.linalg.norm(r)

        if res_norm < tol and abs(energy - energy_prev) < tol:
            return energy, x0
        energy_prev = energy

        # Expand subspace with residual correction vectors
        # For each of the lowest b Ritz pairs, compute residual and add to V
        new_vecs = []
        for i in range(min(b, k)):
            ri = AX[:, i] - eigvals[i] * X[:, i]
            ri_norm = np.linalg.norm(ri)
            if ri_norm > tol * 0.01:
                new_vecs.append(ri / ri_norm)

        if not new_vecs:
            return energy, x0

        # Orthogonalize new vectors against V and each other
        W = np.column_stack(new_vecs)
        # Project out existing subspace: W = W - V (V† W) — BLAS-3
        W = W - V @ (V.conj().T @ W)
        # QR orthogonalize the new block
        Q, R_qr = np.linalg.qr(W)
        # Keep only linearly independent columns
        diag_r = np.abs(np.diag(R_qr))
        good = diag_r > 1e-14 * max(diag_r.max(), 1.0)
        if not np.any(good):
            return energy, x0
        Q = Q[:, good]

        # Check if subspace would exceed max size — restart if so
        if k + Q.shape[1] > max_subspace:
            # Restart: keep best b Ritz vectors
            keep = min(b, k)
            V = X[:, :keep].copy()
            AV = AX[:, :keep].copy()
            # Re-orthogonalize
            V, _ = np.linalg.qr(V)
            AV = np.column_stack([matvec(V[:, j]) for j in range(V.shape[1])])
            continue

        # Expand: append new vectors and their H-images
        AQ = np.column_stack([matvec(Q[:, j]) for j in range(Q.shape[1])])
        V = np.hstack([V, Q])
        AV = np.hstack([AV, AQ])

    return best_energy, best_vec


# ── Newton-Schulz polar decomposition ─────────────────────────────────────────

def newton_schulz_polar(A, tol=None):
    """Thin polar decomposition A = U P via Newton-Schulz iteration.

    All inner loops use the ``@`` operator (BLAS-3 gemm) — no QR or SVD
    for the tall / square case.

    Algorithm (tall / square, m >= n)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Scale:  U_0 = A / ‖A‖_F        (ensures all singular values ≤ 1)
    Iterate: U_{k+1} = ½ U_k (3I - U_k† U_k)
    Stop:   ‖U_{k+1} - U_k‖_F < tol
    Extract: P = U† A

    Wide case (m < n)
    ~~~~~~~~~~~~~~~~~
    Wide matrices arise at chain boundaries where chi_L = 1.  They are
    small, so we fall back to numpy QR for correctness without measurable
    performance loss.

    Parameters
    ----------
    A : ndarray, shape (m, n)
    tol : float, optional
        Convergence tolerance; defaults to ``NS_TOL``.

    Returns
    -------
    U : ndarray, shape (m, k)  where k = min(m, n)
        Left-isometric factor: U† U = I_k.
    P : ndarray, shape (k, n)
        Remainder factor: A ≈ U @ P.
    """
    if tol is None:
        tol = NS_TOL

    m, n = A.shape
    fro = np.linalg.norm(A, 'fro')

    if fro < 1e-300:
        k = min(m, n)
        return (np.zeros((m, k), dtype=A.dtype),
                np.zeros((k, n), dtype=A.dtype))

    if m >= n:
        # ── Standard tall / square Newton-Schulz ──────────────────────────
        # Converges U† U → I_n  (left-isometric factor).
        U = A / fro                       # (m, n); all singular values in (0, 1]
        I_n = np.eye(n, dtype=A.dtype)
        for _ in range(100):              # hard cap; tol usually triggers first
            UtU = U.conj().T @ U          # (n, n) — BLAS-3 gemm
            U_new = 0.5 * (U @ (3.0 * I_n - UtU))  # (m, n) — BLAS-3 gemm
            diff = np.linalg.norm(U_new - U, 'fro')
            U = U_new
            if diff < tol:
                break
        P = U.conj().T @ A               # (n, n) — BLAS-3 gemm
        return U, P

    else:
        # ── Wide fallback: numpy QR ────────────────────────────────────────
        # Wide matrices (m < n) only occur at chain boundaries where chi_L=1;
        # those tensors are tiny (d rows), so QR cost is negligible.
        Q, R = np.linalg.qr(A)           # Q: (m, m) unitary, R: (m, n)
        return Q, R
