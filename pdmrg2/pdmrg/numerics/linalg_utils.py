"""GEMM-optimized linear algebra utilities for PDMRG2.

Replaces memory-bandwidth-bound operations (Lanczos, QR, dense SVD) with
BLAS-3-heavy alternatives that map efficiently onto CPU cache hierarchies
and are structurally ready for GPU porting.

Exposed routines
----------------
block_davidson        Block-Davidson (LOBPCG-style) eigensolver
newton_schulz_polar   Iterative polar decomposition for gauge shifts
rsvd_cholesky         Randomized SVD with Cholesky-QR2 orthogonalization

Top-level configuration
-----------------------
NS_TOL          : float  Newton-Schulz convergence tolerance       (default 1e-10)
RSVD_OVERSAMPLE : int    Oversampling parameter p for rSVD         (default 10)
"""

import numpy as np
from scipy.linalg import eigh, solve_triangular

# ── Top-level configuration variables ─────────────────────────────────────────
NS_TOL = 1e-10       # Newton-Schulz convergence tolerance
RSVD_OVERSAMPLE = 10  # rSVD oversampling parameter p


# ── Block-Davidson (LOBPCG-style) eigensolver ──────────────────────────────────

def block_davidson(matvec, dim, x_init, b=4, max_iter=30, tol=1e-10):
    """Block-Davidson eigensolver for the lowest eigenvalue/vector.

    Replaces single-vector Lanczos with a block of *b* orthogonal trial
    vectors, enabling batched BLAS-3 matmuls (b H-applications per step,
    then a dense b×b Rayleigh-Ritz projection via ``eigh``).

    Parameters
    ----------
    matvec : callable
        Maps 1-D ndarray of length *dim* → 1-D ndarray (H_eff application).
    dim : int
        Hilbert-space dimension (vector length).
    x_init : ndarray, shape (dim,)
        Initial guess; will be normalized internally.
    b : int
        Block size (number of trial vectors).  Default 4.
    max_iter : int
        Maximum Rayleigh-Ritz iterations.
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

    # Build starting block: first column = normalized x_init, rest = random.
    X = np.empty((dim, b), dtype=dtype)
    nrm = np.linalg.norm(x_init)
    X[:, 0] = x_init / (nrm if nrm > 0 else 1.0)
    for i in range(1, b):
        v = rng.standard_normal(dim).astype(dtype)
        # Classical Gram-Schmidt against existing columns
        for j in range(i):
            v -= np.dot(X[:, j], v) * X[:, j]
        nrm_v = np.linalg.norm(v)
        X[:, i] = v / (nrm_v if nrm_v > 0 else 1.0)

    energy_prev = np.inf

    for _ in range(max_iter):
        # Y = H X — b matrix-vector products; collected as (dim, b) dense matrix.
        # In a GPU port these become a single batched gemv / gemm.
        Y = np.column_stack([matvec(X[:, j]) for j in range(b)])  # (dim, b)

        # Rayleigh-Ritz: H_proj = X† Y  — (b, b) BLAS-3 gemm
        H_proj = X.conj().T @ Y                         # (b, b)
        H_proj = 0.5 * (H_proj + H_proj.conj().T)       # symmetrize

        # Dense b×b eigenproblem (negligible cost)
        eigvals, eigvecs = eigh(H_proj)                  # ascending order

        # Update trial block: X_new = X @ eigvecs  — BLAS-3 gemm
        X_new = X @ eigvecs                              # (dim, b)

        energy = float(eigvals[0].real)
        v0 = X_new[:, 0]

        # Residual of the lowest eigenpair: r = (H - E I)|v0⟩
        # Rotate Y to match new basis so we reuse the H applications.
        Y_new = Y @ eigvecs                              # (dim, b)
        r = Y_new[:, 0] - energy * v0
        res_norm = np.linalg.norm(r)

        X = X_new
        if res_norm < tol and abs(energy - energy_prev) < tol:
            break
        energy_prev = energy

    return energy, X[:, 0]


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


# ── Randomized SVD with Cholesky-QR2 ──────────────────────────────────────────

def rsvd_cholesky(M, rank, p=None):
    """Randomized SVD with Cholesky-QR2 orthogonalization.

    Approximates M ≈ U diag(S) Vh at target *rank*, avoiding scipy's
    QR-based orthonormalization in favour of Gram-matrix Cholesky (BLAS-3).

    Algorithm
    ~~~~~~~~~
    1. Sketch:         Y = M Ω           (BLAS-3; Ω is N × (rank+p) Gaussian)
    2. Cholesky-QR2:   Q = chol_qr(Y)   twice for numerical stability
    3. Project:        B = Q† M          (BLAS-3)
    4. Small SVD:      B = Ũ Σ Ṽ†       ((rank+p) × n dense SVD)
    5. Reconstruct:    U = Q Ũ           (BLAS-3)

    .. note::
       **Critical exception** — Do NOT call this for MPI boundary merges.
       Those must use ``accurate_svd`` (exact SVD) because the boundary
       protocol inverts singular values (V = Λ⁻¹), and rSVD's relative
       accuracy on small singular values is insufficient.

    Parameters
    ----------
    M : ndarray, shape (m, n)
        Matrix to decompose.
    rank : int
        Number of singular triplets to retain.
    p : int, optional
        Oversampling parameter.  Defaults to ``RSVD_OVERSAMPLE``.

    Returns
    -------
    U : ndarray, shape (m, rank)
    S : ndarray, shape (rank,)
    Vh : ndarray, shape (rank, n)
    trunc_err : float
        Approximate Σ_{i > rank} σ_i² (truncation error).
    """
    if p is None:
        p = RSVD_OVERSAMPLE

    m, n = M.shape
    k = min(rank + p, min(m, n))      # sketch rank (capped by matrix rank)

    # ── 1. Sketch: Y = M Ω  (BLAS-3 gemm) ───────────────────────────────────
    rng = np.random.default_rng(seed=0)
    Omega = rng.standard_normal((n, k)).astype(M.dtype)
    Y = M @ Omega                      # (m, k)

    # ── 2. Cholesky-QR2  (two passes for numerical stability) ─────────────────
    Q = _cholesky_qr(Y)
    Q = _cholesky_qr(Q)

    # ── 3. Core projection: B = Q† M  (BLAS-3 gemm) ─────────────────────────
    B = Q.conj().T @ M                 # (k, n)

    # ── 4. Small SVD on (k × n) ───────────────────────────────────────────────
    from scipy.linalg import svd as sp_svd
    U_tilde, S, Vht = sp_svd(B, full_matrices=False)   # (k, k), (k,), (k, n)

    # ── 5. Reconstruct: U = Q Ũ  (BLAS-3 gemm) ──────────────────────────────
    U_full = Q @ U_tilde               # (m, k)

    # Truncate to requested rank
    r = min(rank, len(S))
    trunc_err = float(np.sum(S[r:] ** 2))
    return U_full[:, :r], S[:r], Vht[:r, :], trunc_err


def _cholesky_qr(Y):
    """Orthonormalize columns of Y via Gram-matrix Cholesky.

    G = Y† Y,  L = chol(G),  Q = Y (L†)⁻¹

    Falls back to numpy QR if Cholesky is numerically unstable (rank-deficient
    sketch, extremely ill-conditioned input).

    Parameters
    ----------
    Y : ndarray, shape (m, k)

    Returns
    -------
    Q : ndarray, shape (m, k)
        Matrix with orthonormal columns (to working precision).
    """
    G = Y.conj().T @ Y               # (k, k) Gram matrix — BLAS-3 gemm
    G = 0.5 * (G + G.conj().T)       # symmetrize

    # Regularize floating-point negative eigenvalues
    diag_min = float(np.min(np.diag(G).real))
    if diag_min < 0:
        G -= diag_min * np.eye(G.shape[0], dtype=G.dtype)

    try:
        L = np.linalg.cholesky(G)    # lower-triangular: G = L L†
        # Want Q = Y L†⁻¹  so that  Q†Q = L†⁻¹ Y†Y L†⁻† = L†⁻¹ L L† L†⁻† = I
        # Solve  L @ Q† = Y†  (lower triangular)  →  Q† = L⁻¹ Y†
        Qt = solve_triangular(L, Y.conj().T, lower=True)
        Q = Qt.conj().T
    except np.linalg.LinAlgError:
        # Fallback: numpy QR (ill-conditioned sketch)
        Q, _ = np.linalg.qr(Y)

    return Q
