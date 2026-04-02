"""
Eigensolver for effective Hamiltonians in DMRG.

This module provides functions to solve the effective eigenvalue problem
using scipy.sparse.linalg.eigsh.

The effective Hamiltonian is provided as a LinearOperator, and we find
the ground state (lowest eigenvalue).

Author: A2DMRG Implementation
"""

import numpy as np
from scipy.sparse.linalg import eigsh, eigs, ArpackNoConvergence


def _is_complex_dtype(dtype) -> bool:
    """Return True iff dtype is a complex dtype.

    Note: ``np.iscomplexobj(np.dtype('complex128'))`` is False, so we explicitly
    check the dtype kind / subdtype.
    """
    if dtype is None:
        return False
    dt = np.dtype(dtype)
    return dt.kind == 'c' or np.issubdtype(dt, np.complexfloating)


def _random_initial_guess(size: int, dtype):
    v0 = np.random.randn(size)
    if _is_complex_dtype(dtype):
        v0 = v0 + 1j * np.random.randn(size)
    return v0


def solve_effective_hamiltonian(H_eff, v0=None, tol=1e-10, maxiter=None):
    """
    Solve the effective eigenvalue problem to find the ground state.

    Uses scipy.sparse.linalg.eigsh to find the lowest eigenvalue and
    corresponding eigenvector of the effective Hamiltonian H_eff.

    Parameters
    ----------
    H_eff : LinearOperator
        Effective Hamiltonian as a LinearOperator (e.g. from _build_heff_numpy_1site
        or _build_heff_numpy_2site in local_microstep)
    v0 : numpy.ndarray, optional
        Initial guess for the eigenvector (flattened). If None, a random guess is used.
        Shape should be (H_eff.shape[0],)
    tol : float, optional
        Convergence tolerance for eigsh. Default is 1e-10.
    maxiter : int, optional
        Maximum number of iterations. If None, uses eigsh default.

    Returns
    -------
    energy : float
        Ground state energy (lowest eigenvalue)
    eigvec : numpy.ndarray
        Ground state eigenvector (flattened), normalized to unit norm

    Examples
    --------
    >>> from a2dmrg.numerics.local_microstep import _build_heff_numpy_1site
    >>> from a2dmrg.environments.environment import build_environments_incremental
    >>>
    >>> # Build environments and solve for ground state at site i
    >>> L_envs, R_envs, canon = build_environments_incremental(mps_arrays, mpo_arrays)
    >>> H_eff = _build_heff_numpy_1site(L_envs[i], mpo_arrays[i], R_envs[i+1])
    >>> v0 = canon[i].ravel()
    >>> energy, eigvec = solve_effective_hamiltonian(H_eff, v0=v0)
    >>> print(f"Ground state energy: {energy}")
    """
    # If no initial guess provided, create a random one
    if v0 is None:
        v0 = _random_initial_guess(H_eff.shape[0], H_eff.dtype)

    # Normalize initial guess
    v0 = v0 / np.linalg.norm(v0)

    # Choose solver and 'which' selector based on dtype.
    #
    # ARPACK's symmetric routines (dsaupd/dseupd) support which='SA' (smallest
    # algebraic) for REAL symmetric matrices.  For COMPLEX Hermitian operators
    # scipy routes eigsh through eigs internally, but passes 'SA' which is not a
    # valid selector for znaupd — this causes ARPACK error -1 (no convergence).
    # Fix: for complex dtype use eigs(which='SR') directly.  'SR' = smallest real
    # part, which equals the smallest eigenvalue for a Hermitian operator.
    is_complex = _is_complex_dtype(H_eff.dtype)
    _solver = eigs if is_complex else eigsh
    _which  = 'SR' if is_complex else 'SA'

    # For complex Hermitian, ARPACK needs a relaxed tolerance.  The outer DMRG
    # sweep loop checks energy convergence between sweeps; the per-step eigensolver
    # does not need to solve to tol=1e-10 — 1e-7 is more than sufficient for
    # local updates and avoids ARPACK error -1 (no convergence) on complex H.
    arpack_tol = max(tol, 1e-7) if is_complex else tol

    kwargs = {'k': 1, 'which': _which, 'tol': arpack_tol, 'v0': v0}
    if maxiter is not None:
        kwargs['maxiter'] = maxiter

    try:
        eigenvalues, eigenvectors = _solver(H_eff, **kwargs)
    except ArpackNoConvergence as e:
        # ARPACK didn't converge — reuse its best partial result as a warm restart.
        # Using the best-known approximation converges much faster than a random v0.
        if e.eigenvectors is not None and e.eigenvectors.shape[1] > 0:
            v0_retry = e.eigenvectors[:, 0]
            norm = np.linalg.norm(v0_retry)
            if norm > 1e-14:
                v0_retry = v0_retry / norm
            else:
                v0_retry = _random_initial_guess(H_eff.shape[0], H_eff.dtype)
                v0_retry = v0_retry / np.linalg.norm(v0_retry)
        else:
            v0_retry = _random_initial_guess(H_eff.shape[0], H_eff.dtype)
            v0_retry = v0_retry / np.linalg.norm(v0_retry)
        # Use more Krylov vectors for better convergence on retry.
        ncv = min(50, H_eff.shape[0])
        eigenvalues, eigenvectors = _solver(
            H_eff, k=1, which=_which, tol=arpack_tol, v0=v0_retry, ncv=ncv
        )
    except Exception as e:
        # Non-ARPACK failure — fall back to random initial guess.
        print(f"Warning: eigsh failed with error: {e}")
        print("Trying with random initial guess...")
        v0_random = _random_initial_guess(H_eff.shape[0], H_eff.dtype)
        v0_random = v0_random / np.linalg.norm(v0_random)
        eigenvalues, eigenvectors = _solver(H_eff, k=1, which=_which, tol=arpack_tol, v0=v0_random)

    # Extract ground state.  For complex Hermitian H, eigenvalues from eigs() are
    # returned as complex with a tiny imaginary rounding residual — take the real part.
    energy = float(np.real(eigenvalues[0]))
    eigvec = eigenvectors[:, 0]

    # Ensure normalized
    eigvec = eigvec / np.linalg.norm(eigvec)

    return energy, eigvec


def solve_effective_hamiltonian_2site(H_eff, v0=None, tol=1e-10, maxiter=None,
                                       return_reshaped=False, shape=None):
    """
    Solve the two-site effective eigenvalue problem.

    This is a convenience wrapper around solve_effective_hamiltonian that can
    optionally reshape the result back to the two-site tensor form.

    Parameters
    ----------
    H_eff : LinearOperator
        Two-site effective Hamiltonian
    v0 : numpy.ndarray, optional
        Initial guess (flattened)
    tol : float, optional
        Convergence tolerance
    maxiter : int, optional
        Maximum iterations
    return_reshaped : bool, optional
        If True, reshape eigenvector to (chi_L, d, d, chi_R). Default is False.
    shape : tuple, optional
        Shape to reshape to (chi_L, d, d, chi_R). Required if return_reshaped=True.

    Returns
    -------
    energy : float
        Ground state energy
    eigvec : numpy.ndarray
        Ground state eigenvector, either flattened or reshaped depending on
        return_reshaped parameter

    Examples
    --------
    >>> # After building two-site H_eff
    >>> chi_L, d, chi_R = 4, 2, 4
    >>> energy, theta = solve_effective_hamiltonian_2site(
    ...     H_eff, v0=v0, return_reshaped=True, shape=(chi_L, d, d, chi_R)
    ... )
    >>> print(theta.shape)
    (4, 2, 2, 4)
    """
    # Solve the eigenvalue problem
    energy, eigvec = solve_effective_hamiltonian(H_eff, v0=v0, tol=tol, maxiter=maxiter)

    # Optionally reshape
    if return_reshaped:
        if shape is None:
            raise ValueError("Must provide shape parameter when return_reshaped=True")
        eigvec = eigvec.reshape(shape)

    return energy, eigvec
