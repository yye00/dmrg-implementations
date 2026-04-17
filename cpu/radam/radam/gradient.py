"""Euclidean gradient of the Rayleigh quotient ``E = <X|H|X>/<X|X>``.

For each MPS core ``A_i`` of shape ``(chi_L, d, chi_R)``, we return a
tensor of the same shape representing

``grad_i = dE / d A_i*
        = (<X|X>)^{-1} * ( H_eff_i . A_i - E * N_eff_i . A_i )``

where ``H_eff_i`` is the effective Hamiltonian at site ``i``
constructed from the left environment, the local MPO core, and the
right environment; and ``N_eff_i`` is the analogous norm operator.

When ``X`` is right-canonical with the orthogonality center at site 0,
``N_eff_i`` equals the identity for every site ``i``, so the expression
simplifies to ``(<X|X>)^{-1} * (H_eff_i A_i - E N_eff_i A_i)`` with
``<X|X>`` equal to ``|| X[0] ||_F^2`` and ``E = <X|H|X> / <X|X>``.
"""

from __future__ import annotations

import numpy as np

from .environments import (
    build_left_envs_H,
    build_right_envs_H,
    build_left_envs_norm,
    build_right_envs_norm,
)


def energy_only(X, H) -> float:
    """Return the Rayleigh-quotient energy ``<X|H|X> / <X|X>`` as a real float.

    Used by the line search inside R-LBFGS, where re-computing the
    full per-site gradient would be wasteful.
    """
    L = len(X)
    xhx = complex(build_left_envs_H(X, H)[L][0, 0, 0])
    xx = float(np.real(build_left_envs_norm(X)[L][0, 0]))
    if xx <= 0.0:
        raise ValueError(f"<X|X>={xx} is not positive; MPS is degenerate")
    return float(np.real(xhx / xx))


def _apply_heff(L_H, W, R_H, A):
    """Apply the site effective Hamiltonian to ``A``.

    ``L_H``: ``(a_bra, l, a_ket)``, ``W``: ``(l, n, p, q)``,
    ``R_H``: ``(b_bra, n, b_ket)``, ``A``: ``(a_ket, q, b_ket)``.
    Returns ``Hout`` of shape ``(a_bra, p, b_bra)`` -- same shape as ``A``
    if interpreted as bra indices.
    """
    # T1[a_bra, l, q, b_ket] = L_H[a_bra, l, a_ket] * A[a_ket, q, b_ket]
    T1 = np.einsum("alk,kqb->alqb", L_H, A)
    # T2[a_bra, n, p, b_ket] = T1[a_bra, l, q, b_ket] * W[l, n, p, q]
    T2 = np.einsum("alqb,lnpq->anpb", T1, W)
    # out[a_bra, p, c_bra] = T2[a_bra, n, p, b_ket] * R_H[c_bra, n, b_ket]
    out = np.einsum("anpb,cnb->apc", T2, R_H)
    return out


def _apply_norm(L_N, R_N, A):
    """Apply the site effective norm operator to ``A``."""
    # out[a_bra, p, c_bra] = L_N[a_bra, a_ket] * A[a_ket, p, b_ket] * R_N[c_bra, b_ket]
    return np.einsum("ak,kpb,cb->apc", L_N, A, R_N)


def euclidean_gradient(X, H, *, return_energy: bool = True):
    """Return ``(gradient_cores, energy)`` (or just ``gradient_cores``).

    Parameters
    ----------
    X : list of ndarray
        MPS cores.
    H : list of ndarray
        MPO cores.
    return_energy : bool
        If True, also return the current Rayleigh-quotient energy as a
        real float.

    Returns
    -------
    grads : list of ndarray
        One core per site, same shapes as ``X``.  ``grads[i]`` is the
        Euclidean gradient of ``E = <X|H|X> / <X|X>`` with respect to
        ``A_i*``.
    energy : float
        Current energy (only returned if ``return_energy`` is True).
    """
    L = len(X)
    L_H = build_left_envs_H(X, H)
    R_H = build_right_envs_H(X, H)
    L_N = build_left_envs_norm(X)
    R_N = build_right_envs_norm(X)

    # <X|H|X> and <X|X>.
    xhx = complex(L_H[L][0, 0, 0])
    xx = complex(L_N[L][0, 0])
    xx_real = float(np.real(xx))
    if xx_real <= 0.0:
        raise ValueError(f"<X|X>={xx_real} is not positive; MPS is degenerate")
    E = float(np.real(xhx / xx))

    inv_xx = 1.0 / xx_real
    grads = []
    for i in range(L):
        h_times_A = _apply_heff(L_H[i], H[i], R_H[i + 1], X[i])
        n_times_A = _apply_norm(L_N[i], R_N[i + 1], X[i])
        g = inv_xx * (h_times_A - E * n_times_A)
        grads.append(g)

    if return_energy:
        return grads, E
    return grads

