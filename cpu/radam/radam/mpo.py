"""MPO builders for Heisenberg XXX, TFIM, and Josephson junction array.

All MPO cores have shape ``(mpo_L, mpo_R, d_up, d_down)``.  ``d_up`` is
the "ket" physical index (contracted with the MPS core) and ``d_down`` is
the "bra" index (contracted with the conjugated MPS core).

Three models are provided:

* :func:`build_heisenberg_mpo` -- XXX chain
  ``H = J sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})
        + bz sum_i Sz_i``

* :func:`build_tfim_mpo` -- transverse-field Ising
  ``H = -J sum_i Sz_i Sz_{i+1} - hx sum_i Sx_i``

* :func:`build_josephson_mpo` -- Josephson junction array
  ``H = -E_J/2 sum_i (exp(i phi_ext) exp(i phi_i) exp(-i phi_{i+1}) + h.c.)
        + E_C sum_i n_i^2  -  mu sum_i n_i``
  with charge truncation ``|n| <= n_max`` (local dim ``d = 2*n_max + 1``).
  External flux ``phi_ext != 0`` breaks time-reversal symmetry and
  forces complex128 arithmetic.
"""

from __future__ import annotations

import numpy as np


def _spin_half_ops(dtype=np.complex128):
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=dtype)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=dtype)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=dtype)
    I2 = np.eye(2, dtype=dtype)
    return I2, Sx, Sy, Sz


def build_heisenberg_mpo(L: int, j: float = 1.0, bz: float = 0.0, *, dtype=np.complex128):
    """Build the Heisenberg XXX MPO with bond dim 5 in bulk.

    The bulk MPO matrix is

    ::

        [[ I,  0,  0,  0,  0 ],
         [Sx,  0,  0,  0,  0 ],
         [Sy,  0,  0,  0,  0 ],
         [Sz,  0,  0,  0,  0 ],
         [bz*Sz, j*Sx, j*Sy, j*Sz, I ]]

    which represents ``H = j*(SS) + bz*Sz`` in TT form.

    Returns a list of ``L`` cores of shape ``(mpo_L, mpo_R, 2, 2)``.
    Edge cores use trivial bond of size 1 on the appropriate side.
    """
    I2, Sx, Sy, Sz = _spin_half_ops(dtype=dtype)
    W = np.zeros((5, 5, 2, 2), dtype=dtype)
    W[0, 0] = I2
    W[1, 0] = Sx
    W[2, 0] = Sy
    W[3, 0] = Sz
    W[4, 0] = bz * Sz
    W[4, 1] = j * Sx
    W[4, 2] = j * Sy
    W[4, 3] = j * Sz
    W[4, 4] = I2

    cores = []
    for i in range(L):
        if i == 0:
            # Keep only the last row (acts as the accumulator start).
            core = W[4:5, :, :, :].copy()  # shape (1, 5, 2, 2)
        elif i == L - 1:
            # Keep only the first column (acts as the accumulator end).
            core = W[:, 0:1, :, :].copy()  # shape (5, 1, 2, 2)
        else:
            core = W.copy()
        cores.append(core)
    return cores


def build_tfim_mpo(L: int, j: float = 1.0, hx: float = 1.0, *, dtype=np.complex128):
    """Build the transverse-field Ising MPO.

    ``H = -J sum_i Sz_i Sz_{i+1} - hx sum_i Sx_i`` with spin-1/2 operators.

    Bulk bond dim is 3:

    ::

        [[ I,     0,   0 ],
         [Sz,     0,   0 ],
         [-hx*Sx, -J*Sz, I ]]
    """
    I2, Sx, _Sy, Sz = _spin_half_ops(dtype=dtype)
    W = np.zeros((3, 3, 2, 2), dtype=dtype)
    W[0, 0] = I2
    W[1, 0] = Sz
    W[2, 0] = -hx * Sx
    W[2, 1] = -j * Sz
    W[2, 2] = I2

    cores = []
    for i in range(L):
        if i == 0:
            core = W[2:3, :, :, :].copy()
        elif i == L - 1:
            core = W[:, 0:1, :, :].copy()
        else:
            core = W.copy()
        cores.append(core)
    return cores


def build_josephson_mpo(
    L: int,
    *,
    E_J: float = 1.0,
    E_C: float = 0.5,
    mu: float = 0.0,
    n_max: int = 2,
    phi_ext: float = np.pi / 4,
    dtype=np.complex128,
):
    """Build a Josephson junction array MPO.

    Uses charge truncation ``|n| <= n_max`` giving local dimension
    ``d = 2*n_max + 1`` with basis ordered ``|-n_max>, ..., |+n_max>``.
    The operators are:

    * ``n`` = diag(-n_max, ..., +n_max)
    * ``exp(i phi)`` is the charge-raising ladder:
      ``(exp_iphi)[j, i] = 1`` iff ``j = i + 1``.
    * ``exp(-i phi) = (exp_iphi)^dagger``

    Bulk MPO bond dimension is 4 using the standard nearest-neighbour
    automaton:

    ::

        [[ I,            0,                       0,                            0 ],
         [exp_miphi,     0,                       0,                            0 ],
         [exp_iphi,      0,                       0,                            0 ],
         [onsite,        -E_J/2 * e^{i phi_ext} * exp_iphi,
                         -E_J/2 * e^{-i phi_ext} * exp_miphi,       I ]]

    where ``onsite = E_C n^2 - mu n``.

    This matches :func:`benchmarks.lib.models.build_josephson_mpo` (which
    uses quimb's ``SpinHam1D``) up to a global sign convention.
    """
    d = 2 * n_max + 1
    charges = np.arange(-n_max, n_max + 1, dtype=np.float64)

    I_d = np.eye(d, dtype=dtype)
    n_op = np.diag(charges).astype(dtype)
    n2 = n_op @ n_op

    exp_iphi = np.zeros((d, d), dtype=dtype)
    for i in range(d - 1):
        exp_iphi[i + 1, i] = 1.0
    exp_miphi = exp_iphi.conj().T

    flux_phase = np.exp(1j * phi_ext)
    on_site = E_C * n2 - mu * n_op

    W = np.zeros((4, 4, d, d), dtype=dtype)
    # Row 0: completed channel
    W[0, 0] = I_d
    # Row 1: just completed hopping (need exp_miphi to close)
    W[1, 0] = exp_miphi
    # Row 2: just completed hopping (need exp_iphi to close)
    W[2, 0] = exp_iphi
    # Row 3: start channel
    W[3, 0] = on_site
    W[3, 1] = (-E_J / 2.0) * flux_phase * exp_iphi
    W[3, 2] = (-E_J / 2.0) * np.conj(flux_phase) * exp_miphi
    W[3, 3] = I_d

    cores = []
    for i in range(L):
        if i == 0:
            core = W[3:4, :, :, :].copy()       # (1, 4, d, d)
        elif i == L - 1:
            core = W[:, 0:1, :, :].copy()       # (4, 1, d, d)
        else:
            core = W.copy()                     # (4, 4, d, d)
        cores.append(core)
    return cores
