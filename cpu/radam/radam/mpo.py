"""MPO builders for Heisenberg XXX and Transverse-Field Ising.

All MPO cores have shape ``(mpo_L, mpo_R, d_up, d_down)``.  ``d_up`` is
the "ket" physical index (contracted with the MPS core) and ``d_down`` is
the "bra" index (contracted with the conjugated MPS core).

Two models are provided:

* :func:`build_heisenberg_mpo` -- XXX chain
  ``H = J sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})
        + bz sum_i Sz_i``

* :func:`build_tfim_mpo` -- transverse-field Ising
  ``H = -J sum_i Sz_i Sz_{i+1} - hx sum_i Sx_i``
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
