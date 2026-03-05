"""Environment tensor management for DMRG.

Stores and manages the left and right environment tensors that encode
the contraction of MPS, MPO, and conjugate MPS for sites outside the
current optimization window.
"""

import numpy as np
from pdmrg.environments.update import (
    update_left_env, update_right_env,
    init_left_env, init_right_env,
)


class EnvironmentManager:
    """Manages L and R environment tensors for a block of sites.

    Attributes
    ----------
    L_envs : dict[int, ndarray]
        L_envs[i] = left environment at bond i (encodes sites 0..i-1).
    R_envs : dict[int, ndarray]
        R_envs[i] = right environment at bond i (encodes sites i+1..L-1).
    """

    def __init__(self, L_envs=None, R_envs=None):
        self.L_envs = L_envs if L_envs is not None else {}
        self.R_envs = R_envs if R_envs is not None else {}

    def build_initial_envs(self, mps_arrays, mpo_arrays, dtype=np.float64):
        """Build all L and R environments from scratch for a set of sites.

        Parameters
        ----------
        mps_arrays : list of ndarray
            MPS tensors in our convention (left, phys, right), indexed 0..L-1.
        mpo_arrays : list of ndarray
            MPO tensors in our convention (mpo_L, mpo_R, d, d), indexed 0..L-1.
        dtype : dtype
        """
        L = len(mps_arrays)

        # Build left environments from left to right
        # L_envs[0] is the trivial left boundary
        chi_L_0 = mps_arrays[0].shape[0]
        D_0 = mpo_arrays[0].shape[0]
        self.L_envs[0] = init_left_env(chi_L_0, D_0, dtype)

        for i in range(L - 1):
            self.L_envs[i + 1] = update_left_env(
                self.L_envs[i], mps_arrays[i], mpo_arrays[i]
            )

        # Build right environments from right to left
        # R_envs[L-1] is the trivial right boundary
        chi_R_last = mps_arrays[-1].shape[2]
        D_last = mpo_arrays[-1].shape[1]
        self.R_envs[L - 1] = init_right_env(chi_R_last, D_last, dtype)

        for i in range(L - 2, -1, -1):
            self.R_envs[i] = update_right_env(
                self.R_envs[i + 1], mps_arrays[i + 1], mpo_arrays[i + 1]
            )

    def update_left(self, site, A, W):
        """Update left environment after sweeping right past site.

        After optimizing and splitting the two-site wavefunction at
        sites (site, site+1), the left environment for bond site+1 is
        updated using the new left-canonical tensor A at `site`.
        """
        self.L_envs[site + 1] = update_left_env(
            self.L_envs[site], A, W
        )

    def update_right(self, site, B, W):
        """Update right environment after sweeping left past site.

        After optimizing at sites (site-1, site), the right environment
        for bond site-1 is updated using the right-canonical tensor B at `site`.
        """
        self.R_envs[site - 1] = update_right_env(
            self.R_envs[site], B, W
        )

    def get_envs_for_bond(self, site_left, site_right):
        """Get L and R environments for a two-site optimization at (site_left, site_right).

        Returns
        -------
        L_env : ndarray for bond to the left of site_left
        R_env : ndarray for bond to the right of site_right
        """
        return self.L_envs[site_left], self.R_envs[site_right]
