"""Validation tests: PDMRG vs serial DMRG on Heisenberg chain.

These tests require MPI and should be run with:
  mpirun -np 2 python -m pytest pdmrg/tests/test_heisenberg.py -v

For single-process testing, the serial warmup and core algorithms
are validated in test_numerics.py.
"""

import numpy as np
import pytest
import quimb.tensor as qtn

from pdmrg.mps.canonical import get_tensor_data, get_mpo_tensor_data
from pdmrg.environments.update import (
    update_left_env, update_right_env,
    init_left_env, init_right_env,
)
from pdmrg.numerics.eigensolver import optimize_two_site
from pdmrg.numerics.accurate_svd import truncated_svd
from pdmrg.dmrg import serial_warmup


class TestSerialWarmup:
    def test_warmup_improves_energy(self):
        """Serial warmup should produce a reasonable initial state."""
        L = 10
        H = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)

        A, W, E_warmup = serial_warmup(H, L, bond_dim_warmup=20, n_warmup_sweeps=3)

        # Reference
        dmrg = qtn.DMRG2(H, bond_dims=[20, 50], cutoffs=1e-12)
        dmrg.solve(tol=1e-12, verbosity=0)

        # Warmup energy should be within 10% of reference
        assert E_warmup < 0  # Should be negative
        assert abs(E_warmup - dmrg.energy) / abs(dmrg.energy) < 0.1


class TestSerialDMRGHeisenberg:
    def test_l10_convergence(self):
        """Serial DMRG on L=10 Heisenberg converges to reference."""
        L = 10
        max_bond = 30
        H = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)

        dmrg_ref = qtn.DMRG2(H, bond_dims=[30, 50], cutoffs=1e-12)
        dmrg_ref.solve(tol=1e-12, verbosity=0)

        A, W, E = serial_warmup(H, L, bond_dim_warmup=max_bond,
                                 n_warmup_sweeps=8)

        assert abs(E - dmrg_ref.energy) < 1e-4

    def test_energy_per_site(self):
        """Energy per site approaches known infinite-chain limit."""
        L = 20
        H = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)

        A, W, E = serial_warmup(H, L, bond_dim_warmup=40, n_warmup_sweeps=6)

        E_per_site = E / L
        # Known: E/L -> -0.4431... for S=1/2 Heisenberg (infinite L)
        assert E_per_site < -0.40  # Should be close to -0.44
