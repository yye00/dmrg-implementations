"""Regression tests for complex dtype detection in eigensolver.

The eigensolver generates a random initial guess ``v0`` when none is provided.
For complex effective Hamiltonians, ``v0`` must be complex.
"""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator


def _load_eigensolver_module():
    # Load module by path to avoid importing a2dmrg.numerics.__init__ which pulls
    # in quimb/numba.
    base = Path(__file__).resolve().parents[1]
    path = base / "numerics" / "eigensolver.py"
    spec = importlib.util.spec_from_file_location("a2dmrg_numerics_eigensolver", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_random_v0_is_complex_for_complex_operator():
    es = _load_eigensolver_module()

    H_eff = LinearOperator(
        shape=(5, 5),
        matvec=lambda x: x.astype(np.complex128),
        dtype=np.dtype("complex128"),
    )

    captured = {}

    def fake_eigsh(op, **kwargs):
        captured["v0"] = kwargs.get("v0")
        # Return a trivial eigenpair.
        n = op.shape[0]
        vec = np.ones((n, 1), dtype=op.dtype)
        return np.array([0.0]), vec

    # Patch the imported symbol inside the module.
    es.eigsh = fake_eigsh

    es.solve_effective_hamiltonian(H_eff, v0=None)

    assert np.iscomplexobj(captured["v0"])
