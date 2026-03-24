# A2DMRG Performance Rewrite

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the O(L^2) per-sweep bottleneck in A2DMRG by replacing per-site full environment rebuilds with incremental O(L) environment caching, and removing all quimb overhead from the hot path.

**Architecture:** Extract MPS/MPO to raw numpy arrays once. Build all left/right environments via two O(L) sweeps (right-canonicalize then left-sweep). Each micro-step reuses cached environments instead of rebuilding from scratch. Phase 5 compression uses numpy SVD instead of quimb. No backwards compatibility — old code is replaced outright.

**Tech Stack:** numpy, scipy (eigsh), mpi4py, quimb (initialization/finalization only)

**Target accuracy:** 1e-10 vs quimb DMRG2 reference

**Key files in scope:**
- `a2dmrg/a2dmrg/environments/environment.py` — rewrite environment building
- `a2dmrg/a2dmrg/numerics/local_microstep.py` — rewrite micro-steps
- `a2dmrg/a2dmrg/parallel/local_steps.py` — rewrite parallel dispatch
- `a2dmrg/a2dmrg/dmrg.py` — wire new pipeline into main loop
- `a2dmrg/a2dmrg/mps/canonical.py` — add numpy canonicalization

**Critical design decisions (from code review):**
1. **MPO convention**: Use `(D_L, D_R, d_up, d_down)` everywhere via `_get_mpo_array` (observables.py).
   Do NOT use `_reshape_mpo_tensor` (environment.py) which returns `(D_L, d_out, D_R, d_in)` — different axis order.
   `d_up` connects to ket MPS, `d_down` connects to bra MPS.
2. **No double reshaping**: Fast micro-steps build H_eff inline (`_build_heff_numpy_*site`)
   instead of calling `build_effective_hamiltonian_*site` which internally reshapes raw arrays.
3. **Coefficient scaling**: `form_linear_combination` applies coeffs to site 0 only (not all sites).
   Multiplying into every site would give `c^L` amplitude instead of `c`.
4. **Candidate gauge**: Candidates use `canon_arrays` (from incremental env builder), not
   the original non-canonical `mps_arrays`. The optimized tensor lives in the canonical gauge.

---

### Task 1: Canary accuracy test

Establish a fast, reliable accuracy baseline that we run after every change.

**Files:**
- Create: `a2dmrg/a2dmrg/tests/test_canary.py`

- [ ] **Step 1: Write the canary test**

This test runs a2dmrg at L=8, chi=20 (two-site) with np=2 and checks energy against quimb DMRG2. Must complete in <60s even on the slow current implementation.

```python
"""Fast canary test: a2dmrg accuracy at L=8 chi=20 vs quimb DMRG2."""
import numpy as np
import pytest
import quimb.tensor as qtn

from a2dmrg.hamiltonians.heisenberg import build_heisenberg_mpo
from a2dmrg.mpi_compat import MPI


def _quimb_reference(L, chi):
    """Run quimb DMRG2 and return ground state energy."""
    mpo = build_heisenberg_mpo(L)
    dmrg = qtn.DMRG2(mpo, bond_dims=chi, cutoffs=1e-14)
    dmrg.solve(max_sweeps=30, tol=1e-12, verbosity=0)
    return float(np.real(dmrg.energy))


@pytest.mark.mpi
def test_canary_heisenberg_l8():
    """A2DMRG must match quimb DMRG2 within 1e-10 for L=8 chi=20."""
    from a2dmrg.dmrg import a2dmrg_main

    comm = MPI.COMM_WORLD
    L, chi = 8, 20

    mpo = build_heisenberg_mpo(L)
    energy, mps = a2dmrg_main(
        L=L, mpo=mpo, bond_dim=chi, max_sweeps=20, tol=1e-12,
        warmup_sweeps=2, finalize_sweeps=2,
        comm=comm, verbose=False, one_site=False,
    )

    if comm.Get_rank() == 0:
        E_ref = _quimb_reference(L, chi)
        diff = abs(energy - E_ref)
        print(f"A2DMRG: {energy:.12f}  quimb: {E_ref:.12f}  diff: {diff:.2e}")
        assert diff < 1e-10, f"Energy diff {diff:.2e} exceeds 1e-10"


@pytest.mark.mpi
def test_canary_heisenberg_l12():
    """Larger canary: L=12 chi=20."""
    from a2dmrg.dmrg import a2dmrg_main

    comm = MPI.COMM_WORLD
    L, chi = 12, 20

    mpo = build_heisenberg_mpo(L)
    energy, mps = a2dmrg_main(
        L=L, mpo=mpo, bond_dim=chi, max_sweeps=20, tol=1e-12,
        warmup_sweeps=2, finalize_sweeps=2,
        comm=comm, verbose=False, one_site=False,
    )

    if comm.Get_rank() == 0:
        E_ref = _quimb_reference(L, chi)
        diff = abs(energy - E_ref)
        print(f"A2DMRG: {energy:.12f}  quimb: {E_ref:.12f}  diff: {diff:.2e}")
        assert diff < 1e-10, f"Energy diff {diff:.2e} exceeds 1e-10"
```

- [ ] **Step 2: Run the canary to establish baseline**

```bash
cd /home/captain/clawd/work/dmrg-implementations/a2dmrg
mpirun --oversubscribe -np 2 python3 -m pytest a2dmrg/tests/test_canary.py -v -s 2>&1
```

Record: current energy, diff, and wall time. This is our "before" measurement.

- [ ] **Step 3: Commit**

```bash
git add a2dmrg/a2dmrg/tests/test_canary.py
git commit -m "test(a2dmrg): add canary accuracy test vs quimb DMRG2"
```

---

### Task 2: Numpy array extraction utilities

Create reliable functions to extract MPS and MPO tensors from quimb objects into standard numpy arrays. These already exist in `observables.py` but are private (`_get_mps_array`, `_get_mpo_array`). We promote and harden them.

**Files:**
- Create: `a2dmrg/a2dmrg/numerics/numpy_utils.py`
- Create: `a2dmrg/a2dmrg/tests/test_numpy_utils.py`

- [ ] **Step 1: Write tests for array extraction**

```python
"""Tests for numpy MPS/MPO extraction utilities."""
import numpy as np
import pytest
import quimb.tensor as qtn

from a2dmrg.hamiltonians.heisenberg import build_heisenberg_mpo


def test_extract_mps_shapes():
    """Extracted MPS arrays have shape (chi_L, d, chi_R) with chi_L[0]=1, chi_R[-1]=1."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays

    L, chi = 8, 10
    mps = qtn.MPS_rand_state(L, bond_dim=chi, dtype='float64')
    arrays = extract_mps_arrays(mps)

    assert len(arrays) == L
    assert arrays[0].shape[0] == 1, "First site must have chi_L=1"
    assert arrays[-1].shape[2] == 1, "Last site must have chi_R=1"
    for i in range(L):
        assert arrays[i].ndim == 3, f"Site {i} must be 3D (chi_L, d, chi_R)"
        assert arrays[i].shape[1] == 2, f"Physical dim must be 2 (spin-1/2)"
    # Bond dimensions must match between neighbors
    for i in range(L - 1):
        assert arrays[i].shape[2] == arrays[i + 1].shape[0], \
            f"Bond mismatch at ({i},{i+1}): {arrays[i].shape[2]} != {arrays[i+1].shape[0]}"


def test_extract_mpo_shapes():
    """Extracted MPO arrays have shape (D_L, D_R, d, d) with D_L[0]=1, D_R[-1]=1."""
    from a2dmrg.numerics.numpy_utils import extract_mpo_arrays

    L = 8
    mpo = build_heisenberg_mpo(L)
    arrays = extract_mpo_arrays(mpo)

    assert len(arrays) == L
    assert arrays[0].shape[0] == 1, "First MPO site must have D_L=1"
    assert arrays[-1].shape[1] == 1, "Last MPO site must have D_R=1"
    for i in range(L):
        assert arrays[i].ndim == 4, f"MPO site {i} must be 4D"
        # Convention: (mpo_L, mpo_R, d_up, d_down)
        d = arrays[i].shape[2]
        assert arrays[i].shape[3] == d, "Physical dims must match"
    # Bond dimensions must match between neighbors
    for i in range(L - 1):
        assert arrays[i].shape[1] == arrays[i + 1].shape[0], \
            f"MPO bond mismatch at ({i},{i+1}): {arrays[i].shape[1]} != {arrays[i+1].shape[0]}"


def test_extract_mps_preserves_energy():
    """Energy computed from extracted arrays must match quimb's energy."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays, extract_mpo_arrays
    from a2dmrg.numerics.observables import compute_energy

    L = 8
    mpo = build_heisenberg_mpo(L)
    dmrg = qtn.DMRG2(mpo, bond_dims=20, cutoffs=1e-14)
    dmrg.solve(max_sweeps=10, tol=1e-12, verbosity=0)
    mps = dmrg._k

    E_quimb = float(np.real(dmrg.energy))
    E_obs = compute_energy(mps, mpo)

    assert abs(E_quimb - E_obs) < 1e-10, \
        f"Energy mismatch: quimb={E_quimb}, observables={E_obs}"


def test_extract_complex_mps():
    """Complex MPS extraction works for Josephson-type models."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays

    L = 6
    mps = qtn.MPS_rand_state(L, bond_dim=8, dtype='complex128')
    arrays = extract_mps_arrays(mps)

    for i in range(L):
        assert arrays[i].dtype == np.complex128
```

- [ ] **Step 2: Run tests — should fail (module doesn't exist)**

```bash
cd /home/captain/clawd/work/dmrg-implementations/a2dmrg
python3 -m pytest a2dmrg/tests/test_numpy_utils.py -v 2>&1
```

- [ ] **Step 3: Implement numpy_utils.py**

```python
"""Numpy array extraction from quimb MPS/MPO objects.

Standard conventions:
- MPS: (chi_left, d, chi_right) — always 3D, even at boundaries (chi=1)
- MPO: (D_left, D_right, d_up, d_down) — always 4D, even at boundaries (D=1)
  d_up connects to ket MPS, d_down connects to bra MPS.
  This matches observables.py's _get_mpo_array convention and compute_energy() contractions.
"""

import numpy as np
import quimb.tensor as qtn


def extract_mps_arrays(mps):
    """Extract MPS tensors as list of numpy arrays in (chi_L, d, chi_R) format.

    All tensors are returned as 3D arrays. Boundary sites are padded to 3D:
    - Site 0: (1, d, chi_R)
    - Site L-1: (chi_L, d, 1)
    """
    L = mps.L
    arrays = []
    for i in range(L):
        t = mps[i]
        data = np.asarray(t.data)
        inds = t.inds
        phys_name = mps.site_ind_id.format(i)
        phys_pos = list(inds).index(phys_name)

        if data.ndim == 2:
            if i == 0:
                # (d, chi_R) or (chi_R, d)
                if phys_pos == 0:
                    arrays.append(data[None, :, :])
                else:
                    arrays.append(data.T[None, :, :])
            else:
                # (chi_L, d) or (d, chi_L)
                if phys_pos == 1:
                    arrays.append(data[:, :, None])
                else:
                    arrays.append(data.T[:, :, None])
        elif data.ndim == 3:
            # Reorder to (chi_L, d, chi_R)
            if phys_pos == 1:
                arrays.append(data)
            elif phys_pos == 0:
                arrays.append(data.transpose(1, 0, 2))
            else:
                arrays.append(data.transpose(0, 2, 1))
        elif data.ndim == 1:
            arrays.append(data[None, :, None])
        else:
            raise ValueError(f"Unexpected ndim={data.ndim} at site {i}")

    return arrays


def extract_mpo_arrays(mpo):
    """Extract MPO tensors as list of numpy arrays in (D_L, D_R, d_up, d_down) format.

    Uses the same convention as observables.py's _get_mpo_array:
    - axis 0: mpo_left bond
    - axis 1: mpo_right bond
    - axis 2: upper physical (connects to ket MPS)
    - axis 3: lower physical (connects to bra MPS)

    This convention matches compute_energy() contractions exactly.
    """
    from a2dmrg.numerics.observables import _get_mpo_array
    L = mpo.L
    return [_get_mpo_array(mpo, i) for i in range(L)]
```

- [ ] **Step 4: Run tests — should pass**

```bash
python3 -m pytest a2dmrg/tests/test_numpy_utils.py -v 2>&1
```

- [ ] **Step 5: Commit**

```bash
git add a2dmrg/a2dmrg/numerics/numpy_utils.py a2dmrg/a2dmrg/tests/test_numpy_utils.py
git commit -m "feat(a2dmrg): add numpy MPS/MPO extraction utilities"
```

---

### Task 3: Numpy canonicalization and incremental environment builder

This is the core fix. Build all L[i] and R[i] via two O(L) sweeps instead of O(L) per site.

**Algorithm:**
1. Extract MPS to numpy arrays
2. Right-canonicalize (sweep right-to-left with QR), building R[i] along the way
3. Left-sweep: at each site i, the MPS is in i-orthogonal form. Build L[i+1] and advance.

After this, L[i] and R[i+1] are the correct environments for site i's eigensolve.

**Files:**
- Modify: `a2dmrg/a2dmrg/environments/environment.py`
- Create: `a2dmrg/a2dmrg/tests/test_incremental_envs.py`

- [ ] **Step 1: Write tests for incremental environment builder**

```python
"""Tests for incremental environment building."""
import numpy as np
import pytest
import quimb.tensor as qtn

from a2dmrg.hamiltonians.heisenberg import build_heisenberg_mpo


def _brute_force_energy_at_site(mps_arrays, mpo_arrays, L_env, R_env, site):
    """Compute <psi|H|psi> / <psi|psi> using L[site], W[site], R[site+1] and center tensor.

    NOTE: Does NOT use build_effective_hamiltonian_1site (which would double-reshape
    pre-extracted MPO arrays). Instead, directly contracts L @ A @ W @ R @ A* inline.
    """
    A = mps_arrays[site]
    W = mpo_arrays[site]  # (D_L, D_R, d_up, d_down)
    A_conj = A.conj()

    # Same contraction as _update_left_env_numpy / compute_energy_numpy:
    # L[bra,mpo,ket] @ A[ket,d,ket'] -> X[bra,mpo,d,ket']
    X = np.tensordot(L_env, A, axes=(2, 0))
    # X @ W: contract mpo with D_L, d_ket with d_up
    Y = np.tensordot(X, W, axes=([1, 2], [0, 2]))
    # Y[bra,ket',D_R,d_down] @ R[bra_R,mpo_R,ket_R]
    Z = np.tensordot(Y, R_env, axes=([1, 2], [2, 1]))
    # Z[bra, d_down, bra_R] @ A*[chi_L, d, chi_R]
    # Contract bra with chi_L, d_down with d, bra_R with chi_R
    Hv_dot_v = np.tensordot(Z, A_conj, axes=([0, 1, 2], [0, 1, 2]))
    energy = float(np.real(Hv_dot_v))

    # Normalize: compute <psi|psi> via L @ A @ A* @ R (no MPO)
    T = np.tensordot(L_env[:, 0, :], A, axes=(1, 0))  # use mpo=0 slice for identity
    # Actually easier: just use norm of center tensor (canonical form => norm = 1)
    norm_sq = float(np.real(np.dot(A.ravel().conj(), A.ravel())))
    return energy / norm_sq


def test_incremental_envs_match_full_energy():
    """Environments from incremental builder must give correct site energies."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays, extract_mpo_arrays
    from a2dmrg.environments.environment import build_environments_incremental
    from a2dmrg.numerics.observables import compute_energy

    L = 8
    mpo = build_heisenberg_mpo(L)
    dmrg = qtn.DMRG2(mpo, bond_dims=20, cutoffs=1e-14)
    dmrg.solve(max_sweeps=10, tol=1e-12, verbosity=0)
    mps = dmrg._k

    E_ref = compute_energy(mps, mpo)
    mps_arrays = extract_mps_arrays(mps)
    mpo_arrays = extract_mpo_arrays(mpo)

    L_envs, R_envs, canon_arrays = build_environments_incremental(mps_arrays, mpo_arrays)

    # At each site in the canonical sweep, the local Rayleigh quotient
    # should equal the total energy (property of mixed-canonical form)
    for i in range(L):
        E_site = _brute_force_energy_at_site(
            canon_arrays, mpo_arrays, L_envs[i], R_envs[i + 1], i
        )
        assert abs(E_site - E_ref) < 1e-8, \
            f"Site {i}: E_site={E_site:.10f} vs E_ref={E_ref:.10f}, diff={abs(E_site-E_ref):.2e}"


def test_incremental_envs_boundary_shapes():
    """Boundary environments must be 1x1x1."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays, extract_mpo_arrays
    from a2dmrg.environments.environment import build_environments_incremental

    L = 6
    mpo = build_heisenberg_mpo(L)
    mps = qtn.MPS_rand_state(L, bond_dim=10, dtype='float64')
    mps_arrays = extract_mps_arrays(mps)
    mpo_arrays = extract_mpo_arrays(mpo)

    L_envs, R_envs, _ = build_environments_incremental(mps_arrays, mpo_arrays)

    assert L_envs[0].shape == (1, 1, 1)
    assert R_envs[L].shape == (1, 1, 1)
    assert len(L_envs) == L + 1
    assert len(R_envs) == L + 1


def test_incremental_envs_complex():
    """Works with complex128 dtype."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays, extract_mpo_arrays
    from a2dmrg.environments.environment import build_environments_incremental

    L = 6
    mpo = build_heisenberg_mpo(L)
    mps = qtn.MPS_rand_state(L, bond_dim=10, dtype='complex128')
    mps_arrays = extract_mps_arrays(mps)
    mpo_arrays = extract_mpo_arrays(mpo)

    L_envs, R_envs, canon_arrays = build_environments_incremental(mps_arrays, mpo_arrays)

    for i in range(L + 1):
        assert L_envs[i].dtype == np.complex128
        assert R_envs[i].dtype == np.complex128
```

- [ ] **Step 2: Run tests — should fail**

```bash
python3 -m pytest a2dmrg/tests/test_incremental_envs.py -v 2>&1
```

- [ ] **Step 3: Implement `build_environments_incremental` in environment.py**

Add this function to `a2dmrg/a2dmrg/environments/environment.py`:

```python
def build_environments_incremental(mps_arrays, mpo_arrays):
    """Build all left and right environments in O(L) via two sweeps.

    Algorithm:
    1. Right-canonicalize MPS arrays (sweep right-to-left with QR).
       Build R_envs[i] from right-canonical tensors.
    2. Left-sweep: at each site i, the MPS is in i-orthogonal form.
       Build L_envs[i+1] by left-canonicalizing site i.

    After this, L_envs[i] and R_envs[i+1] are the correct environments
    for site i's eigensolve in i-orthogonal form.

    Parameters
    ----------
    mps_arrays : list of ndarray
        MPS tensors in (chi_L, d, chi_R) format.
    mpo_arrays : list of ndarray
        MPO tensors in (D_L, D_R, d_up, d_down) format (from extract_mpo_arrays).

    Returns
    -------
    L_envs : list of ndarray
        L_envs[i] has shape (chi_bra, D, chi_ket) for bond i.
    R_envs : list of ndarray
        R_envs[i] has shape (chi_bra, D, chi_ket) for bond i.
    canon_arrays : list of ndarray
        MPS tensors after canonicalization sweep. At site i during
        the left-sweep, canon_arrays[i] is the center tensor in
        i-orthogonal form.
    """
    L = len(mps_arrays)
    dtype = mps_arrays[0].dtype

    # Copy arrays (we will modify them during canonicalization)
    arrays = [a.copy() for a in mps_arrays]

    # --- Step 1: Right-canonicalize, build R_envs ---
    R_envs = [None] * (L + 1)
    R_envs[L] = np.ones((1, 1, 1), dtype=dtype)

    for i in range(L - 1, 0, -1):
        chi_L, d, chi_R = arrays[i].shape
        # Reshape to (chi_L, d*chi_R) and take QR of transpose
        mat = arrays[i].reshape(chi_L, d * chi_R)
        Q, R = np.linalg.qr(mat.T)
        # Q^T is left factor (goes into arrays[i]), R^T absorbed left
        arrays[i] = Q.T.reshape(-1, d, chi_R)
        # Absorb R^T into arrays[i-1] from the right
        # arrays[i-1] has shape (chi_L2, d2, chi_R2) where chi_R2 = chi_L
        arrays[i - 1] = np.tensordot(arrays[i - 1], R.T, axes=(2, 0))
        # Build R_env[i]
        R_envs[i] = _update_right_env_numpy(R_envs[i + 1], arrays[i], mpo_arrays[i])

    # Build R_envs[0] from site 0 (which has absorbed all gauge from right-canonicalization)
    R_envs[0] = _update_right_env_numpy(R_envs[1], arrays[0], mpo_arrays[0])

    # --- Step 2: Left-sweep, build L_envs ---
    # At this point: arrays[0] is the center (all gauge accumulated),
    # arrays[1..L-1] are right-canonical.
    L_envs = [None] * (L + 1)
    L_envs[0] = np.ones((1, 1, 1), dtype=dtype)

    # Save the center tensors at each site for eigensolves
    canon_arrays = [None] * L

    for i in range(L):
        # arrays[i] IS the center tensor for i-orthogonal form
        canon_arrays[i] = arrays[i].copy()

        # Left-canonicalize site i to advance to site i+1
        if i < L - 1:
            chi_L, d, chi_R = arrays[i].shape
            mat = arrays[i].reshape(chi_L * d, chi_R)
            Q, R = np.linalg.qr(mat)
            new_chi = Q.shape[1]
            arrays[i] = Q.reshape(chi_L, d, new_chi)
            # Absorb R into arrays[i+1]
            arrays[i + 1] = np.tensordot(R, arrays[i + 1], axes=(1, 0))
            # Build L_env[i+1] from the left-canonical tensor
            L_envs[i + 1] = _update_left_env_numpy(L_envs[i], arrays[i], mpo_arrays[i])
        else:
            # Last site: no advancement needed
            pass

    return L_envs, R_envs, canon_arrays


def _update_left_env_numpy(L_env, A, W):
    """Update left environment: L_new = contract(L, A, W, A*).

    L_env: (bra, mpo, ket)
    A: (chi_L, d, chi_R) — ket tensor (left-canonical)
    W: (D_L, D_R, d_up, d_down) — MPO tensor

    Convention: d_up connects to ket, d_down connects to bra.
    Same contraction pattern as compute_energy() in observables.py.

    Returns L_new: (bra', mpo', ket')
    """
    A_conj = A.conj()
    # Step 1: L[bra,mpo,ket] @ A[ket,d,ket'] -> X[bra,mpo,d_ket,ket']
    X = np.tensordot(L_env, A, axes=(2, 0))
    # Step 2: X[bra,mpo,d_ket,ket'] @ W[D_L,D_R,d_up,d_down]
    # Contract mpo (X axis 1) with D_L (W axis 0), d_ket (X axis 2) with d_up (W axis 2)
    Y = np.tensordot(X, W, axes=([1, 2], [0, 2]))
    # Y shape: (bra, ket', D_R, d_down)
    # Step 3: Y[bra,ket',D_R,d_down] @ A*[chi_L,d,chi_R]
    # Contract bra (Y axis 0) with chi_L (A* axis 0), d_down (Y axis 3) with d (A* axis 1)
    L_new = np.tensordot(Y, A_conj, axes=([0, 3], [0, 1]))
    # L_new shape: (ket', D_R, chi_R_conj) = (ket', mpo', bra')
    # Transpose to (bra', mpo', ket')
    return L_new.transpose(2, 1, 0)


def _update_right_env_numpy(R_env, B, W):
    """Update right environment: R_new = contract(R, B, W, B*).

    R_env: (bra, mpo, ket)
    B: (chi_L, d, chi_R) — ket tensor (right-canonical)
    W: (D_L, D_R, d_up, d_down) — MPO tensor

    Convention: d_up connects to ket, d_down connects to bra.

    Returns R_new: (bra', mpo', ket')
    """
    B_conj = B.conj()
    # Step 1: B[chi_L,d,chi_R] @ R[bra,mpo,ket] -> X[chi_L,d_ket,bra,mpo]
    X = np.tensordot(B, R_env, axes=(2, 2))
    # Step 2: X[chi_L,d_ket,bra,mpo] @ W[D_L,D_R,d_up,d_down]
    # Contract d_ket (X axis 1) with d_up (W axis 2), mpo (X axis 3) with D_R (W axis 1)
    Y = np.tensordot(X, W, axes=([1, 3], [2, 1]))
    # Y shape: (chi_L, bra, D_L, d_down)
    # Step 3: Y[chi_L,bra,D_L,d_down] @ B*[chi_L,d,chi_R]
    # Contract bra (Y axis 1) with chi_R (B* axis 2), d_down (Y axis 3) with d (B* axis 1)
    R_new = np.tensordot(Y, B_conj, axes=([1, 3], [2, 1]))
    # R_new shape: (chi_L, D_L, chi_L_conj) = (ket_L, mpo_L, bra_L)
    # Transpose to (bra_L, mpo_L, ket_L)
    return R_new.transpose(2, 1, 0)
```

**IMPORTANT:** The contraction index orders in `_update_left_env_numpy` and `_update_right_env_numpy` must be validated carefully against the existing `_update_left_env` / environment code. After implementation, verify by comparing outputs against the existing `build_left_environments` / `build_right_environments` for a known MPS.

- [ ] **Step 4: Run tests — should pass**

```bash
python3 -m pytest a2dmrg/tests/test_incremental_envs.py -v 2>&1
```

If the site-energy test fails, the contraction indices are wrong. Debug by comparing `_update_left_env_numpy` output against the existing `build_left_environments` output element-by-element.

- [ ] **Step 5: Commit**

```bash
git add a2dmrg/a2dmrg/environments/environment.py a2dmrg/a2dmrg/tests/test_incremental_envs.py
git commit -m "feat(a2dmrg): add O(L) incremental environment builder"
```

---

### Task 4: Rewrite local micro-steps to use cached environments

Replace the per-site `mps.copy() + canonize() + full_env_rebuild` with a single call using pre-built environments.

**Files:**
- Modify: `a2dmrg/a2dmrg/numerics/local_microstep.py`
- Create: `a2dmrg/a2dmrg/tests/test_fast_microstep.py`

- [ ] **Step 1: Write tests for fast micro-steps**

```python
"""Tests for fast micro-steps using cached environments."""
import numpy as np
import pytest
import quimb.tensor as qtn

from a2dmrg.hamiltonians.heisenberg import build_heisenberg_mpo


def test_fast_microstep_1site_energy():
    """Fast 1-site micro-step must produce lower energy than input."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays, extract_mpo_arrays
    from a2dmrg.environments.environment import build_environments_incremental
    from a2dmrg.numerics.local_microstep import fast_microstep_1site
    from a2dmrg.numerics.observables import compute_energy

    L = 8
    mpo = build_heisenberg_mpo(L)
    dmrg = qtn.DMRG2(mpo, bond_dims=20, cutoffs=1e-14)
    dmrg.solve(max_sweeps=5, tol=1e-8, verbosity=0)
    mps = dmrg._k

    E_before = compute_energy(mps, mpo)
    mps_arrays = extract_mps_arrays(mps)
    mpo_arrays = extract_mpo_arrays(mpo)
    L_envs, R_envs, canon_arrays = build_environments_incremental(mps_arrays, mpo_arrays)

    # Optimize site 4
    site = 4
    optimized_tensor, eigval = fast_microstep_1site(
        canon_arrays[site], mpo_arrays[site],
        L_envs[site], R_envs[site + 1],
        site, L,
    )
    assert optimized_tensor.shape == canon_arrays[site].shape
    # Eigenvalue should be <= current energy (variational principle)
    assert eigval <= E_before + 1e-10


def test_fast_microstep_2site_energy():
    """Fast 2-site micro-step must produce lower energy than input."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays, extract_mpo_arrays
    from a2dmrg.environments.environment import build_environments_incremental
    from a2dmrg.numerics.local_microstep import fast_microstep_2site
    from a2dmrg.numerics.observables import compute_energy

    L = 8
    mpo = build_heisenberg_mpo(L)
    dmrg = qtn.DMRG2(mpo, bond_dims=20, cutoffs=1e-14)
    dmrg.solve(max_sweeps=5, tol=1e-8, verbosity=0)
    mps = dmrg._k

    E_before = compute_energy(mps, mpo)
    mps_arrays = extract_mps_arrays(mps)
    mpo_arrays = extract_mpo_arrays(mpo)
    L_envs, R_envs, canon_arrays = build_environments_incremental(mps_arrays, mpo_arrays)

    site = 3
    U, SVh, eigval = fast_microstep_2site(
        canon_arrays[site], canon_arrays[site + 1],
        mpo_arrays[site], mpo_arrays[site + 1],
        L_envs[site], R_envs[site + 2],
        site, L, max_bond=20,
    )
    assert eigval <= E_before + 1e-10
```

- [ ] **Step 2: Run tests — should fail**

```bash
python3 -m pytest a2dmrg/tests/test_fast_microstep.py -v 2>&1
```

- [ ] **Step 3: Implement fast micro-step functions**

Add to `a2dmrg/a2dmrg/numerics/local_microstep.py`:

```python
def _build_heff_numpy_1site(L_env, W, R_env):
    """Build 1-site effective Hamiltonian from pre-extracted numpy arrays.

    Unlike effective_ham.build_effective_hamiltonian_1site, this takes
    already-extracted numpy arrays and does NOT call _reshape_mpo_tensor.

    L_env: (bra, mpo, ket)
    W: (D_L, D_R, d_up, d_down) — from extract_mpo_arrays
    R_env: (bra, mpo, ket)

    Contraction: same as compute_energy / _update_left_env_numpy
    """
    from scipy.sparse.linalg import LinearOperator

    bra_L, mpo_L, ket_L = L_env.shape
    bra_R, mpo_R, ket_R = R_env.shape
    d = W.shape[2]  # d_up = d_down = physical dim
    size = ket_L * d * ket_R

    def matvec(v):
        psi = v.reshape(ket_L, d, ket_R)
        # L[bra,mpo,ket] @ psi[ket,d,ket'] -> X[bra,mpo,d,ket']
        X = np.tensordot(L_env, psi, axes=(2, 0))
        # X[bra,mpo,d_ket,ket'] @ W[D_L,D_R,d_up,d_down]
        # Contract mpo with D_L, d_ket with d_up
        Y = np.tensordot(X, W, axes=([1, 2], [0, 2]))
        # Y[bra,ket',D_R,d_down] @ R[bra_R,mpo_R,ket_R]
        # Contract ket' with ket_R, D_R with mpo_R
        # But we need: result = (bra_L, d_bra, bra_R) for the output
        # Y[bra, ket', D_R, d_down] @ R[c, V, d]
        # Contract D_R (Y axis 2) with mpo_R (R axis 1), ket' (Y axis 1) with ket_R (R axis 2)
        result = np.tensordot(Y, R_env, axes=([1, 2], [2, 1]))
        # result shape: (bra, d_down, bra_R) = (bra_L, d_bra, bra_R)
        return result.ravel()

    return LinearOperator(shape=(size, size), matvec=matvec, dtype=L_env.dtype)


def _build_heff_numpy_2site(L_env, W1, W2, R_env):
    """Build 2-site effective Hamiltonian from pre-extracted numpy arrays.

    L_env: (bra, mpo, ket)
    W1, W2: (D_L, D_R, d_up, d_down) — from extract_mpo_arrays
    R_env: (bra, mpo, ket)
    """
    from scipy.sparse.linalg import LinearOperator

    bra_L, mpo_L, ket_L = L_env.shape
    bra_R, mpo_R, ket_R = R_env.shape
    d1 = W1.shape[2]
    d2 = W2.shape[2]
    size = ket_L * d1 * d2 * ket_R

    def matvec(v):
        theta = v.reshape(ket_L, d1, d2, ket_R)
        # L[bra,mpo,ket] @ theta[ket,d1,d2,ket_R'] -> X[bra,mpo,d1,d2,ket_R']
        X = np.tensordot(L_env, theta, axes=(2, 0))
        # X[bra,mpo,d1,d2,ket_R'] @ W1[D_L,D_R,d_up,d_down]
        # Contract mpo with D_L, d1 with d_up
        Y = np.tensordot(X, W1, axes=([1, 2], [0, 2]))
        # Y[bra,d2,ket_R',D_R1,d_down1]
        # Y @ W2[D_L2,D_R2,d_up2,d_down2]
        # Contract d2 (Y axis 1) with d_up2 (W2 axis 2), D_R1 (Y axis 3) with D_L2 (W2 axis 0)
        Z = np.tensordot(Y, W2, axes=([1, 3], [2, 0]))
        # Z[bra,ket_R',d_down1,D_R2,d_down2]
        # Z @ R[bra_R,mpo_R,ket_R]
        # Contract ket_R' (Z axis 1) with ket_R (R axis 2), D_R2 (Z axis 3) with mpo_R (R axis 1)
        result = np.tensordot(Z, R_env, axes=([1, 3], [2, 1]))
        # result[bra, d_down1, d_down2, bra_R]
        return result.ravel()

    return LinearOperator(shape=(size, size), matvec=matvec, dtype=L_env.dtype)


def fast_microstep_1site(center_tensor, W, L_env, R_env, site, L, tol=1e-10):
    """One-site micro-step using pre-built environments.

    No MPS copy, no canonization, no environment rebuild.

    Parameters
    ----------
    center_tensor : ndarray, shape (chi_L, d, chi_R)
        Center tensor in i-orthogonal form.
    W : ndarray, shape (D_L, D_R, d_up, d_down)
        MPO tensor at this site (from extract_mpo_arrays).
    L_env : ndarray, shape (bra, mpo, ket)
        Left environment at bond i.
    R_env : ndarray, shape (bra, mpo, ket)
        Right environment at bond i+1.
    site : int
        Site index.
    L : int
        Total number of sites.
    tol : float
        Eigensolver tolerance.

    Returns
    -------
    optimized_tensor : ndarray, shape (chi_L, d, chi_R)
    eigenvalue : float
    """
    from .eigensolver import solve_effective_hamiltonian

    H_eff = _build_heff_numpy_1site(L_env, W, R_env)
    v0 = center_tensor.ravel()
    energy, eigvec = solve_effective_hamiltonian(H_eff, v0=v0, tol=tol)
    optimized = eigvec.reshape(center_tensor.shape)
    return optimized, float(np.real(energy))


def fast_microstep_2site(tensor_i, tensor_ip1, W_i, W_ip1,
                         L_env, R_env, site, L,
                         max_bond=None, cutoff=0.0, tol=1e-10):
    """Two-site micro-step using pre-built environments.

    Parameters
    ----------
    tensor_i : ndarray, shape (chi_L, d1, chi_M)
    tensor_ip1 : ndarray, shape (chi_M, d2, chi_R)
    W_i, W_ip1 : MPO tensors, shape (D_L, D_R, d_up, d_down)
    L_env : left environment at bond i
    R_env : right environment at bond i+2
    site : int
    L : int
    max_bond : int or None
    cutoff : float
    tol : float

    Returns
    -------
    U_tensor : ndarray, shape (chi_L, d1, chi_new)
    SVh_tensor : ndarray, shape (chi_new, d2, chi_R)
    eigenvalue : float
    """
    from .eigensolver import solve_effective_hamiltonian

    chi_L, d1, chi_M = tensor_i.shape
    _, d2, chi_R = tensor_ip1.shape

    # Form two-site theta
    theta = np.tensordot(tensor_i, tensor_ip1, axes=(2, 0))

    H_eff = _build_heff_numpy_2site(L_env, W_i, W_ip1, R_env)
    v0 = theta.ravel()
    energy, eigvec = solve_effective_hamiltonian(H_eff, v0=v0, tol=tol)

    # SVD split
    theta_opt = eigvec.reshape(chi_L * d1, d2 * chi_R)
    U, S, Vh = np.linalg.svd(theta_opt, full_matrices=False)

    # Truncate
    if max_bond is not None:
        chi_new = min(max_bond, len(S))
    else:
        chi_new = len(S)
    if cutoff > 0:
        mask = S > cutoff
        chi_new = min(chi_new, max(1, mask.sum()))

    U = U[:, :chi_new]
    S = S[:chi_new]
    Vh = Vh[:chi_new, :]

    U_tensor = U.reshape(chi_L, d1, chi_new)
    SVh_tensor = (np.diag(S) @ Vh).reshape(chi_new, d2, chi_R)

    return U_tensor, SVh_tensor, float(np.real(energy))
```

- [ ] **Step 4: Run tests — should pass**

```bash
python3 -m pytest a2dmrg/tests/test_fast_microstep.py -v 2>&1
```

- [ ] **Step 5: Commit**

```bash
git add a2dmrg/a2dmrg/numerics/local_microstep.py a2dmrg/a2dmrg/tests/test_fast_microstep.py
git commit -m "feat(a2dmrg): add fast micro-steps using cached environments"
```

---

### Task 5: Rewrite parallel local steps to use cached environments

Replace `parallel_local_microsteps` to pre-build environments once, then dispatch eigensolves to ranks.

**Files:**
- Modify: `a2dmrg/a2dmrg/parallel/local_steps.py`

- [ ] **Step 1: Rewrite `parallel_local_microsteps`**

Replace the body of `parallel_local_microsteps` in `local_steps.py`:

```python
def parallel_local_microsteps(mps, mpo, comm,
                              microstep_type="two_site",
                              max_bond=None, cutoff=0.0, tol=1e-10):
    """Phase 2: parallel local micro-steps with O(L) environment caching.

    All ranks build environments from the same MPS (replicated, O(L) work).
    Each rank then solves its assigned sites using cached environments.
    No MPS copies, no per-site canonicalization.
    """
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays, extract_mpo_arrays
    from a2dmrg.environments.environment import build_environments_incremental
    from a2dmrg.numerics.local_microstep import fast_microstep_1site, fast_microstep_2site
    from .distribute import distribute_sites

    rank = comm.Get_rank()
    n_procs = comm.Get_size()
    L = mps.L

    # Extract to numpy (all ranks do this — MPS is already broadcast)
    mps_arrays = extract_mps_arrays(mps)
    mpo_arrays = extract_mpo_arrays(mpo)

    # Build all environments in O(L) — all ranks, replicated
    L_envs, R_envs, canon_arrays = build_environments_incremental(mps_arrays, mpo_arrays)

    # Distribute sites
    my_sites = distribute_sites(L, n_procs, rank)

    results = {}

    if microstep_type == "one_site":
        for site in my_sites:
            opt_tensor, eigval = fast_microstep_1site(
                canon_arrays[site], mpo_arrays[site],
                L_envs[site], R_envs[site + 1],
                site, L, tol=tol,
            )
            # Build candidate: use canonicalized arrays (not original mps_arrays!)
            # The optimized tensor lives in the canonical gauge where L_envs/R_envs
            # were built, so we must use canon_arrays as the base.
            candidate_arrays = [a.copy() for a in canon_arrays]
            candidate_arrays[site] = opt_tensor
            results[site] = (candidate_arrays, eigval)

    elif microstep_type == "two_site":
        for site in my_sites:
            if site >= L - 1:
                continue
            U, SVh, eigval = fast_microstep_2site(
                canon_arrays[site], canon_arrays[site + 1],
                mpo_arrays[site], mpo_arrays[site + 1],
                L_envs[site], R_envs[site + 2],
                site, L, max_bond=max_bond, cutoff=cutoff, tol=tol,
            )
            # Use canon_arrays as base (gauge consistency)
            candidate_arrays = [a.copy() for a in canon_arrays]
            candidate_arrays[site] = U
            candidate_arrays[site + 1] = SVh
            results[site] = (candidate_arrays, eigval)
    else:
        raise ValueError(f"Unknown microstep_type: {microstep_type}")

    return results
```

**NOTE:** The results format changes from `(quimb_mps, energy)` to `(list_of_numpy_arrays, energy)`. This will require corresponding changes in `gather_local_results` and `prepare_candidate_mps_list` (Task 6).

- [ ] **Step 2: Update `gather_local_results` and `prepare_candidate_mps_list`**

These functions in the same file need to handle the new format:

```python
def gather_local_results(local_results, comm):
    """Gather results from all ranks. Results are (numpy_arrays, energy) tuples."""
    all_results_list = comm.allgather(local_results)
    all_results = {}
    for results_from_rank in all_results_list:
        all_results.update(results_from_rank)
    return all_results


def prepare_candidate_mps_list(mps, all_local_results):
    """Prepare candidate list as numpy array lists.

    Returns list of (numpy_arrays_list, energy) where first is the original.
    """
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays
    original_arrays = extract_mps_arrays(mps)
    candidates = [original_arrays]

    for site_idx in sorted(all_local_results.keys()):
        candidate_arrays, _ = all_local_results[site_idx]
        candidates.append(candidate_arrays)

    return candidates
```

- [ ] **Step 3: Commit (integration will be tested in Task 6)**

```bash
git add a2dmrg/a2dmrg/parallel/local_steps.py
git commit -m "feat(a2dmrg): rewrite parallel local steps with cached environments"
```

---

### Task 6: Rewrite coarse-space to work with numpy arrays

The coarse-space functions (cross-energy, overlap, filtering) need to work with numpy array lists instead of quimb MPS.

**Files:**
- Modify: `a2dmrg/a2dmrg/numerics/observables.py`
- Modify: `a2dmrg/a2dmrg/parallel/coarse_space.py`
- Create: `a2dmrg/a2dmrg/tests/test_numpy_coarse.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for numpy-based coarse-space computation."""
import numpy as np
import pytest
import quimb.tensor as qtn

from a2dmrg.hamiltonians.heisenberg import build_heisenberg_mpo


def test_numpy_energy_matches_quimb():
    """Energy from numpy arrays must match observables.compute_energy."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays, extract_mpo_arrays
    from a2dmrg.numerics.observables import compute_energy, compute_energy_numpy

    L = 8
    mpo = build_heisenberg_mpo(L)
    dmrg = qtn.DMRG2(mpo, bond_dims=20, cutoffs=1e-14)
    dmrg.solve(max_sweeps=10, tol=1e-12, verbosity=0)

    E_quimb = compute_energy(dmrg._k, mpo)
    mps_arrays = extract_mps_arrays(dmrg._k)
    mpo_arrays = extract_mpo_arrays(mpo)
    E_numpy = compute_energy_numpy(mps_arrays, mpo_arrays)

    assert abs(E_quimb - E_numpy) < 1e-12


def test_numpy_overlap():
    """Overlap of MPS with itself must be ~1 (after normalization)."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays
    from a2dmrg.numerics.observables import compute_overlap_numpy

    L = 6
    mps = qtn.MPS_rand_state(L, bond_dim=10, dtype='float64')
    mps /= mps.norm()
    arrays = extract_mps_arrays(mps)

    overlap = compute_overlap_numpy(arrays, arrays)
    assert abs(overlap - 1.0) < 1e-10
```

- [ ] **Step 2: Implement numpy energy and overlap in observables.py**

Add to `a2dmrg/a2dmrg/numerics/observables.py`:

```python
def compute_energy_numpy(mps_arrays, mpo_arrays, normalize=True):
    """Compute <psi|H|psi> from numpy arrays.

    mps_arrays: list of (chi_L, d, chi_R) from extract_mps_arrays
    mpo_arrays: list of (D_L, D_R, d_up, d_down) from extract_mpo_arrays
    Same contraction pattern as compute_energy() in observables.py.
    """
    L = len(mps_arrays)
    dtype = mps_arrays[0].dtype
    L_env = np.ones((1, 1, 1), dtype=dtype)

    for i in range(L):
        A = mps_arrays[i]
        W = mpo_arrays[i]
        A_conj = A.conj()
        X = np.tensordot(L_env, A, axes=(2, 0))
        Y = np.tensordot(X, W, axes=([1, 2], [0, 2]))
        L_env = np.tensordot(Y, A_conj, axes=([0, 3], [0, 1]))
        L_env = L_env.transpose(2, 1, 0)

    energy = float(np.real(np.trace(L_env[:, 0, :])))
    if normalize:
        norm_sq = compute_overlap_numpy(mps_arrays, mps_arrays)
        energy /= abs(norm_sq)
    return energy


def compute_cross_energy_numpy(bra_arrays, mpo_arrays, ket_arrays):
    """Compute <bra|H|ket> from numpy arrays (unnormalized)."""
    L = len(bra_arrays)
    dtype = np.result_type(bra_arrays[0].dtype, ket_arrays[0].dtype)
    L_env = np.ones((1, 1, 1), dtype=dtype)

    for i in range(L):
        A_ket = ket_arrays[i]
        A_bra_conj = bra_arrays[i].conj()
        W = mpo_arrays[i]
        X = np.tensordot(L_env, A_ket, axes=(2, 0))
        Y = np.tensordot(X, W, axes=([1, 2], [0, 2]))
        L_env = np.tensordot(Y, A_bra_conj, axes=([0, 3], [0, 1]))
        L_env = L_env.transpose(2, 1, 0)

    return complex(np.trace(L_env[:, 0, :]))


def compute_overlap_numpy(bra_arrays, ket_arrays):
    """Compute <bra|ket> from numpy arrays."""
    L = len(bra_arrays)
    dtype = np.result_type(bra_arrays[0].dtype, ket_arrays[0].dtype)
    T = np.ones((1, 1), dtype=dtype)

    for i in range(L):
        A_bra_conj = bra_arrays[i].conj()
        A_ket = ket_arrays[i]
        X = np.tensordot(T, A_bra_conj, axes=(0, 0))
        T = np.tensordot(X, A_ket, axes=([0, 1], [0, 1]))

    return complex(np.trace(T))
```

- [ ] **Step 3: Update coarse_space.py to use numpy functions**

The candidate_mps_list is now `list[list[ndarray]]` instead of `list[quimb.MPS]`. Three specific changes:

**3a.** In `filter_redundant_candidates` (around line 76), replace:
```python
from a2dmrg.numerics.observables import compute_overlap
overlap = compute_overlap(candidate, retained)
```
with:
```python
from a2dmrg.numerics.observables import compute_overlap_numpy
overlap = compute_overlap_numpy(candidate, retained)
```

**3b.** In `build_coarse_matrices` (around lines 199, 283), replace:
```python
from a2dmrg.numerics.observables import compute_cross_energy, compute_overlap
H_coarse[i, j] = compute_cross_energy(bra, mpo, ket)
S_coarse[i, j] = compute_overlap(bra, ket)
```
with:
```python
from a2dmrg.numerics.observables import compute_cross_energy_numpy, compute_overlap_numpy
H_coarse[i, j] = compute_cross_energy_numpy(bra, mpo_arrays, ket)
S_coarse[i, j] = compute_overlap_numpy(bra, ket)
```
Note: `build_coarse_matrices` must now accept `mpo_arrays` (pre-extracted) instead of the quimb `mpo`.

**3c.** Fix dtype detection: replace `for tensor in first_mps.tensors` (line ~216) with:
```python
dtype = candidates[0][0].dtype  # dtype of first site tensor of first candidate
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest a2dmrg/tests/test_numpy_coarse.py -v 2>&1
```

- [ ] **Step 5: Commit**

```bash
git add a2dmrg/a2dmrg/numerics/observables.py a2dmrg/a2dmrg/parallel/coarse_space.py \
       a2dmrg/a2dmrg/tests/test_numpy_coarse.py
git commit -m "feat(a2dmrg): numpy-based coarse-space energy and overlap"
```

---

### Task 7: Numpy TT-SVD compression (replace quimb.compress)

Phase 5 uses `quimb.compress()` which takes 35s at L=20 chi=50 due to tensor object overhead. Replace with pure numpy.

**Files:**
- Create: `a2dmrg/a2dmrg/numerics/compression.py`
- Create: `a2dmrg/a2dmrg/tests/test_numpy_compression.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for numpy TT-SVD compression."""
import numpy as np
import pytest
import quimb.tensor as qtn


def test_compress_preserves_state():
    """Compressed MPS must have overlap > 0.999 with original."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays
    from a2dmrg.numerics.compression import tt_svd_compress
    from a2dmrg.numerics.observables import compute_overlap_numpy

    L = 8
    mps = qtn.MPS_rand_state(L, bond_dim=30, dtype='float64')
    mps /= mps.norm()
    arrays = extract_mps_arrays(mps)

    compressed = tt_svd_compress(arrays, max_bond=20)
    overlap = compute_overlap_numpy(arrays, compressed)
    assert abs(overlap) > 0.99


def test_compress_respects_max_bond():
    """All bonds in compressed MPS must be <= max_bond."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays
    from a2dmrg.numerics.compression import tt_svd_compress

    L = 10
    mps = qtn.MPS_rand_state(L, bond_dim=50, dtype='float64')
    arrays = extract_mps_arrays(mps)
    max_bond = 20

    compressed = tt_svd_compress(arrays, max_bond=max_bond)
    for i in range(L - 1):
        chi = compressed[i].shape[2]
        assert chi <= max_bond, f"Bond {i}-{i+1}: chi={chi} > max_bond={max_bond}"


def test_compress_normalizes():
    """Compressed MPS must be normalized."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays
    from a2dmrg.numerics.compression import tt_svd_compress
    from a2dmrg.numerics.observables import compute_overlap_numpy

    L = 8
    mps = qtn.MPS_rand_state(L, bond_dim=30, dtype='float64')
    arrays = extract_mps_arrays(mps)

    compressed = tt_svd_compress(arrays, max_bond=20, normalize=True)
    norm = compute_overlap_numpy(compressed, compressed)
    assert abs(norm - 1.0) < 1e-10
```

- [ ] **Step 2: Implement TT-SVD compression**

Create `a2dmrg/a2dmrg/numerics/compression.py`:

```python
"""Numpy TT-SVD compression for MPS arrays."""
import numpy as np


def tt_svd_compress(arrays, max_bond, normalize=True):
    """Compress MPS arrays via left-to-right TT-SVD.

    Parameters
    ----------
    arrays : list of ndarray
        MPS tensors in (chi_L, d, chi_R) format.
    max_bond : int
        Maximum bond dimension after compression.
    normalize : bool
        Normalize the result.

    Returns
    -------
    compressed : list of ndarray
        Compressed MPS tensors.
    """
    L = len(arrays)
    result = [a.copy() for a in arrays]

    # Left-to-right SVD sweep
    for i in range(L - 1):
        chi_L, d, chi_R = result[i].shape
        mat = result[i].reshape(chi_L * d, chi_R)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        # Truncate to max_bond
        chi_new = min(max_bond, len(S))
        U = U[:, :chi_new]
        S = S[:chi_new]
        Vh = Vh[:chi_new, :]

        result[i] = U.reshape(chi_L, d, chi_new)
        # Absorb S @ Vh into next tensor
        SVh = np.diag(S) @ Vh
        result[i + 1] = np.tensordot(SVh, result[i + 1], axes=(1, 0))

    if normalize:
        # Normalize last tensor
        norm = np.linalg.norm(result[-1])
        if norm > 1e-15:
            result[-1] = result[-1] / norm

    return result
```

- [ ] **Step 3: Run tests**

```bash
python3 -m pytest a2dmrg/tests/test_numpy_compression.py -v 2>&1
```

- [ ] **Step 4: Commit**

```bash
git add a2dmrg/a2dmrg/numerics/compression.py a2dmrg/a2dmrg/tests/test_numpy_compression.py
git commit -m "feat(a2dmrg): add numpy TT-SVD compression"
```

---

### Task 8: Wire everything into dmrg.py main loop

Connect all the new components into the main `a2dmrg_main` function. This replaces the quimb-heavy hot path with the numpy pipeline.

**Files:**
- Modify: `a2dmrg/a2dmrg/dmrg.py`
- Modify: `a2dmrg/a2dmrg/parallel/linear_combination.py`

- [ ] **Step 1: Update Phase 2 in dmrg.py**

Replace the Phase 2 block (around lines 406-428) to use the rewritten `parallel_local_microsteps` which now uses cached environments internally. No changes needed to the call site if the function signature stayed the same.

- [ ] **Step 2: Update Phase 3 (coarse-space) to accept numpy arrays**

The `prepare_candidate_mps_list` now returns `list[list[ndarray]]`. Update `build_coarse_matrices` and `filter_redundant_candidates` calls to pass `mpo_arrays` (numpy) instead of `mpo` (quimb). Add `mpo_arrays` extraction at the top of the sweep loop (once).

- [ ] **Step 3: Update Phase 4 (linear combination) to work with numpy arrays**

Rewrite `form_linear_combination` in `linear_combination.py` to work with `list[list[ndarray]]`:

```python
def form_linear_combination(candidate_arrays_list, coeffs):
    """Form weighted sum of candidate MPS (numpy arrays).

    IMPORTANT: The coefficient for each candidate is applied to site 0 ONLY.
    An MPS state |psi> = A[0] A[1] ... A[L-1], so multiplying c into every
    site would give c^L |psi> instead of c |psi>.

    Parameters
    ----------
    candidate_arrays_list : list of list of ndarray
        Each inner list is an MPS as numpy arrays.
    coeffs : ndarray
        Weights for each candidate.

    Returns
    -------
    combined : list of ndarray
        Combined MPS arrays (bonds may grow by factor of n_candidates).
    """
    n_cands = len(candidate_arrays_list)
    L = len(candidate_arrays_list[0])

    combined = []
    for i in range(L):
        # Apply coefficient to site 0 only (avoids c^L scaling bug)
        if i == 0:
            tensors = [coeffs[j] * candidate_arrays_list[j][i] for j in range(n_cands)]
        else:
            tensors = [candidate_arrays_list[j][i] for j in range(n_cands)]

        # Block-diagonal concatenation along bond dimensions
        if i == 0:
            # First site: (1, d, chi_R) — concatenate along chi_R
            combined.append(np.concatenate(tensors, axis=2))
        elif i == L - 1:
            # Last site: (chi_L, d, 1) — concatenate along chi_L
            combined.append(np.concatenate(tensors, axis=0))
        else:
            # Middle: block-diagonal (chi_L*n, d, chi_R*n)
            total_L = sum(t.shape[0] for t in tensors)
            d = tensors[0].shape[1]
            total_R = sum(t.shape[2] for t in tensors)
            block = np.zeros((total_L, d, total_R), dtype=tensors[0].dtype)
            row, col = 0, 0
            for t in tensors:
                block[row:row+t.shape[0], :, col:col+t.shape[2]] = t
                row += t.shape[0]
                col += t.shape[2]
            combined.append(block)

    return combined
```

- [ ] **Step 4: Update Phase 5 (compression) to use numpy TT-SVD**

Replace `mps.compress(max_bond=bond_dim, cutoff=0.0)` and `mps /= mps.norm()` with:

```python
from a2dmrg.numerics.compression import tt_svd_compress
from a2dmrg.numerics.observables import compute_energy_numpy
from a2dmrg.numerics.numpy_utils import arrays_to_quimb_mps

combined_arrays = form_linear_combination(candidate_arrays_list, coeffs)
compressed_arrays = tt_svd_compress(combined_arrays, max_bond=bond_dim, normalize=True)
energy_after_compression = compute_energy_numpy(compressed_arrays, mpo_arrays)

# Convert back to quimb MPS for next sweep iteration.
# This is needed because parallel_local_microsteps() calls mps.L and
# extract_mps_arrays(mps) which require a quimb MPS object.
mps = arrays_to_quimb_mps(compressed_arrays)
```

The quimb MPS conversion is the loop variable for the next sweep. This conversion is lightweight (just wrapping arrays) and ensures compatibility with the sweep loop. Finalization sweeps also require quimb MPS.

- [ ] **Step 5: Add `arrays_to_quimb_mps` utility in numpy_utils.py**

```python
def arrays_to_quimb_mps(arrays, dtype=None):
    """Convert list of numpy arrays back to quimb MPS."""
    import quimb.tensor as qtn

    L = len(arrays)
    tensors = []
    for i in range(L):
        a = arrays[i]
        if dtype is not None:
            a = a.astype(dtype)
        # Convert to quimb's expected shapes
        if i == 0:
            data = a[0, :, :]  # (d, chi_R)
            inds = (f'k{i}', f'b{i}-{i+1}')
        elif i == L - 1:
            data = a[:, :, 0]  # (chi_L, d)
            inds = (f'b{i-1}-{i}', f'k{i}')
        else:
            data = a  # (chi_L, d, chi_R)
            inds = (f'b{i-1}-{i}', f'k{i}', f'b{i}-{i+1}')
        tensors.append(qtn.Tensor(data=data, inds=inds, tags={f'I{i}'}))

    tn = qtn.TensorNetwork(tensors)
    return qtn.MatrixProductState.from_TN(
        tn, site_tag_id='I{}', site_ind_id='k{}', cyclic=False, L=L
    )
```

- [ ] **Step 6: Run canary test**

```bash
cd /home/captain/clawd/work/dmrg-implementations/a2dmrg
mpirun --oversubscribe -np 2 python3 -m pytest a2dmrg/tests/test_canary.py -v -s 2>&1
```

**This is the critical checkpoint.** If the canary passes with 1e-10 accuracy, the rewrite is correct. Record timing — should be dramatically faster than baseline.

- [ ] **Step 7: Commit**

```bash
git add a2dmrg/a2dmrg/dmrg.py a2dmrg/a2dmrg/parallel/linear_combination.py \
       a2dmrg/a2dmrg/numerics/numpy_utils.py
git commit -m "feat(a2dmrg): wire numpy pipeline into main loop"
```

---

### Task 9: Run full accuracy validation

Run all existing tests plus new accuracy tests across multiple L and chi values.

**Files:**
- Modify: `a2dmrg/a2dmrg/tests/test_canary.py` (add more cases)

- [ ] **Step 1: Add comprehensive accuracy tests**

Add to `test_canary.py`:

```python
@pytest.mark.mpi
@pytest.mark.parametrize("L,chi", [(8, 20), (12, 20), (12, 50), (20, 20), (20, 50)])
def test_accuracy_sweep(L, chi):
    """Parametrized accuracy test across L and chi values."""
    from a2dmrg.dmrg import a2dmrg_main
    comm = MPI.COMM_WORLD
    mpo = build_heisenberg_mpo(L)

    energy, mps = a2dmrg_main(
        L=L, mpo=mpo, bond_dim=chi, max_sweeps=20, tol=1e-12,
        warmup_sweeps=2, finalize_sweeps=2,
        comm=comm, verbose=False, one_site=False,
    )

    if comm.Get_rank() == 0:
        E_ref = _quimb_reference(L, chi)
        diff = abs(energy - E_ref)
        print(f"L={L} chi={chi}: A2DMRG={energy:.12f} quimb={E_ref:.12f} diff={diff:.2e}")
        assert diff < 1e-10, f"L={L} chi={chi}: diff {diff:.2e} > 1e-10"
```

- [ ] **Step 2: Run the full accuracy sweep**

```bash
mpirun --oversubscribe -np 2 python3 -m pytest a2dmrg/tests/test_canary.py -v -s 2>&1
```

Record all timings and accuracy numbers.

- [ ] **Step 3: Run existing test suite to check for regressions**

```bash
# Component tests (no MPI)
python3 -m pytest a2dmrg/tests/ -v -m "not mpi" --timeout=120 2>&1

# MPI tests
mpirun --oversubscribe -np 2 python3 -m pytest a2dmrg/tests/ -v -m mpi --timeout=300 2>&1
```

Fix any failures.

- [ ] **Step 4: Commit**

```bash
git add -A a2dmrg/
git commit -m "test(a2dmrg): full accuracy validation after performance rewrite"
```

---

### Task 10: Performance measurement and cleanup

Measure the improvement and clean up dead code.

- [ ] **Step 1: Timing comparison**

Run the canary test with timing and compare against the baseline from Task 1.

```bash
time mpirun --oversubscribe -np 2 python3 -m pytest a2dmrg/tests/test_canary.py::test_canary_heisenberg_l12 -v -s 2>&1
```

- [ ] **Step 2: Remove dead code**

Delete the old `local_microstep_1site` and `local_microstep_2site` functions (the ones that did per-site canonize + full env rebuild). Delete `_transform_to_i_orthogonal`. Delete the old `build_left_environments` and `build_right_environments` if no longer used.

- [ ] **Step 3: Run all tests one final time**

```bash
python3 -m pytest a2dmrg/tests/ -v -m "not mpi" --timeout=120 2>&1
mpirun --oversubscribe -np 2 python3 -m pytest a2dmrg/tests/ -v -m mpi --timeout=300 2>&1
```

- [ ] **Step 4: Commit**

```bash
git add -A a2dmrg/
git commit -m "refactor(a2dmrg): remove old O(L^2) environment code, cleanup"
```
