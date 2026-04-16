# radam -- Riemannian Adam for MPS / Tensor Train (CPU, numpy)

A self-contained CPU + numpy implementation of the **Riemannian Adam**
(R-Adam) optimizer applied to the fixed-rank Tensor Train (MPS)
manifold, for ground-state problems on 1D quantum Hamiltonians.

No `quimb`, no `torch`, no `jax` -- just `numpy`.

This is *not* a DMRG variant.  There is no local eigensolve, no
two-site update, and no sweeping.  Every epoch computes the Euclidean
gradient at every MPS core simultaneously, projects it onto the
tangent space of the TT manifold, updates the first / second Adam
moments, and retracts back to the manifold by re-canonicalization.

For a quasi-Newton optimizer with much tighter convergence on the
same manifold (quasi-second-order, reaches ~1e-9 vs DMRG on the small
grid), see the sibling package
[`cpu/rlbfgs/`](../rlbfgs/README.md).  R-Adam here is first-order and
makes more sense as a cheap warmup or when you need a robust, simple
optimizer.


## Why Riemannian?

Applying Euclidean Adam directly to the cores of an MPS pulls the
state off its canonical gauge and inflates the bond dimension.  The
set of TTs of fixed bond dim `chi` is a curved smooth manifold;
R-Adam restricts all gradient and momentum updates to the tangent
space at the current state, and uses a retraction to return to the
manifold at the end of each step.  The variance term in Adam becomes
a single global scalar (the squared Riemannian-gradient norm) rather
than an element-wise quantity, because a manifold has no global
coordinate system.


## Algorithm (R-Adam step)

1. **Euclidean gradient** `grad_i = dE / dA_i*` via left/right MPO
   and norm environments (`radam/gradient.py`).
2. **Tangent-space projection** onto the right-canonical gauge
   (`radam/projection.py`):
   `G_i <- V_i - (V_i A_i^H) A_i` for `i >= 1`.
3. **Vector transport** of the previous momentum by re-applying the
   current projection.
4. **First-moment update**:
   `M_i <- beta1 * M_i + (1 - beta1) * G_i`.
5. **Scalar second-moment update**:
   `v <- beta2 * v + (1 - beta2) * sum_i ||G_i||_F^2`.
6. **Bias correction** (standard Adam).
7. **Tangent step**
   `Delta_i = -lr * M_hat_i / (sqrt(v_hat) + eps)`.
8. **Retraction**: `X <- right_canonicalize([A_i + Delta_i])`,
   then normalize `<X|X> = 1` to prevent radial drift across
   iterations.

Convergence criterion: Riemannian gradient norm below `tol`.


## Representation

* MPS core: `(chi_L, d, chi_R)` numpy array.
* MPO core: `(mpo_L, mpo_R, d_up, d_dn)`.  `d_up` contracts with the
  **bra** physical index, `d_dn` with the **ket**.

The optimizer keeps `X` in *right-canonical* form with the
orthogonality center at site 0.  The momentum list `M` matches `X`
core-shape-for-core-shape.


## Supported Hamiltonians

* `build_heisenberg_mpo(L, j, bz)` -- XXX chain plus a longitudinal
  field.  Bond dim 5, real.
* `build_tfim_mpo(L, j, hx)` -- transverse-field Ising.  Bond dim 3,
  real.
* `build_josephson_mpo(L, E_J, E_C, mu, n_max, phi_ext)` -- Josephson
  junction array with external flux.  Bond dim 4, complex (``phi_ext``
  breaks time-reversal symmetry).  Matches
  `benchmarks/lib/models.py::build_josephson_mpo` exactly on an ED
  cross-check.


## Quickstart

```bash
cd cpu/radam
pip install -e .

# CLI:
python -m radam heisenberg --L 10 --chi 16 --epochs 500 --lr 1e-2
python -m radam tfim       --L 10 --chi 16 --epochs 500 --lr 1e-2 --hx 1.0

# Example script:
python examples/run_heisenberg.py

# Tests:
pytest
```


## Programmatic use

```python
from radam.driver import run_heisenberg

result = run_heisenberg(
    L=16, chi=32, j=1.0, bz=0.0,
    lr=5e-3, max_epochs=500, lr_schedule="cosine",
    seed=0, tol=1e-7,
)
# result["mps"], result["energy"], result["grad_norm"], result["history"], ...
```


## Files

| File                    | Purpose                                        |
|-------------------------|------------------------------------------------|
| `radam/mps.py`          | MPS init, canonical forms, norm, inner product |
| `radam/mpo.py`          | Heisenberg / TFIM / Josephson MPO builders     |
| `radam/environments.py` | Left/right MPO and norm environments           |
| `radam/gradient.py`     | Euclidean gradient of Rayleigh quotient        |
| `radam/projection.py`   | Tangent-space projection + vector transport    |
| `radam/retraction.py`   | Retraction (add delta + re-canonicalize + norm)|
| `radam/optimizer.py`    | Per-step R-Adam update                         |
| `radam/driver.py`       | Top-level loop, LR schedule, logging           |
| `radam/__main__.py`     | `python -m radam ...` CLI                      |


## Benchmark harness integration

This package is registered as the `radam` implementation in
`benchmarks/lib/registry.py`.  The companion `rlbfgs` package is
registered separately as `rlbfgs`.

```bash
python3 benchmarks/run.py list                         # both show up
python3 benchmarks/run.py validate --impl radam
python3 benchmarks/run.py validate --impl quimb-dmrg1,radam,rlbfgs   # head-to-head
```

**Small-scale proof-of-concept only.**  Pure CPU numpy; the challenge
grid in `run_mi300x_challenge.py` (chi=64-512, L=50-500) is out of
scope.  Per-model defaults are tuned for the small grid; benchmark-
grade convergence at larger sizes requires re-tuning, and no GPU port
exists.


## Things to watch

* **Learning rate**: R-Adam is sensitive to `lr`.  Too large and the
  retraction distorts the state; too small and the optimizer stalls.
  Start with `1e-3` -- `1e-2` and reduce with cosine annealing.
* **Convergence**: track the Riemannian gradient norm, not just the
  energy.  The energy may plateau long before the state is at a
  critical point.
* **Gauge drift**: the tangent-space projection assumes right-canonical
  gauge.  The retraction re-canonicalizes every step, so drift is
  bounded.  If you skip retraction you must expect the projection to
  degrade.
* **Bond dimension**: this retraction does not truncate -- the
  element-wise core update preserves bond dim.  A bond-dim-flexible
  retraction (TT-sum followed by SVD compression) could be plugged
  into `retract_and_recanonicalize`.
