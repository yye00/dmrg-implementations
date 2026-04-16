# radam -- Riemannian Adam **and Riemannian L-BFGS** for MPS / Tensor Train (CPU, numpy)

A self-contained CPU + numpy implementation of two manifold optimizers
on the fixed-rank Tensor Train (MPS) manifold:

* **R-Adam** -- first-order, momentum + scalar variance, simultaneous
  all-core updates (the original prompt).
* **R-LBFGS** -- quasi-second-order, two-loop recursion lifted to the
  manifold via vector transport, with Strong-Wolfe line search,
  cautious update, and a metric-preconditioning step that uses the
  per-bond ``L_norm[i]`` weights to make the L-BFGS Hessian estimate
  consistent with the manifold's induced metric.
* **Warm-start** driver: short R-Adam warmup followed by
  metric-preconditioned R-LBFGS polish.

No `quimb`, no `torch`, no `jax` -- just `numpy`.

These are *not* DMRG variants.  There is no local eigensolve, no
two-site update, and no sweeping.  Every epoch computes the Euclidean
gradient at every MPS core simultaneously, projects it onto the
tangent space of the TT manifold, updates either Adam moments
(R-Adam) or the L-BFGS Hessian estimate from past gradient pairs
(R-LBFGS), and retracts back to the manifold by re-canonicalization.


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

See the math write-up at the top of the source files
(`radam/gradient.py`, `radam/projection.py`, `radam/retraction.py`,
`radam/optimizer.py`).


## Algorithms

### R-Adam step

1. **Euclidean gradient** `grad_i = dE / dA_i*` for every site, via
   left/right MPO and norm environments (`radam/gradient.py`).
2. **Tangent-space projection** onto the right-canonical gauge
   (`radam/projection.py`):
   `G_i <- V_i - (V_i A_i^H) A_i` for `i >= 1`.
3. **Vector transport** of the previous momentum by re-applying the
   current projection.
4. **First-moment update**:
   `M_i <- beta1 * M_i + (1 - beta1) * G_i`.
5. **Scalar second-moment update**:
   `v <- beta2 * v + (1 - beta2) * sum_i || G_i ||_F^2`.
6. **Bias correction** (standard Adam).
7. **Tangent step**
   `Delta_i = -lr * M_hat_i / (sqrt(v_hat) + eps)`.
8. **Retraction**: `X <- right_canonicalize([A_i + Delta_i]);
   X[0] <- X[0] / sqrt(<X|X>)` (the explicit normalization at the end
   prevents radial drift, which otherwise corrupts L-BFGS curvature
   pairs across iterations).

Convergence criterion: `sqrt(v_hat)` / Riemannian gradient norm below
`tol`.

### R-LBFGS step (with metric preconditioning)

1. Compute the Euclidean gradient `grad_i = dE/dA_i*`.
2. **Project** to the right-canonical tangent space:
   ``g_phys = P_X(grad)``.
3. **Precondition** with the manifold metric weights
   ``L_norm[i]`` (left norm environments, computed alongside the
   gradient).  Because ``L_norm[i]^{-1}`` preserves the gauge-orthogonal
   condition ``V_i A_i^H = 0`` at sites ``i >= 1``, no second
   projection is needed:
   ``g = L_norm^{-1} . g_phys``.
4. **Two-loop recursion** with on-the-fly vector transport (each
   stored ``(s, y)`` pair is re-projected to the current tangent space
   at the time of use).
5. **Strong Wolfe line search** (Nocedal & Wright Algorithm 3.5/3.6).
   The slope condition uses ``g_phys`` (not the preconditioned ``g``)
   because that is what predicts ``dE/dalpha``.
6. **Pair update** ``s = alpha * direction``,
   ``y = g_new - P_{X_new}(g)`` with the **cautious** rule
   (skip if ``<s, y> <= eps * ||s|| * ||y||``).

The convergence criterion uses the un-preconditioned Riemannian
gradient norm ``||g_phys||`` (the physical gradient).

### Warm-start driver

Short R-Adam warmup gets the state into a good basin of attraction;
metric-preconditioned R-LBFGS then delivers quasi-second-order
convergence to high precision.  See ``radam/driver.py::run_warmstart_rlbfgs``.

## Convergence (CPU, numpy, single-threaded BLAS)

Measured on this implementation against ``quimb`` single-site DMRG
running on the same problem:

| Problem                    | DMRG1 chi=20 (ref)       | warm-start R-LBFGS       | gap     | wall  |
|----------------------------|--------------------------|--------------------------|---------|-------|
| Heisenberg L=12 chi=20     | -5.142090632840526       | -5.142090628199          | 4.6e-9  | ~6 min |
| Josephson  L=8  chi=20 d=5 | -2.84379784155192        | -2.84379784018189        | 1.4e-9  | ~25 min |

The Josephson run uses ``polish_ridge=1e-4`` (the random
initialization can have rank-deficient ``L_norm[i]``, so a generous
ridge keeps the preconditioner stable) and 800 R-Adam warmup epochs
followed by 1500 R-LBFGS polish epochs.  See
``benchmarks/lib/runners/radam_runner.py::_DEFAULT_WARMSTART_HYPERPARAMS``
for the exact parameters used by ``--impl radam-warmstart``.

R-LBFGS is **substantially slower** than DMRG1 (DMRG1 reaches the
same energy in ~2 seconds on the Josephson problem); this is a
proof-of-concept that quasi-second-order convergence on the manifold
is achievable with simultaneous all-core updates, *not* a claim of
performance parity.  The motivation for the all-core update pattern
is GPU saturation, not CPU competitiveness.


## Representation

* MPS core: `(chi_L, d, chi_R)` numpy array.
* MPO core: `(mpo_L, mpo_R, d_up, d_dn)`.  `d_up` contracts with the
  **bra** physical index, `d_dn` with the **ket**.

The optimizer keeps `X` in *right-canonical* form with the orthogonality
center at site 0.  The momentum list `M` matches `X` core-shape-for-
core-shape.


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

Adding a new model means producing a list of MPO cores with the shape
convention above.


## Benchmark harness integration

Two implementations are registered in `benchmarks/lib/registry.py`,
both routing through `benchmarks/lib/runners/radam_runner.py`:

* `radam` -- plain Riemannian Adam.  Cheap per epoch, but not
  competitive with DMRG on convergence.
* `radam-warmstart` -- short R-Adam warmup followed by
  metric-preconditioned R-LBFGS polish.  Reaches < 1e-9 of single-site
  DMRG on the small grid.

```bash
# List all implementations:
python3 benchmarks/run.py list

# Plain R-Adam (fast, loose):
python3 benchmarks/run.py validate --impl radam

# Warm-start R-LBFGS (slow, tight: 1e-9-ish):
python3 benchmarks/run.py validate --impl radam-warmstart

# Both at once for head-to-head comparison:
python3 benchmarks/run.py validate --impl quimb-dmrg1,radam,radam-warmstart
```

**Small-scale proof-of-concept only.**  This is pure CPU numpy; the
challenge grid in `run_mi300x_challenge.py` (chi=64-512, L=50-500) is
out of scope.  Per-model defaults in the runner are tuned for the
small problem sizes only; they will not give benchmark-grade
convergence at larger sizes without re-tuning, and no GPU port exists.


## Quickstart

```bash
cd cpu/radam
pip install -e .

# CLI:
python -m radam heisenberg --L 10 --chi 16 --epochs 500 --lr 1e-2
python -m radam tfim --L 10 --chi 16 --epochs 500 --lr 1e-2 --hx 1.0

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

# result["mps"]: list of cores (right-canonical)
# result["energy"], result["grad_norm"], result["history"], ...
```


## Files

| File | Purpose |
| --- | --- |
| `radam/mps.py`          | MPS init, canonical forms, norm, inner product |
| `radam/mpo.py`          | Heisenberg / TFIM / Josephson MPO builders |
| `radam/environments.py` | Left/right MPO and norm environments |
| `radam/gradient.py`     | Euclidean gradient + metric preconditioner |
| `radam/tangent.py`      | Tangent-vector arithmetic helpers |
| `radam/projection.py`   | Tangent-space projection + vector transport |
| `radam/retraction.py`   | Retraction (add delta + re-canonicalize + normalize) |
| `radam/optimizer.py`    | Per-step R-Adam update |
| `radam/lbfgs.py`        | R-LBFGS step, two-loop, Strong Wolfe line search |
| `radam/driver.py`       | Top-level loops, LR schedule, warm-start, logging |
| `radam/__main__.py`     | `python -m radam ...` CLI |
| `tests/`                | Unit + smoke tests (30 passing) |
| `examples/`             | Standalone scripts |


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
