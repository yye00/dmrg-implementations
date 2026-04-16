# rlbfgs -- Riemannian L-BFGS for MPS / Tensor Train (CPU, numpy)

A self-contained CPU + numpy implementation of the **Riemannian L-BFGS**
(R-LBFGS) quasi-Newton optimizer on the fixed-rank Tensor Train (MPS)
manifold, for ground-state problems on 1D quantum Hamiltonians.

No `quimb`, no `torch`, no `jax` -- just `numpy`.

This is *not* a DMRG variant.  There is no local eigensolve, no
two-site update, and no sweeping.  Every iteration computes the
Euclidean gradient at every MPS core simultaneously, projects it onto
the tangent space, builds a search direction from the standard
L-BFGS two-loop recursion (with vector transport), and retracts back
to the manifold by re-canonicalization.  Once near the minimum, a
metric-preconditioned variant delivers quasi-second-order convergence
at comparable *iteration counts* to DMRG's local eigensolve -- the
trade is that each iteration touches all cores at once, keeping the
update pattern GPU-friendly.

The companion `radam` package implements the first-order Riemannian
Adam optimizer for the same manifold; it can serve as a warmup for
`rlbfgs` but is not required -- `run_rlbfgs_warmstart` does its own
warmup using unpreconditioned L-BFGS.

## Algorithm

Each R-LBFGS iteration performs:

1. **Riemannian gradient**: `g = P_X(dE/dA*)` via left/right MPO and
   norm environments.
2. **Two-loop recursion** (Nocedal & Wright 7.4) over the last `m`
   (`s`, `y`) pairs, with each pair **transported** to the current
   tangent space at the time of use (`ρ` recomputed after transport).
3. **Descent guard**: if `<g, d> >= 0` the Hessian estimate is
   non-PSD; reset history and fall back to `d = -g`.
4. **Strong Wolfe line search** (Nocedal & Wright Alg 3.5/3.6) with
   NaN/degeneracy guards.  The slope condition uses the
   *un-preconditioned* physical gradient since that predicts
   `dE/dα`.
5. **Cautious update**: skip `(s, y)` if
   `<s, y> <= eps * ‖s‖ * ‖y‖`.
6. **Retraction**: core-wise `A_i + Δ_i`, then `right_canonicalize`
   and normalize `<X|X> = 1` (the explicit normalization prevents
   radial drift that would corrupt the L-BFGS curvature pairs).

The **preconditioned** variant (enabled in the polish phase of
`run_rlbfgs_warmstart`) applies `L_norm[i]^{-1}` to the projected
gradient at each site.  This is the natural Riemannian metric on the
TT manifold at a right-canonical gauge and makes the L-BFGS Hessian
estimate consistent with the manifold's curvature.  Because
`L_norm[i]^{-1}` preserves the gauge-orthogonality condition
`V_i @ A_i^† = 0` at sites `i >= 1`, no second projection is needed.

## Convergence (CPU, numpy, single-threaded BLAS)

Measured against `quimb` single-site DMRG at the same bond dim:

| Problem                    | DMRG1 chi=20 (ref)  | R-LBFGS warmstart  | gap     |
|----------------------------|---------------------|--------------------|---------|
| Heisenberg L=12 chi=20     | -5.142090632840526  | -5.14209063...     | ~5e-9   |
| Josephson  L=8  chi=20 d=5 | -2.84379784155192   | -2.84379784...     | ~1e-9   |

R-LBFGS is **substantially slower than DMRG1 on CPU** (DMRG1 hits the
same energy in ~2 s on Josephson; R-LBFGS takes minutes).  The
motivation for the all-cores-simultaneous update pattern is GPU
saturation, not CPU performance parity; this package is a CPU proof
of concept that the convergence properties hold.

## Supported Hamiltonians

* `build_heisenberg_mpo(L, j, bz)` -- XXX chain.
* `build_tfim_mpo(L, j, hx)` -- transverse-field Ising.
* `build_josephson_mpo(L, E_J, E_C, mu, n_max, phi_ext)` -- Josephson
  junction array with external flux (complex128).

## Quickstart

```bash
cd cpu/rlbfgs
pip install -e .

# CLI:
python -m rlbfgs heisenberg --L 12 --chi 20 --warmup-epochs 300 --polish-epochs 400
python -m rlbfgs josephson  --L 8  --chi 20 --n-max 2 \
    --warmup-epochs 800 --polish-epochs 2000 --polish-ridge 1e-4

# Example:
python examples/run_josephson.py

# Tests:
pytest
```

## Programmatic use

```python
from rlbfgs.driver import run_josephson

result = run_josephson(
    L=8, chi=20, n_max=2,
    warmup_epochs=800, polish_epochs=2000,
    polish_ridge=1e-4, polish_tol=1e-13,
    seed=0,
)
# result["mps"], result["energy"], result["grad_norm"], ...
```

You can also pass a pre-optimized MPS as `initial_mps=` (e.g., from
an R-Adam warmup or a prior DMRG run) to `run_rlbfgs_warmstart`.

## Files

| File                        | Purpose                                                |
|-----------------------------|--------------------------------------------------------|
| `rlbfgs/mps.py`             | MPS init, canonical forms, norm, inner product         |
| `rlbfgs/mpo.py`             | Heisenberg / TFIM / Josephson MPO builders             |
| `rlbfgs/environments.py`    | Left/right MPO and norm environments                   |
| `rlbfgs/gradient.py`        | Euclidean gradient + metric preconditioner             |
| `rlbfgs/tangent.py`         | Tangent-vector arithmetic helpers                      |
| `rlbfgs/projection.py`      | Tangent-space projection + vector transport            |
| `rlbfgs/retraction.py`      | Retraction (add delta + re-canonicalize + normalize)   |
| `rlbfgs/optimizer.py`       | R-LBFGS step, two-loop, Strong Wolfe line search       |
| `rlbfgs/driver.py`          | `run_rlbfgs`, `run_rlbfgs_warmstart` (two-phase)       |
| `rlbfgs/__main__.py`        | `python -m rlbfgs ...` CLI                             |

## Benchmark integration

This package is registered as the `rlbfgs` implementation in
`benchmarks/lib/registry.py` via `benchmarks/lib/runners/rlbfgs_runner.py`.
Runs only on the small benchmark grid; the challenge grid is out of
scope.

```bash
python3 benchmarks/run.py validate --impl rlbfgs
python3 benchmarks/run.py validate --impl quimb-dmrg1,radam,rlbfgs   # head-to-head
```
