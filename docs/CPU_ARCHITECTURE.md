# CPU DMRG Implementations — Architecture & Tensor Contraction Reference

## Implementations

| Package | Algorithm | Reference |
|---------|-----------|-----------|
| `pdmrg/` | Real-Space Parallel DMRG | Stoudenmire & White 2013 |
| `pdmrg-opt/` | Same + BLAS-3 GPU-prep methods | Stoudenmire & White 2013 |
| `a2dmrg/` | Additive Two-Level DMRG | Grigori & Hassan 2025 |

All three require `np >= 2` (parallel-only algorithms).

---

## PDMRG vs PDMRG-OPT Differences

PDMRG is the reference implementation. PDMRG-OPT is identical except for two BLAS-3 substitutions:

| Component | PDMRG (reference) | PDMRG-OPT (GPU-prep) |
|-----------|-------------------|-------------------|
| Eigensolver | `scipy.sparse.linalg.eigsh` (Lanczos) | `block_davidson` with `eigsh` fallback |
| Canonization | `np.linalg.qr` | `newton_schulz_polar` with QR fallback |
| SVD (local sweep) | `truncated_svd` (exact) | `truncated_svd` (exact) |
| SVD (boundary merge) | `accurate_svd` (exact) | `accurate_svd` (exact) |
| V computation | `compute_v_from_svd` (exact) | `compute_v_from_svd` (exact) |

### Block Davidson (`linalg_utils.py`)
- Subspace expansion with restart (proper Davidson, not subspace iteration)
- Starts from `b=4` trial vectors, expands by adding residual corrections
- Restarts when subspace exceeds `b * 8` vectors
- Deterministic (seed=42), converges to machine precision
- All projection operations use `@` (BLAS-3 gemm)
- Falls back to `eigsh` on any exception

### Newton-Schulz Polar (`linalg_utils.py`)
- Iterative polar decomposition: U_{k+1} = 1/2 U_k (3I - U_k^H U_k)
- Converges to tol=1e-10 (machine precision for float64)
- All operations use `@` (BLAS-3 gemm) — no QR/SVD in the loop
- Falls back to `np.linalg.qr` for wide matrices (m < n, chain boundaries)
- Only affects gauge choice, not physics

### What was removed from PDMRG-OPT
- `rsvd_cholesky` — randomized SVD, errors compound across sweeps
- Was never safe for boundary merges (V = Lambda^-1 inverts singular values)

---

## Tensor Contraction Strategy

All three implementations use hand-coded numpy contractions. No opt_einsum, cotengra, or quimb contraction engine in the hot path.

### Methods Used

| Method | Where | Purpose |
|--------|-------|---------|
| `np.tensordot` | Environment updates, H_eff application | Core workhorse — chained pairwise contractions |
| `np.einsum` | Two-site tensor fusion (`'ijk,klm->ijlm'`) | Simple contractions, test references |
| `@` operator | SVD reconstruction, linear algebra | Matrix-matrix only (not tensor contractions) |

### Index Conventions

```
MPS tensor:     (left_bond, phys, right_bond)         — (chi_L, d, chi_R)
MPO tensor:     (mpo_left, mpo_right, phys_up, phys_down) — (D_L, D_R, d, d)
Two-site theta: (chi_L, d_left, d_right, chi_R)
L environment:  (bra_bond, mpo_bond, ket_bond)        — (chi, D, chi)
R environment:  (bra_bond, mpo_bond, ket_bond)        — (chi, D, chi)
```

A2DMRG uses a different MPO convention: `(mpo_L, phys_out, mpo_R, phys_in)`.

### H_eff Application (Hot Path)

The effective Hamiltonian application contracts 5 tensors (called once per Lanczos/Davidson iteration):

```
result[a,p,q,f] = L[a,w,c] * theta[c,s,r,e] * W_L[w,m,p,s] * W_R[m,n,q,r] * R[f,n,e]
```

Contraction order (outside-in, matches ITensor/TeNPy/Block2):

| Step | Operation | Intermediate | Cost |
|------|-----------|-------------|------|
| 1 | L[a,w,c] x theta[c,s,r,e] | X[a,w,s,r,e] | O(chi^2 D d^2) |
| 2 | X x W_L[w,m,p,s] | Y[a,r,e,m,p] | O(chi D^2 d^3) |
| 3 | Y x W_R[m,n,q,r] | Z[a,e,p,n,q] | O(chi D^2 d^3) |
| 4 | Z x R[f,n,e] | result[a,p,q,f] | O(chi^2 D d^2) |

Dominant cost: **O(chi^2 D d^2)** per application. This is the textbook-optimal ordering for chi >> d, D.

### Environment Updates

Left environment update (3 tensors, called once per sweep step):

```
L_new[a',w',c'] = L[a,w,c] * A*[a,s',a'] * W[w,w',s',s] * A[c,s,c']
```

| Step | Operation | Cost |
|------|-----------|------|
| 1 | L[a,w,c] x A[c,s,c'] | O(chi^2 D d) |
| 2 | X x W[w,w',s',s] | O(chi D^2 d^2) |
| 3 | Y x A*[a,s',a'] | O(chi^2 D d) |

Right environment update is symmetric.

### Contraction Path Optimality

The hand-coded paths follow the known-optimal ordering for standard DMRG (chi >> d, D moderate). However:

1. **No formal guarantee** — paths were chosen by convention, not by an optimizer
2. **No fusion** — each `tensordot` allocates an intermediate array
3. **No BLAS-3 batching** — each contraction is a separate gemm call
4. **No path verification** — opt_einsum/cotengra could validate or improve

For the GPU port, contraction paths should be verified with cotengra and potentially fused into single hipTensor calls.

---

## Algorithm Structure

### PDMRG / PDMRG-OPT Sweep Pattern (Stoudenmire & White 2013)

```
Phase 0: Serial warmup (quimb DMRG2 on rank 0)
Phase 1: Distribute MPS across ranks with V = Lambda^-1

Per sweep:
  1. Local optimization sweeps (parallel, independent)
     - Standard 2-site DMRG within each rank's block
     - Staggered: even ranks sweep right, odd ranks sweep left
  2. Merge at even boundaries (0<->1, 2<->3, ...)
     - Exchange boundary tensors via MPI
     - Compute V = Lambda^-1 from exact SVD
     - Two-site optimization at boundary bond
  3. Local optimization sweeps (opposite direction)
  4. Merge at odd boundaries (1<->2, 3<->4, ...)
  5. Convergence check (global energy via MPI Allreduce)
```

### A2DMRG Sweep Pattern (Grigori & Hassan 2025)

```
Phase 0: Serial warmup (quimb DMRG2 on rank 0)
Phase 1: Distribute MPS across ranks

Per sweep:
  1. Local micro-steps (parallel, independent)
     - One-site or two-site optimization within blocks
  2. Coarse-space construction
     - Build overlap and Hamiltonian matrices across ranks
     - Solve generalized eigenvalue problem (SVD-regularized)
  3. Linear combination of local solutions
  4. Global compression (quimb MPS compress)
  5. Convergence check
```

---

## Precision Policy

- **Never single precision** — all arrays are float64 or complex128
- A2DMRG validates dtype at entry: `ValueError` if not float64/complex128
- SVD for boundary merges always uses `accurate_svd` (recursive refinement for small singular values)
- V = Lambda^-1 uses `compute_v_from_svd` with regularization floor at 1e-12

---

## File Layout

```
pdmrg/pdmrg/
  dmrg.py              — Main algorithm (942 lines)
  mps/canonical.py     — QR-based canonization, quimb conversion
  numerics/
    eigensolver.py     — eigsh wrapper
    effective_ham.py   — H_eff LinearOperator (tensordot chain)
    accurate_svd.py    — Exact SVD with recursive refinement
  environments/
    update.py          — L/R environment update (tensordot chain)
  parallel/
    merge.py           — Boundary merge (V = Lambda^-1)
    distribute.py      — MPS distribution across ranks
    communication.py   — MPI exchange utilities

pdmrg-opt/pdmrg/
  (same structure as pdmrg, plus:)
  numerics/linalg_utils.py  — block_davidson, newton_schulz_polar

a2dmrg/a2dmrg/
  dmrg.py              — Main algorithm
  numerics/
    effective_ham.py   — H_eff (different MPO convention)
    local_microstep.py — Local optimization steps
    coarse_eigenvalue.py — Coarse-space GEV solver
  environments/
    environment.py     — L/R environment construction
```
