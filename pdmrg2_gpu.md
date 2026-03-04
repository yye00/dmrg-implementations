
---

# Application Specification: GEMM-Optimized PDMRG (CPU Optimization Phase)

**Note:** This document describes CPU-level optimizations using GEMM-heavy algorithms. It is NOT a GPU implementation roadmap, but rather prepares the mathematical architecture for potential future GPU porting.

## 1. Objective

Refactor the internal local-block optimization and gauge-shifting mechanics of the existing Parallel DMRG (PDMRG) implementation. The goal is to replace sequential, memory-bandwidth-bound linear algebra operations (Lanczos, standard QR, standard SVD) with GEMM-heavy (BLAS-3) algorithms. This prepares the mathematical architecture for future GPU porting while immediately improving CPU cache performance.

## 2. Component Upgrades

### 2.1. The Local Solver: Block-Davidson (LOBPCG)

**Current State:** The local 2-site optimization step uses a single-vector Krylov solver (e.g., `scipy.sparse.linalg.eigsh`), which relies on matrix-vector products.
**Target State:** Implement a Block-Davidson or Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG) solver.
**Implementation Directives:**

* **Input Modification:** Instead of feeding a single 1D tensor representing the current state guess, initialize a block of $b$ orthogonal random vectors (e.g., $b=4$ or $b=8$), stored as a dense matrix $X$.
* **Hamiltonian Application:** Modify the `LinearOperator` or tensor contraction function that applies the effective Hamiltonian $H_{eff}$. It must now accept the block $X$ and perform a batched tensor contraction to compute $Y = H_{eff} X$.
* **Rayleigh-Ritz Projection:** * Compute the small projected Hamiltonian: 
$$H_{proj} = X^\dagger Y$$


* Solve the dense eigenvalue problem for this small $b \times b$ matrix using `scipy.linalg.eigh`.
* Update the block $X$ with the lowest eigenvectors and compute the residual. Iterate until convergence.



### 2.2. Matrix-Free Gauge Shift: Newton-Schulz Polar Decomposition

**Current State:** Shifting the orthogonality center during internal sweeps uses standard QR or SVD (e.g., `scipy.linalg.qr`).
**Target State:** Replace this with the iterative Newton-Schulz Polar Decomposition to isolate the orthogonal (unitary) tensor $U$ and the positive semi-definite tensor $P$.
**Implementation Directives:**

* **Scaling:** Given the flattened $m \times n$ tensor $A$, safely scale it to ensure spectral radius requirements:

$$U_0 = \frac{A}{\|A\|_F}$$


* **Iteration Loop:** Implement the following Newton-Schulz update strictly using `numpy.matmul` or the `@` operator:

$$U_{k+1} = \frac{1}{2} U_k (3I - U_k^\dagger U_k)$$


* **Convergence:** Halt the loop when the Frobenius norm of the difference $\|U_{k+1} - U_k\|_F$ falls below a specified tolerance (e.g., 1e-10).
* **Extraction:** Compute the remaining tensor (which will be absorbed into the next site):

$$P = U^\dagger A$$



### 2.3. Bond Truncation: Randomized SVD with Cholesky-QR2

**Current State:** Bond truncation utilizes an exact, deterministic dense SVD (`scipy.linalg.svd`).
**Target State:** Implement Randomized SVD (rSVD) leveraging Cholesky-QR2 to maximize matrix-multiplication density.
**Implementation Directives:**

* **Sketching:** For an $N \times N$ tensor $A$ with a target truncation rank $m$ and oversampling parameter $p$ (e.g., $p=10$), generate a random Gaussian matrix $\Omega$ of size $N \times (m+p)$. Compute the sketch:

$$Y = A \Omega$$


* **Cholesky-QR2:** Avoid `scipy.linalg.qr`. Instead, orthonormalize $Y$ using the Gram matrix:
* $$G = Y^\dagger Y$$


* Compute the Cholesky decomposition: 
$$L = \text{cholesky}(G)$$


* Get the orthogonal basis: 
$$Q = Y (L^\dagger)^{-1}$$


* *Stability Check:* If $Y$ is ill-conditioned, repeat the $G$ and $L$ steps once more on $Q$ (this constitutes Cholesky-QR2) to guarantee orthogonality.


* **Core Projection:** Project $A$ into the small subspace:

$$B = Q^\dagger A$$


* **Small SVD:** Perform a standard `scipy.linalg.svd` on the tiny $(m+p) \times N$ matrix $B$ to yield $\tilde{U}$, $\Sigma$, and $V^\dagger$.
* **Reconstruction:** Form the final truncated left tensor:

$$U = Q \tilde{U}$$



## 3. Critical Exception: The MPI Boundary Merge

**Context:** When the spatial blocks meet in the PDMRG protocol, they merge using a diagonal matrix $V = \Lambda^{-1}$, where $\Lambda$ contains the singular values of the shared bond.


**Constraint:** As noted in the foundational PDMRG literature, standard vendor-provided SVD routines often yield poor relative accuracy for the smallest singular values. Because these tiny values are inverted ($\lambda^{-1} \gg 1$), numerical errors blow up massively. Furthermore, introducing $V = \Lambda^{-1}$ at shared bonds amplifies these relative errors.
**Implementation Directive:** * **Do NOT use the rSVD for the boundary merge step.**

* The code must branch here. For all *internal* block sweeps, use the fast rSVD outlined in Section 2.3.
* When executing the cross-node boundary merge requiring $V = \Lambda^{-1}$, the application must fall back to a high-precision, exact SVD (or the recursive SVD algorithm described by Stoudenmire & White) to ensure the tail-end singular values are calculated with strict relative accuracy before inversion.

## 4. Integration Guidelines for Claude Code

1. **Isolate Linear Algebra:** Create a `linalg_utils.py` module containing `block_davidson`, `newton_schulz_polar`, and `rsvd_cholesky`.
2. **Monkey-Patching `quimb`:** Instead of rewriting `quimb`'s core tensor objects, subclass or wrap the specific DMRG sweep methods (e.g., `_canonize_window` and `_truncate_window`) to intercept the linear algebra calls and route them to your new utilities.
3. **Tolerances:** Expose the Newton-Schulz convergence tolerance and the rSVD oversampling parameter $p$ as top-level configuration variables for easy tuning.

---
