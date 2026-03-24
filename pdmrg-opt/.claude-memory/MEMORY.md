
## Session 2026-02-25 — pdmrg-opt GEMM-optimization implemented

- Created `pdmrg-opt/pdmrg/numerics/linalg_utils.py`: `block_davidson` (LOBPCG-style, b=4), `newton_schulz_polar` (NS iteration, tol=NS_TOL=1e-10), `rsvd_cholesky` (sketch→Cholesky-QR2→small SVD, p=RSVD_OVERSAMPLE=10)
- Bug fixed in `_cholesky_qr`: Cholesky solve must use `lower=True` (solve `L @ Q† = Y†`), not `lower=False` (was computing Q = Y L⁻¹ instead of correct Q = Y L†⁻¹)
- Updated `eigensolver.py` (block_davidson replaces eigsh), `canonical.py` (newton_schulz_polar replaces QR), `dmrg.py` (rsvd_cholesky in local_sweep; NS in canonize_block/rebuild_boundary functions)
- Critical exception preserved: `merge.py` boundary merges still use `accurate_svd` (exact recursive SVD) for V = Λ⁻¹ inversion
- Benchmark result (L=16, bond_dim=30): PDMRG-OPT np=1 is **1.58× faster** than PDMRG np=1 (1.35s vs 2.13s); all |ΔE| < 1e-10; scaling np=1→4 gives 1.48× for both variants
