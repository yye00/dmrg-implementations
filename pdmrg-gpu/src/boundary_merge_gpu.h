#pragma once

#include <hip/hip_runtime.h>
#include "stream_segment.h"
#include "accurate_svd_gpu.h"
#include "heff_optimized_gpu.h"

/**
 * BoundaryMergeGPU: Implements exact SVD boundary reconciliation
 *
 * Maps CPU merge_boundary_tensors from pdmrg/parallel/merge.py to GPU.
 * Implements the critical V = Lambda^-1 prescription (Stoudenmire & White Eq. 5):
 *
 *   theta = psi_left . diag(V) . psi_right   (Eq. 5)
 *
 * Then optimizes theta with Lanczos eigensolver, splits with exact SVD,
 * and computes new V = 1/S for next iteration.
 *
 * CRITICAL: Uses exact SVD (AccurateSVD_GPU), NOT rSVD!
 */
class BoundaryMergeGPU {
public:
    /**
     * Constructor
     *
     * @param max_bond   Maximum bond dimension after SVD truncation
     * @param max_iter   Maximum Lanczos iterations for optimization
     * @param tol        Lanczos convergence tolerance
     */
    BoundaryMergeGPU(int max_bond, int max_iter = 30, double tol = 1e-10);

    ~BoundaryMergeGPU();

    /**
     * Main merge function - mirrors CPU merge_boundary_tensors
     *
     * Implements the 4-step boundary reconciliation:
     *   1. Form theta = psi_left . diag(V) . psi_right
     *   2. Optimize theta with Lanczos (or skip if converged)
     *   3. Split with exact SVD
     *   4. Compute V_new = 1/S (with regularization)
     *
     * @param left               Left boundary data (from left segment)
     * @param right              Right boundary data (from right segment)
     * @param energy             [out] Energy from optimization
     * @param trunc_err          [out] Truncation error from SVD
     * @param skip_optimization  If true, skip Lanczos (just compute energy)
     * @param stream             HIP stream for async operations
     */
    void merge(BoundaryData* left, BoundaryData* right,
               double& energy, double& trunc_err,
               bool skip_optimization = false,
               hipStream_t stream = 0);

private:
    // ============================================================================
    // Step 1: Form theta = psi_left . diag(V) . psi_right
    // ============================================================================

    /**
     * Form two-site wavefunction from boundary tensors
     *
     * Implements Eq. 5: theta[a,s1,s2,b] = psi_left[a,s1,c] * V[c] * psi_right[c,s2,b]
     *
     * This is a critical step - V bridges the independently-evolved segments.
     */
    void form_theta_from_boundary(
        const double* d_psi_left,    // (chi_L, d, chi_bond)
        const double* d_psi_right,   // (chi_bond, d, chi_R)
        const double* d_V,           // (chi_bond,)
        double* d_theta,             // [out] (chi_L, d, d, chi_R)
        int chi_L, int d, int chi_R, int chi_bond,
        hipStream_t stream
    );

    // ============================================================================
    // Step 2: Optimize with Lanczos eigensolver
    // ============================================================================

    /**
     * Optimize two-site wavefunction using Lanczos
     *
     * Finds ground state of H_eff|theta> where:
     *   H_eff = L_env × W_left × W_right × R_env
     *
     * If skip_optimization=true, just computes energy without optimization.
     */
    void optimize_two_site_gpu(
        const double* d_L_env,       // (D_mpo, chi_L, chi_L)
        const double* d_R_env,       // (D_mpo, chi_R, chi_R)
        const double* d_W_left,      // (D_mpo, d, d, D_mpo)
        const double* d_W_right,     // (D_mpo, d, d, D_mpo)
        double* d_theta,             // [in/out] (chi_L, d, d, chi_R)
        double& energy,              // [out]
        int chi_L, int d, int chi_R, int D_mpo,
        bool skip_optimization,
        hipStream_t stream
    );

    // ============================================================================
    // Step 3: Split with exact SVD
    // ============================================================================

    /**
     * Split optimized theta using exact SVD
     *
     * M = theta.reshape(chi_L*d, d*chi_R)
     * U, S, Vh = accurate_svd(M)
     * Truncate to max_bond
     * A_left = U[:, :k].reshape(chi_L, d, k)
     * A_right = (S @ Vh)[:k, :].reshape(k, d, chi_R)
     *
     * CRITICAL: Uses AccurateSVD_GPU (exact), not rSVD!
     */
    void split_with_svd(
        const double* d_theta_opt,   // (chi_L, d, d, chi_R)
        double* d_A_left_new,        // [out] (chi_L, d, k)
        double* d_A_right_new,       // [out] (k, d, chi_R)
        double* d_S,                 // [out] (k,) - singular values
        double& trunc_err,           // [out]
        int& k_out,                  // [out] - truncated bond dim
        int chi_L, int d, int chi_R,
        hipStream_t stream,
        double* d_A_right_canonical  // [out] (k, d, chi_R) - Vh for R_env
    );

    // ============================================================================
    // Step 4: Compute V = 1/S with regularization
    // ============================================================================

    /**
     * Compute V = 1/S for next iteration
     *
     * V[i] = 1 / max(S[i], regularization)
     *
     * Regularization prevents numerical blowup from small singular values.
     * Default: 1e-12 (same as CPU implementation)
     */
    void compute_v_from_s(
        const double* d_S,           // (k,) - singular values
        double* d_V,                 // [out] (k,) - V = 1/S
        int k,
        double regularization = 1e-12,
        hipStream_t stream = 0
    );

    // ============================================================================
    // Helper: Apply H_eff
    // ============================================================================

    /**
     * Apply effective Hamiltonian to theta
     *
     * result = L_env × W_left × theta × W_right × R_env
     *
     * Uses OptimizedHeff from Phase 1.
     */
    void apply_heff(
        const double* d_L_env,
        const double* d_R_env,
        const double* d_W_left,
        const double* d_W_right,
        const double* d_theta,
        double* d_result,
        int chi_L, int d, int chi_R, int D_mpo,
        hipStream_t stream
    );

    // ============================================================================
    // Lanczos Eigensolver
    // ============================================================================

    /**
     * Lanczos algorithm to find ground state
     *
     * Iteratively builds Krylov subspace and solves tridiagonal eigenvalue problem.
     * Returns lowest eigenvalue and corresponding eigenvector.
     */
    void lanczos_eigensolver(
        const double* d_L_env,
        const double* d_R_env,
        const double* d_W_left,
        const double* d_W_right,
        double* d_theta,             // [in/out] initial guess → ground state
        double& energy,              // [out]
        int chi_L, int d, int chi_R, int D_mpo,
        hipStream_t stream
    );

    // ============================================================================
    // Member Variables
    // ============================================================================

    int max_bond_;
    int max_iter_;
    double tol_;

    // Phase 1 components
    AccurateSVD_GPU* svd_;
    OptimizedHeff* heff_;

    // rocBLAS/hipBLAS handle
    rocblas_handle rocblas_h_;

    // Workspace for intermediate tensors
    // Allocated once, reused across merges
    double* d_theta_;           // (chi_L, d, d, chi_R) - max size
    double* d_theta_opt_;       // (chi_L, d, d, chi_R) - optimized
    double* d_M_;               // (chi_L*d, d*chi_R) - reshaped for SVD
    double* d_H_theta_;         // (chi_L, d, d, chi_R) - H|theta>
    double* d_V_psi_right_;     // (chi_bond, d, chi_R) - V * psi_right

    // Lanczos workspace
    double* d_lanczos_v_;       // Lanczos vectors
    double* d_lanczos_alpha_;   // Tridiagonal diagonal
    double* d_lanczos_beta_;    // Tridiagonal off-diagonal

    size_t workspace_size_;
    int max_chi_L_;
    int max_chi_R_;
    int max_d_;
    int max_D_mpo_;

    bool is_initialized_;

    // ============================================================================
    // Initialization
    // ============================================================================

    void allocate_workspace(int chi_L, int d, int chi_R, int D_mpo);
    void free_workspace();
};
