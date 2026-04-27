#ifndef DMRG_GPU_BASE_H
#define DMRG_GPU_BASE_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include "scalar_traits.h"

/**
 * GPU-native DMRG — NAIVE BASELINE (single-site)
 *
 * Competent first-pass GPU implementation of single-site DMRG. All linear
 * algebra runs on device (rocBLAS GEMM/GEMV/AXPY/DOT/NRM2 in device-pointer
 * mode, rocSOLVER `dsteqr` for the Lanczos tridiagonal eigensolve,
 * rocSOLVER `gesvd_auto` for the SVD, device-side truncation kernels for
 * S-scaling and column extraction). No host-pointer rocBLAS, no CPU
 * LAPACK calls, no per-iteration heap allocations, no host roundtrips of
 * device-resident data.
 *
 * Compared to DMRGGPU (the optimized variant), this baseline omits:
 *   - dual-stream pipelining (apply_heff vs env_update overlap),
 *   - HIP graph capture for the Lanczos inner loop,
 *   - randomized SVD (RSVD),
 *   - batched GEMM and the GpuOpts ablation framework,
 *   - sparse-MPO compaction,
 *   - D_PAD MFMA-friendly padding.
 * The baseline uses single-GEMM-per-pair patterns where the optimized variant
 * uses gemm_batched, a single rocBLAS handle on a single stream, and the
 * standard non-fused Lanczos kernels. It is naive in algorithmic structure
 * but does not waste time on CPU work that has a one-line GPU equivalent.
 *
 * Templated on Scalar: double (real) or hipDoubleComplex (complex128).
 */
template<typename Scalar>
class DMRGGPUBase {
    using Traits = ScalarTraits<Scalar>;
    using RealType = typename Traits::RealType;

public:
    DMRGGPUBase(int L, int d, int chi_max, int D_mpo, double tol = 1e-10);
    ~DMRGGPUBase();

    double run(int n_sweeps);

    void initialize_mps_random(double scale = 0.1);
    void set_mpo(const std::vector<Scalar*>& h_mpo_tensors);

    double get_energy() const { return energy_; }
    void get_mps(std::vector<std::vector<Scalar>>& h_mps) const;
    int chi_L(int site) const { return bond_dims_[site]; }
    int chi_R(int site) const { return bond_dims_[site + 1]; }
    void set_quiet(bool) {}  // no-op

private:
    // System parameters
    int L_, d_, chi_max_, D_mpo_;
    double tol_;
    double energy_;

    // Bond dimensions: bond_dims_[i] = dim of bond between site i-1 and site i
    std::vector<int> bond_dims_;

    // GPU tensor data
    std::vector<Scalar*> d_mps_tensors_;
    std::vector<Scalar*> d_mpo_tensors_;
    std::vector<Scalar*> d_L_envs_;
    std::vector<Scalar*> d_R_envs_;

    // W_left[site]: (D*d, d*D) for left env & H_eff
    // W_right[site]: (D*d, d*D) for right env
    std::vector<Scalar*> d_W_left_;
    std::vector<Scalar*> d_W_right_;

    std::vector<int> L_env_alloc_chi_;
    std::vector<int> R_env_alloc_chi_;

    // GPU handles
    hipStream_t stream_;
    rocblas_handle rocblas_h_;

    // Contraction intermediates
    Scalar* d_T1_;
    Scalar* d_T2_;

    // Lanczos workspace (pre-allocated)
    Scalar* d_theta_;
    Scalar* d_heff_result_;
    Scalar* d_lanczos_v_;
    Scalar* d_ritz_coeffs_;
    int theta_size_max_;
    int max_lanczos_iter_;

    // Device-pointer-mode scratch for rocBLAS BLAS-1 results (one scalar each).
    Scalar*   d_dot_result_;      // <v_i | w>          (per Lanczos iter)
    RealType* d_nrm2_result_;     // ||w||              (per Lanczos iter)
    RealType* d_inv_nrm_;         // 1/||w||            (computed by inv_real_kernel)
    Scalar*   d_neg_alpha_;       // -alpha_i           (axpy multiplier)
    Scalar*   d_neg_overlap_;     // -<v_j | w>         (reorth axpy multiplier)
    Scalar*   d_neg_beta_scalars_;// -beta_i (per iter, indexed array)

    // Per-iteration alpha/beta arrays on device.
    RealType* d_alpha_dev_;       // [max_lanczos_iter_]
    RealType* d_beta_dev_;        // [max_lanczos_iter_]

    // rocSOLVER dsteqr workspaces — fully on-device tridiagonal eigensolve.
    double*      d_steqr_D_;      // diagonal (overwritten with eigenvalues)
    double*      d_steqr_E_;      // subdiagonal (overwritten)
    double*      d_steqr_C_;      // eigenvector matrix (max_iter × max_iter)
    rocblas_int* d_steqr_info_;   // rocsolver info output

    // SVD workspace (pre-allocated at max size). Truncation runs on device via
    // extract_cols_kernel and scale_rows_by_diag_kernel from common/scalar_traits.h;
    // only the singular-value vector S is read back to host (small: <= chi_max
    // doubles per call) for the truncation-rank decision.
    Scalar* d_svd_A_;
    Scalar* d_svd_U_;
    RealType* d_svd_S_;
    Scalar* d_svd_Vh_;
    Scalar* d_svd_work_;          // device-side scratch for S*Vh (or U*S)
    RealType* d_svd_E_;
    int* d_svd_info_;

    // Tiny host buffer used only for the truncation-rank decision (one D2H
    // of the singular values per SVD; <= chi_max * 8 bytes, control-flow scalar).
    std::vector<RealType> h_svd_S_;

    // Core algorithm
    void build_initial_environments();
    void update_left_env(int site);
    void update_right_env(int site);
    void ensure_L_env_alloc(int idx, int chi);
    void ensure_R_env_alloc(int idx, int chi);

    double optimize_site(int site, char direction);
    void form_theta(int site, Scalar* d_theta);
    void apply_heff(int site, const Scalar* d_theta, Scalar* d_result);
    double lanczos_eigensolver(int site, Scalar* d_theta);
    void svd_and_update_mps(int site, Scalar* d_theta, char direction);

    double sweep_left_to_right();
    double sweep_right_to_left();

    void allocate_mps_tensor(int site, int cL, int cR);
    void free_gpu_resources();
};

// Include template implementation
#include "dmrg_gpu_base_impl.h"

#endif // DMRG_GPU_BASE_H
