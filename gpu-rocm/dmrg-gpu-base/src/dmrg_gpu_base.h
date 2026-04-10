#ifndef DMRG_GPU_BASE_H
#define DMRG_GPU_BASE_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include "scalar_traits.h"

/**
 * GPU-native DMRG — NAIVE BASELINE (single-site)
 *
 * Unoptimized reference implementation of single-site DMRG on GPU.
 * Uses only rocBLAS single-GEMM calls (no gemm_batched), host-pointer mode
 * throughout, CPU LAPACK dstev for the Lanczos tridiagonal eigensolve, and
 * rocSOLVER gesvd followed by host-side truncation for the SVD. No fused
 * MPO tensors, no custom kernels (except the complex-conjugate helper that
 * is required for correctness of the bra contraction), no device-pointer
 * Lanczos, no batched pointer setup.
 *
 * This class exists to provide a reference baseline for measuring the
 * speedup of the optimizations used in DMRGGPU (dmrg-gpu).
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

    // Host-side Lanczos tridiagonal (CPU LAPACK dstev)
    std::vector<double> h_alpha_;
    std::vector<double> h_beta_;
    std::vector<double> h_steqr_work_;  // workspace for dstev
    std::vector<double> h_steqr_Z_;     // eigenvectors from dstev

    // SVD workspace (pre-allocated at max size)
    Scalar* d_svd_A_;
    Scalar* d_svd_U_;
    RealType* d_svd_S_;
    Scalar* d_svd_Vh_;
    RealType* d_svd_E_;
    int* d_svd_info_;

    // Host workspace for SVD results (copied back from GPU)
    std::vector<Scalar> h_svd_U_, h_svd_Vh_, h_svd_tmp_;
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
