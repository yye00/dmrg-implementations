#ifndef DMRG2_GPU_BASE_H
#define DMRG2_GPU_BASE_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <string>
#include "scalar_traits.h"

/**
 * GPU-native two-site DMRG — NAIVE BASELINE
 *
 * Unoptimized reference implementation of two-site DMRG on GPU.
 * Differences from DMRG2GPU (dmrg2-gpu):
 *   - No fused two-site MPO (d_WW): apply_heff contracts W_left and W_right
 *     in separate steps via single GEMM calls.
 *   - No gemm_batched: per-(w,s) single rocBLAS gemm in for-loops.
 *   - No device-pointer-mode Lanczos: host-pointer mode throughout, CPU
 *     LAPACK dstev for the tridiagonal eigenproblem.
 *   - No custom kernels (except the complex-conjugate helper needed for
 *     correctness of the bra contraction on complex envs).
 *   - rocSOLVER gesvd followed by host-side truncation + host-side scaling.
 *
 * Hard-coded defaults: tolerances, Lanczos iteration counts, and algorithm
 * choices are baked in — there is no CLI or API for toggling optimizations.
 *
 * Templated on Scalar: double (real) or hipDoubleComplex (complex128).
 */
template<typename Scalar>
class DMRG2GPUBase {
    using Traits = ScalarTraits<Scalar>;
    using RealType = typename Traits::RealType;

public:
    DMRG2GPUBase(int L, int d, int chi_max, int D_mpo, double tol = 1e-10);
    ~DMRG2GPUBase();

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

    // Bond dimensions
    std::vector<int> bond_dims_;

    // GPU tensor data
    std::vector<Scalar*> d_mps_tensors_;
    std::vector<Scalar*> d_mpo_tensors_;
    std::vector<Scalar*> d_L_envs_;
    std::vector<Scalar*> d_R_envs_;

    // W_left / W_right matrices for single-site env updates AND apply_heff
    std::vector<Scalar*> d_W_left_;
    std::vector<Scalar*> d_W_right_;

    std::vector<int> L_env_alloc_chi_;
    std::vector<int> R_env_alloc_chi_;

    // GPU handles
    hipStream_t stream_;
    rocblas_handle rocblas_h_;

    // Contraction intermediates — larger than dmrg-gpu-base: must hold
    // (chi*d) × (D*d*chi) for the two-site H_eff path.
    Scalar* d_T1_;
    Scalar* d_T2_;
    Scalar* d_T3_;

    // Lanczos workspace
    Scalar* d_theta_;
    Scalar* d_heff_result_;
    Scalar* d_lanczos_v_;
    Scalar* d_ritz_coeffs_;
    int theta_size_max_;
    int max_lanczos_iter_;

    // Host-side Lanczos tridiagonal (CPU LAPACK dstev)
    std::vector<double> h_alpha_;
    std::vector<double> h_beta_;
    std::vector<double> h_steqr_work_;
    std::vector<double> h_steqr_Z_;

    // SVD workspace
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

    void form_theta_two_site(int site);
    void apply_heff_two_site(int site, const Scalar* d_theta_in, Scalar* d_result);
    double lanczos_eigensolver(int site, Scalar* d_theta, int theta_size);
    void svd_split(int site, Scalar* d_theta, char direction);

    double optimize_bond(int site, char direction);
    double sweep_left_to_right();
    double sweep_right_to_left();

    void allocate_mps_tensor(int site, int cL, int cR);
    void free_gpu_resources();
};

// Include template implementation
#include "dmrg2_gpu_base_impl.h"

#endif // DMRG2_GPU_BASE_H
