#ifndef DMRG_GPU_H
#define DMRG_GPU_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include "scalar_traits.h"

/**
 * GPU-native DMRG - Single-site optimization
 *
 * Templated on Scalar: double (real) or cuDoubleComplex (complex128).
 *
 * ALL tensor contractions on GPU via cuBLAS gemm.
 * ALL linear algebra on GPU via cuBLAS/cuSOLVER.
 * Only CPU work: control flow, convergence checks on scalars,
 *                small tridiagonal eigensolve (Lanczos, ~100 elements),
 *                loops over small MPO bond dimension (D=5, d=2) to dispatch GEMMs.
 */
template<typename Scalar>
class DMRGGPU {
    using Traits = ScalarTraits<Scalar>;
    using RealType = typename Traits::RealType;

public:
    DMRGGPU(int L, int d, int chi_max, int D_mpo, double tol = 1e-10);
    ~DMRGGPU();

    double run(int n_sweeps);

    void initialize_mps_random(double scale = 0.1);
    void initialize_mps_product();
    void initialize_mps_neel();
    void set_mpo(const std::vector<Scalar*>& h_mpo_tensors);

    double get_energy() const { return energy_; }
    void get_mps(std::vector<std::vector<Scalar>>& h_mps) const;
    int chi_L(int site) const { return bond_dims_[site]; }
    int chi_R(int site) const { return bond_dims_[site + 1]; }
    void set_quiet(bool) {}  // no-op, all output removed except final summary

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
    cudaStream_t stream_;
    cublasHandle_t cublas_h_;
    cusolverDnHandle_t cusolver_h_;

    // Contraction intermediates
    Scalar* d_T1_;
    Scalar* d_T2_;

    // Lanczos workspace (pre-allocated)
    Scalar* d_theta_;
    Scalar* d_heff_result_;
    Scalar* d_lanczos_v_;
    Scalar* d_ritz_coeffs_;  // Ritz coefficients converted to Scalar for gemv
    int theta_size_max_;
    int max_lanczos_iter_;

    // Device scalars for sync-free Lanczos (device pointer mode)
    Scalar* d_dot_result_;
    RealType* d_nrm2_result_;
    Scalar* d_neg_alpha_;
    Scalar* d_neg_overlap_;
    RealType* d_inv_nrm_;
    RealType* d_alpha_dev_;     // alpha[iter] on device
    RealType* d_beta_dev_;      // beta[iter] on device
    Scalar* d_neg_beta_scalars_; // -beta[iter] as Scalar for axpy
    Scalar* d_const_one_;
    Scalar* d_const_zero_;
    Scalar* d_const_neg_one_;

    // Batched GEMM pointer arrays (on device)
    Scalar** d_batch_A_;
    Scalar** d_batch_B_;
    Scalar** d_batch_C_;

    // SVD workspace (pre-allocated at max size)
    Scalar* d_svd_A_;
    Scalar* d_svd_U_;
    RealType* d_svd_S_;      // singular values always real
    Scalar* d_svd_Vh_;
    Scalar* d_svd_work_;
    RealType* d_svd_rwork_;  // rwork for complex SVD
    int* d_svd_info_;
    int svd_lwork_;          // cuSOLVER workspace size

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
#include "dmrg_gpu_impl.h"

#endif // DMRG_GPU_H
