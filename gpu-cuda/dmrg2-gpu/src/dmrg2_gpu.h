#ifndef DMRG2_GPU_H
#define DMRG2_GPU_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <string>
#include "scalar_traits.h"

/**
 * GPU-native DMRG - Two-site optimization
 *
 * Templated on Scalar: double (real) or cuDoubleComplex (complex128).
 *
 * Two-site DMRG optimizes pairs of adjacent MPS tensors, allowing
 * bond dimensions to grow adaptively via SVD truncation.
 *
 * ALL tensor contractions on GPU via cuBLAS gemm.
 * Fused two-site MPO (WW) precomputed for efficient H_eff application.
 */
template<typename Scalar>
class DMRG2GPU {
    using Traits = ScalarTraits<Scalar>;
    using RealType = typename Traits::RealType;

public:
    DMRG2GPU(int L, int d, int chi_max, int D_mpo, double tol = 1e-10);
    ~DMRG2GPU();

    double run(int n_sweeps);

    void initialize_mps_random(double scale = 0.1);
    void initialize_mps_product();
    void initialize_mps_neel();

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

    // W_left[site]: (D*d, d*D) for left env update
    // W_right[site]: (D*d, d*D) for right env update
    std::vector<Scalar*> d_W_left_;
    std::vector<Scalar*> d_W_right_;

    // Fused two-site MPO: WW[bond] is (D*d*d, d*d*D) for bond (site, site+1)
    std::vector<Scalar*> d_WW_;

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
    Scalar* d_ritz_coeffs_;
    int theta_size_max_;
    int max_lanczos_iter_;

    // Device scalars for sync-free Lanczos (device pointer mode)
    Scalar* d_dot_result_;
    RealType* d_nrm2_result_;
    Scalar* d_neg_alpha_;
    Scalar* d_neg_overlap_;
    RealType* d_inv_nrm_;
    RealType* d_alpha_dev_;
    RealType* d_beta_dev_;
    Scalar* d_neg_beta_scalars_;
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
    RealType* d_svd_S_;
    Scalar* d_svd_Vh_;
    RealType* d_svd_rwork_;   // rwork for complex SVD (5*min(m,n) doubles)
    int* d_svd_info_;
    Scalar* d_svd_work_;
    int svd_lwork_;           // cuSOLVER SVD workspace size

    // CPU workspace (for receiving GPU SVD results and truncation/scaling)
    std::vector<Scalar> h_svd_U_, h_svd_Vh_, h_svd_tmp_;
    std::vector<RealType> h_svd_S_;

    // Core algorithm
    void build_initial_environments();
    void update_left_env(int site);
    void update_right_env(int site);
    void ensure_L_env_alloc(int idx, int chi);
    void ensure_R_env_alloc(int idx, int chi);

    void precompute_fused_mpo(const std::vector<Scalar*>& h_mpo_tensors);
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
#include "dmrg2_gpu_impl.h"

#endif // DMRG2_GPU_H
