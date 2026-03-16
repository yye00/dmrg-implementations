#ifndef DMRG2_GPU_H
#define DMRG2_GPU_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <string>
#include "scalar_traits.h"

/**
 * GPU-native DMRG - Two-site optimization
 *
 * Templated on Scalar: double (real) or hipDoubleComplex (complex128).
 *
 * Two-site DMRG optimizes pairs of adjacent MPS tensors, allowing
 * bond dimensions to grow adaptively via SVD truncation.
 *
 * ALL tensor contractions on GPU via rocBLAS gemm.
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
    void set_cpu_svd(bool use_cpu) { use_cpu_svd_ = use_cpu; }
    void set_rsvd(bool use_rsvd) { use_rsvd_ = use_rsvd; }

    int chi_L(int site) const { return bond_dims_[site]; }
    int chi_R(int site) const { return bond_dims_[site + 1]; }

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

    // Batched GEMM pointer arrays (on device)
    Scalar** d_batch_A_;
    Scalar** d_batch_B_;
    Scalar** d_batch_C_;

    // SVD workspace (pre-allocated at max size)
    Scalar* d_svd_A_;
    Scalar* d_svd_U_;
    RealType* d_svd_S_;
    Scalar* d_svd_Vh_;
    RealType* d_svd_E_;
    int* d_svd_info_;
    Scalar* d_svd_work_;

    // CPU SVD workspace (pre-allocated)
    std::vector<Scalar> h_svd_A_, h_svd_U_, h_svd_Vh_, h_svd_work_, h_svd_tmp_;
    std::vector<RealType> h_svd_S_;
    std::vector<RealType> h_svd_rwork_;
    bool use_cpu_svd_;

    // Randomized truncated SVD workspace
    bool use_rsvd_;
    int rsvd_oversampling_;
    Scalar* d_rsvd_omega_;    // (n, k+p) random projection matrix on GPU
    Scalar* d_rsvd_Y_;        // (m, k+p) projected result on GPU
    Scalar* d_rsvd_Q_;        // (m, k+p) QR factor on GPU
    Scalar* d_rsvd_B_;        // (k+p, n) projected matrix on GPU
    Scalar* d_rsvd_ipiv_;     // (k+p) QR pivot/tau on GPU (rocSOLVER)
    Scalar* d_rsvd_U_full_;   // (m, k+p) for U = Q @ U_small on GPU
    std::vector<Scalar> h_rsvd_B_;
    std::vector<Scalar> h_rsvd_U_small_;  // (k+p, k+p) from SVD of B

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
    void rsvd_split(int site, Scalar* d_theta, char direction);

    double optimize_bond(int site, char direction);
    double sweep_left_to_right();
    double sweep_right_to_left();

    void allocate_mps_tensor(int site, int cL, int cR);
    void free_gpu_resources();
};

// Include template implementation
#include "dmrg2_gpu_impl.h"

#endif // DMRG2_GPU_H
