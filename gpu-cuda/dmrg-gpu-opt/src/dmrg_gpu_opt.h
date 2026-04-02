#ifndef DMRG_GPU_OPT_H
#define DMRG_GPU_OPT_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <string>
#include "scalar_traits.h"

/**
 * GPU-native DMRG - Single-site optimization with Newton-Schulz + Block-Davidson
 *
 * Templated on Scalar: double (real) or cuDoubleComplex (complex128).
 *
 * Key algorithmic changes from dmrg-gpu:
 * 1. Newton-Schulz polar decomposition replaces SVD for bond splitting
 *    (BLAS-3 GEMM-heavy, GPU-native), handles both tall and wide matrices
 * 2. Block-Davidson eigensolver replaces Lanczos (BLAS-3 dominant)
 * 3. Dimension padding to MFMA-16 multiples for tile alignment
 * 4. Batched Step-3 GEMMs reduce kernel launch overhead
 *
 * ALL tensor contractions on GPU via cuBLAS gemm.
 * Only CPU work: control flow, convergence checks on scalars,
 *                small eigendecompositions (Davidson projected H, NS P^H P),
 *                loops over small MPO bond dimension (D=5, d=2) to dispatch GEMMs.
 */

// Round up to next multiple of 16 for tile alignment
static inline int pad_mfma16(int x) { return (x + 15) & ~15; }

template<typename Scalar>
class DMRGGPUOpt {
    using Traits = ScalarTraits<Scalar>;
    using RealType = typename Traits::RealType;

public:
    DMRGGPUOpt(int L, int d, int chi_max, int D_mpo, double tol = 1e-10);
    ~DMRGGPUOpt();

    double run(int n_sweeps);

    void initialize_mps_random(double scale = 0.1);
    void initialize_mps_product();
    void initialize_mps_neel();

    void set_mpo(const std::vector<Scalar*>& h_mpo_tensors);

    double get_energy() const { return energy_; }
    void get_mps(std::vector<std::vector<Scalar>>& h_mps) const;

    int chi_L(int site) const { return bond_dims_[site]; }
    int chi_R(int site) const { return bond_dims_[site + 1]; }

private:
    // System parameters
    int L_, d_, chi_max_, chi_max_user_, D_mpo_;
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

    // Contraction intermediates
    Scalar* d_T1_;
    Scalar* d_T2_;

    // Lanczos workspace (pre-allocated, used as fallback)
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

    // Step-3 batched GEMM pointer arrays (D*d entries for apply_heff/env update)
    Scalar** d_batch3_A_;
    Scalar** d_batch3_B_;
    Scalar** d_batch3_C_;

    // SVD workspace (reused as NS scratch)
    Scalar* d_svd_A_;
    Scalar* d_svd_U_;
    RealType* d_svd_S_;
    Scalar* d_svd_Vh_;

    // CPU SVD workspace (fallback)
    std::vector<Scalar> h_svd_A_, h_svd_U_, h_svd_Vh_, h_svd_work_, h_svd_tmp_;
    std::vector<RealType> h_svd_S_;
    std::vector<RealType> h_svd_rwork_;

    // Newton-Schulz workspace
    Scalar* d_ns_U_;        // (chi_max*d, chi_max) NS iterate
    Scalar* d_ns_U_new_;    // (chi_max*d, chi_max) NS next iterate
    Scalar* d_ns_gram_;     // (chi_max, chi_max) U^H U
    Scalar* d_ns_P_;        // (chi_max, chi_max) PSD factor
    std::vector<Scalar> h_ns_PtP_;
    std::vector<RealType> h_ns_eigvals_;
    std::vector<Scalar> h_ns_syev_work_;
    std::vector<RealType> h_ns_syev_rwork_;

    // Block-Davidson workspace
    int davidson_b_;
    int davidson_max_sub_;
    Scalar* d_dav_V_;
    Scalar* d_dav_AV_;
    Scalar* d_dav_work_;
    Scalar* d_dav_work2_;
    std::vector<Scalar> h_dav_H_proj_;
    std::vector<RealType> h_dav_eigvals_;
    std::vector<Scalar> h_dav_eigvecs_;

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

    // Newton-Schulz polar decomposition
    void newton_schulz_left(Scalar* d_A, int m, int n,
                            Scalar* d_U, Scalar* d_P,
                            double tol, int* out_iters);

    // NS-based SVD and MPS update
    void ns_svd_and_update_mps(int site, Scalar* d_theta, char direction);

    // SVD fallback (CPU LAPACK)
    void svd_fallback(int site, Scalar* d_theta, char direction);

    // Block-Davidson eigensolver
    double block_davidson_eigensolver(int site, Scalar* d_theta);

    double sweep_left_to_right();
    double sweep_right_to_left();

    void allocate_mps_tensor(int site, int cL, int cR);
    void free_gpu_resources();
};

// Include template implementation
#include "dmrg_gpu_opt_impl.h"

#endif // DMRG_GPU_OPT_H
