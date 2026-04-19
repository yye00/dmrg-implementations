#ifndef DMRG2_GPU_H
#define DMRG2_GPU_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "scalar_traits.h"
#include "../../common/gpu_opts.h"

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
    int chi_L(int site) const { return bond_dims_[site]; }
    int chi_R(int site) const { return bond_dims_[site + 1]; }
    void set_quiet(bool) {}  // no-op

    GpuOpts& opts() { return opts_; }
    const GpuOpts& opts() const { return opts_; }

private:
    // System parameters
    int L_, d_, chi_max_, D_mpo_;
    int D_mpo_actual_;
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

    // SPARSE_MPO: per-site nnz row/col lists for WW (two-site fused MPO)
    std::vector<int*> d_WW_nnz_rows_;
    std::vector<int*> d_WW_nnz_cols_;
    std::vector<int>  ww_nnz_rows_count_;
    std::vector<int>  ww_nnz_cols_count_;

    std::vector<int> L_env_alloc_chi_;
    std::vector<int> R_env_alloc_chi_;

    // GPU handles — dual-stream pipeline (env update ∥ absorb)
    hipStream_t stream_;
    hipStream_t stream_env_;
    rocblas_handle rocblas_h_;
    rocblas_handle rocblas_h_env_;
    hipEvent_t event_canon_ready_;
    hipEvent_t event_env_done_;

    // Env-stream scratch (independent of stream_'s d_T1_/d_T2_)
    Scalar* d_T1_env_;
    Scalar* d_T2_env_;
    Scalar** d_batch_A_env_;
    Scalar** d_batch_B_env_;
    Scalar** d_batch_C_env_;

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

    // rocsolver tridiagonal eigensolver workspace (replaces CPU LAPACK dstev)
    double* d_steqr_D_;
    double* d_steqr_E_;
    double* d_steqr_C_;
    rocblas_int* d_steqr_info_;
    Scalar* d_const_one_;
    Scalar* d_const_zero_;
    Scalar* d_const_neg_one_;

    // Batched GEMM pointer arrays (on device)
    Scalar** d_batch_A_;
    Scalar** d_batch_B_;
    Scalar** d_batch_C_;

    // Length-D_mpo vector of ones on device; used as the reduction vector for
    // Step 3 of apply_heff_two_site (R3-F1 full-batched collapse).
    Scalar* d_ones_D_;

    // SVD workspace (pre-allocated at max size)
    Scalar* d_svd_A_;
    Scalar* d_svd_U_;
    RealType* d_svd_S_;
    Scalar* d_svd_Vh_;
    RealType* d_svd_E_;
    int* d_svd_info_;
    Scalar* d_svd_work_;
    // R3-F2: device scalars required by rocsolver_gesvdj (it writes residual
    // and n_sweeps to GPU memory — host pointers cause silent hangs).
    double* d_svdj_residual_;
    rocblas_int* d_svdj_n_sweeps_;

    // CPU workspace (for receiving GPU SVD results and truncation/scaling)
    std::vector<Scalar> h_svd_U_, h_svd_Vh_, h_svd_tmp_;
    std::vector<RealType> h_svd_S_;

    // RSVD workspace (allocated only when opts_.rsvd is on)
    static constexpr int RSVD_OVERSAMPLE_ = 10;
    Scalar* d_rsvd_omega_   = nullptr;
    Scalar* d_rsvd_Y_       = nullptr;
    Scalar* d_rsvd_tau_     = nullptr;
    Scalar* d_rsvd_B_       = nullptr;
    Scalar* d_rsvd_U_small_ = nullptr;
    int     rsvd_r_max_     = 0;

    // LANCZOS_GRAPH: cached HIP-graph exec per (site, cL, cR) for apply_heff
    Scalar* d_heff_input_ = nullptr;
    std::unordered_map<uint64_t, hipGraphExec_t> apply_heff_graph_cache_;
    static inline uint64_t graph_key(int site, int cL, int cR) {
        return ((uint64_t)(uint32_t)site << 40) |
               ((uint64_t)(uint32_t)cL   << 20) |
                (uint64_t)(uint32_t)cR;
    }

    // Ablation flags + phase timers
    GpuOpts opts_;
    PhaseTimer t_lanczos_;
    PhaseTimer t_apply_heff_;
    PhaseTimer t_svd_;
    PhaseTimer t_absorb_;
    PhaseTimer t_env_update_;
    void init_timers();
    void report_timers();

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
