#ifndef DMRG_GPU_H
#define DMRG_GPU_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include "scalar_traits.h"

/**
 * GPU-native DMRG - Single-site optimization
 *
 * Templated on Scalar: double (real) or hipDoubleComplex (complex128).
 *
 * ALL tensor contractions on GPU via rocBLAS gemm.
 * ALL linear algebra on GPU via rocBLAS/rocSOLVER.
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
    //
    // Dual-stream pipeline (forward sweep):
    //   stream_      : Lanczos + SVD + absorb(S*Vh) into MPS[site+1]
    //   stream_env_  : update_left_env(site) → L[site+1]
    // After SVD writes U into MPS[site], event_canon_ready_ is recorded on
    // stream_ and stream_env_ waits on it before starting env_update. The env
    // update then runs concurrently with the absorb GEMM on stream_ (they touch
    // disjoint memory — env_update reads MPS[site]/L[site]/W and writes
    // L[site+1]; absorb reads S*Vh and MPS[site+1] and writes MPS[site+1]).
    // Before the next site's Lanczos, stream_ waits on event_env_done_ so that
    // apply_heff sees the updated L[site+1]. Symmetric setup for the backward
    // sweep (update_right_env).
    hipStream_t stream_;
    hipStream_t stream_env_;
    rocblas_handle rocblas_h_;
    rocblas_handle rocblas_h_env_;
    hipEvent_t event_canon_ready_;   // signaled after MPS[site] (U or Vh) is written
    hipEvent_t event_env_done_;      // signaled after L[site+1] / R[site] is written
    bool env_update_pending_ = false;

    // Contraction intermediates (main stream)
    Scalar* d_T1_;
    Scalar* d_T2_;
    // Contraction intermediates (env stream) — must be disjoint from main
    // stream's scratch so absorb and env_update can run concurrently.
    Scalar* d_T1_env_;
    Scalar* d_T2_env_;

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

    // rocsolver tridiagonal eigensolver workspace (replaces CPU LAPACK dstev)
    double* d_steqr_D_;         // diagonal (overwritten with eigenvalues)
    double* d_steqr_E_;         // subdiagonal (overwritten)
    double* d_steqr_C_;         // eigenvector matrix (max_iter × max_iter)
    rocblas_int* d_steqr_info_; // rocsolver info output
    Scalar* d_const_one_;
    Scalar* d_const_zero_;
    Scalar* d_const_neg_one_;

    // Batched GEMM pointer arrays (on device) — main stream
    Scalar** d_batch_A_;
    Scalar** d_batch_B_;
    Scalar** d_batch_C_;
    // Batched GEMM pointer arrays (on device) — env stream
    Scalar** d_batch_A_env_;
    Scalar** d_batch_B_env_;
    Scalar** d_batch_C_env_;

    // Length-D_mpo vector of ones on device; used as the reduction vector for
    // Step 3 of apply_heff (R3-F1 full-batched collapse).
    Scalar* d_ones_D_;

    // SVD workspace (pre-allocated at max size)
    Scalar* d_svd_A_;
    Scalar* d_svd_U_;
    RealType* d_svd_S_;      // singular values always real
    Scalar* d_svd_Vh_;
    RealType* d_svd_E_;      // superdiagonal always real
    int* d_svd_info_;
    Scalar* d_svd_work_;
    // R3-F2: device scalars required by rocsolver_gesvdj.
    double* d_svdj_residual_;
    rocblas_int* d_svdj_n_sweeps_;

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
