#ifndef PDMRG_GPU_H
#define PDMRG_GPU_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <string>
#include "scalar_traits.h"

/**
 * Stream-Parallel DMRG (PDMRG) on GPU
 *
 * Partitions the chain into P segments, each assigned to a HIP stream.
 * Segments run independent two-site DMRG sweeps concurrently, then
 * boundary merges couple them via accurate SVD (V = 1/S).
 *
 * Templated on Scalar: double (real) or hipDoubleComplex (complex128).
 * Based on dmrg2-gpu architecture: same rocBLAS GEMM patterns, Lanczos, SVD.
 */
template<typename Scalar>
class PDMRGGPU {
    using Traits = ScalarTraits<Scalar>;
    using RealType = typename Traits::RealType;

public:
    PDMRGGPU(int L, int d, int chi_max, int D_mpo, int n_segments, double tol = 1e-10);
    ~PDMRGGPU();

    // Setup
    void set_mpo(const std::vector<Scalar*>& h_mpo_tensors);
    void initialize_mps_random(double scale = 0.1);

    // Run
    double run(int n_outer_sweeps, int n_local_sweeps = 2, int n_warmup = 3);

    // Results
    double get_energy() const { return energy_; }
    void get_mps(std::vector<std::vector<Scalar>>& h_mps) const;
    void set_cpu_svd(bool use_cpu) { use_cpu_svd_ = use_cpu; }

    int chi_L(int site) const { return bond_dims_[site]; }
    int chi_R(int site) const { return bond_dims_[site + 1]; }

private:
    // System parameters
    int L_, d_, chi_max_, D_mpo_;
    double tol_;
    double energy_;

    // Bond dimensions: bond_dims_[i] = dim of bond between site i-1 and i
    std::vector<int> bond_dims_;

    // === Chain partitioning ===
    int n_segments_;
    std::vector<int> seg_first_;      // seg_first_[k] = first site of segment k
    std::vector<int> seg_last_;       // seg_last_[k] = last site of segment k (inclusive)
    std::vector<int> boundary_bonds_; // [n_segments-1] bond site at each boundary

    // === HIP streams and rocBLAS handles (one per segment) ===
    std::vector<hipStream_t> streams_;
    std::vector<rocblas_handle> handles_;

    // === Global GPU data (shared across streams) ===
    std::vector<Scalar*> d_mps_tensors_;
    std::vector<Scalar*> d_mpo_tensors_;
    std::vector<Scalar*> d_L_envs_;
    std::vector<Scalar*> d_R_envs_;
    std::vector<Scalar*> d_W_left_;
    std::vector<Scalar*> d_W_right_;
    std::vector<Scalar*> d_WW_;       // fused two-site MPO
    std::vector<int> L_env_alloc_chi_;
    std::vector<int> R_env_alloc_chi_;

    // === Per-stream workspace ===
    struct StreamWorkspace {
        Scalar* d_theta;
        Scalar* d_heff_result;
        Scalar* d_T1;
        Scalar* d_T2;
        Scalar* d_lanczos_v;
        Scalar* d_ritz_coeffs;
        Scalar** d_batch_A;
        Scalar** d_batch_B;
        Scalar** d_batch_C;
        // Pinned host pointer arrays (avoid heap alloc in hot path)
        Scalar** h_batch_A_pinned;
        Scalar** h_batch_B_pinned;
        Scalar** h_batch_C_pinned;
        // Cached apply_heff Step 1 A/C pointers (constant per site)
        Scalar** d_heff_batch_A;
        Scalar** d_heff_batch_C;
        int heff_cached_site;
        // Lanczos device-pointer-mode scalars
        Scalar* d_dot_result;        // 1 element: raw dot product result
        RealType* d_nrm2_result;     // 1 element: raw nrm2 result
        Scalar* d_neg_alpha;         // 1 element: -alpha for axpy
        Scalar* d_neg_overlap;       // 1 element: -overlap for reorth
        RealType* d_inv_nrm;         // 1 element: 1/nrm for scal
        RealType* d_alpha_dev;       // [max_iter]: alpha values
        RealType* d_beta_dev;        // [max_iter]: beta values
        Scalar* d_neg_beta_scalars;  // [max_iter]: -beta as Scalar for axpy
        Scalar* d_const_one;         // constant 1.0
        Scalar* d_const_zero;        // constant 0.0
        Scalar* d_const_neg_one;     // constant -1.0
        // GPU SVD
        Scalar* d_svd_A;
        Scalar* d_svd_U;
        RealType* d_svd_S;
        Scalar* d_svd_Vh;
        RealType* d_svd_E;
        int* d_svd_info;
        Scalar* d_svd_work;
        // CPU SVD
        std::vector<Scalar> h_svd_A, h_svd_U, h_svd_Vh, h_svd_work, h_svd_tmp;
        std::vector<RealType> h_svd_S;
        std::vector<RealType> h_svd_rwork;
    };
    std::vector<StreamWorkspace> workspaces_;

    bool use_cpu_svd_;
    int theta_size_max_;
    int max_lanczos_iter_;

    // === Core methods (stream-aware: si = stream index) ===
    void form_theta_two_site(int site, int si);
    void apply_heff_two_site(int site, const Scalar* d_in, Scalar* d_out, int si);
    double lanczos_eigensolver(int site, Scalar* d_theta, int theta_size, int si);
    void svd_split(int site, Scalar* d_theta, char direction, int si);
    double optimize_bond(int site, char direction, int si);

    // === Environment updates (stream-aware) ===
    void update_left_env(int site, int si);
    void update_right_env(int site, int si);
    void ensure_L_env_alloc(int idx, int chi);
    void ensure_R_env_alloc(int idx, int chi);

    // === Initialization ===
    void build_initial_environments();
    void precompute_fused_mpo(const std::vector<Scalar*>& h_mpo_tensors);
    void allocate_mps_tensor(int site, int cL, int cR);

    // === Sweep methods ===
    double sweep_LR_full();        // full-chain L→R (warmup), stream 0
    double sweep_RL_full();        // full-chain R→L (warmup), stream 0
    void segment_sweep_LR(int seg_idx);  // local L→R within segment
    void segment_sweep_RL(int seg_idx);  // local R→L within segment

    // === Partitioning ===
    void partition_chain();

    // === Memory management ===
    void allocate_stream_workspaces();
    void free_gpu_resources();
};

// Include template implementation
#include "pdmrg_gpu_impl.h"

#endif // PDMRG_GPU_H
