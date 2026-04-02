#ifndef PDMRG_GPU_H
#define PDMRG_GPU_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <string>
#include "scalar_traits.h"

/**
 * Stream-Parallel DMRG (PDMRG) on GPU
 *
 * Partitions the chain into P segments, each assigned to a CUDA stream.
 * Segments run independent two-site DMRG sweeps concurrently, then
 * boundary merges couple them via accurate SVD (V = 1/S).
 *
 * Templated on Scalar: double (real) or cuDoubleComplex (complex128).
 * Based on dmrg2-gpu architecture: same cuBLAS GEMM patterns, Lanczos, SVD.
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
    void set_rsvd(bool use_rsvd) { use_rsvd_ = use_rsvd; }
    void set_quiet(bool) {}  // no-op

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

    // === Stoudenmire boundary state (V = Lambda^-1 at each boundary) ===
    struct BoundaryState {
        std::vector<RealType> V;  // V = 1/S, length = chi at boundary bond
        int chi;                  // current bond dimension at this boundary
    };
    std::vector<BoundaryState> boundary_states_;

    // === CUDA streams and cuBLAS/cuSOLVER handles (one per segment) ===
    std::vector<cudaStream_t> streams_;
    std::vector<cublasHandle_t> handles_;
    std::vector<cusolverDnHandle_t> cusolver_handles_;

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
        RealType* d_svd_E;          // rwork for complex SVD
        int* d_svd_info;
        Scalar* d_svd_work;
        int svd_lwork;              // cuSOLVER workspace size
        // CPU SVD
        std::vector<Scalar> h_svd_A, h_svd_U, h_svd_Vh, h_svd_work, h_svd_tmp;
        std::vector<RealType> h_svd_S;
        std::vector<RealType> h_svd_rwork;
        // Randomized truncated SVD (GPU QR)
        Scalar* d_rsvd_omega;     // (n, r) random projection on GPU
        Scalar* d_rsvd_Y;         // (m, r) projected result on GPU
        Scalar* d_rsvd_Q;         // (m, r) QR factor on GPU
        Scalar* d_rsvd_B;         // (r, n) projected matrix on GPU
        Scalar* d_rsvd_ipiv;      // (r) QR tau on GPU (cuSOLVER)
        Scalar* d_rsvd_U_full;    // (m, r) for U = Q @ U_small on GPU
        Scalar* d_qr_work;        // cuSOLVER QR workspace
        int qr_lwork;             // cuSOLVER QR workspace size
        Scalar* d_orgqr_work;     // cuSOLVER orgqr workspace
        int orgqr_lwork;          // cuSOLVER orgqr workspace size
        std::vector<Scalar> h_rsvd_B;          // (r, n) host copy
        std::vector<Scalar> h_rsvd_U_small;    // (r, r) from SVD of B
    };
    std::vector<StreamWorkspace> workspaces_;

    bool use_cpu_svd_;
    bool use_rsvd_;
    bool lanczos_use_1site_;  // when true, Lanczos calls apply_heff_single_site
    int rsvd_oversampling_;
    int theta_size_max_;
    int max_lanczos_iter_;

    // === Core methods (stream-aware: si = stream index) ===
    // Two-site (for main PDMRG sweeps and boundary merge)
    void form_theta_two_site(int site, int si);
    void apply_heff_two_site(int site, const Scalar* d_in, Scalar* d_out, int si);
    double lanczos_eigensolver(int site, Scalar* d_theta, int theta_size, int si);
    void svd_split(int site, Scalar* d_theta, char direction, int si);
    void rsvd_split(int site, Scalar* d_theta, char direction, int si);
    double optimize_bond(int site, char direction, int si);

    // Single-site (for warmup and polish -- cheaper eigsh problem)
    void apply_heff_single_site(int site, const Scalar* d_in, Scalar* d_out, int si);
    void svd_split_single_site(int site, Scalar* d_theta, char direction, int si);
    double optimize_site_single(int site, char direction, int si);

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
    double sweep_LR_full();        // full-chain L->R two-site, stream 0
    double sweep_RL_full();        // full-chain R->L two-site, stream 0
    double sweep_LR_full_1site();  // full-chain L->R single-site (warmup/polish), stream 0
    double sweep_RL_full_1site();  // full-chain R->L single-site (warmup/polish), stream 0
    void segment_sweep_LR(int seg_idx);  // local L->R within segment
    void segment_sweep_RL(int seg_idx);  // local R->L within segment
    double merge_and_optimize_boundaries(int parity = -1);  // Stoudenmire boundary coupling
    void form_theta_with_V(int site, int boundary_idx, int si);  // theta = psi_L . diag(V) . psi_R
    void initialize_boundary_states();  // V = ones at startup

    // === Partitioning ===
    void partition_chain();

    // === Memory management ===
    void allocate_stream_workspaces();
    void free_gpu_resources();
};

// Include template implementation
#include "pdmrg_gpu_impl.h"

#endif // PDMRG_GPU_H
