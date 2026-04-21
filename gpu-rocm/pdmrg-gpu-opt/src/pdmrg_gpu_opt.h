#ifndef PDMRG_GPU_OPT_H
#define PDMRG_GPU_OPT_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <cstdio>
#include "scalar_traits.h"
#include "../../common/gpu_opts.h"

/**
 * PDMRG-GPU-OPT: Stream-Parallel DMRG with Block-Davidson
 *
 * Key algorithmic changes from pdmrg-gpu:
 * 1. Block-Davidson eigensolver replaces Lanczos (BLAS-3 dominant)
 *
 * Templated on Scalar: double (real) or hipDoubleComplex (complex128).
 * 2. Dimension padding to MFMA-16 multiples for MI300X tile alignment
 * 3. Strided batched Step-3 GEMMs to reduce kernel launch overhead
 */

// Round up to next multiple of 16 for MI300X MFMA FP64 tile alignment
static inline int pad_mfma16(int x) { return (x + 15) & ~15; }

template<typename Scalar>
class PDMRGGPUOpt {
    using Traits = ScalarTraits<Scalar>;
    using RealType = typename Traits::RealType;

public:
    PDMRGGPUOpt(int L, int d, int chi_max, int D_mpo, int n_segments, double tol = 1e-10);
    ~PDMRGGPUOpt();

    // Setup
    void set_mpo(const std::vector<Scalar*>& h_mpo_tensors);
    void initialize_mps_random(double scale = 0.1);

    // Run
    double run(int n_outer_sweeps, int n_local_sweeps = 2, int n_warmup = 3);

    // Results
    double get_energy() const { return energy_; }
    void get_mps(std::vector<std::vector<Scalar>>& h_mps) const;
    void set_cpu_svd(bool use_cpu) { use_cpu_svd_ = use_cpu; }
    void set_use_davidson(bool use_dav) {
        use_davidson_ = use_dav;
        // LANCZOS_GRAPH + Block-Davidson is incompatible: apply_heff_two_site
        // is called with AV + j*dim (variable output pointer per subspace
        // column); graph capture locks in the first-seen address and replays
        // write to the stale pointer, hanging Rayleigh-Ritz. Force the flag
        // off here so Lanczos mode can keep using graph capture unaffected.
        if (use_davidson_ && opts_.lanczos_graph) {
            std::fprintf(stderr,
                "[pdmrg-gpu-opt] LANCZOS_GRAPH=1 incompatible with --davidson; "
                "disabling graph capture.\n");
            opts_.lanczos_graph = false;
        }
    }
    void set_rsvd(bool use_rsvd) { use_rsvd_ = use_rsvd; }
    void set_use_batched_sweep(bool b) { use_batched_sweep_ = b; }
    void set_use_chebyshev(bool b) { use_chebyshev_ = b; }
    void set_quiet(bool) {}  // no-op

    // Ablation controls (defaults loaded from DMRG_GPU_OPT_* env vars in ctor)
    GpuOpts& opts() { return opts_; }
    const GpuOpts& opts() const { return opts_; }

    int chi_L(int site) const { return bond_dims_[site]; }
    int chi_R(int site) const { return bond_dims_[site + 1]; }

private:
    // System parameters
    int L_, d_, chi_max_, chi_max_user_, D_mpo_;
    int D_mpo_actual_;      // user-supplied MPO bond dim; D_mpo_ may be padded (D_PAD)
    double tol_;
    double energy_;

    // Bond dimensions
    std::vector<int> bond_dims_;

    // === Chain partitioning ===
    int n_segments_;
    std::vector<int> seg_first_;
    std::vector<int> seg_last_;
    std::vector<int> boundary_bonds_;

    // === Stoudenmire boundary state (V = Λ⁻¹ at each boundary) ===
    struct BoundaryState {
        std::vector<RealType> V;  // V = 1/S, length = chi at boundary bond
        int chi;                  // current bond dimension at this boundary
    };
    std::vector<BoundaryState> boundary_states_;

    // === HIP streams and rocBLAS handles ===
    std::vector<hipStream_t> streams_;
    std::vector<rocblas_handle> handles_;

    // === Worker stream pool for concurrent independent GEMMs ===
    int n_workers_;
    std::vector<std::vector<hipStream_t>> worker_streams_;     // [segment][worker]
    std::vector<std::vector<rocblas_handle>> worker_handles_;  // [segment][worker]
    std::vector<std::vector<hipEvent_t>> worker_done_events_;  // [segment][worker]
    std::vector<hipEvent_t> step_done_events_;                 // [segment]

    // === GPU tensors ===
    std::vector<Scalar*> d_mps_tensors_;
    std::vector<Scalar*> d_mpo_tensors_;
    std::vector<Scalar*> d_L_envs_;
    std::vector<Scalar*> d_R_envs_;
    std::vector<Scalar*> d_W_left_;
    std::vector<Scalar*> d_W_right_;
    std::vector<Scalar*> d_WW_;

    // SPARSE_MPO: per-site nnz row/col lists. Class-level (shared across
    // segments) because they depend only on W / WW for that site.
    // Single-site (W_left): used by apply_heff_single_site.
    std::vector<int*> d_WL_nnz_rows_;
    std::vector<int*> d_WL_nnz_cols_;
    std::vector<int>  wl_nnz_rows_count_;
    std::vector<int>  wl_nnz_cols_count_;
    // Two-site (WW): used by apply_heff_two_site.
    std::vector<int*> d_WW_nnz_rows_;
    std::vector<int*> d_WW_nnz_cols_;
    std::vector<int>  ww_nnz_rows_count_;
    std::vector<int>  ww_nnz_cols_count_;
    // Host-side nnz lists (for host-side pointer setup paths: single-site
    // Step 1/3 and two-site Step 3 beta-accumulation loop).
    std::vector<std::vector<int>> h_WL_nnz_rows_;
    std::vector<std::vector<int>> h_WL_nnz_cols_;
    std::vector<std::vector<int>> h_WW_nnz_rows_;
    std::vector<std::vector<int>> h_WW_nnz_cols_;

    std::vector<int> L_env_alloc_chi_;
    std::vector<int> R_env_alloc_chi_;

    // === Per-stream workspace ===
    struct StreamWorkspace {
        // Contraction intermediates
        Scalar* d_theta;
        Scalar* d_heff_result;
        Scalar* d_T1;
        Scalar* d_T2;

        // Batched GEMM pointer arrays
        Scalar** d_batch_A;
        Scalar** d_batch_B;
        Scalar** d_batch_C;
        Scalar** d_heff_batch_A;
        Scalar** d_heff_batch_C;
        int heff_cached_site;

        // === Block-Davidson workspace ===
        Scalar* d_dav_V;        // (theta_size_max, max_subspace) subspace basis
        Scalar* d_dav_AV;       // (theta_size_max, max_subspace) H × basis
        Scalar* d_dav_work;     // (theta_size_max, davidson_b) scratch
        Scalar* d_dav_work2;    // (theta_size_max, davidson_b) scratch 2
        // Host buffers for small eigenproblem
        std::vector<Scalar> h_dav_H_proj;      // (max_sub, max_sub)
        std::vector<RealType> h_dav_eigvals;   // (max_sub)
        std::vector<Scalar> h_dav_eigvecs;     // (max_sub, max_sub)
        std::vector<RealType> h_dav_syev_work; // workspace for dsyev rwork
        std::vector<Scalar> h_dav_V_copy;      // for restart QR on host

        // === Lanczos fallback workspace ===
        Scalar* d_lanczos_v;
        Scalar* d_ritz_coeffs;
        Scalar* d_dot_result;
        RealType* d_nrm2_result;
        Scalar* d_neg_alpha;
        Scalar* d_neg_overlap;
        RealType* d_inv_nrm;
        RealType* d_alpha_dev;
        RealType* d_beta_dev;
        Scalar* d_neg_beta_scalars;
        Scalar* d_const_one;
        Scalar* d_const_zero;
        Scalar* d_const_neg_one;

        // === SVD workspace ===
        Scalar* d_svd_A;
        Scalar* d_svd_U;
        RealType* d_svd_S;
        Scalar* d_svd_Vh;
        RealType* d_svd_E;
        int* d_svd_info;
        Scalar* d_svd_work;
        std::vector<Scalar> h_svd_A, h_svd_U, h_svd_Vh, h_svd_work, h_svd_tmp;
        std::vector<RealType> h_svd_S;
        std::vector<RealType> h_svd_rwork;

        // === rSVD workspace (Halko-Martinsson-Tropp) ===
        Scalar* d_rsvd_omega;    // (n, r) random projection on GPU
        Scalar* d_rsvd_Y;       // (m, r) Y = theta @ Omega on GPU
        Scalar* d_rsvd_Q;       // (m, r) QR result on GPU
        Scalar* d_rsvd_B;       // (r, n) B = Q^H @ theta; reused for U_small upload
        Scalar* d_rsvd_ipiv;    // (r) QR tau on GPU (rocSOLVER)
        Scalar* d_rsvd_U_full;  // (m, r) U = Q @ U_small on GPU
        std::vector<Scalar> h_rsvd_B;       // host buffer for CPU SVD of small B
        std::vector<Scalar> h_rsvd_U_small; // (r, r) from SVD of B

        // LANCZOS_GRAPH: per-segment fixed-address bounce buffer + cached
        // HIP-graph execs keyed by (site, cL, cR). Only allocated/populated
        // when opts_.lanczos_graph is on.
        Scalar* d_heff_input = nullptr;
        std::unordered_map<uint64_t, hipGraphExec_t> apply_heff_graph_cache;
    };
    std::vector<StreamWorkspace> workspaces_;

    // LANCZOS_GRAPH: packed (site, cL, cR) key for per-(shape) graph caching.
    static inline uint64_t graph_key(int site, int cL, int cR) {
        return ((uint64_t)(uint32_t)site << 40) |
               ((uint64_t)(uint32_t)cL   << 20) |
                (uint64_t)(uint32_t)cR;
    }

    // Ablation flags + phase timers
    GpuOpts opts_;
    PhaseTimer t_lanczos_;      // full lanczos_eigensolver call
    PhaseTimer t_apply_heff_;   // each apply_heff invocation
    PhaseTimer t_svd_;          // SVD bond splitting
    PhaseTimer t_absorb_;       // scale + absorb GEMM
    PhaseTimer t_env_update_;   // update_left_env / update_right_env
    void init_timers();
    void report_timers();

    bool use_cpu_svd_;
    bool use_davidson_;
    bool use_rsvd_;
    bool lanczos_use_1site_;  // when true, Lanczos calls apply_heff_single_site
    bool use_batched_sweep_;  // cross-segment batched GEMM in lock-step sweep
    bool use_chebyshev_;      // Chebyshev-filtered subspace iteration eigensolver
    int rsvd_oversampling_;
    int theta_size_max_;
    int max_lanczos_iter_;
    int davidson_b_;
    int davidson_max_sub_;

    // Cross-segment batched GEMM pointer arrays (for batched sweep mode)
    Scalar** d_xs_batch_A_;   // size: n_segments * D_mpo * d * d
    Scalar** d_xs_batch_B_;
    Scalar** d_xs_batch_C_;

    // === Core methods ===
    void form_theta_two_site(int site, int si);
    void apply_heff_two_site(int site, const Scalar* d_in, Scalar* d_out, int si);
    double block_davidson_eigensolver(int site, Scalar* d_theta, int theta_size, int si);
    double lanczos_eigensolver(int site, Scalar* d_theta, int theta_size, int si);
    double chebyshev_eigensolver(int site, Scalar* d_theta, int theta_size, int si);
    void svd_split(int site, Scalar* d_theta, char direction, int si);
    void rsvd_split(int site, Scalar* d_theta, char direction, int si);
    double optimize_bond(int site, char direction, int si);

    // Single-site (for warmup and polish — cheaper eigsh problem)
    void apply_heff_single_site(int site, const Scalar* d_in, Scalar* d_out, int si);
    void svd_split_single_site(int site, Scalar* d_theta, char direction, int si);
    double optimize_site_single(int site, char direction, int si);

    // === Environment updates ===
    void update_left_env(int site, int si);
    void update_right_env(int site, int si);
    void ensure_L_env_alloc(int idx, int chi);
    void ensure_R_env_alloc(int idx, int chi);

    // === Initialization ===
    void build_initial_environments();
    void precompute_fused_mpo(const std::vector<Scalar*>& h_mpo_tensors);
    void allocate_mps_tensor(int site, int cL, int cR);

    // === Sweep methods ===
    double sweep_LR_full();        // full-chain L→R two-site, stream 0
    double sweep_RL_full();        // full-chain R→L two-site, stream 0
    double sweep_LR_full_1site();  // full-chain L→R single-site (warmup/polish), stream 0
    double sweep_RL_full_1site();  // full-chain R→L single-site (warmup/polish), stream 0
    void segment_sweep_LR(int seg_idx);
    void segment_sweep_RL(int seg_idx);
    void batched_segment_sweep(bool even_LR);  // lock-step cross-segment batched sweep
    double batched_lanczos_eigensolver(const int* sites, const int* seg_indices,
                                        int n_batch, Scalar** d_thetas, int theta_size);
    void batched_apply_heff_two_site(const int* sites, const int* seg_indices,
                                      int n_batch, const Scalar** d_thetas_in, Scalar** d_results);
    double merge_and_optimize_boundaries(int parity = -1);  // Stoudenmire boundary coupling
    void form_theta_with_V(int site, int boundary_idx, int si);  // θ = ψ_L · diag(V) · ψ_R
    void initialize_boundary_states();  // V = ones at startup

    // === Partitioning ===
    void partition_chain();

    // === Memory management ===
    void allocate_stream_workspaces();
    void free_gpu_resources();
};

#include "pdmrg_gpu_opt_impl.h"

#endif // PDMRG_GPU_OPT_H
