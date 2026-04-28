#ifndef DMRG2_GPU_OPT_H
#define DMRG2_GPU_OPT_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include "scalar_traits.h"
#include "../../common/gpu_opts.h"

/**
 * GPU-native DMRG - Two-site optimization with Block-Davidson
 *
 * Templated on Scalar: double (real) or hipDoubleComplex (complex128).
 *
 * Tier: -opt is the "FURTHER algorithmic improvements" tier per the project's
 * three-tier model (see dmrg-gpu-opt.h docstring for the full taxonomy).
 * Block-Davidson REPLACES Lanczos (different algorithm, BLAS-3 dominant) —
 * this is intentional algorithmic divergence from dmrg2-gpu, not a bug.
 *
 * Algorithm-level differences from dmrg2-gpu:
 * 1. Block-Davidson eigensolver replaces Lanczos (BLAS-3 dominant).
 * 2. Dimension padding to MFMA-16 multiples for MI300X tile alignment.
 * 3. Strided batched Step-3 GEMMs reduce kernel launch overhead.
 *
 * Correctness baseline matches dmrg2-gpu (post round-4 C6 backport):
 * - Zero per-sweep host LAPACK on default code path (rocsolver_dsteqr for
 *   tridiagonal, rocsolver_gesvd_auto for SVD, on-device truncate + scale).
 * - Two-site fused MPO (WW) precomputed for efficient H_eff application.
 *
 * Paper status: excluded from G1 baseline campaign per §6.4 analytical-bound
 * treatment. Binary still ships and must be at-least-correct.
 */

// Round up to next multiple of 16 for MI300X MFMA FP64 tile alignment
static inline int pad_mfma16(int x) { return (x + 15) & ~15; }

template<typename Scalar>
class DMRG2GPUOpt {
    using Traits = ScalarTraits<Scalar>;
    using RealType = typename Traits::RealType;

public:
    DMRG2GPUOpt(int L, int d, int chi_max, int D_mpo, double tol = 1e-10);
    ~DMRG2GPUOpt();

    double run(int n_sweeps);

    void initialize_mps_random(double scale = 0.1);
    void initialize_mps_product();
    void initialize_mps_neel();

    void set_mpo(const std::vector<Scalar*>& h_mpo_tensors);

    double get_energy() const { return energy_; }
    void get_mps(std::vector<std::vector<Scalar>>& h_mps) const;

    // Ablation controls (defaults loaded from DMRG_GPU_OPT_* env vars in ctor)
    GpuOpts& opts() { return opts_; }
    const GpuOpts& opts() const { return opts_; }

    // Public API setters — parity with pdmrg-gpu-opt (round-5 J2). Setters
    // for flags genuinely N/A in two-site DMRG (set_use_batched_sweep,
    // set_use_chebyshev) are intentionally absent rather than no-ops.
    void set_cpu_svd(bool use_cpu) { use_cpu_svd_ = use_cpu; }
    // Toggling Davidson on disables lanczos_graph (graph capture is
    // incompatible with Davidson's variable output pointer per subspace
    // column). Toggling Davidson OFF re-enables lanczos_graph if the user
    // had it on at construction — symmetric round-trip for benchmark
    // switches.
    void set_use_davidson(bool use_dav) {
        use_davidson_ = use_dav;
        if (use_dav && opts_.lanczos_graph) {
            opts_.lanczos_graph = false;
            lanczos_graph_was_user_enabled_ = true;
        } else if (!use_dav && lanczos_graph_was_user_enabled_) {
            opts_.lanczos_graph = true;
        }
    }
    void set_rsvd(bool use_rsvd) { use_rsvd_ = use_rsvd; }
    void set_quiet(bool) {}  // no-op (matches pdmrg-gpu-opt API surface)

    int chi_L(int site) const { return bond_dims_[site]; }
    int chi_R(int site) const { return bond_dims_[site + 1]; }

private:
    // System parameters
    int L_, d_, chi_max_, chi_max_user_, D_mpo_;
    int D_mpo_actual_;      // user-supplied MPO bond dim; D_mpo_ may be padded (D_PAD)
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

    // SPARSE_MPO: per-bond nnz row/col lists for WW (two-site fused MPO)
    std::vector<int*> d_WW_nnz_rows_;
    std::vector<int*> d_WW_nnz_cols_;
    std::vector<int>  ww_nnz_rows_count_;
    std::vector<int>  ww_nnz_cols_count_;
    // Host-side nnz lists (needed for host-pointer setup path)
    std::vector<std::vector<int>> h_WW_nnz_rows_;
    std::vector<std::vector<int>> h_WW_nnz_cols_;

    std::vector<int> L_env_alloc_chi_;
    std::vector<int> R_env_alloc_chi_;

    // GPU handles — dual-stream pipeline (round-6 J2 port from dmrg-gpu).
    // svd_split_fallback records event_canon_ready_ once the canonical MPS
    // tensor is written; stream_env_ waits on it and runs update_left/right_env
    // in parallel with the trailing scale on stream_. Before the next
    // iteration's form_theta_two_site + Lanczos, stream_ waits on
    // event_env_done_.
    hipStream_t stream_;
    hipStream_t stream_env_;
    rocblas_handle rocblas_h_;
    rocblas_handle rocblas_h_env_;
    hipEvent_t event_canon_ready_;
    hipEvent_t event_env_done_;
    bool env_update_pending_ = false;

    // Contraction intermediates (main stream)
    Scalar* d_T1_;
    Scalar* d_T2_;
    // Contraction intermediates (env stream) — disjoint from main-stream
    // scratch so absorb and env_update can run concurrently.
    Scalar* d_T1_env_;
    Scalar* d_T2_env_;

    // Lanczos workspace (pre-allocated, used as fallback)
    Scalar* d_theta_;
    Scalar* d_heff_result_;
    Scalar* d_lanczos_v_;
    Scalar* d_ritz_coeffs_;
    int theta_size_max_;
    int max_lanczos_iter_;

    // Batched GEMM pointer arrays (on device) — main + env stream variants
    Scalar** d_batch_A_;
    Scalar** d_batch_B_;
    Scalar** d_batch_C_;
    Scalar** d_batch_A_env_;
    Scalar** d_batch_B_env_;
    Scalar** d_batch_C_env_;

    // SVD workspace (on-device — replaces the previous host-LAPACK svd_split_fallback)
    Scalar* d_svd_A_;
    Scalar* d_svd_U_;
    RealType* d_svd_S_;
    Scalar* d_svd_Vh_;
    RealType* d_svd_E_;
    int* d_svd_info_;
    Scalar* d_svd_work_;
    double* d_svdj_residual_ = nullptr;
    rocblas_int* d_svdj_n_sweeps_ = nullptr;

    // Tridiagonal eigensolve workspace (rocsolver_dsteqr — replaces the prior
    // 2× host LAPACK dstev_ in lanczos_eigensolver, matching pdmrg-gpu-opt
    // 8dbd1b8 + dmrg-gpu-opt c5 backport).
    double*      d_steqr_D_ = nullptr;
    double*      d_steqr_E_ = nullptr;
    double*      d_steqr_C_ = nullptr;
    rocblas_int* d_steqr_info_ = nullptr;

    // Randomized SVD workspace (Halko-Martinsson-Tropp, on-device). Allocated
    // only when use_rsvd_ flag is on (round-5 J2 port from dmrg2-gpu).
    static constexpr int RSVD_OVERSAMPLE_ = 10;
    Scalar*  d_rsvd_omega_   = nullptr;
    Scalar*  d_rsvd_Y_       = nullptr;
    Scalar*  d_rsvd_tau_     = nullptr;
    Scalar*  d_rsvd_B_       = nullptr;
    Scalar*  d_rsvd_U_small_ = nullptr;
    int      rsvd_r_max_     = 0;

    // CPU SVD workspace (legacy — only used by the init-time workspace query;
    // runtime SVD is fully on-device).
    std::vector<Scalar> h_svd_A_, h_svd_U_, h_svd_Vh_, h_svd_work_, h_svd_tmp_;
    std::vector<RealType> h_svd_S_;
    std::vector<RealType> h_svd_rwork_;

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

    // LANCZOS_GRAPH: cached HIP-graph exec per (site, cL, cR) for apply_heff.
    // d_heff_input_ is a fixed-address bounce buffer so captured graphs can
    // read from a constant address across Lanczos iterations. Only allocated
    // when opts_.lanczos_graph is on.
    Scalar* d_heff_input_ = nullptr;
    std::unordered_map<uint64_t, hipGraphExec_t> apply_heff_graph_cache_;
    static inline uint64_t graph_key(int site, int cL, int cR) {
        return ((uint64_t)(uint32_t)site << 40) |
               ((uint64_t)(uint32_t)cL   << 20) |
                (uint64_t)(uint32_t)cR;
    }

    // Ablation flags + phase timers
    GpuOpts opts_;
    // Public-API-controlled flags. Defaults match the historical -opt
    // behavior (Block-Davidson primary, no RSVD, on-device SVD).
    bool use_cpu_svd_ = false;       // opt-in CPU LAPACK SVD path (legacy)
    bool use_davidson_ = true;       // -opt's defining choice; false → Lanczos
    bool use_rsvd_ = false;          // opt-in randomized SVD (round-5 port)
    // Tracks whether the ctor disabled lanczos_graph because Davidson was
    // on, so set_use_davidson(false) can re-enable it symmetrically.
    bool lanczos_graph_was_user_enabled_ = false;
    PhaseTimer t_lanczos_;      // full lanczos_eigensolver call
    PhaseTimer t_apply_heff_;   // each apply_heff invocation
    PhaseTimer t_svd_;          // SVD bond splitting
    PhaseTimer t_absorb_;       // scale + absorb GEMM
    PhaseTimer t_env_update_;   // update_left_env / update_right_env
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

    // Bond splitting (CPU LAPACK)
    void svd_split_fallback(int site, Scalar* d_theta, char direction);

    // Block-Davidson eigensolver
    double block_davidson_eigensolver(int site, Scalar* d_theta, int theta_size);

    double optimize_bond(int site, char direction);
    double sweep_left_to_right();
    double sweep_right_to_left();

    void allocate_mps_tensor(int site, int cL, int cR);
    void free_gpu_resources();
};

// Include template implementation
#include "dmrg2_gpu_opt_impl.h"

#endif // DMRG2_GPU_OPT_H
