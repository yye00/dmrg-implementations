#ifndef PDMRG_GPU_BASE_H
#define PDMRG_GPU_BASE_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <string>
#include "scalar_traits.h"
#include "../../common/accurate_svd_gpu.h"

/**
 * Stream-Parallel DMRG (PDMRG) on GPU — NAIVE BASELINE
 *
 * Competent first-pass GPU implementation of stream-parallel two-site DMRG.
 * Preserves PDMRG's defining feature — chain partitioned into P segments
 * running concurrently on P HIP streams — and runs all linear algebra on
 * device. The two-site fused MPO (WW) is precomputed once per bond at
 * set_mpo() time. The Lanczos eigensolver runs in device-pointer mode with
 * rocSOLVER `dsteqr` for the tridiagonal eigenproblem. The SVD uses
 * rocSOLVER `gesvd_auto` and device-side truncation kernels. No CPU
 * computation in the inner sweep loop.
 *
 * Compared to PDMRGGPU (the paper-reference variant), this baseline omits:
 *   - HIP graph capture for the Lanczos inner loop,
 *   - randomized SVD (RSVD),
 *   - batched GEMM and the GpuOpts ablation framework,
 *   - sparse-MPO compaction,
 *   - D_PAD MFMA-friendly padding,
 *   - the on-device WW precompute kernel (uses host-side compute then H2D).
 * The baseline DOES use the on-device Stoudenmire `accurate_svd_gpu` at
 * segment-boundary merges (J1 lock — Stoudenmire is part of pdmrg's
 * algorithm, not an optimization, and is therefore mandatory in every pdmrg
 * variant including this baseline).
 * The baseline uses non-blocking streams (required for correct concurrent
 * boundary merges; not an optimization), single-GEMM-per-pair patterns
 * where the optimized variant uses gemm_batched, and the standard non-fused
 * Lanczos kernels.
 *
 * CLAUDE.md compliance: warmup and polish sweeps are SINGLE-SITE
 * (sweep_LR_full_1site / sweep_RL_full_1site). The number of warmup and
 * polish sweeps is configurable via run() parameters (default 1 each;
 * caller must supply n_polish explicitly to override).
 *
 * Templated on Scalar: double (real) or hipDoubleComplex (complex128).
 */
template<typename Scalar>
class PDMRGGPUBase {
    using Traits = ScalarTraits<Scalar>;
    using RealType = typename Traits::RealType;

public:
    PDMRGGPUBase(int L, int d, int chi_max, int D_mpo, int n_segments, double tol = 1e-10);
    ~PDMRGGPUBase();

    void set_mpo(const std::vector<Scalar*>& h_mpo_tensors);
    void initialize_mps_random(double scale = 0.1);

    // Run PDMRG. CLAUDE.md compliant defaults (n_warmup=1, n_polish=0,
    // single-site warmup and polish). All counts are configurable so the
    // benchmark harness can pass them explicitly per CLAUDE.md rule.
    double run(int n_outer_sweeps,
               int n_local_sweeps = 2,
               int n_warmup = 1,
               int n_polish = 0);

    double get_energy() const { return energy_; }
    void get_mps(std::vector<std::vector<Scalar>>& h_mps) const;

    int chi_L(int site) const { return bond_dims_[site]; }
    int chi_R(int site) const { return bond_dims_[site + 1]; }

    void set_quiet(bool) {}  // no-op

private:
    // === System parameters ===
    int L_, d_, chi_max_, D_mpo_;
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
        std::vector<RealType> V;
        int chi;
    };
    std::vector<BoundaryState> boundary_states_;

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

    // Per-bond two-site fused MPO (WW), precomputed once at set_mpo() time.
    // d_WW_[bond] has shape (D*d², D*d²) for each adjacent (site, site+1) pair.
    // Eliminates the per-Lanczos-iteration host roundtrip pattern.
    std::vector<Scalar*> d_WW_;

    std::vector<int> L_env_alloc_chi_;
    std::vector<int> R_env_alloc_chi_;

    // === Per-stream workspace ===
    // All buffers below live on device. Each stream has its own copy so
    // multiple segment sweeps can run concurrently without interfering.
    struct StreamWorkspace {
        // Contraction intermediates
        Scalar* d_T1;
        Scalar* d_T2;
        Scalar* d_T3;   // generic device scratch (kept for parity with -gpu)

        // Lanczos
        Scalar* d_theta;
        Scalar* d_heff_result;
        Scalar* d_lanczos_v;
        Scalar* d_ritz_coeffs;

        // Device-pointer-mode scratch for rocBLAS BLAS-1 results.
        Scalar*   d_dot_result;
        RealType* d_nrm2_result;
        RealType* d_inv_nrm;
        Scalar*   d_neg_alpha;
        Scalar*   d_neg_overlap;
        Scalar*   d_neg_beta_scalars;  // [max_lanczos_iter]

        // Per-iteration alpha/beta arrays on device.
        RealType* d_alpha_dev;         // [max_lanczos_iter]
        RealType* d_beta_dev;          // [max_lanczos_iter]

        // rocSOLVER dsteqr workspaces — fully on-device tridiagonal eigensolve.
        double*      d_steqr_D;
        double*      d_steqr_E;
        double*      d_steqr_C;        // (max_iter × max_iter)
        rocblas_int* d_steqr_info;

        // SVD workspace. Truncation runs on device via extract_cols_kernel
        // and scale_rows/cols_by_diag_kernel from common/scalar_traits.h.
        Scalar* d_svd_A;
        Scalar* d_svd_U;
        RealType* d_svd_S;
        Scalar* d_svd_Vh;
        Scalar* d_svd_work;            // device scratch for S*Vh (or U*S)
        RealType* d_svd_E;
        int* d_svd_info;
        // rocsolver_gesvdj output scalars (residual + sweep count, on device).
        double*      d_svdj_residual;
        rocblas_int* d_svdj_n_sweeps;

        // Pre-allocated device scratch for form_theta_with_V (psi_R copy).
        Scalar* d_psi_R;
        // Tiny host buffer used only for the truncation-rank decision.
        std::vector<RealType> h_svd_S;

        // GPU-native accurate SVD (Stoudenmire recursive) for boundary merges.
        // Stoudenmire is part of pdmrg's algorithmic definition — without it
        // V = Λ⁻¹ amplifies SVD's poor relative accuracy on small singular
        // values, contaminating subsequent sweeps. NOT an "advanced
        // optimization" gated by the -base charter; mandatory for every
        // pdmrg variant. See common/accurate_svd_gpu.h.
        AsvdScratch<Scalar> asvd;
    };
    std::vector<StreamWorkspace> workspaces_;

    int theta_size_max_;
    int max_lanczos_iter_;
    bool lanczos_use_1site_;  // when true, Lanczos calls apply_heff_single_site

    // === Two-site core methods (stream-aware via si) ===
    void form_theta_two_site(int site, int si);
    void apply_heff_two_site(int site, const Scalar* d_in, Scalar* d_out, int si);
    double lanczos_eigensolver(int site, Scalar* d_theta, int theta_size, int si);
    void svd_split(int site, Scalar* d_theta, char direction, int si);
    double optimize_bond(int site, char direction, int si);

    // === Single-site methods (for warmup and polish) ===
    void apply_heff_single_site(int site, const Scalar* d_in, Scalar* d_out, int si);
    void svd_split_single_site(int site, Scalar* d_theta, char direction, int si);
    double optimize_site_single(int site, char direction, int si);

    // === Environment updates ===
    void update_left_env(int site, int si);
    void update_right_env(int site, int si);
    void ensure_L_env_alloc(int idx, int chi);
    void ensure_R_env_alloc(int idx, int chi);

    // === Initialization ===
    void precompute_WW();   // host-side per-bond WW build at set_mpo time
    void build_initial_environments();
    void allocate_mps_tensor(int site, int cL, int cR);
    void partition_chain();
    void initialize_boundary_states();

    // === Sweep methods ===
    double sweep_LR_full();
    double sweep_RL_full();
    double sweep_LR_full_1site();
    double sweep_RL_full_1site();
    void segment_sweep_LR(int seg_idx);
    void segment_sweep_RL(int seg_idx);
    double merge_and_optimize_boundaries(int parity = -1);
    void form_theta_with_V(int site, int boundary_idx, int si);

    // === Memory management ===
    void allocate_stream_workspaces();
    void free_gpu_resources();
};

// Include template implementation
#include "pdmrg_gpu_base_impl.h"

#endif // PDMRG_GPU_BASE_H
