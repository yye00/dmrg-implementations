#ifndef PDMRG_GPU_BASE_H
#define PDMRG_GPU_BASE_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <string>
#include "scalar_traits.h"

/**
 * Stream-Parallel DMRG (PDMRG) on GPU — NAIVE BASELINE
 *
 * Unoptimized reference implementation of stream-parallel two-site DMRG.
 * Preserves PDMRG's defining feature — chain partitioned into P segments
 * running concurrently on P HIP streams — but strips every per-stream
 * optimization from the `pdmrg-gpu` implementation:
 *
 *   - No fused two-site MPO cache (d_WW): the WW tensor is rebuilt on
 *     the HOST inside apply_heff_two_site on every Lanczos iteration
 *     and uploaded back to the GPU. Cost is never amortized.
 *   - No gemm_batched: all contractions in apply_heff / update_*_env
 *     use nested for-loops of single rocBLAS gemm calls.
 *   - No device-pointer-mode Lanczos: host-pointer mode throughout, with
 *     CPU LAPACK dstev for the tridiagonal eigenproblem.
 *   - No custom GPU kernels (except the complex-conjugate helper needed
 *     for bra correctness on complex environments).
 *   - No randomized SVD, no on-device SVD truncation.
 *   - No boundary accurate-SVD (Stoudenmire V = Λ⁻¹ still used, but the
 *     singular values come from the plain rocSOLVER gesvd + host truncation).
 *   - Single-site warmup and polish sweeps still exist because they are
 *     part of the PDMRG algorithm (not an optimization), but they run
 *     through the same naive matvec/Lanczos/SVD paths.
 *
 * Hard-coded defaults:
 *   - n_warmup       = 3   (single-site warmup sweeps before PDMRG loop)
 *   - n_outer        = 20  (outer PDMRG iterations; chosen by caller)
 *   - n_local        = 2   (inner local sweeps per outer iteration)
 *   - Polish sweeps  = 10  (two-site full-chain polish after PDMRG loop)
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

    // Run PDMRG. n_local_sweeps and n_warmup are hard-coded if the caller
    // omits them; n_outer_sweeps is always supplied explicitly.
    double run(int n_outer_sweeps, int n_local_sweeps = 2, int n_warmup = 3);

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

    std::vector<int> L_env_alloc_chi_;
    std::vector<int> R_env_alloc_chi_;

    // === Per-stream workspace — minimal naive set ===
    struct StreamWorkspace {
        // Contraction intermediates
        Scalar* d_T1;
        Scalar* d_T2;
        Scalar* d_T3;  // per-call WW scratch (D*d² × D*d²)

        // Lanczos
        Scalar* d_theta;
        Scalar* d_heff_result;
        Scalar* d_lanczos_v;
        Scalar* d_ritz_coeffs;

        // Host-side Lanczos workspace (CPU LAPACK dstev)
        std::vector<double> h_alpha;
        std::vector<double> h_beta;
        std::vector<double> h_steqr_work;
        std::vector<double> h_steqr_Z;

        // SVD workspace (rocSOLVER gesvd + host-side truncation)
        Scalar* d_svd_A;
        Scalar* d_svd_U;
        RealType* d_svd_S;
        Scalar* d_svd_Vh;
        RealType* d_svd_E;
        int* d_svd_info;
        std::vector<Scalar> h_svd_U, h_svd_Vh, h_svd_tmp;
        std::vector<RealType> h_svd_S;
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
