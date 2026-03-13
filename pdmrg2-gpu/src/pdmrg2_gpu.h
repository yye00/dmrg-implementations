#ifndef PDMRG2_GPU_H
#define PDMRG2_GPU_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <string>
#include "scalar_traits.h"

/**
 * PDMRG2-GPU: Stream-Parallel DMRG with Newton-Schulz + Block-Davidson
 *
 * Key algorithmic changes from pdmrg-gpu:
 * 1. Newton-Schulz polar decomposition replaces QR/SVD for canonicalization
 *    and bond splitting (BLAS-3 GEMM-heavy, GPU-native)
 * 2. Block-Davidson eigensolver replaces Lanczos (BLAS-3 dominant)
 *
 * Templated on Scalar: double (real) or hipDoubleComplex (complex128).
 */
template<typename Scalar>
class PDMRG2GPU {
    using Traits = ScalarTraits<Scalar>;
    using RealType = typename Traits::RealType;

public:
    PDMRG2GPU(int L, int d, int chi_max, int D_mpo, int n_segments, double tol = 1e-10);
    ~PDMRG2GPU();

    // Setup
    void set_mpo(const std::vector<Scalar*>& h_mpo_tensors);
    void initialize_mps_random(double scale = 0.1);

    // Run
    double run(int n_outer_sweeps, int n_local_sweeps = 2, int n_warmup = 3);

    // Results
    double get_energy() const { return energy_; }
    void get_mps(std::vector<std::vector<Scalar>>& h_mps) const;
    void set_cpu_svd(bool use_cpu) { use_cpu_svd_ = use_cpu; }
    void set_use_ns_split(bool use_ns) { use_ns_split_ = use_ns; }

    int chi_L(int site) const { return bond_dims_[site]; }
    int chi_R(int site) const { return bond_dims_[site + 1]; }

private:
    // System parameters
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

    // === HIP streams and rocBLAS handles ===
    std::vector<hipStream_t> streams_;
    std::vector<rocblas_handle> handles_;

    // === GPU tensors ===
    std::vector<Scalar*> d_mps_tensors_;
    std::vector<Scalar*> d_mpo_tensors_;
    std::vector<Scalar*> d_L_envs_;
    std::vector<Scalar*> d_R_envs_;
    std::vector<Scalar*> d_W_left_;
    std::vector<Scalar*> d_W_right_;
    std::vector<Scalar*> d_WW_;
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

        // === Newton-Schulz workspace ===
        Scalar* d_ns_U;        // (max_ns_m, max_ns_n) iterate
        Scalar* d_ns_U_new;    // (max_ns_m, max_ns_n) next iterate
        Scalar* d_ns_gram;     // (max_ns_n, max_ns_n) U^H U or Q Q^H
        Scalar* d_ns_P;        // (max_ns_n, max_ns_n) remainder

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

        // NS-split eigendecomp workspace
        std::vector<Scalar> h_ns_PtP;
        std::vector<RealType> h_ns_eigvals;
        std::vector<Scalar> h_ns_syev_work;
        std::vector<RealType> h_ns_syev_rwork;
    };
    std::vector<StreamWorkspace> workspaces_;

    bool use_cpu_svd_;
    bool use_ns_split_;
    int theta_size_max_;
    int max_lanczos_iter_;
    int davidson_b_;
    int davidson_max_sub_;

    // === Core methods ===
    void form_theta_two_site(int site, int si);
    void apply_heff_two_site(int site, const Scalar* d_in, Scalar* d_out, int si);
    double block_davidson_eigensolver(int site, Scalar* d_theta, int theta_size, int si);
    double lanczos_eigensolver(int site, Scalar* d_theta, int theta_size, int si);
    void ns_split(int site, Scalar* d_theta, char direction, int si);
    void svd_split(int site, Scalar* d_theta, char direction, int si);
    double optimize_bond(int site, char direction, int si);

    // === Newton-Schulz polar decomposition ===
    // Left variant (tall/square, m >= n): A = U @ P, U^H U = I
    void newton_schulz_left(Scalar* d_A, int m, int n, Scalar* d_U, Scalar* d_P,
                            int si, double tol = 1e-10, int* out_iters = nullptr);
    // Right variant (wide, m < n): A = L @ Q, Q Q^H = I
    void newton_schulz_right(Scalar* d_A, int m, int n, Scalar* d_L, Scalar* d_Q,
                             int si, double tol = 1e-10, int* out_iters = nullptr);

    // === Canonicalization via Newton-Schulz ===
    void left_canonize_site(int site, int si);
    void right_canonize_site(int site, int si);
    void canonize_segment_right(int seg_idx);  // right-canonize all sites in segment
    void canonize_segment_left(int seg_idx);   // left-canonize all sites in segment

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
    double sweep_LR_full();
    double sweep_RL_full();
    void segment_sweep_LR(int seg_idx);
    void segment_sweep_RL(int seg_idx);
    double merge_and_optimize_boundaries(int parity = -1);  // Stoudenmire boundary coupling

    // === Partitioning ===
    void partition_chain();

    // === Memory management ===
    void allocate_stream_workspaces();
    void free_gpu_resources();
};

#include "pdmrg2_gpu_impl.h"

#endif // PDMRG2_GPU_H
