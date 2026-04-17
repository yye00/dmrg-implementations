#ifndef RLBFGS_GPU_H
#define RLBFGS_GPU_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <string>
#include "scalar_traits.h"

/**
 * RLBFGS-GPU: Riemannian L-BFGS optimizer for MPS / TT ground state search.
 *
 * Right-canonical MPS gauge with orthogonality center at site 0.
 * Optional DMRG1 warmstart sweeps to seed the optimizer from a Lanczos
 * single-site local solve.
 *
 * Templated on Scalar: double (real) or hipDoubleComplex (complex128).
 * Single-stream design — no segmentation, no worker pool.
 *
 * Algorithm (ported from cpu/rlbfgs/rlbfgs/optimizer.py):
 *   1. Riemannian gradient g_k = P_{X_k}(grad_E_k)
 *   2. Two-loop recursion with on-the-fly vector transport
 *   3. Armijo backtracking line search (sufficient-decrease)
 *   4. Cautious pair update: push (s,y) iff <s,y> > eps * |s||y|
 */

template<typename Scalar>
class RLBFGSGPU {
    using Traits = ScalarTraits<Scalar>;
    using RealType = typename Traits::RealType;

public:
    struct Config {
        int max_epochs = 200;
        int n_warmup = 0;           // DMRG1 single-site warmstart sweeps (<=2 per project rule)
        int history_size = 10;      // L-BFGS memory m
        double grad_tol = 1e-10;    // L-BFGS with line search can reach much tighter than Adam
        double energy_tol = 1e-14;
        // Armijo line search
        double c1 = 1e-4;
        double alpha_init = 1.0;
        double beta = 0.5;          // backtrack shrink factor
        int ls_max_iter = 25;
        double min_alpha = 1e-14;
        // Cautious update
        double cautious_eps = 1e-10;
        bool verbose = true;
        int log_every = 1;
    };

    RLBFGSGPU(int L, int d, int chi_max, int D_mpo, double tol = 1e-10);
    ~RLBFGSGPU();

    void set_mpo(const std::vector<Scalar*>& h_mpo_tensors);
    void initialize_mps_random(double scale = 0.1, int seed = 42);

    double run(const Config& cfg);

    double get_energy() const { return energy_; }
    int    get_epochs() const { return n_epochs_done_; }
    void get_mps(std::vector<std::vector<Scalar>>& h_mps) const;
    void set_cpu_svd(bool use_cpu) { use_cpu_svd_ = use_cpu; }

    int chi_L(int site) const { return bond_dims_[site]; }
    int chi_R(int site) const { return bond_dims_[site + 1]; }

private:
    int L_, d_, chi_max_, D_mpo_;
    double tol_;
    double energy_;
    int n_epochs_done_;

    std::vector<int> bond_dims_;

    hipStream_t stream_;
    rocblas_handle handle_;

    // === GPU tensors (identical layout to radam-gpu) ===
    std::vector<Scalar*> d_mps_;
    std::vector<Scalar*> d_mps_alloc_sz_;
    std::vector<Scalar*> d_mpo_;
    std::vector<Scalar*> d_W_left_;
    std::vector<Scalar*> d_W_right_;
    std::vector<Scalar*> d_L_H_;
    std::vector<Scalar*> d_R_H_;
    std::vector<Scalar*> d_L_N_;

    // L-BFGS state
    //   d_grad_[i]      : current Riemannian gradient g_k (reused for step direction)
    //   d_g_prev_[i]    : previous gradient g_{k-1} (for y_k = g_k - transport(g_{k-1}))
    //   d_dir_[i]       : current search direction d_k = -H_k g_k
    //   d_hist_s_[j][i] : stored tangent vector s_j for pair j (per-site core)
    //   d_hist_y_[j][i] : stored tangent vector y_j for pair j
    //   d_trial_mps_[i] : trial MPS during line search
    //   d_trial_g_[i]   : gradient at trial point
    std::vector<Scalar*> d_grad_;
    std::vector<Scalar*> d_g_prev_;
    std::vector<Scalar*> d_dir_;
    std::vector<Scalar*> d_trial_mps_;
    std::vector<Scalar*> d_trial_g_;
    std::vector<std::vector<Scalar*>> d_hist_s_;
    std::vector<std::vector<Scalar*>> d_hist_y_;
    std::vector<double>               h_hist_rho_;   // host-side 1/<s,y>
    int hist_count_ = 0;      // number of stored pairs (<= history_size)
    int hist_head_  = 0;      // ring-buffer write head

    // Scratch (same as radam-gpu)
    Scalar* d_T1_;
    Scalar* d_T2_;
    Scalar* d_heff_result_;
    Scalar* d_theta_;
    Scalar** d_batch_A_;
    Scalar** d_batch_B_;
    Scalar** d_batch_C_;
    Scalar* d_dot_result_;
    RealType* d_nrm2_result_;

    // Lanczos workspace
    int max_lanczos_iter_;
    int theta_size_max_;
    Scalar* d_lanczos_v_;
    Scalar* d_ritz_coeffs_;
    Scalar* d_neg_alpha_;
    Scalar* d_neg_overlap_;
    RealType* d_inv_nrm_;
    RealType* d_alpha_dev_;
    RealType* d_beta_dev_;
    Scalar* d_neg_beta_scalars_;
    Scalar* d_const_one_;
    Scalar* d_const_zero_;
    Scalar* d_const_neg_one_;

    // SVD workspace
    std::vector<Scalar>   h_svd_A_, h_svd_U_, h_svd_Vh_, h_svd_work_, h_svd_tmp_;
    std::vector<RealType> h_svd_S_, h_svd_rwork_;
    Scalar* d_svd_work_;

    int bond_size_max_;

    bool use_cpu_svd_;

    // === Core methods ===

    void allocate_gpu_memory();
    void free_gpu_memory();

    void upload_and_fuse_mpo(const std::vector<Scalar*>& h_mpo_tensors);

    void right_canonicalize_mps();
    void normalize_site0();

    void update_left_H_env(int site);
    void update_right_H_env(int site);
    void build_initial_H_envs();
    void build_all_L_N_envs();

    void apply_heff(int site, const Scalar* d_in, Scalar* d_out);
    void apply_norm(int site, const Scalar* d_in, Scalar* d_out);

    double compute_energy_from_envs();
    double compute_all_gradients();

    // Generic tangent projection onto tangent space at d_mps_.
    // Projects the supplied core array in-place.
    void tangent_project_cores(std::vector<Scalar*>& cores);

    // Tangent-space ops across all MPS cores (operate on vectors of per-site pointers)
    double tangent_inner_real(const std::vector<Scalar*>& A,
                              const std::vector<Scalar*>& B) const;
    double tangent_norm_sq(const std::vector<Scalar*>& A) const;
    void   tangent_scale(std::vector<Scalar*>& A, double alpha);
    void   tangent_axpy(double alpha,
                        const std::vector<Scalar*>& X,
                        std::vector<Scalar*>& Y);           // Y += alpha * X
    void   tangent_copy(const std::vector<Scalar*>& Src,
                        std::vector<Scalar*>& Dst);
    void   tangent_zero(std::vector<Scalar*>& A);

    // L-BFGS two-loop: fills d_dir_ with search direction (= -H_k g_k)
    void lbfgs_two_loop();

    // Build d_trial_mps_ = retract(d_mps_ + alpha * d_dir_) and right-canonicalize.
    // On return, envs (H and N) are rebuilt against d_trial_mps_ (via temporary swap).
    double trial_energy_at(double alpha);
    // Commit trial as current: swap d_mps_ ↔ d_trial_mps_, rebuild envs.
    void commit_trial();

    // DMRG1 warmstart
    double lanczos_eigensolver(int site, Scalar* d_theta, int theta_size);
    void   svd_split_single_site(int site, Scalar* d_theta, char direction);
    double optimize_site_single(int site, char direction);
    double dmrg1_sweep_LR();
    double dmrg1_sweep_RL();

    size_t max_site_size() const { return (size_t)chi_max_ * d_ * chi_max_; }
};

#include "rlbfgs_gpu_impl.h"

#endif // RLBFGS_GPU_H
