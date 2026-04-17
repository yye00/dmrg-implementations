#ifndef RADAM_GPU_H
#define RADAM_GPU_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <string>
#include "scalar_traits.h"

/**
 * RADAM-GPU: Riemannian Adam optimizer for MPS / TT ground state search.
 *
 * Right-canonical MPS gauge with orthogonality center at site 0.
 * Optional DMRG1 warmstart sweeps to seed the optimizer from a Lanczos
 * single-site local solve.
 *
 * Templated on Scalar: double (real) or hipDoubleComplex (complex128).
 * Single-stream design — no segmentation, no worker pool.
 */

template<typename Scalar>
class RAdamGPU {
    using Traits = ScalarTraits<Scalar>;
    using RealType = typename Traits::RealType;

public:
    struct Config {
        int max_epochs = 500;
        int n_warmup = 0;            // DMRG1 single-site warmstart sweeps
        double lr = 1e-3;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-8;
        double grad_tol = 1e-9;
        double energy_tol = 1e-12;
        bool cosine_lr = false;
        bool verbose = true;
        int log_every = 10;
    };

    RAdamGPU(int L, int d, int chi_max, int D_mpo, double tol = 1e-10);
    ~RAdamGPU();

    void set_mpo(const std::vector<Scalar*>& h_mpo_tensors);
    void initialize_mps_random(double scale = 0.1, int seed = 42);

    // Optimization driver. Returns final energy.
    double run(const Config& cfg);

    // Inspection
    double get_energy() const { return energy_; }
    int    get_epochs() const { return n_epochs_done_; }
    void get_mps(std::vector<std::vector<Scalar>>& h_mps) const;
    void set_cpu_svd(bool use_cpu) { use_cpu_svd_ = use_cpu; }

    int chi_L(int site) const { return bond_dims_[site]; }
    int chi_R(int site) const { return bond_dims_[site + 1]; }

private:
    // Problem parameters
    int L_, d_, chi_max_, D_mpo_;
    double tol_;
    double energy_;
    int n_epochs_done_;

    // Bond dimensions (tapered near edges, capped at chi_max_)
    std::vector<int> bond_dims_;

    // Single stream + rocBLAS handle
    hipStream_t stream_;
    rocblas_handle handle_;

    // === GPU tensors ===
    std::vector<Scalar*> d_mps_;          // per-site MPS cores, flat (chi_L, d, chi_R)
    std::vector<Scalar*> d_mps_alloc_sz_; // dummy — we always allocate the max
    std::vector<Scalar*> d_mpo_;          // per-site MPO, shape (D, d, d, D) col-major
    std::vector<Scalar*> d_W_left_;       // fused W_left per site: (D*d, d*D)
    std::vector<Scalar*> d_W_right_;      // fused W_right per site: (D*d, d*D)
    std::vector<Scalar*> d_L_H_;          // left H-env [0..L], shape (chi, D_mpo, chi)
    std::vector<Scalar*> d_R_H_;          // right H-env [0..L]
    std::vector<Scalar*> d_L_N_;          // left norm-env [0..L], shape (chi, chi)

    // Radam state (device-resident where scalar, host-resident where scalar)
    std::vector<Scalar*> d_M_;            // first-moment per site
    std::vector<Scalar*> d_grad_;         // reused for Euclidean gradient and step direction

    // Scratch
    Scalar* d_T1_;
    Scalar* d_T2_;
    Scalar* d_heff_result_;
    Scalar* d_theta_;
    Scalar** d_batch_A_;
    Scalar** d_batch_B_;
    Scalar** d_batch_C_;
    Scalar* d_dot_result_;
    RealType* d_nrm2_result_;

    // Lanczos workspace (DMRG1 warmstart)
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

    // SVD workspace (CPU SVD path for canonicalization)
    std::vector<Scalar>   h_svd_A_, h_svd_U_, h_svd_Vh_, h_svd_work_, h_svd_tmp_;
    std::vector<RealType> h_svd_S_, h_svd_rwork_;
    Scalar* d_svd_work_;  // device scratch for absorbing S*Vh / U*S into neighbors

    // Bond-max workspace
    int bond_size_max_;   // max theta size (chi_max*d*chi_max)

    bool use_cpu_svd_;

    // === Core methods ===

    // Memory
    void allocate_gpu_memory();
    void free_gpu_memory();

    // MPO fusion (from pdmrg-gpu-opt set_mpo/precompute_fused_mpo, minus WW two-site fusion)
    void upload_and_fuse_mpo(const std::vector<Scalar*>& h_mpo_tensors);

    // MPS canonicalization
    void right_canonicalize_mps();            // CPU QR sweep (copy D2H, QR, copy back)
    void normalize_site0();                   // divide d_mps_[0] by sqrt(<X|X>)

    // Env builders (single-stream, patterns from pdmrg-gpu-opt update_*_env)
    void update_left_H_env(int site);
    void update_right_H_env(int site);
    void build_initial_H_envs();              // sets boundaries, runs L→R then R→L chain
    void build_all_L_N_envs();                // left norm-envs L_N_[0..L]

    // Effective Hamiltonian (single-site, port of apply_heff_single_site)
    void apply_heff(int site, const Scalar* d_in, Scalar* d_out);

    // Single-site norm operator: out = L_N_[site] * in * (R_N=identity for right-canonical)
    void apply_norm(int site, const Scalar* d_in, Scalar* d_out);

    // Gradient
    double compute_energy_from_envs();
    double compute_all_gradients();           // returns current energy E
    void   tangent_project_inplace();         // in-place project d_grad_[i] onto tangent at d_mps_
    double grad_frobenius_norm_sq();

    // Adam update: overwrites d_grad_[i] with step direction Delta_i (see radam_gpu_impl.h)
    void adam_update(int step, double lr, double beta1, double beta2, double eps,
                     double& v_state, double& grad_norm_out);

    // Retraction
    void retract_and_recanonicalize();

    // DMRG1 warmstart helpers (copied / simplified from pdmrg-gpu-opt)
    double lanczos_eigensolver(int site, Scalar* d_theta, int theta_size);
    void   svd_split_single_site(int site, Scalar* d_theta, char direction);
    double optimize_site_single(int site, char direction);
    double dmrg1_sweep_LR();
    double dmrg1_sweep_RL();

    // MPS per-site size helpers
    size_t max_site_size() const { return (size_t)chi_max_ * d_ * chi_max_; }
};

#include "radam_gpu_impl.h"

#endif // RADAM_GPU_H
