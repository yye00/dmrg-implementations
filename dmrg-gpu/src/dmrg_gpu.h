#ifndef DMRG_GPU_H
#define DMRG_GPU_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <hiptensor/hiptensor.h>
#include <hiptensor/hiptensor_types.h>
#include <vector>
#include <string>
#include <map>
#include <utility>

/**
 * GPU-native DMRG - Single-site optimization
 *
 * ALL tensor contractions run on GPU via hipTensor.
 * ALL linear algebra runs on GPU via rocBLAS/rocSOLVER.
 * Only CPU work: control flow, convergence checks on scalars,
 *                small tridiagonal eigensolve (Lanczos, ~100 elements).
 */
class DMRGGPU {
public:
    DMRGGPU(int L, int d, int chi_max, int D_mpo, double tol = 1e-10);
    ~DMRGGPU();

    double run(int n_sweeps);

    void initialize_mps_random(double scale = 0.1);
    void initialize_mps_product();
    void initialize_mps_neel();
    void load_mps_from_file(const std::string& filename);

    void set_mpo(const std::vector<double*>& h_mpo_tensors);

    double get_energy() const { return energy_; }
    void get_mps(std::vector<std::vector<double>>& h_mps) const;

    int chi_L(int site) const { return bond_dims_[site]; }
    int chi_R(int site) const { return bond_dims_[site + 1]; }

private:
    // System parameters
    int L_, d_, chi_max_, D_mpo_;
    double tol_;
    double energy_;

    // Bond dimensions: bond_dims_[i] = dim of bond between site i-1 and site i
    std::vector<int> bond_dims_;

    // GPU tensor data
    std::vector<double*> d_mps_tensors_;
    std::vector<double*> d_mpo_tensors_;
    std::vector<double*> d_L_envs_;
    std::vector<double*> d_R_envs_;

    std::vector<int> L_env_alloc_chi_;
    std::vector<int> R_env_alloc_chi_;

    // GPU handles
    hipStream_t stream_;
    rocblas_handle rocblas_h_;

    // hipTensor handle and contraction intermediates
    hiptensorHandle_t ht_handle_;
    double* d_T1_;  // max-sized contraction intermediate
    double* d_T2_;  // max-sized contraction intermediate

    // Cached hipTensor plans keyed by (dim1, dim2)
    struct CachedPlans {
        hiptensorPlan_t plan1, plan2, plan3;
        void* ws1 = nullptr;
        void* ws2 = nullptr;
        void* ws3 = nullptr;
        uint64_t ws_sz1 = 0, ws_sz2 = 0, ws_sz3 = 0;
    };

    std::map<std::pair<int,int>, CachedPlans*> heff_plan_cache_;
    std::map<std::pair<int,int>, CachedPlans*> lenv_plan_cache_;
    std::map<std::pair<int,int>, CachedPlans*> renv_plan_cache_;

    CachedPlans* get_heff_plans(int chi_L, int chi_R);
    CachedPlans* get_lenv_plans(int chi_in, int chi_out);
    CachedPlans* get_renv_plans(int chi_in, int chi_out);

    // Lanczos / optimization workspace
    double* d_theta_;
    double* d_heff_result_;
    int theta_size_max_;

    // SVD workspace (pre-allocated at max size)
    double* d_svd_A_;      // copy of theta for SVD (rocsolver overwrites)
    double* d_svd_U_;      // left singular vectors
    double* d_svd_S_;      // singular values
    double* d_svd_Vh_;     // right singular vectors (V^T)
    double* d_svd_E_;      // superdiagonal workspace
    int*    d_svd_info_;   // SVD convergence info
    double* d_svd_work_;   // temp for S*Vh or U*S

    // Core algorithm
    void build_initial_environments();
    void update_left_env(int site);
    void update_right_env(int site);
    void ensure_L_env_alloc(int idx, int chi);
    void ensure_R_env_alloc(int idx, int chi);

    double optimize_site(int site, char direction);
    void form_theta(int site, double* d_theta);
    void apply_heff(int site, const double* d_theta, double* d_result);
    double lanczos_eigensolver(int site, double* d_theta);
    void svd_and_update_mps(int site, double* d_theta, char direction);

    double sweep_left_to_right();
    double sweep_right_to_left();

    void allocate_mps_tensor(int site, int cL, int cR);
    void free_gpu_resources();
};

#endif // DMRG_GPU_H
