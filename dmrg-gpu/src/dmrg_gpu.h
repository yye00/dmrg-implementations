#ifndef DMRG_GPU_H
#define DMRG_GPU_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <string>

class AccurateSVD_GPU;

/**
 * Reference DMRG on GPU - Single-site optimization
 * 
 * Standard sequential DMRG algorithm:
 * - No parallelism (unlike PDMRG)
 * - No canonicalization complexity
 * - Clean baseline for validation
 */
class DMRGGPU {
public:
    DMRGGPU(int L, int d, int chi_max, int D_mpo, double tol = 1e-10);
    ~DMRGGPU();

    // Main algorithm
    double run(int n_sweeps);
    
    // MPS initialization
    void initialize_mps_random(double scale = 0.1);
    void initialize_mps_product();  // All spin up
    void initialize_mps_neel();     // Alternating up/down
    void load_mps_from_file(const std::string& filename);
    
    // MPO setup
    void set_mpo(const std::vector<double*>& h_mpo_tensors);
    
    // Access results
    double get_energy() const { return energy_; }
    void get_mps(std::vector<std::vector<double>>& h_mps) const;

    // Bond dimension at a bond (between site i-1 and site i)
    // bond_dims_[0] = 1 (left boundary), bond_dims_[L] = 1 (right boundary)
    int chi_L(int site) const { return bond_dims_[site]; }
    int chi_R(int site) const { return bond_dims_[site + 1]; }

private:
    // System parameters
    int L_;           // Chain length
    int d_;           // Physical dimension (2 for spin-1/2)
    int chi_max_;     // Maximum bond dimension
    int D_mpo_;       // MPO bond dimension
    double tol_;      // Convergence tolerance
    
    // Current state
    double energy_;
    
    // Bond dimensions: bond_dims_[i] = dimension of bond between site i-1 and site i
    // bond_dims_[0] = 1 (left boundary)
    // bond_dims_[L] = 1 (right boundary)
    std::vector<int> bond_dims_;
    
    // GPU data
    std::vector<double*> d_mps_tensors_;  // MPS tensors [L]
    std::vector<double*> d_mpo_tensors_;  // MPO tensors [L]
    std::vector<double*> d_L_envs_;       // Left environments [L+1]
    std::vector<double*> d_R_envs_;       // Right environments [L+1]
    
    // Allocation tracking for environments
    std::vector<int> L_env_alloc_chi_;  // Allocated chi for L_envs
    std::vector<int> R_env_alloc_chi_;  // Allocated chi for R_envs
    
    // GPU resources
    hipStream_t stream_;
    rocblas_handle rocblas_h_;
    AccurateSVD_GPU* svd_;
    
    // Workspace buffers
    double* d_theta_;        // Current site wavefunction
    double* d_heff_result_;  // H_eff application result
    int theta_size_max_;
    
    // Environment building
    void build_initial_environments();
    void update_left_env(int site);   // Update L[site+1] from L[site], MPS[site], MPO[site]
    void update_right_env(int site);  // Update R[site] from R[site+1], MPS[site], MPO[site]
    void ensure_L_env_alloc(int idx, int chi);
    void ensure_R_env_alloc(int idx, int chi);
    
    // Site optimization
    double optimize_site(int site, char direction);  // 'L' or 'R'
    void form_theta(int site, double* d_theta);
    void apply_heff(int site, const double* d_theta, double* d_result);
    double lanczos_eigensolver(int site, double* d_theta);
    void svd_and_update_mps(int site, double* d_theta, char direction);
    
    // Sweeping
    double sweep_left_to_right();
    double sweep_right_to_left();
    
    // Utilities
    void allocate_mps_tensor(int site, int cL, int cR);
    void free_gpu_resources();
};

#endif // DMRG_GPU_H
