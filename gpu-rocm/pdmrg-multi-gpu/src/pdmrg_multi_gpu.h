#ifndef PDMRG_MULTI_GPU_H
#define PDMRG_MULTI_GPU_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <string>
#include "scalar_traits.h"

/**
 * Multi-GPU Stream-Parallel DMRG (PDMRG) across multiple MI300X devices
 *
 * Extends pdmrg-gpu: each segment is assigned to a separate GPU device
 * instead of separate HIP streams on the same device.
 *
 * - Each device owns its segment's MPS tensors, environments, and workspace
 * - MPO tensors are replicated to all devices (read-only, small)
 * - Boundary merges use P2P access or explicit D2D staging
 * - Warmup and polish phases gather MPS to device 0 for full-chain sweeps
 *
 * Templated on Scalar: double (real) or hipDoubleComplex (complex128).
 */
template<typename Scalar>
class PDMRGMultiGPU {
    using Traits = ScalarTraits<Scalar>;
    using RealType = typename Traits::RealType;

public:
    PDMRGMultiGPU(int L, int d, int chi_max, int D_mpo, int n_devices, double tol = 1e-10);
    ~PDMRGMultiGPU();

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

    // === Multi-GPU device management ===
    int n_devices_;                     // number of GPU devices to use
    int n_available_devices_;           // total GPUs available on system
    std::vector<bool> peer_access_;     // peer_access_[i*n + j] = P2P enabled i→j

    // === Chain partitioning ===
    int n_segments_;                    // == n_devices_
    std::vector<int> seg_first_;        // seg_first_[k] = first site of segment k
    std::vector<int> seg_last_;         // seg_last_[k] = last site of segment k (inclusive)
    std::vector<int> boundary_bonds_;   // [n_segments-1] bond site at each boundary
    std::vector<int> site_to_device_;   // site_to_device_[site] = device index owning this site

    // === Stoudenmire boundary state (V = Λ⁻¹ at each boundary) ===
    struct BoundaryState {
        std::vector<RealType> V;  // V = 1/S, length = chi at boundary bond
        int chi;                  // current bond dimension at this boundary
    };
    std::vector<BoundaryState> boundary_states_;

    // === Per-device resources ===
    struct DeviceContext {
        int device_id;
        hipStream_t stream;
        rocblas_handle handle;

        // MPS tensors for sites in this device's segment
        // Indexed by LOCAL site: d_mps[local_site]
        std::vector<Scalar*> d_mps;

        // MPO tensors (replicated on each device)
        std::vector<Scalar*> d_mpo;
        std::vector<Scalar*> d_W_left;
        std::vector<Scalar*> d_W_right;
        std::vector<Scalar*> d_WW;

        // Environment tensors for this device's segment range
        // L_envs[local_idx], R_envs[local_idx]
        // For segment [first, last]: L_envs has indices 0..seg_len+1, R_envs same
        std::vector<Scalar*> d_L_envs;
        std::vector<Scalar*> d_R_envs;
        std::vector<int> L_env_alloc_chi;
        std::vector<int> R_env_alloc_chi;

        // Workspace (same as pdmrg-gpu StreamWorkspace)
        Scalar* d_theta;
        Scalar* d_heff_result;
        Scalar* d_T1;
        Scalar* d_T2;
        Scalar* d_lanczos_v;
        Scalar* d_ritz_coeffs;
        Scalar** d_batch_A;
        Scalar** d_batch_B;
        Scalar** d_batch_C;
        Scalar** h_batch_A_pinned;
        Scalar** h_batch_B_pinned;
        Scalar** h_batch_C_pinned;
        Scalar** d_heff_batch_A;
        Scalar** d_heff_batch_C;
        int heff_cached_site;
        Scalar* d_dot_result;
        RealType* d_nrm2_result;
        Scalar* d_neg_alpha;
        Scalar* d_neg_overlap;
        RealType* d_inv_nrm;
        RealType* d_alpha_dev;
        RealType* d_beta_dev;
        Scalar* d_neg_beta_scalars;
        double* d_steqr_D;
        double* d_steqr_E;
        double* d_steqr_C;
        rocblas_int* d_steqr_info;
        Scalar* d_const_one;
        Scalar* d_const_zero;
        Scalar* d_const_neg_one;
        // GPU SVD
        Scalar* d_svd_A;
        Scalar* d_svd_U;
        RealType* d_svd_S;
        Scalar* d_svd_Vh;
        RealType* d_svd_E;
        int* d_svd_info;
        Scalar* d_svd_work;
        // CPU SVD (fallback)
        std::vector<Scalar> h_svd_A, h_svd_U, h_svd_Vh, h_svd_work, h_svd_tmp;
        std::vector<RealType> h_svd_S;
        std::vector<RealType> h_svd_rwork;
        // Randomized truncated SVD
        Scalar* d_rsvd_omega;
        Scalar* d_rsvd_Y;
        Scalar* d_rsvd_Q;
        Scalar* d_rsvd_B;
        Scalar* d_rsvd_ipiv;
        Scalar* d_rsvd_U_full;
        std::vector<Scalar> h_rsvd_B;
        std::vector<Scalar> h_rsvd_U_small;

        // Segment info
        int seg_first, seg_last;
        int seg_len;     // = seg_last - seg_first + 1

        // Staging buffer for boundary merge cross-device copies
        Scalar* d_boundary_staging;
    };
    std::vector<DeviceContext> devices_;

    // === Device 0 resources for warmup/polish full-chain sweeps ===
    // During warmup/polish, all MPS/env are gathered to device 0
    std::vector<Scalar*> d0_mps_tensors_;     // full chain on device 0
    std::vector<Scalar*> d0_L_envs_;
    std::vector<Scalar*> d0_R_envs_;
    std::vector<int> d0_L_env_alloc_chi_;
    std::vector<int> d0_R_env_alloc_chi_;

    bool use_cpu_svd_;
    bool use_rsvd_;
    bool lanczos_use_1site_;
    int rsvd_oversampling_;
    int theta_size_max_;
    int max_lanczos_iter_;

    // === Core methods (device-aware: di = device index) ===
    // Two-site
    void form_theta_two_site(int site, int di);
    void apply_heff_two_site(int site, const Scalar* d_in, Scalar* d_out, int di);
    double lanczos_eigensolver(int site, Scalar* d_theta, int theta_size, int di);
    void svd_split(int site, Scalar* d_theta, char direction, int di);
    void rsvd_split(int site, Scalar* d_theta, char direction, int di);
    double optimize_bond(int site, char direction, int di);

    // Single-site (warmup/polish)
    void apply_heff_single_site(int site, const Scalar* d_in, Scalar* d_out, int di);
    void svd_split_single_site(int site, Scalar* d_theta, char direction, int di);
    double optimize_site_single(int site, char direction, int di);

    // === Environment updates (device-aware) ===
    void update_left_env(int site, int di);
    void update_right_env(int site, int di);
    void ensure_L_env_alloc(int site, int chi, int di);
    void ensure_R_env_alloc(int site, int chi, int di);

    // === Multi-GPU data access helpers ===
    // Get device-local pointer for a global site's MPS tensor
    Scalar* get_mps(int site, int di);
    Scalar* get_L_env(int idx, int di);
    Scalar* get_R_env(int idx, int di);
    Scalar* get_WW(int site, int di);
    Scalar* get_W_left(int site, int di);
    Scalar* get_W_right(int site, int di);
    int local_site(int global_site, int di) const;
    int local_env_idx(int global_idx, int di) const;

    // === Initialization ===
    void build_initial_environments();
    void precompute_fused_mpo(const std::vector<Scalar*>& h_mpo_tensors);
    void allocate_mps_tensor(int site, int cL, int cR, int di);

    // === Cross-device transfers ===
    void gather_mps_to_device0();        // scatter MPS from all devices to device 0
    void scatter_mps_from_device0();     // scatter MPS from device 0 back to owning devices
    void gather_envs_to_device0();       // gather environments to device 0
    void scatter_envs_from_device0();    // scatter environments back to owning devices
    void copy_boundary_mps_to_device(int boundary_idx, int target_device);

    // === Sweep methods ===
    // Full-chain sweeps on device 0 (warmup/polish)
    double sweep_LR_full_1site();
    double sweep_RL_full_1site();
    double sweep_LR_full();   // two-site (for polish)
    double sweep_RL_full();   // two-site (for polish)
    // Segment sweeps (each on its own device)
    void segment_sweep_LR(int seg_idx);
    void segment_sweep_RL(int seg_idx);
    // Boundary merge
    double merge_and_optimize_boundaries(int parity = -1);
    void form_theta_with_V(int site, int boundary_idx, int di);
    void initialize_boundary_states();

    // === Multi-GPU setup ===
    void setup_peer_access();
    void partition_chain();
    void allocate_device_resources();
    void free_gpu_resources();
};

// Include template implementation
#include "pdmrg_multi_gpu_impl.h"

#endif // PDMRG_MULTI_GPU_H
