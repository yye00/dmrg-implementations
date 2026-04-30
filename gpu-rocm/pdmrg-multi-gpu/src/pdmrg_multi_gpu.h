#ifndef PDMRG_MULTI_GPU_H
#define PDMRG_MULTI_GPU_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <string>
#include "scalar_traits.h"
#include "../../common/gpu_opts.h"
#include "../../common/accurate_svd_gpu.h"
#include "../../common/pointer_mode_guard.h"

/**
 * Multi-GPU Stream-Parallel DMRG (PDMRG) across multiple MI300X devices
 *
 * Companion variant to pdmrg-gpu: each segment is assigned to a separate
 * GPU device instead of separate HIP streams on the same device.
 *
 * - Each device owns its segment's MPS tensors, environments, and workspace
 * - MPO tensors are replicated to all devices (read-only, small)
 * - Boundary merges use P2P access or explicit D2D staging
 * - Warmup and polish phases gather MPS to device 0 for full-chain sweeps
 *
 * GpuOpts scope (round-12 doc fix): only `rsvd` and `profile` are
 * honoured. pdmrg-multi-gpu is Lanczos-only (no Davidson path), so
 * `lanczos_graph` is N-A here. sparse_mpo / fuse_lanczos / d_pad /
 * device_k are NOT wired —
 * pdmrg-multi-gpu's optimization charter is multi-device parallelism,
 * not single-device kernel ablations. For those, run pdmrg-gpu-opt on
 * one device. The opts_ field is kept so env-var loading still works
 * uniformly, but unsupported flags are no-ops.
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
    void initialize_mps_product();
    void initialize_mps_neel();

    // Run. PDMRG-rules-2026-04-15 lock: warmup and polish MUST be single-site,
    // capped at 2 each, and zero is a supported configuration. The CLI driver
    // is required to pass --warmup and --polish explicitly so that compiled-in
    // defaults cannot drift unnoticed across rebench campaigns.
    double run(int n_outer_sweeps, int n_local_sweeps = 2, int n_warmup = 1, int n_polish = 0);

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
        // Boundary R_env must be built from canonical Vh, not S·Vh — pre-
        // allocated swap buffer (round-8 self-audit C-new1-ext; mirrors
        // pdmrg-gpu and pdmrg-gpu-opt).
        Scalar* d_Vh_canonical;
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
        Scalar* d_rsvd_U_small; // (r, r) on-device U_small from rocsolver_gesvd of B

        // GPU-native accurate SVD scratch for Stoudenmire boundary merges
        // (algorithmic upgrade — multi-gpu previously used plain svd_split).
        AsvdScratch<Scalar> asvd;

        // Segment info
        int seg_first, seg_last;
        int seg_len;     // = seg_last - seg_first + 1
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

    // Environment-driven opt-in flags (DMRG_GPU_OPT_*). load_from_env() is
    // called in the constructor. Per the class docstring: only `rsvd` and
    // `profile` are honoured here. Other flags load but are unused —
    // pdmrg-multi-gpu's optimization charter is multi-device parallelism,
    // not single-device kernel ablations (use pdmrg-gpu-opt for those).
    GpuOpts opts_;

    // Phase timers (round-12 instrumentation). Recorded only when
    // opts_.profile is on. Single-stream-pair instrumentation: events
    // record on devices_[0].stream during full-chain warmup/polish; the
    // segment_sweep paths (per-device parallel) are NOT timed to keep the
    // panel cost-free in the parallel hot path.
    PhaseTimer t_lanczos_;
    PhaseTimer t_apply_heff_;
    PhaseTimer t_svd_;
    PhaseTimer t_absorb_;
    PhaseTimer t_env_update_;
    void init_timers() {
        t_lanczos_.init("lanczos", opts_.profile);
        t_apply_heff_.init("apply_heff", opts_.profile);
        t_svd_.init("svd", opts_.profile);
        t_absorb_.init("absorb", opts_.profile);
        t_env_update_.init("env_update", opts_.profile);
    }
    void report_timers() {
        if (!opts_.profile) return;
        std::fprintf(stderr, "== pdmrg-multi-gpu phase timers (device 0 only) ==\n");
        // Skip uninstrumented phases — printing 0.00 ms / 0 calls would be
        // noise. Currently lanczos / apply_heff / svd are wired; absorb
        // and env_update remain available for forward instrumentation
        // without changing the panel layout.
        auto print_if_used = [](const char* lbl, PhaseTimer& t) {
            if (t.calls() == 0) return;
            std::fprintf(stderr, "  %-12s : %8.2f ms (%d calls)\n",
                         lbl, t.total_ms(), t.calls());
        };
        print_if_used("lanczos",    t_lanczos_);
        print_if_used("apply_heff", t_apply_heff_);
        print_if_used("svd",        t_svd_);
        print_if_used("absorb",     t_absorb_);
        print_if_used("env_update", t_env_update_);
    }
public:
    GpuOpts& opts() { return opts_; }
private:

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

    // === Sweep methods ===
    // Full-chain sweeps on device 0 (warmup/polish)
    double sweep_LR_full_1site();
    double sweep_RL_full_1site();
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
