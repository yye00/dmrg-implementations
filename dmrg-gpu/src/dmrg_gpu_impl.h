#ifndef DMRG_GPU_IMPL_H
#define DMRG_GPU_IMPL_H

#include <rocsolver/rocsolver.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << hipGetErrorString(err) << std::endl; \
            throw std::runtime_error("HIP error"); \
        } \
    } while(0)

#define ROCBLAS_CHECK(call) \
    do { \
        rocblas_status status = call; \
        if (status != rocblas_status_success) { \
            std::cerr << "rocBLAS error in " << __FILE__ << ":" << __LINE__ \
                      << " - status " << status << std::endl; \
            throw std::runtime_error("rocBLAS error"); \
        } \
    } while(0)

// Profiling counters (reset per sweep pair)
static double prof_lanczos_ms = 0, prof_svd_ms = 0, prof_env_ms = 0;
static int prof_lanczos_iters = 0, prof_site_count = 0;
static int prof_heff_calls = 0;

// ============================================================================
// Constructor
// ============================================================================

template<typename Scalar>
DMRGGPU<Scalar>::DMRGGPU(int L, int d, int chi_max, int D_mpo, double tol)
    : L_(L), d_(d), chi_max_(chi_max), D_mpo_(D_mpo), tol_(tol), energy_(0.0) {

    // Bond dimensions
    bond_dims_.resize(L + 1);
    bond_dims_[0] = 1;
    bond_dims_[L] = 1;
    for (int i = 1; i < L; i++) {
        double exact_dim = pow((double)d_, std::min(i, L - i));
        bond_dims_[i] = (exact_dim > chi_max_) ? chi_max_ : (int)exact_dim;
    }

    // GPU handles
    HIP_CHECK(hipStreamCreate(&stream_));
    ROCBLAS_CHECK(rocblas_create_handle(&rocblas_h_));
    ROCBLAS_CHECK(rocblas_set_stream(rocblas_h_, stream_));

    // HIP Graph state
    heff_graph_ = nullptr;
    heff_graph_exec_ = nullptr;
    heff_graph_site_ = -1;

    // Device pointer mode handle for graph capture
    ROCBLAS_CHECK(rocblas_create_handle(&rocblas_h_device_));
    ROCBLAS_CHECK(rocblas_set_stream(rocblas_h_device_, stream_));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(rocblas_h_device_, rocblas_pointer_mode_device));

    // Persistent device-side scalars for graph capture
    HIP_CHECK(hipMalloc(&d_scalar_one_, sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_scalar_zero_, sizeof(Scalar)));
    Scalar h_one = Traits::one(), h_zero = Traits::zero();
    HIP_CHECK(hipMemcpy(d_scalar_one_, &h_one, sizeof(Scalar), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_scalar_zero_, &h_zero, sizeof(Scalar), hipMemcpyHostToDevice));

    // Contraction intermediates
    int t_max = D_mpo_ * d_ * chi_max_ * chi_max_;
    HIP_CHECK(hipMalloc(&d_T1_, t_max * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_T2_, t_max * sizeof(Scalar)));

    // MPS tensors
    d_mps_tensors_.resize(L, nullptr);
    for (int i = 0; i < L; i++) {
        allocate_mps_tensor(i, chi_L(i), chi_R(i));
    }

    // MPO tensors
    d_mpo_tensors_.resize(L, nullptr);

    // W matrices (allocated in set_mpo)
    d_W_left_.resize(L, nullptr);
    d_W_right_.resize(L, nullptr);

    // Environments
    d_L_envs_.resize(L + 1, nullptr);
    d_R_envs_.resize(L + 1, nullptr);
    L_env_alloc_chi_.resize(L + 1, 0);
    R_env_alloc_chi_.resize(L + 1, 0);

    for (int i = 0; i <= L; i++) {
        int chi_alloc = (i == 0 || i == L) ? 1 : chi_max_;
        int sz = chi_alloc * D_mpo_ * chi_alloc;
        HIP_CHECK(hipMalloc(&d_L_envs_[i], sz * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_R_envs_[i], sz * sizeof(Scalar)));
        HIP_CHECK(hipMemset(d_L_envs_[i], 0, sz * sizeof(Scalar)));
        HIP_CHECK(hipMemset(d_R_envs_[i], 0, sz * sizeof(Scalar)));
        L_env_alloc_chi_[i] = chi_alloc;
        R_env_alloc_chi_[i] = chi_alloc;
    }

    // Lanczos workspace
    theta_size_max_ = chi_max_ * d_ * chi_max_;
    max_lanczos_iter_ = std::min(100, theta_size_max_);
    HIP_CHECK(hipMalloc(&d_theta_, theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_theta_staging_, theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_heff_result_, theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_lanczos_v_, (size_t)max_lanczos_iter_ * theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_ritz_coeffs_, max_lanczos_iter_ * sizeof(Scalar)));

    // Batched GEMM pointer arrays — Step 1 (D*d entries)
    int batch1_max = D_mpo_ * d_;
    HIP_CHECK(hipMalloc(&d_batch_A_, batch1_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_B_, batch1_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_C_, batch1_max * sizeof(Scalar*)));

    // Batched GEMM pointer arrays — Step 3 (D*d entries, one set per wp)
    int batch3_max = D_mpo_ * d_;
    HIP_CHECK(hipMalloc(&d_step3_A_, batch3_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_step3_B_, batch3_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_step3_C_, batch3_max * sizeof(Scalar*)));

    // Pinned host pointer arrays (persistent addresses for graph capture)
    HIP_CHECK(hipHostMalloc(&h_pin_A_, batch1_max * sizeof(Scalar*)));
    HIP_CHECK(hipHostMalloc(&h_pin_B_, batch1_max * sizeof(Scalar*)));
    HIP_CHECK(hipHostMalloc(&h_pin_C_, batch1_max * sizeof(Scalar*)));
    HIP_CHECK(hipHostMalloc(&h_pin_A3_, batch3_max * sizeof(Scalar*)));
    HIP_CHECK(hipHostMalloc(&h_pin_B3_, batch3_max * sizeof(Scalar*)));
    HIP_CHECK(hipHostMalloc(&h_pin_C3_, batch3_max * sizeof(Scalar*)));

    // SVD workspace
    int svd_max_dim = chi_max_ * d_;
    HIP_CHECK(hipMalloc(&d_svd_A_,    theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_U_,    svd_max_dim * chi_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_S_,    chi_max_ * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_svd_Vh_,   chi_max_ * svd_max_dim * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_E_,    chi_max_ * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_svd_info_, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_svd_work_, theta_size_max_ * sizeof(Scalar)));

    // Host workspace for SVD results (copied back from GPU)
    h_svd_U_.resize(svd_max_dim * chi_max_);
    h_svd_S_.resize(chi_max_);
    h_svd_Vh_.resize(chi_max_ * svd_max_dim);
    h_svd_tmp_.resize(svd_max_dim * chi_max_);
}

// ============================================================================
// Destructor
// ============================================================================

template<typename Scalar>
DMRGGPU<Scalar>::~DMRGGPU() {
    free_gpu_resources();
}

template<typename Scalar>
void DMRGGPU<Scalar>::free_gpu_resources() {
    for (auto ptr : d_mps_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d_mpo_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d_W_left_) if (ptr) hipFree(ptr);
    for (auto ptr : d_W_right_) if (ptr) hipFree(ptr);
    for (auto ptr : d_L_envs_) if (ptr) hipFree(ptr);
    for (auto ptr : d_R_envs_) if (ptr) hipFree(ptr);

    if (d_theta_) hipFree(d_theta_);
    if (d_theta_staging_) hipFree(d_theta_staging_);
    if (d_heff_result_) hipFree(d_heff_result_);
    if (d_lanczos_v_) hipFree(d_lanczos_v_);
    if (d_ritz_coeffs_) hipFree(d_ritz_coeffs_);
    if (d_T1_) hipFree(d_T1_);
    if (d_T2_) hipFree(d_T2_);
    if (d_batch_A_) hipFree(d_batch_A_);
    if (d_batch_B_) hipFree(d_batch_B_);
    if (d_batch_C_) hipFree(d_batch_C_);
    if (d_step3_A_) hipFree(d_step3_A_);
    if (d_step3_B_) hipFree(d_step3_B_);
    if (d_step3_C_) hipFree(d_step3_C_);
    if (d_svd_A_) hipFree(d_svd_A_);
    if (d_svd_U_) hipFree(d_svd_U_);
    if (d_svd_S_) hipFree(d_svd_S_);
    if (d_svd_Vh_) hipFree(d_svd_Vh_);
    if (d_svd_E_) hipFree(d_svd_E_);
    if (d_svd_info_) hipFree(d_svd_info_);
    if (d_svd_work_) hipFree(d_svd_work_);

    // Pinned host memory
    if (h_pin_A_) hipHostFree(h_pin_A_);
    if (h_pin_B_) hipHostFree(h_pin_B_);
    if (h_pin_C_) hipHostFree(h_pin_C_);
    if (h_pin_A3_) hipHostFree(h_pin_A3_);
    if (h_pin_B3_) hipHostFree(h_pin_B3_);
    if (h_pin_C3_) hipHostFree(h_pin_C3_);

    // HIP Graph
    if (heff_graph_exec_) hipGraphExecDestroy(heff_graph_exec_);
    if (heff_graph_) hipGraphDestroy(heff_graph_);
    if (d_scalar_one_) hipFree(d_scalar_one_);
    if (d_scalar_zero_) hipFree(d_scalar_zero_);

    rocblas_destroy_handle(rocblas_h_device_);
    rocblas_destroy_handle(rocblas_h_);
    hipStreamDestroy(stream_);
}

// ============================================================================
// Memory management
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::allocate_mps_tensor(int site, int cL, int cR) {
    if (d_mps_tensors_[site]) HIP_CHECK(hipFree(d_mps_tensors_[site]));
    HIP_CHECK(hipMalloc(&d_mps_tensors_[site], cL * d_ * cR * sizeof(Scalar)));
}

template<typename Scalar>
void DMRGGPU<Scalar>::ensure_L_env_alloc(int idx, int chi) {
    if (chi > L_env_alloc_chi_[idx]) {
        if (d_L_envs_[idx]) HIP_CHECK(hipFree(d_L_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_L_envs_[idx], sz * sizeof(Scalar)));
        L_env_alloc_chi_[idx] = chi;
    }
}

template<typename Scalar>
void DMRGGPU<Scalar>::ensure_R_env_alloc(int idx, int chi) {
    if (chi > R_env_alloc_chi_[idx]) {
        if (d_R_envs_[idx]) HIP_CHECK(hipFree(d_R_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_R_envs_[idx], sz * sizeof(Scalar)));
        R_env_alloc_chi_[idx] = chi;
    }
}

// ============================================================================
// MPS initialization
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::initialize_mps_random(double scale) {
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        std::vector<Scalar> h_A(size);
        for (int j = 0; j < size; j++) {
            h_A[j] = Traits::scale_by_real(scale, Traits::random_val());
        }
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(Scalar), hipMemcpyHostToDevice));
    }
}

template<typename Scalar>
void DMRGGPU<Scalar>::initialize_mps_product() {
    for (int i = 0; i < L_; i++) {
        int cL = chi_L(i), cR = chi_R(i);
        int size = cL * d_ * cR;
        std::vector<Scalar> h_A(size, Traits::zero());
        int chi_min = std::min(cL, cR);
        for (int a = 0; a < chi_min; a++) {
            h_A[a + 0*cL + a*cL*d_] = Traits::one();
        }
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(Scalar), hipMemcpyHostToDevice));
    }
}

template<typename Scalar>
void DMRGGPU<Scalar>::initialize_mps_neel() {
    for (int i = 0; i < L_; i++) {
        int cL = chi_L(i), cR = chi_R(i);
        int size = cL * d_ * cR;
        std::vector<Scalar> h_A(size, Traits::zero());
        int spin = (i % 2 == 0) ? 0 : 1;
        int chi_min = std::min(cL, cR);
        for (int a = 0; a < chi_min; a++) {
            h_A[a + spin*cL + a*cL*d_] = Traits::one();
        }
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(Scalar), hipMemcpyHostToDevice));
    }
}

template<typename Scalar>
void DMRGGPU<Scalar>::set_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    int D = D_mpo_, d = d_;
    for (int i = 0; i < L_; i++) {
        int size = D * d * d * D;
        HIP_CHECK(hipMalloc(&d_mpo_tensors_[i], size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_mpo_tensors_[i], h_mpo_tensors[i],
                            size * sizeof(Scalar), hipMemcpyHostToDevice));

        // Precompute W_left and W_right matrices
        int wm_size = D * d * d * D;
        std::vector<Scalar> h_WL(wm_size, Traits::zero());
        std::vector<Scalar> h_WR(wm_size, Traits::zero());
        for (int w = 0; w < D; w++)
            for (int s = 0; s < d; s++)
                for (int sp = 0; sp < d; sp++)
                    for (int wp = 0; wp < D; wp++) {
                        Scalar val = h_mpo_tensors[i][w + s*D + sp*D*d + wp*D*d*d];
                        h_WL[(w*d+s) + (wp*d+sp) * D * d] = val;
                        h_WR[(wp*d+s) + (w*d+sp) * D * d] = val;
                    }
        HIP_CHECK(hipMalloc(&d_W_left_[i], wm_size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_W_left_[i], h_WL.data(),
                            wm_size * sizeof(Scalar), hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&d_W_right_[i], wm_size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_W_right_[i], h_WR.data(),
                            wm_size * sizeof(Scalar), hipMemcpyHostToDevice));
    }
}

// ============================================================================
// GEMM-based tensor contractions
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::apply_heff(int site, const Scalar* d_theta_in, Scalar* d_result) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    Scalar* L_env = d_L_envs_[site];
    Scalar* R_env = d_R_envs_[site + 1];
    Scalar* W_mat = d_W_left_[site];
    Scalar* V = d_T1_;
    Scalar* U = d_T2_;

    // Step 1: V_ws[a',b] = L_w^T[a',a] * theta_s[a,b]  (batched GEMM)
    {
        for (int w = 0; w < D; w++)
            for (int s = 0; s < d; s++) {
                int ws = w * d + s;
                h_pin_A_[ws] = L_env + w * cL;
                h_pin_B_[ws] = const_cast<Scalar*>(d_theta_in) + s * cL;
                h_pin_C_[ws] = V + ws * cL * cR;
            }
        HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_pin_A_, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_pin_B_, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_pin_C_, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
            Traits::op_t, rocblas_operation_none,
            cL, cR, cL,
            &one,
            (const Scalar**)d_batch_A_, cL * D,
            (const Scalar**)d_batch_B_, cL * d,
            &zero_val,
            d_batch_C_, cL,
            D * d));
    }

    // Step 2: U = V * W_matrix
    ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
        rocblas_operation_none, rocblas_operation_none,
        cL * cR, d * D, D * d,
        &one,
        V, cL * cR,
        W_mat, D * d,
        &zero_val,
        U, cL * cR));

    // Step 3: result_s'[a',b'] = sum_w' U_{w'd+s'}[a',b] * R_w'[b,b']
    // D batched calls (batch_count=d), wp accumulates, sp independent within each batch.
    for (int wp = 0; wp < D; wp++) {
        Scalar beta = (wp == 0) ? Traits::zero() : Traits::one();
        for (int sp = 0; sp < d; sp++) {
            int ws_out = wp * d + sp;
            h_pin_A3_[sp] = U + ws_out * cL * cR;
            h_pin_B3_[sp] = R_env + wp * cR;
            h_pin_C3_[sp] = d_result + sp * cL;
        }
        HIP_CHECK(hipMemcpyAsync(d_step3_A_, h_pin_A3_, d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_step3_B_, h_pin_B3_, d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_step3_C_, h_pin_C3_, d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
            rocblas_operation_none, rocblas_operation_none,
            cL, cR, cR,
            &one,
            (const Scalar**)d_step3_A_, cL,
            (const Scalar**)d_step3_B_, cR * D,
            &beta,
            d_step3_C_, cL * d,
            d));
    }
}

// ============================================================================
// HIP Graph: apply_heff with pre-uploaded pointers (for graph capture)
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::apply_heff_graph(int site) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    int D = D_mpo_, d = d_;

    // Uses rocblas_h_device_ with device pointer mode — all alpha/beta are device pointers.
    // d_scalar_one_ and d_scalar_zero_ are persistent device memory.

    Scalar* V = d_T1_;
    Scalar* U = d_T2_;

    // Step 1: batched GEMM (pointer arrays already on device from setup_heff_graph)
    ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_device_,
        Traits::op_t, rocblas_operation_none,
        cL, cR, cL,
        d_scalar_one_,
        (const Scalar**)d_batch_A_, cL * D,
        (const Scalar**)d_batch_B_, cL * d,
        d_scalar_zero_,
        d_batch_C_, cL,
        D * d));

    // Step 2: dense GEMM
    ROCBLAS_CHECK(Traits::gemm(rocblas_h_device_,
        rocblas_operation_none, rocblas_operation_none,
        cL * cR, d * D, D * d,
        d_scalar_one_,
        V, cL * cR,
        d_W_left_[site], D * d,
        d_scalar_zero_,
        U, cL * cR));

    // Step 3: D batched calls (pointer arrays already on device)
    // wp=0: beta=zero, wp>0: beta=one (accumulate)
    ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_device_,
        rocblas_operation_none, rocblas_operation_none,
        cL, cR, cR,
        d_scalar_one_,
        (const Scalar**)(d_step3_A_), cL,
        (const Scalar**)(d_step3_B_), cR * D,
        d_scalar_zero_,
        d_step3_C_, cL * d,
        d));
    for (int wp = 1; wp < D; wp++) {
        ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_device_,
            rocblas_operation_none, rocblas_operation_none,
            cL, cR, cR,
            d_scalar_one_,
            (const Scalar**)(d_step3_A_ + wp * d), cL,
            (const Scalar**)(d_step3_B_ + wp * d), cR * D,
            d_scalar_one_,  // beta=1 for accumulation
            d_step3_C_ + wp * d, cL * d,
            d));
    }
}

// ============================================================================
// HIP Graph: setup and capture apply_heff for a given site
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::setup_heff_graph(int site) {
    if (heff_graph_site_ == site) return;  // already cached for this site

    // Destroy old graph
    if (heff_graph_exec_) { HIP_CHECK(hipGraphExecDestroy(heff_graph_exec_)); heff_graph_exec_ = nullptr; }
    if (heff_graph_) { HIP_CHECK(hipGraphDestroy(heff_graph_)); heff_graph_ = nullptr; }

    int cL = chi_L(site), cR = chi_R(site);
    int D = D_mpo_, d = d_;

    // Pre-compute Step 1 pointer arrays (using d_theta_staging_ as fixed theta base)
    for (int w = 0; w < D; w++)
        for (int s = 0; s < d; s++) {
            int ws = w * d + s;
            h_pin_A_[ws] = d_L_envs_[site] + w * cL;
            h_pin_B_[ws] = d_theta_staging_ + s * cL;
            h_pin_C_[ws] = d_T1_ + ws * cL * cR;
        }
    HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_pin_A_, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_pin_B_, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_pin_C_, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));

    // Pre-compute ALL Step 3 pointer arrays (D batches × d entries each)
    Scalar* R_env = d_R_envs_[site + 1];
    for (int wp = 0; wp < D; wp++)
        for (int sp = 0; sp < d; sp++) {
            int idx = wp * d + sp;
            h_pin_A3_[idx] = d_T2_ + (wp * d + sp) * cL * cR;
            h_pin_B3_[idx] = R_env + wp * cR;
            h_pin_C3_[idx] = d_heff_result_ + sp * cL;
        }
    HIP_CHECK(hipMemcpyAsync(d_step3_A_, h_pin_A3_, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_step3_B_, h_pin_B3_, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_step3_C_, h_pin_C3_, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));

    // Capture apply_heff_graph into a HIP Graph
    HIP_CHECK(hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal));
    apply_heff_graph(site);
    HIP_CHECK(hipStreamEndCapture(stream_, &heff_graph_));
    HIP_CHECK(hipGraphInstantiate(&heff_graph_exec_, heff_graph_, nullptr, nullptr, 0));

    heff_graph_site_ = site;
}

// ============================================================================
// Left environment update
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::update_left_env(int site) {
    int chi_in = bond_dims_[site];
    int chi_out = bond_dims_[site + 1];
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    ensure_L_env_alloc(site + 1, chi_out);

    Scalar* L_env = d_L_envs_[site];
    Scalar* A = d_mps_tensors_[site];
    Scalar* W_mat = d_W_left_[site];
    Scalar* L_new = d_L_envs_[site + 1];
    Scalar* V = d_T1_;
    Scalar* U = d_T2_;

    // Step 1: V_ws[a',b] = L_w^T[a',a] * A_s[a,b]  (batched GEMM)
    {
        for (int w = 0; w < D; w++)
            for (int s = 0; s < d; s++) {
                int ws = w * d + s;
                h_pin_A_[ws] = L_env + w * chi_in;
                h_pin_B_[ws] = A + s * chi_in;
                h_pin_C_[ws] = V + ws * chi_in * chi_out;
            }
        HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_pin_A_, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_pin_B_, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_pin_C_, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
            Traits::op_t, rocblas_operation_none,
            chi_in, chi_out, chi_in,
            &one,
            (const Scalar**)d_batch_A_, chi_in * D,
            (const Scalar**)d_batch_B_, chi_in * d,
            &zero_val,
            d_batch_C_, chi_in,
            D * d));
    }

    // Step 2: U = V * W_matrix
    ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
        rocblas_operation_none, rocblas_operation_none,
        chi_in * chi_out, d * D, D * d,
        &one,
        V, chi_in * chi_out,
        W_mat, D * d,
        &zero_val,
        U, chi_in * chi_out));

    // Step 3: L_new_w'[b,b'] = sum_{s'} U^H[ws',b] * A[s',b']
    // wp INDEPENDENT (batch), sp ACCUMULATES (sequential loop).
    // d batched calls, batch_count=D each.
    for (int sp = 0; sp < d; sp++) {
        Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
        for (int wp = 0; wp < D; wp++) {
            int ws_out = wp * d + sp;
            h_pin_A3_[wp] = U + ws_out * chi_in * chi_out;
            h_pin_B3_[wp] = A + sp * chi_in;
            h_pin_C3_[wp] = L_new + wp * chi_out;
        }
        HIP_CHECK(hipMemcpyAsync(d_step3_A_, h_pin_A3_, D*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_step3_B_, h_pin_B3_, D*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_step3_C_, h_pin_C3_, D*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
            Traits::op_h, rocblas_operation_none,
            chi_out, chi_out, chi_in,
            &one,
            (const Scalar**)d_step3_A_, chi_in,
            (const Scalar**)d_step3_B_, chi_in * d,
            &beta,
            d_step3_C_, chi_out * D,
            D));
    }

    // For complex: L_new = conj(U^H * A) = U^T * conj(A), the correct bra contraction
    if constexpr (Traits::is_complex) {
        conjugate_inplace(L_new, chi_out * D * chi_out, stream_);
    }
}

// ============================================================================
// Right environment update
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::update_right_env(int site) {
    int chi_in = bond_dims_[site + 1];
    int chi_out = bond_dims_[site];
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    ensure_R_env_alloc(site, chi_out);

    Scalar* A = d_mps_tensors_[site];
    Scalar* R_env = d_R_envs_[site + 1];
    Scalar* W_mat = d_W_right_[site];
    Scalar* R_new = d_R_envs_[site];
    Scalar* V = d_T1_;
    Scalar* U = d_T2_;

    // Step 1: V_ws[a,b'] = A_s[a,b] * R_w'[b,b']  (batched GEMM)
    {
        for (int wp = 0; wp < D; wp++)
            for (int s = 0; s < d; s++) {
                int ws = wp * d + s;
                h_pin_A_[ws] = A + s * chi_out;
                h_pin_B_[ws] = R_env + wp * chi_in;
                h_pin_C_[ws] = V + ws * chi_out * chi_in;
            }
        HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_pin_A_, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_pin_B_, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_pin_C_, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
            rocblas_operation_none, rocblas_operation_none,
            chi_out, chi_in, chi_in,
            &one,
            (const Scalar**)d_batch_A_, chi_out * d,
            (const Scalar**)d_batch_B_, chi_in * D,
            &zero_val,
            d_batch_C_, chi_out,
            D * d));
    }

    // Step 2: U = V * W_matrix
    ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
        rocblas_operation_none, rocblas_operation_none,
        chi_out * chi_in, d * D, D * d,
        &one,
        V, chi_out * chi_in,
        W_mat, D * d,
        &zero_val,
        U, chi_out * chi_in));

    // Step 3: R_new_w[a,a'] = sum_s' U_ws'[a,b'] * A_s'^H[b',a']
    // w INDEPENDENT (batch), sp ACCUMULATES (sequential loop).
    // d batched calls, batch_count=D each.
    for (int sp = 0; sp < d; sp++) {
        Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
        for (int w = 0; w < D; w++) {
            int ws_out = w * d + sp;
            h_pin_A3_[w] = U + ws_out * chi_out * chi_in;
            h_pin_B3_[w] = A + sp * chi_out;
            h_pin_C3_[w] = R_new + w * chi_out;
        }
        HIP_CHECK(hipMemcpyAsync(d_step3_A_, h_pin_A3_, D*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_step3_B_, h_pin_B3_, D*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_step3_C_, h_pin_C3_, D*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
            rocblas_operation_none, Traits::op_h,
            chi_out, chi_out, chi_in,
            &one,
            (const Scalar**)d_step3_A_, chi_out,
            (const Scalar**)d_step3_B_, chi_out * d,
            &beta,
            d_step3_C_, chi_out * D,
            D));
    }
}

// ============================================================================
// Environment building
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::build_initial_environments() {
    // L[0] = trivial left boundary: (1, D_mpo, 1), L[0][0,0,0] = 1
    {
        std::vector<Scalar> h_L(D_mpo_, Traits::zero());
        h_L[0] = Traits::one();
        HIP_CHECK(hipMemcpy(d_L_envs_[0], h_L.data(),
                            D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // R[L] = trivial right boundary: (1, D_mpo, 1), R[L][0,D-1,0] = 1
    {
        std::vector<Scalar> h_R(D_mpo_, Traits::zero());
        h_R[D_mpo_ - 1] = Traits::one();
        HIP_CHECK(hipMemcpy(d_R_envs_[L_], h_R.data(),
                            D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // Build all R environments from right to left
    for (int i = L_ - 1; i >= 0; i--) {
        update_right_env(i);
    }
}

// ============================================================================
// Theta formation and Lanczos
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::form_theta(int site, Scalar* d_theta) {
    int size = chi_L(site) * d_ * chi_R(site);
    HIP_CHECK(hipMemcpy(d_theta, d_mps_tensors_[site],
                        size * sizeof(Scalar), hipMemcpyDeviceToDevice));
}

template<typename Scalar>
double DMRGGPU<Scalar>::lanczos_eigensolver(int site, Scalar* d_theta) {
    int n = chi_L(site) * d_ * chi_R(site);
    int max_iter = std::min(max_lanczos_iter_, n);
    double tol_lanczos = 1e-12;
    double tol_eig_conv = 1e-12;

    Scalar* d_lanczos_v = d_lanczos_v_;

    // Alpha and beta are always real for Hermitian operators
    std::vector<double> h_alpha(max_iter);
    std::vector<double> h_beta(max_iter);

    // v[0] = theta / ||theta||
    double norm;
    ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_theta, 1, &norm));

    if (norm < 1e-14) {
        std::vector<Scalar> h_init(n);
        srand(42 + site);
        for (int i = 0; i < n; i++) h_init[i] = Traits::random_val();
        HIP_CHECK(hipMemcpy(d_theta, h_init.data(), n * sizeof(Scalar), hipMemcpyHostToDevice));
        ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_theta, 1, &norm));
    }

    double inv_norm = 1.0 / norm;
    ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, &inv_norm, d_theta, 1));
    HIP_CHECK(hipMemcpy(d_lanczos_v, d_theta, n * sizeof(Scalar), hipMemcpyDeviceToDevice));

    double prev_energy = 1e30;
    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        Scalar* d_vi = d_lanczos_v + iter * n;

        // w = H|v_i>
        apply_heff(site, d_vi, d_heff_result_);

        // alpha_i = <v_i|w> (real for Hermitian H)
        Scalar alpha_result;
        ROCBLAS_CHECK(Traits::dot(rocblas_h_, n, d_vi, 1, d_heff_result_, 1, &alpha_result));
        double alpha_i = Traits::real_part(alpha_result);
        h_alpha[iter] = alpha_i;

        // w = w - alpha_i * v_i
        Scalar neg_alpha = Traits::make_scalar(-alpha_i);
        ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, &neg_alpha, d_vi, 1, d_heff_result_, 1));

        // w = w - beta_{i-1} * v_{i-1}
        if (iter > 0) {
            Scalar neg_beta = Traits::make_scalar(-h_beta[iter - 1]);
            Scalar* d_vim1 = d_lanczos_v + (iter - 1) * n;
            ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, &neg_beta, d_vim1, 1, d_heff_result_, 1));
        }

        // Full reorthogonalization via gemv
        if (iter > 0) {
            Scalar one_val = Traits::one(), zero_sc = Traits::zero(), neg_one = Traits::neg(Traits::one());
            ROCBLAS_CHECK(Traits::gemv(rocblas_h_, Traits::op_h,
                n, iter + 1, &one_val,
                d_lanczos_v, n,
                d_heff_result_, 1,
                &zero_sc, d_ritz_coeffs_, 1));
            ROCBLAS_CHECK(Traits::gemv(rocblas_h_, rocblas_operation_none,
                n, iter + 1, &neg_one,
                d_lanczos_v, n,
                d_ritz_coeffs_, 1,
                &one_val, d_heff_result_, 1));
        } else {
            Scalar overlap;
            ROCBLAS_CHECK(Traits::dot(rocblas_h_, n, d_lanczos_v, 1, d_heff_result_, 1, &overlap));
            Scalar neg_overlap = Traits::neg(overlap);
            ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, &neg_overlap, d_lanczos_v, 1, d_heff_result_, 1));
        }

        // beta_i = ||w||
        double beta_i;
        ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_heff_result_, 1, &beta_i));
        h_beta[iter] = beta_i;

        if (beta_i < tol_lanczos) {
            iter++;
            break;
        }

        // Eigenvalue convergence check (every 3 iterations after iter >= 4)
        if (iter >= 4 && iter % 3 == 0) {
            int ncheck = iter + 1;
            std::vector<double> h_D_chk(ncheck), h_E_chk(ncheck);
            std::copy(h_alpha.begin(), h_alpha.begin() + ncheck, h_D_chk.begin());
            for (int i = 0; i < ncheck - 1; i++) h_E_chk[i] = h_beta[i];
            h_E_chk[ncheck - 1] = 0.0;
            const char jobz_n = 'N';
            const int n_chk = ncheck;
            std::vector<double> h_work_chk(1);
            int info_chk = 0;
            dstev_(&jobz_n, &n_chk, h_D_chk.data(), h_E_chk.data(), nullptr, &n_chk, h_work_chk.data(), &info_chk);
            if (info_chk == 0) {
                double cur_energy = h_D_chk[0];
                if (std::abs(cur_energy - prev_energy) < tol_eig_conv) {
                    iter++;
                    break;
                }
                prev_energy = cur_energy;
            }
        }

        // v_{i+1} = w / beta_i
        if (iter + 1 < max_iter) {
            Scalar* d_vip1 = d_lanczos_v + (iter + 1) * n;
            double scale = 1.0 / beta_i;
            HIP_CHECK(hipMemcpy(d_vip1, d_heff_result_, n * sizeof(Scalar), hipMemcpyDeviceToDevice));
            ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, &scale, d_vip1, 1));
        }
    }

    int niter = iter;
    prof_lanczos_iters += niter;
    prof_heff_calls += niter;

    // Solve tridiagonal eigenvalue problem on CPU (always real)
    std::vector<double> h_D(niter), h_E(niter), h_Z(niter * niter);
    std::vector<double> h_work(std::max(1, 2*niter - 2));
    int lapack_info = 0;

    std::copy(h_alpha.begin(), h_alpha.begin() + niter, h_D.begin());
    for (int i = 0; i < niter - 1; i++) h_E[i] = h_beta[i];
    if (niter > 0) h_E[niter - 1] = 0.0;

    const char jobz = 'V';
    const int n_lapack = niter;
    const int ldz = niter;

    dstev_(&jobz, &n_lapack, h_D.data(), h_E.data(), h_Z.data(), &ldz, h_work.data(), &lapack_info);

    if (lapack_info != 0) {
        throw std::runtime_error("LAPACK dstev failed with info = " + std::to_string(lapack_info));
    }

    double energy = h_D[0];

    // Reconstruct ground state: |theta> = sum_i c[i] |v_i>
    // Ritz coefficients are real (from dstev); convert to Scalar for gemv
    std::vector<Scalar> h_ritz_scalar(niter);
    for (int i = 0; i < niter; i++) {
        h_ritz_scalar[i] = Traits::make_scalar(h_Z[i]);
    }
    HIP_CHECK(hipMemcpy(d_ritz_coeffs_, h_ritz_scalar.data(), niter * sizeof(Scalar), hipMemcpyHostToDevice));

    Scalar one_sc = Traits::one(), zero_sc = Traits::zero();
    ROCBLAS_CHECK(Traits::gemv(
        rocblas_h_, rocblas_operation_none,
        n, niter, &one_sc,
        d_lanczos_v, n,
        d_ritz_coeffs_, 1,
        &zero_sc, d_theta, 1
    ));

    // Normalize
    ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_theta, 1, &norm));
    inv_norm = 1.0 / norm;
    ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, &inv_norm, d_theta, 1));

    return energy;
}

// ============================================================================
// SVD and MPS update
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::svd_and_update_mps(int site, Scalar* d_theta, char direction) {
    int cL = chi_L(site);
    int cR = chi_R(site);

    int m, n_svd;
    if (direction == 'R') { m = cL * d_; n_svd = cR; }
    else                  { m = cL;      n_svd = d_ * cR; }
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);

    // GPU SVD via rocsolver gesvd
    HIP_CHECK(hipMemcpy(d_svd_A_, d_theta, m * n_svd * sizeof(Scalar), hipMemcpyDeviceToDevice));

    Traits::rocsolver_gesvd(rocblas_h_,
        rocblas_svect_singular, rocblas_svect_singular,
        m, n_svd,
        d_svd_A_, m,
        d_svd_S_,
        d_svd_U_, m,
        d_svd_Vh_, full_k,
        d_svd_E_,
        rocblas_outofplace,
        d_svd_info_);

    HIP_CHECK(hipMemcpy(h_svd_U_.data(), d_svd_U_, m * full_k * sizeof(Scalar), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_svd_S_.data(), d_svd_S_, full_k * sizeof(RealType), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_svd_Vh_.data(), d_svd_Vh_, full_k * n_svd * sizeof(Scalar), hipMemcpyDeviceToHost));

    Scalar* h_U_data = h_svd_U_.data();
    RealType* h_S_data = h_svd_S_.data();
    Scalar* h_Vh_data = h_svd_Vh_.data();

    // Truncation (CPU, tiny loop)
    int new_k = k;
    for (int i = 0; i < new_k; i++) {
        if (h_S_data[i] < 1e-14) { new_k = i; break; }
    }
    if (new_k == 0) new_k = 1;

    if (direction == 'R') {
        int new_chi_R = new_k;

        // Compute S*Vh on CPU
        for (int j = 0; j < n_svd; j++)
            for (int i = 0; i < new_k; i++)
                h_svd_tmp_[i + j * new_k] = Traits::scale_by_real(h_S_data[i], h_Vh_data[i + j * full_k]);

        // Upload U[:, :new_k]
        allocate_mps_tensor(site, cL, new_chi_R);
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], h_U_data,
                            m * new_chi_R * sizeof(Scalar), hipMemcpyHostToDevice));

        // Absorb S*Vh into A[site+1]
        if (site + 1 < L_) {
            HIP_CHECK(hipMemcpy(d_svd_work_, h_svd_tmp_.data(),
                                new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));
            int next_cR = chi_R(site + 1);
            Scalar one = Traits::one(), zero_val = Traits::zero();
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                new_k, d_ * next_cR, cR, &one,
                d_svd_work_, new_k,
                d_mps_tensors_[site + 1], cR, &zero_val,
                d_T1_, new_k));
            allocate_mps_tensor(site + 1, new_chi_R, next_cR);
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site + 1], d_T1_,
                                new_k * d_ * next_cR * sizeof(Scalar), hipMemcpyDeviceToDevice));
        }
        bond_dims_[site + 1] = new_chi_R;

    } else {  // direction == 'L'
        int new_chi_L = new_k;

        // Upload Vh[:new_k, :]
        allocate_mps_tensor(site, new_chi_L, cR);
        if (new_chi_L == full_k) {
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site], h_Vh_data,
                                full_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));
        } else {
            for (int j = 0; j < n_svd; j++)
                for (int i = 0; i < new_chi_L; i++)
                    h_svd_tmp_[i + j * new_chi_L] = h_Vh_data[i + j * full_k];
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site], h_svd_tmp_.data(),
                                new_chi_L * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));
        }

        // Compute U*S on CPU
        for (int j = 0; j < new_k; j++)
            for (int i = 0; i < m; i++)
                h_svd_tmp_[i + j * m] = Traits::scale_by_real(h_S_data[j], h_U_data[i + j * m]);

        // Absorb U*S into A[site-1]
        if (site > 0) {
            HIP_CHECK(hipMemcpy(d_svd_work_, h_svd_tmp_.data(),
                                m * new_k * sizeof(Scalar), hipMemcpyHostToDevice));
            int prev_cL = chi_L(site - 1);
            Scalar one = Traits::one(), zero_val = Traits::zero();
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                prev_cL * d_, new_k, m, &one,
                d_mps_tensors_[site - 1], prev_cL * d_,
                d_svd_work_, m, &zero_val,
                d_T1_, prev_cL * d_));
            allocate_mps_tensor(site - 1, prev_cL, new_chi_L);
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site - 1], d_T1_,
                                prev_cL * d_ * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice));
        }
        bond_dims_[site] = new_chi_L;
    }
}

// ============================================================================
// Site optimization
// ============================================================================

template<typename Scalar>
double DMRGGPU<Scalar>::optimize_site(int site, char direction) {
    form_theta(site, d_theta_);

    HIP_CHECK(hipStreamSynchronize(stream_));
    auto t0 = std::chrono::high_resolution_clock::now();
    double energy = lanczos_eigensolver(site, d_theta_);
    HIP_CHECK(hipStreamSynchronize(stream_));
    auto t1 = std::chrono::high_resolution_clock::now();
    svd_and_update_mps(site, d_theta_, direction);
    HIP_CHECK(hipStreamSynchronize(stream_));
    auto t2 = std::chrono::high_resolution_clock::now();

    prof_lanczos_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    prof_svd_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
    prof_site_count++;
    return energy;
}

// ============================================================================
// Sweep methods
// ============================================================================

template<typename Scalar>
double DMRGGPU<Scalar>::sweep_left_to_right() {
    double energy = 0.0;

    for (int site = 0; site < L_ - 1; site++) {
        energy = optimize_site(site, 'R');
        HIP_CHECK(hipStreamSynchronize(stream_));
        auto te0 = std::chrono::high_resolution_clock::now();
        update_left_env(site);
        HIP_CHECK(hipStreamSynchronize(stream_));
        auto te1 = std::chrono::high_resolution_clock::now();
        prof_env_ms += std::chrono::duration<double, std::milli>(te1 - te0).count();
    }
    // Optimize last site without SVD
    {
        int site = L_ - 1;
        form_theta(site, d_theta_);
        energy = lanczos_eigensolver(site, d_theta_);
        int sz = chi_L(site) * d_ * chi_R(site);
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], d_theta_, sz * sizeof(Scalar),
                            hipMemcpyDeviceToDevice));
    }

    return energy;
}

template<typename Scalar>
double DMRGGPU<Scalar>::sweep_right_to_left() {
    double energy = 0.0;

    for (int site = L_ - 1; site >= 1; site--) {
        energy = optimize_site(site, 'L');
        HIP_CHECK(hipStreamSynchronize(stream_));
        auto te0 = std::chrono::high_resolution_clock::now();
        update_right_env(site);
        HIP_CHECK(hipStreamSynchronize(stream_));
        auto te1 = std::chrono::high_resolution_clock::now();
        prof_env_ms += std::chrono::duration<double, std::milli>(te1 - te0).count();
    }
    // Optimize first site without SVD
    {
        int site = 0;
        form_theta(site, d_theta_);
        energy = lanczos_eigensolver(site, d_theta_);
        int sz = chi_L(site) * d_ * chi_R(site);
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], d_theta_, sz * sizeof(Scalar),
                            hipMemcpyDeviceToDevice));
    }

    return energy;
}

// ============================================================================
// Main algorithm
// ============================================================================

template<typename Scalar>
double DMRGGPU<Scalar>::run(int n_sweeps) {
    const char* type_name = Traits::is_complex ? "complex128" : "float64";
    printf("=== GPU-Native DMRG (rocBLAS GEMM, %s) ===\n", type_name);
    printf("L = %d, d = %d, chi_max = %d, D_mpo = %d\n", L_, d_, chi_max_, D_mpo_);
    printf("Running %d sweeps...\n\n", n_sweeps);

    auto t_start = std::chrono::high_resolution_clock::now();

    printf("Building initial environments...\n");
    build_initial_environments();

    auto t_envs = std::chrono::high_resolution_clock::now();
    double env_time = std::chrono::duration<double>(t_envs - t_start).count();
    printf("  Environment build: %.3f s\n\n", env_time);

    double energy_prev = 0.0;

    for (int sweep = 0; sweep < n_sweeps; sweep++) {
        prof_lanczos_ms = prof_svd_ms = prof_env_ms = 0;
        prof_lanczos_iters = prof_site_count = prof_heff_calls = 0;

        auto t_sweep = std::chrono::high_resolution_clock::now();

        double energy_LR = sweep_left_to_right();
        double energy_RL = sweep_right_to_left();

        auto t_sweep_end = std::chrono::high_resolution_clock::now();
        double sweep_time = std::chrono::duration<double>(t_sweep_end - t_sweep).count();

        energy_ = energy_RL;
        double dE = std::abs(energy_ - energy_prev);

        double other_ms = sweep_time*1000.0 - prof_lanczos_ms - prof_svd_ms - prof_env_ms;
        printf("Sweep %3d: E = %.12f, dE = %.2e, time = %.3f s\n",
               sweep, energy_, dE, sweep_time);
        printf("  Profile: lanczos=%.0fms (%d iters, %d heff) svd=%.0fms env=%.0fms other=%.0fms\n",
               prof_lanczos_ms, prof_lanczos_iters, prof_heff_calls,
               prof_svd_ms, prof_env_ms, other_ms);

        if (dE < tol_ && sweep > 0) {
            printf("Converged after %d sweeps!\n", sweep + 1);
            break;
        }

        energy_prev = energy_;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();
    printf("\nTotal wall time: %.3f s\n", total_time);

    return energy_;
}

// ============================================================================
// Utility methods
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::get_mps(std::vector<std::vector<Scalar>>& h_mps) const {
    h_mps.resize(L_);
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        h_mps[i].resize(size);
        HIP_CHECK(hipMemcpy(h_mps[i].data(), d_mps_tensors_[i],
                            size * sizeof(Scalar), hipMemcpyDeviceToHost));
    }
}

#endif // DMRG_GPU_IMPL_H
