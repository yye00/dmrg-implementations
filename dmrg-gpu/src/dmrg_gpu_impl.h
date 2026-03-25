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

#define HT_CHECK(call) \
    do { \
        hiptensorStatus_t ht_status = call; \
        if (ht_status != HIPTENSOR_STATUS_SUCCESS) { \
            std::cerr << "hipTensor error in " << __FILE__ << ":" << __LINE__ \
                      << " - status " << (int)ht_status << ": " \
                      << hiptensorGetErrorString(ht_status) << std::endl; \
            throw std::runtime_error("hipTensor error"); \
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
    HIP_CHECK(hipMalloc(&d_heff_result_, theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_lanczos_v_, (size_t)max_lanczos_iter_ * theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_ritz_coeffs_, max_lanczos_iter_ * sizeof(Scalar)));

    // Batched GEMM pointer arrays
    int batch_max = D_mpo_ * d_;
    HIP_CHECK(hipMalloc(&d_batch_A_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_B_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_C_, batch_max * sizeof(Scalar*)));

    // hipTensor handle and workspace
    HT_CHECK(hiptensorCreate(&ht_handle_));
    ht_workspace_ = nullptr;
    ht_workspace_size_ = 0;
    // Pre-allocate a generous workspace (resize if needed)
    ht_workspace_size_ = (size_t)D_mpo_ * d_ * chi_max_ * chi_max_ * sizeof(Scalar) * 2;
    HIP_CHECK(hipMalloc(&ht_workspace_, ht_workspace_size_));
    // Conjugate buffer for complex env updates
    HIP_CHECK(hipMalloc(&d_conj_buf_, chi_max_ * d_ * chi_max_ * sizeof(Scalar)));

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
    if (d_heff_result_) hipFree(d_heff_result_);
    if (d_lanczos_v_) hipFree(d_lanczos_v_);
    if (d_ritz_coeffs_) hipFree(d_ritz_coeffs_);
    if (d_T1_) hipFree(d_T1_);
    if (d_T2_) hipFree(d_T2_);
    if (d_batch_A_) hipFree(d_batch_A_);
    if (d_batch_B_) hipFree(d_batch_B_);
    if (d_batch_C_) hipFree(d_batch_C_);
    // Destroy cached hiptensor plans
    for (auto& [k, e] : ht_plan_cache_) {
        hiptensorDestroyPlan(e.plan);
        hiptensorDestroyOperationDescriptor(e.opDesc);
        hiptensorDestroyTensorDescriptor(e.descD);
        hiptensorDestroyTensorDescriptor(e.descB);
        hiptensorDestroyTensorDescriptor(e.descA);
    }
    ht_plan_cache_.clear();
    if (ht_workspace_) hipFree(ht_workspace_);
    if (d_conj_buf_) hipFree(d_conj_buf_);
    hiptensorDestroy(ht_handle_);
    if (d_svd_A_) hipFree(d_svd_A_);
    if (d_svd_U_) hipFree(d_svd_U_);
    if (d_svd_S_) hipFree(d_svd_S_);
    if (d_svd_Vh_) hipFree(d_svd_Vh_);
    if (d_svd_E_) hipFree(d_svd_E_);
    if (d_svd_info_) hipFree(d_svd_info_);
    if (d_svd_work_) hipFree(d_svd_work_);

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
// hipTensor contraction helper
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::ht_contract(
    const Scalar* A_data, int rankA, const int64_t* extA, const int32_t* modesA,
    const Scalar* B_data, int rankB, const int64_t* extB, const int32_t* modesB,
    Scalar* D_data, int rankD, const int64_t* extD, const int32_t* modesD)
{
    // Build cache key: [rankA, extA..., modesA..., rankB, extB..., modesB..., rankD, extD..., modesD...]
    std::vector<int64_t> key;
    key.reserve(3 + rankA*2 + rankB*2 + rankD*2);
    key.push_back(rankA);
    for (int i = 0; i < rankA; i++) key.push_back(extA[i]);
    for (int i = 0; i < rankA; i++) key.push_back(modesA[i]);
    key.push_back(rankB);
    for (int i = 0; i < rankB; i++) key.push_back(extB[i]);
    for (int i = 0; i < rankB; i++) key.push_back(modesB[i]);
    key.push_back(rankD);
    for (int i = 0; i < rankD; i++) key.push_back(extD[i]);
    for (int i = 0; i < rankD; i++) key.push_back(modesD[i]);

    auto it = ht_plan_cache_.find(key);
    if (it == ht_plan_cache_.end()) {
        // Cache miss: create descriptors + plan
        hiptensorDataType_t dtype = Traits::is_complex ? HIPTENSOR_C_64F : HIPTENSOR_R_64F;
        hiptensorComputeDescriptor_t ctype = Traits::is_complex
            ? HIPTENSOR_COMPUTE_DESC_C64F : HIPTENSOR_COMPUTE_DESC_64F;

        HtPlanEntry entry;
        HT_CHECK(hiptensorCreateTensorDescriptor(ht_handle_, &entry.descA,
            rankA, extA, NULL, dtype, 256));
        HT_CHECK(hiptensorCreateTensorDescriptor(ht_handle_, &entry.descB,
            rankB, extB, NULL, dtype, 256));
        HT_CHECK(hiptensorCreateTensorDescriptor(ht_handle_, &entry.descD,
            rankD, extD, NULL, dtype, 256));

        HT_CHECK(hiptensorCreateContraction(ht_handle_, &entry.opDesc,
            entry.descA, modesA, HIPTENSOR_OP_IDENTITY,
            entry.descB, modesB, HIPTENSOR_OP_IDENTITY,
            entry.descD, modesD, HIPTENSOR_OP_IDENTITY,
            entry.descD, modesD, ctype));

        hiptensorPlanPreference_t pref;
        HT_CHECK(hiptensorCreatePlanPreference(ht_handle_, &pref,
            HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE));

        uint64_t worksize = 0;
        HT_CHECK(hiptensorEstimateWorkspaceSize(ht_handle_, entry.opDesc,
            pref, HIPTENSOR_WORKSPACE_DEFAULT, &worksize));

        if (worksize > ht_workspace_size_) {
            if (ht_workspace_) hipFree(ht_workspace_);
            HIP_CHECK(hipMalloc(&ht_workspace_, worksize));
            ht_workspace_size_ = worksize;
        }

        HT_CHECK(hiptensorCreatePlan(ht_handle_, &entry.plan, entry.opDesc, pref, ht_workspace_size_));
        hiptensorDestroyPlanPreference(pref);

        it = ht_plan_cache_.emplace(std::move(key), entry).first;
    }

    // Execute contraction using cached plan
    Scalar alpha = Traits::one();
    Scalar beta = Traits::zero();
    HT_CHECK(hiptensorContract(ht_handle_, it->second.plan,
        &alpha, A_data, B_data,
        &beta, D_data, D_data,
        ht_workspace_, ht_workspace_size_, stream_));
}

template<typename Scalar>
void DMRGGPU<Scalar>::make_conjugate(const Scalar* src, Scalar* dst, int n) {
    if constexpr (Traits::is_complex) {
        HIP_CHECK(hipMemcpyAsync(dst, src, n * sizeof(Scalar),
                                  hipMemcpyDeviceToDevice, stream_));
        conjugate_inplace(dst, n, stream_);
    } else {
        HIP_CHECK(hipMemcpyAsync(dst, src, n * sizeof(Scalar),
                                  hipMemcpyDeviceToDevice, stream_));
    }
}

// ============================================================================
// hipTensor-based tensor contractions
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::apply_heff(int site, const Scalar* d_theta_in, Scalar* d_result) {
    int64_t cL = chi_L(site);
    int64_t cR = chi_R(site);
    int64_t D = D_mpo_, d = d_;

    // Mode labels: a=ket-left, p=bra-left, w=mpo-in, q=mpo-out,
    //              s=phys-ket, t=phys-bra, b=ket-right, r=bra-right
    // Step 1: T1[p, w, s, b] = sum_a L[a, w, p] * theta[a, s, b]
    {
        int64_t extL[] = {cL, D, cL};
        int32_t modesL[] = {'a', 'w', 'p'};
        int64_t extTh[] = {cL, d, cR};
        int32_t modesTh[] = {'a', 's', 'b'};
        int64_t extT1[] = {cL, D, d, cR};
        int32_t modesT1[] = {'p', 'w', 's', 'b'};
        ht_contract(d_L_envs_[site], 3, extL, modesL,
                    d_theta_in, 3, extTh, modesTh,
                    d_T1_, 4, extT1, modesT1);
    }

    // Step 2: T2[p, b, t, q] = sum_{w,s} T1[p, w, s, b] * W[w, s, t, q]
    {
        int64_t extT1[] = {cL, D, d, cR};
        int32_t modesT1[] = {'p', 'w', 's', 'b'};
        int64_t extW[] = {D, d, d, D};
        int32_t modesW[] = {'w', 's', 't', 'q'};
        int64_t extT2[] = {cL, cR, d, D};
        int32_t modesT2[] = {'p', 'b', 't', 'q'};
        ht_contract(d_T1_, 4, extT1, modesT1,
                    d_mpo_tensors_[site], 4, extW, modesW,
                    d_T2_, 4, extT2, modesT2);
    }

    // Step 3: result[p, t, r] = sum_{b,q} T2[p, b, t, q] * R[b, q, r]
    {
        int64_t extT2[] = {cL, cR, d, D};
        int32_t modesT2[] = {'p', 'b', 't', 'q'};
        int64_t extR[] = {cR, D, cR};
        int32_t modesR[] = {'b', 'q', 'r'};
        int64_t extRes[] = {cL, d, cR};
        int32_t modesRes[] = {'p', 't', 'r'};
        ht_contract(d_T2_, 4, extT2, modesT2,
                    d_R_envs_[site + 1], 3, extR, modesR,
                    d_result, 3, extRes, modesRes);
    }
}

// ============================================================================
// Left environment update
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::update_left_env(int site) {
    int64_t chi_in = bond_dims_[site];
    int64_t chi_out = bond_dims_[site + 1];
    int64_t D = D_mpo_, d = d_;

    ensure_L_env_alloc(site + 1, chi_out);

    // Step 1: T1[p, w, s, b] = sum_a L[a, w, p] * A[a, s, b]
    {
        int64_t extL[] = {chi_in, D, chi_in};
        int32_t modesL[] = {'a', 'w', 'p'};
        int64_t extA[] = {chi_in, d, chi_out};
        int32_t modesA[] = {'a', 's', 'b'};
        int64_t extT1[] = {chi_in, D, d, chi_out};
        int32_t modesT1[] = {'p', 'w', 's', 'b'};
        ht_contract(d_L_envs_[site], 3, extL, modesL,
                    d_mps_tensors_[site], 3, extA, modesA,
                    d_T1_, 4, extT1, modesT1);
    }

    // Step 2: T2[p, b, t, q] = sum_{w,s} T1[p, w, s, b] * W[w, s, t, q]
    {
        int64_t extT1[] = {chi_in, D, d, chi_out};
        int32_t modesT1[] = {'p', 'w', 's', 'b'};
        int64_t extW[] = {D, d, d, D};
        int32_t modesW[] = {'w', 's', 't', 'q'};
        int64_t extT2[] = {chi_in, chi_out, d, D};
        int32_t modesT2[] = {'p', 'b', 't', 'q'};
        ht_contract(d_T1_, 4, extT1, modesT1,
                    d_mpo_tensors_[site], 4, extW, modesW,
                    d_T2_, 4, extT2, modesT2);
    }

    // Step 3: L_new[b, q, r] = sum_{p,t} T2[p, b, t, q] * conj(A)[p, t, r]
    {
        int a_size = chi_in * d * chi_out;
        const Scalar* A_bra;
        if constexpr (Traits::is_complex) {
            make_conjugate(d_mps_tensors_[site], d_conj_buf_, a_size);
            A_bra = d_conj_buf_;
        } else {
            A_bra = d_mps_tensors_[site];
        }
        int64_t extT2[] = {chi_in, chi_out, d, D};
        int32_t modesT2[] = {'p', 'b', 't', 'q'};
        int64_t extAbar[] = {chi_in, d, chi_out};
        int32_t modesAbar[] = {'p', 't', 'r'};
        int64_t extLnew[] = {chi_out, D, chi_out};
        int32_t modesLnew[] = {'b', 'q', 'r'};
        ht_contract(d_T2_, 4, extT2, modesT2,
                    A_bra, 3, extAbar, modesAbar,
                    d_L_envs_[site + 1], 3, extLnew, modesLnew);
    }
}

// ============================================================================
// Right environment update
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::update_right_env(int site) {
    int64_t chi_in = bond_dims_[site + 1];
    int64_t chi_out = bond_dims_[site];
    int64_t D = D_mpo_, d = d_;

    ensure_R_env_alloc(site, chi_out);

    // Step 1: T1[a, s, q, r] = sum_b A[a, s, b] * R[b, q, r]
    {
        int64_t extA[] = {chi_out, d, chi_in};
        int32_t modesA[] = {'a', 's', 'b'};
        int64_t extR[] = {chi_in, D, chi_in};
        int32_t modesR[] = {'b', 'q', 'r'};
        int64_t extT1[] = {chi_out, d, D, chi_in};
        int32_t modesT1[] = {'a', 's', 'q', 'r'};
        ht_contract(d_mps_tensors_[site], 3, extA, modesA,
                    d_R_envs_[site + 1], 3, extR, modesR,
                    d_T1_, 4, extT1, modesT1);
    }

    // Step 2: T2[a, r, w, t] = sum_{s,q} W[w, s, t, q] * T1[a, s, q, r]
    {
        int64_t extW[] = {D, d, d, D};
        int32_t modesW[] = {'w', 's', 't', 'q'};
        int64_t extT1[] = {chi_out, d, D, chi_in};
        int32_t modesT1[] = {'a', 's', 'q', 'r'};
        int64_t extT2[] = {chi_out, chi_in, D, d};
        int32_t modesT2[] = {'a', 'r', 'w', 't'};
        ht_contract(d_mpo_tensors_[site], 4, extW, modesW,
                    d_T1_, 4, extT1, modesT1,
                    d_T2_, 4, extT2, modesT2);
    }

    // Step 3: R_new[a, w, p] = sum_{r,t} T2[a, r, w, t] * conj(A)[p, t, r]
    {
        int a_size = chi_out * d * chi_in;
        const Scalar* A_bra;
        if constexpr (Traits::is_complex) {
            make_conjugate(d_mps_tensors_[site], d_conj_buf_, a_size);
            A_bra = d_conj_buf_;
        } else {
            A_bra = d_mps_tensors_[site];
        }
        int64_t extT2[] = {chi_out, chi_in, D, d};
        int32_t modesT2[] = {'a', 'r', 'w', 't'};
        int64_t extAbar[] = {chi_out, d, chi_in};
        int32_t modesAbar[] = {'p', 't', 'r'};
        int64_t extRnew[] = {chi_out, D, chi_out};
        int32_t modesRnew[] = {'a', 'w', 'p'};
        ht_contract(d_T2_, 4, extT2, modesT2,
                    A_bra, 3, extAbar, modesAbar,
                    d_R_envs_[site], 3, extRnew, modesRnew);
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
    printf("=== GPU-Native DMRG (hipTensor + rocBLAS, %s) ===\n", type_name);
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
