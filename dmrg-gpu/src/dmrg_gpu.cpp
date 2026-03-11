#include "dmrg_gpu.h"
#include <rocsolver/rocsolver.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <fstream>
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

#define HIPTENSOR_CHECK(call) \
    do { \
        hiptensorStatus_t st = call; \
        if (st != HIPTENSOR_STATUS_SUCCESS) { \
            std::cerr << "hipTensor error in " << __FILE__ << ":" << __LINE__ \
                      << " - status " << st << std::endl; \
            throw std::runtime_error("hipTensor error"); \
        } \
    } while(0)

// LAPACK tridiagonal eigensolver (kept for Lanczos - negligible cost)
extern "C" void dstev_(const char* jobz, const int* n, double* d, double* e,
                       double* z, const int* ldz, double* work, int* info);

// Mode labels for hipTensor contractions
enum : int32_t {
    M_a  = 10,   // left bond (ket)
    M_w  = 11,   // MPO left bond
    M_ap = 12,   // left bond (bra)
    M_s  = 13,   // physical (ket)
    M_b  = 14,   // right bond (ket)
    M_sp = 15,   // physical (bra)
    M_wp = 16,   // MPO right bond
    M_bp = 17,   // right bond (bra)
};

// ============================================================================
// Helper: create a single hipTensor contraction plan
// ============================================================================
static hiptensorPlan_t create_contraction_plan(
    hiptensorHandle_t handle,
    int nA, const int64_t* extA, const int32_t* modesA,
    int nB, const int64_t* extB, const int32_t* modesB,
    int nC, const int64_t* extC, const int32_t* modesC,
    void** d_workspace, uint64_t* ws_size)
{
    hiptensorTensorDescriptor_t descA, descB, descC;
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(handle, &descA,
        nA, extA, NULL, HIPTENSOR_R_64F, 0));
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(handle, &descB,
        nB, extB, NULL, HIPTENSOR_R_64F, 0));
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(handle, &descC,
        nC, extC, NULL, HIPTENSOR_R_64F, 0));

    hiptensorOperationDescriptor_t opDesc;
    HIPTENSOR_CHECK(hiptensorCreateContraction(handle, &opDesc,
        descA, modesA, HIPTENSOR_OP_IDENTITY,
        descB, modesB, HIPTENSOR_OP_IDENTITY,
        descC, modesC, HIPTENSOR_OP_IDENTITY,
        descC, modesC,
        HIPTENSOR_COMPUTE_DESC_64F));

    hiptensorPlanPreference_t pref;
    HIPTENSOR_CHECK(hiptensorCreatePlanPreference(handle, &pref,
        HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE));

    hiptensorPlan_t plan;
    HIPTENSOR_CHECK(hiptensorCreatePlan(handle, &plan, opDesc, pref, 0));

    *ws_size = 0;
    HIPTENSOR_CHECK(hiptensorPlanGetAttribute(handle, plan,
        HIPTENSOR_PLAN_REQUIRED_WORKSPACE, ws_size, sizeof(*ws_size)));
    *d_workspace = nullptr;
    if (*ws_size > 0) HIP_CHECK(hipMalloc(d_workspace, *ws_size));

    // Plans capture all info; descriptors can be destroyed
    HIPTENSOR_CHECK(hiptensorDestroyPlanPreference(pref));
    HIPTENSOR_CHECK(hiptensorDestroyOperationDescriptor(opDesc));
    HIPTENSOR_CHECK(hiptensorDestroyTensorDescriptor(descC));
    HIPTENSOR_CHECK(hiptensorDestroyTensorDescriptor(descB));
    HIPTENSOR_CHECK(hiptensorDestroyTensorDescriptor(descA));

    return plan;
}

// Helper to destroy a CachedPlans and free its workspace
static void destroy_cached_plans(DMRGGPU::CachedPlans* p) {
    if (!p) return;
    hiptensorDestroyPlan(p->plan1);
    hiptensorDestroyPlan(p->plan2);
    hiptensorDestroyPlan(p->plan3);
    if (p->ws1) hipFree(p->ws1);
    if (p->ws2) hipFree(p->ws2);
    if (p->ws3) hipFree(p->ws3);
    delete p;
}

// ============================================================================
// Constructor
// ============================================================================

DMRGGPU::DMRGGPU(int L, int d, int chi_max, int D_mpo, double tol)
    : L_(L), d_(d), chi_max_(chi_max), D_mpo_(D_mpo), tol_(tol), energy_(0.0) {

    // Bond dimensions
    bond_dims_.resize(L + 1);
    bond_dims_[0] = 1;
    bond_dims_[L] = 1;
    for (int i = 1; i < L; i++) {
        bond_dims_[i] = std::min(chi_max_, (int)pow(d_, std::min(i, L - i)));
    }

    // GPU handles
    HIP_CHECK(hipStreamCreate(&stream_));
    ROCBLAS_CHECK(rocblas_create_handle(&rocblas_h_));
    ROCBLAS_CHECK(rocblas_set_stream(rocblas_h_, stream_));

    // hipTensor handle
    HIPTENSOR_CHECK(hiptensorCreate(&ht_handle_));

    // Contraction intermediates (max-sized: D_mpo * d * chi_max^2)
    int t_max = D_mpo_ * d_ * chi_max_ * chi_max_;
    HIP_CHECK(hipMalloc(&d_T1_, t_max * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_T2_, t_max * sizeof(double)));

    // MPS tensors
    d_mps_tensors_.resize(L, nullptr);
    for (int i = 0; i < L; i++) {
        allocate_mps_tensor(i, chi_L(i), chi_R(i));
    }

    // MPO tensors
    d_mpo_tensors_.resize(L, nullptr);

    // Environments (allocate interior at chi_max to avoid reallocation)
    d_L_envs_.resize(L + 1, nullptr);
    d_R_envs_.resize(L + 1, nullptr);
    L_env_alloc_chi_.resize(L + 1, 0);
    R_env_alloc_chi_.resize(L + 1, 0);

    for (int i = 0; i <= L; i++) {
        int chi_alloc = (i == 0 || i == L) ? 1 : chi_max_;
        int sz = chi_alloc * D_mpo_ * chi_alloc;
        HIP_CHECK(hipMalloc(&d_L_envs_[i], sz * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_R_envs_[i], sz * sizeof(double)));
        HIP_CHECK(hipMemset(d_L_envs_[i], 0, sz * sizeof(double)));
        HIP_CHECK(hipMemset(d_R_envs_[i], 0, sz * sizeof(double)));
        L_env_alloc_chi_[i] = chi_alloc;
        R_env_alloc_chi_[i] = chi_alloc;
    }

    // Lanczos workspace
    theta_size_max_ = chi_max_ * d_ * chi_max_;
    HIP_CHECK(hipMalloc(&d_theta_, theta_size_max_ * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_heff_result_, theta_size_max_ * sizeof(double)));

    // SVD workspace (pre-allocated at max dimensions)
    int svd_max_dim = chi_max_ * d_;  // max of m or n across both sweep dirs
    HIP_CHECK(hipMalloc(&d_svd_A_,    theta_size_max_ * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_svd_U_,    svd_max_dim * chi_max_ * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_svd_S_,    chi_max_ * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_svd_Vh_,   chi_max_ * svd_max_dim * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_svd_E_,    chi_max_ * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_svd_info_, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_svd_work_, theta_size_max_ * sizeof(double)));
}

// ============================================================================
// Destructor
// ============================================================================

DMRGGPU::~DMRGGPU() {
    free_gpu_resources();
}

void DMRGGPU::free_gpu_resources() {
    for (auto ptr : d_mps_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d_mpo_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d_L_envs_) if (ptr) hipFree(ptr);
    for (auto ptr : d_R_envs_) if (ptr) hipFree(ptr);

    if (d_theta_) hipFree(d_theta_);
    if (d_heff_result_) hipFree(d_heff_result_);
    if (d_T1_) hipFree(d_T1_);
    if (d_T2_) hipFree(d_T2_);
    if (d_svd_A_) hipFree(d_svd_A_);
    if (d_svd_U_) hipFree(d_svd_U_);
    if (d_svd_S_) hipFree(d_svd_S_);
    if (d_svd_Vh_) hipFree(d_svd_Vh_);
    if (d_svd_E_) hipFree(d_svd_E_);
    if (d_svd_info_) hipFree(d_svd_info_);
    if (d_svd_work_) hipFree(d_svd_work_);

    for (auto& [k, v] : heff_plan_cache_) destroy_cached_plans(v);
    for (auto& [k, v] : lenv_plan_cache_) destroy_cached_plans(v);
    for (auto& [k, v] : renv_plan_cache_) destroy_cached_plans(v);

    hiptensorDestroy(ht_handle_);
    rocblas_destroy_handle(rocblas_h_);
    hipStreamDestroy(stream_);
}

// ============================================================================
// Memory management
// ============================================================================

void DMRGGPU::allocate_mps_tensor(int site, int cL, int cR) {
    if (d_mps_tensors_[site]) HIP_CHECK(hipFree(d_mps_tensors_[site]));
    HIP_CHECK(hipMalloc(&d_mps_tensors_[site], cL * d_ * cR * sizeof(double)));
}

void DMRGGPU::ensure_L_env_alloc(int idx, int chi) {
    if (chi > L_env_alloc_chi_[idx]) {
        if (d_L_envs_[idx]) HIP_CHECK(hipFree(d_L_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_L_envs_[idx], sz * sizeof(double)));
        L_env_alloc_chi_[idx] = chi;
    }
}

void DMRGGPU::ensure_R_env_alloc(int idx, int chi) {
    if (chi > R_env_alloc_chi_[idx]) {
        if (d_R_envs_[idx]) HIP_CHECK(hipFree(d_R_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_R_envs_[idx], sz * sizeof(double)));
        R_env_alloc_chi_[idx] = chi;
    }
}

// ============================================================================
// MPS initialization (host -> device copies, unchanged)
// ============================================================================

void DMRGGPU::initialize_mps_random(double scale) {
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        std::vector<double> h_A(size);
        for (int j = 0; j < size; j++) {
            h_A[j] = scale * (2.0 * rand() / RAND_MAX - 1.0);
        }
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(double), hipMemcpyHostToDevice));
    }
}

void DMRGGPU::initialize_mps_product() {
    for (int i = 0; i < L_; i++) {
        int cL = chi_L(i), cR = chi_R(i);
        int size = cL * d_ * cR;
        std::vector<double> h_A(size, 0.0);
        int chi_min = std::min(cL, cR);
        for (int a = 0; a < chi_min; a++) {
            h_A[a + 0*cL + a*cL*d_] = 1.0;
        }
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(double), hipMemcpyHostToDevice));
    }
}

void DMRGGPU::initialize_mps_neel() {
    for (int i = 0; i < L_; i++) {
        int cL = chi_L(i), cR = chi_R(i);
        int size = cL * d_ * cR;
        std::vector<double> h_A(size, 0.0);
        int spin = (i % 2 == 0) ? 0 : 1;
        int chi_min = std::min(cL, cR);
        for (int a = 0; a < chi_min; a++) {
            h_A[a + spin*cL + a*cL*d_] = 1.0;
        }
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(double), hipMemcpyHostToDevice));
    }
}

void DMRGGPU::set_mpo(const std::vector<double*>& h_mpo_tensors) {
    for (int i = 0; i < L_; i++) {
        int size = D_mpo_ * d_ * d_ * D_mpo_;
        HIP_CHECK(hipMalloc(&d_mpo_tensors_[i], size * sizeof(double)));
        HIP_CHECK(hipMemcpy(d_mpo_tensors_[i], h_mpo_tensors[i],
                            size * sizeof(double), hipMemcpyHostToDevice));
    }
}

// ============================================================================
// hipTensor plan creation (cached by dimensions)
// ============================================================================

// H_eff plans: T1 = L*theta, T2 = W*T1, result = R*T2
DMRGGPU::CachedPlans* DMRGGPU::get_heff_plans(int cL, int cR) {
    auto key = std::make_pair(cL, cR);
    auto it = heff_plan_cache_.find(key);
    if (it != heff_plan_cache_.end()) return it->second;

    auto* p = new CachedPlans();
    int D = D_mpo_, d = d_;

    // Step 1: T1[w, a', s, b] = sum_a L[a, w, a'] * theta[a, s, b]
    {
        int64_t eA[] = {(int64_t)cL, (int64_t)D, (int64_t)cL};
        int32_t mA[] = {M_a, M_w, M_ap};
        int64_t eB[] = {(int64_t)cL, (int64_t)d, (int64_t)cR};
        int32_t mB[] = {M_a, M_s, M_b};
        int64_t eC[] = {(int64_t)D, (int64_t)cL, (int64_t)d, (int64_t)cR};
        int32_t mC[] = {M_w, M_ap, M_s, M_b};
        p->plan1 = create_contraction_plan(ht_handle_,
            3, eA, mA, 3, eB, mB, 4, eC, mC, &p->ws1, &p->ws_sz1);
    }

    // Step 2: T2[a', s', w', b] = sum_{w,s} W[w, s, s', w'] * T1[w, a', s, b]
    {
        int64_t eA[] = {(int64_t)D, (int64_t)d, (int64_t)d, (int64_t)D};
        int32_t mA[] = {M_w, M_s, M_sp, M_wp};
        int64_t eB[] = {(int64_t)D, (int64_t)cL, (int64_t)d, (int64_t)cR};
        int32_t mB[] = {M_w, M_ap, M_s, M_b};
        int64_t eC[] = {(int64_t)cL, (int64_t)d, (int64_t)D, (int64_t)cR};
        int32_t mC[] = {M_ap, M_sp, M_wp, M_b};
        p->plan2 = create_contraction_plan(ht_handle_,
            4, eA, mA, 4, eB, mB, 4, eC, mC, &p->ws2, &p->ws_sz2);
    }

    // Step 3: result[a', s', b'] = sum_{w',b} R[b, w', b'] * T2[a', s', w', b]
    {
        int64_t eA[] = {(int64_t)cR, (int64_t)D, (int64_t)cR};
        int32_t mA[] = {M_b, M_wp, M_bp};
        int64_t eB[] = {(int64_t)cL, (int64_t)d, (int64_t)D, (int64_t)cR};
        int32_t mB[] = {M_ap, M_sp, M_wp, M_b};
        int64_t eC[] = {(int64_t)cL, (int64_t)d, (int64_t)cR};
        int32_t mC[] = {M_ap, M_sp, M_bp};
        p->plan3 = create_contraction_plan(ht_handle_,
            3, eA, mA, 4, eB, mB, 3, eC, mC, &p->ws3, &p->ws_sz3);
    }

    heff_plan_cache_[key] = p;
    return p;
}

// Left environment update plans
DMRGGPU::CachedPlans* DMRGGPU::get_lenv_plans(int chi_in, int chi_out) {
    auto key = std::make_pair(chi_in, chi_out);
    auto it = lenv_plan_cache_.find(key);
    if (it != lenv_plan_cache_.end()) return it->second;

    auto* p = new CachedPlans();
    int D = D_mpo_, d = d_;

    // Step 1: T1[w, a', s, b] = sum_a L[a, w, a'] * A[a, s, b]
    {
        int64_t eA[] = {(int64_t)chi_in, (int64_t)D, (int64_t)chi_in};
        int32_t mA[] = {M_a, M_w, M_ap};
        int64_t eB[] = {(int64_t)chi_in, (int64_t)d, (int64_t)chi_out};
        int32_t mB[] = {M_a, M_s, M_b};
        int64_t eC[] = {(int64_t)D, (int64_t)chi_in, (int64_t)d, (int64_t)chi_out};
        int32_t mC[] = {M_w, M_ap, M_s, M_b};
        p->plan1 = create_contraction_plan(ht_handle_,
            3, eA, mA, 3, eB, mB, 4, eC, mC, &p->ws1, &p->ws_sz1);
    }

    // Step 2: T2[a', s', w', b] = sum_{w,s} W[w, s, s', w'] * T1[w, a', s, b]
    {
        int64_t eA[] = {(int64_t)D, (int64_t)d, (int64_t)d, (int64_t)D};
        int32_t mA[] = {M_w, M_s, M_sp, M_wp};
        int64_t eB[] = {(int64_t)D, (int64_t)chi_in, (int64_t)d, (int64_t)chi_out};
        int32_t mB[] = {M_w, M_ap, M_s, M_b};
        int64_t eC[] = {(int64_t)chi_in, (int64_t)d, (int64_t)D, (int64_t)chi_out};
        int32_t mC[] = {M_ap, M_sp, M_wp, M_b};
        p->plan2 = create_contraction_plan(ht_handle_,
            4, eA, mA, 4, eB, mB, 4, eC, mC, &p->ws2, &p->ws_sz2);
    }

    // Step 3: L_new[b, w', b'] = sum_{a',s'} A*[a', s', b'] * T2[a', s', w', b]
    {
        int64_t eA[] = {(int64_t)chi_in, (int64_t)d, (int64_t)chi_out};
        int32_t mA[] = {M_ap, M_sp, M_bp};
        int64_t eB[] = {(int64_t)chi_in, (int64_t)d, (int64_t)D, (int64_t)chi_out};
        int32_t mB[] = {M_ap, M_sp, M_wp, M_b};
        int64_t eC[] = {(int64_t)chi_out, (int64_t)D, (int64_t)chi_out};
        int32_t mC[] = {M_b, M_wp, M_bp};
        p->plan3 = create_contraction_plan(ht_handle_,
            3, eA, mA, 4, eB, mB, 3, eC, mC, &p->ws3, &p->ws_sz3);
    }

    lenv_plan_cache_[key] = p;
    return p;
}

// Right environment update plans
DMRGGPU::CachedPlans* DMRGGPU::get_renv_plans(int chi_in, int chi_out) {
    auto key = std::make_pair(chi_in, chi_out);
    auto it = renv_plan_cache_.find(key);
    if (it != renv_plan_cache_.end()) return it->second;

    auto* p = new CachedPlans();
    int D = D_mpo_, d = d_;

    // Step 1: T1[a, s, w', b'] = sum_b A[a, s, b] * R[b, w', b']
    {
        int64_t eA[] = {(int64_t)chi_out, (int64_t)d, (int64_t)chi_in};
        int32_t mA[] = {M_a, M_s, M_b};
        int64_t eB[] = {(int64_t)chi_in, (int64_t)D, (int64_t)chi_in};
        int32_t mB[] = {M_b, M_wp, M_bp};
        int64_t eC[] = {(int64_t)chi_out, (int64_t)d, (int64_t)D, (int64_t)chi_in};
        int32_t mC[] = {M_a, M_s, M_wp, M_bp};
        p->plan1 = create_contraction_plan(ht_handle_,
            3, eA, mA, 3, eB, mB, 4, eC, mC, &p->ws1, &p->ws_sz1);
    }

    // Step 2: T2[a, s', w, b'] = sum_{s,w'} W[w, s, s', w'] * T1[a, s, w', b']
    {
        int64_t eA[] = {(int64_t)D, (int64_t)d, (int64_t)d, (int64_t)D};
        int32_t mA[] = {M_w, M_s, M_sp, M_wp};
        int64_t eB[] = {(int64_t)chi_out, (int64_t)d, (int64_t)D, (int64_t)chi_in};
        int32_t mB[] = {M_a, M_s, M_wp, M_bp};
        int64_t eC[] = {(int64_t)chi_out, (int64_t)d, (int64_t)D, (int64_t)chi_in};
        int32_t mC[] = {M_a, M_sp, M_w, M_bp};
        p->plan2 = create_contraction_plan(ht_handle_,
            4, eA, mA, 4, eB, mB, 4, eC, mC, &p->ws2, &p->ws_sz2);
    }

    // Step 3: R_new[a, w, a'] = sum_{s',b'} A*[a', s', b'] * T2[a, s', w, b']
    {
        int64_t eA[] = {(int64_t)chi_out, (int64_t)d, (int64_t)chi_in};
        int32_t mA[] = {M_ap, M_sp, M_bp};
        int64_t eB[] = {(int64_t)chi_out, (int64_t)d, (int64_t)D, (int64_t)chi_in};
        int32_t mB[] = {M_a, M_sp, M_w, M_bp};
        int64_t eC[] = {(int64_t)chi_out, (int64_t)D, (int64_t)chi_out};
        int32_t mC[] = {M_a, M_w, M_ap};
        p->plan3 = create_contraction_plan(ht_handle_,
            3, eA, mA, 4, eB, mB, 3, eC, mC, &p->ws3, &p->ws_sz3);
    }

    renv_plan_cache_[key] = p;
    return p;
}

// ============================================================================
// GPU Environment building
// ============================================================================

void DMRGGPU::build_initial_environments() {
    // L[0] = trivial left boundary: (1, D_mpo, 1), L[0][0,0,0] = 1
    {
        std::vector<double> h_L(D_mpo_, 0.0);
        h_L[0] = 1.0;
        HIP_CHECK(hipMemcpy(d_L_envs_[0], h_L.data(),
                            D_mpo_ * sizeof(double), hipMemcpyHostToDevice));
    }

    // R[L] = trivial right boundary: (1, D_mpo, 1), R[L][0,D-1,0] = 1
    {
        std::vector<double> h_R(D_mpo_, 0.0);
        h_R[D_mpo_ - 1] = 1.0;
        HIP_CHECK(hipMemcpy(d_R_envs_[L_], h_R.data(),
                            D_mpo_ * sizeof(double), hipMemcpyHostToDevice));
    }

    // Build all R environments from right to left (uses GPU contractions)
    for (int i = L_ - 1; i >= 0; i--) {
        update_right_env(i);
    }
}

// GPU left environment update via hipTensor
void DMRGGPU::update_left_env(int site) {
    int chi_in = bond_dims_[site];
    int chi_out = bond_dims_[site + 1];

    ensure_L_env_alloc(site + 1, chi_out);

    auto* plans = get_lenv_plans(chi_in, chi_out);
    double alpha = 1.0, beta = 0.0;

    // Step 1: T1 = L * A
    HIPTENSOR_CHECK(hiptensorContract(ht_handle_, plans->plan1,
        &alpha, d_L_envs_[site], d_mps_tensors_[site],
        &beta, d_T1_, d_T1_,
        plans->ws1, plans->ws_sz1, stream_));

    // Step 2: T2 = W * T1
    HIPTENSOR_CHECK(hiptensorContract(ht_handle_, plans->plan2,
        &alpha, d_mpo_tensors_[site], d_T1_,
        &beta, d_T2_, d_T2_,
        plans->ws2, plans->ws_sz2, stream_));

    // Step 3: L_new = A* . T2 (A* = A for real tensors)
    HIPTENSOR_CHECK(hiptensorContract(ht_handle_, plans->plan3,
        &alpha, d_mps_tensors_[site], d_T2_,
        &beta, d_L_envs_[site + 1], d_L_envs_[site + 1],
        plans->ws3, plans->ws_sz3, stream_));
}

// GPU right environment update via hipTensor
void DMRGGPU::update_right_env(int site) {
    int chi_in = bond_dims_[site + 1];
    int chi_out = bond_dims_[site];

    ensure_R_env_alloc(site, chi_out);

    auto* plans = get_renv_plans(chi_in, chi_out);
    double alpha = 1.0, beta = 0.0;

    // Step 1: T1 = A * R
    HIPTENSOR_CHECK(hiptensorContract(ht_handle_, plans->plan1,
        &alpha, d_mps_tensors_[site], d_R_envs_[site + 1],
        &beta, d_T1_, d_T1_,
        plans->ws1, plans->ws_sz1, stream_));

    // Step 2: T2 = W * T1
    HIPTENSOR_CHECK(hiptensorContract(ht_handle_, plans->plan2,
        &alpha, d_mpo_tensors_[site], d_T1_,
        &beta, d_T2_, d_T2_,
        plans->ws2, plans->ws_sz2, stream_));

    // Step 3: R_new = A* . T2
    HIPTENSOR_CHECK(hiptensorContract(ht_handle_, plans->plan3,
        &alpha, d_mps_tensors_[site], d_T2_,
        &beta, d_R_envs_[site], d_R_envs_[site],
        plans->ws3, plans->ws_sz3, stream_));
}

// ============================================================================
// GPU H_eff application via hipTensor (THE CRITICAL HOT PATH)
// ============================================================================

void DMRGGPU::form_theta(int site, double* d_theta) {
    int size = chi_L(site) * d_ * chi_R(site);
    HIP_CHECK(hipMemcpy(d_theta, d_mps_tensors_[site],
                        size * sizeof(double), hipMemcpyDeviceToDevice));
}

void DMRGGPU::apply_heff(int site, const double* d_theta_in, double* d_result) {
    int cL = chi_L(site);
    int cR = chi_R(site);

    auto* plans = get_heff_plans(cL, cR);
    double alpha = 1.0, beta = 0.0;

    // Step 1: T1[w, a', s, b] = sum_a L[a, w, a'] * theta[a, s, b]
    HIPTENSOR_CHECK(hiptensorContract(ht_handle_, plans->plan1,
        &alpha, d_L_envs_[site], d_theta_in,
        &beta, d_T1_, d_T1_,
        plans->ws1, plans->ws_sz1, stream_));

    // Step 2: T2[a', s', w', b] = sum_{w,s} W[w, s, s', w'] * T1[w, a', s, b]
    HIPTENSOR_CHECK(hiptensorContract(ht_handle_, plans->plan2,
        &alpha, d_mpo_tensors_[site], d_T1_,
        &beta, d_T2_, d_T2_,
        plans->ws2, plans->ws_sz2, stream_));

    // Step 3: result[a', s', b'] = sum_{w',b} R[b, w', b'] * T2[a', s', w', b]
    HIPTENSOR_CHECK(hiptensorContract(ht_handle_, plans->plan3,
        &alpha, d_R_envs_[site + 1], d_T2_,
        &beta, d_result, d_result,
        plans->ws3, plans->ws_sz3, stream_));
}

// ============================================================================
// Lanczos eigensolver with FULL reorthogonalization
// (GPU BLAS for vectors, CPU LAPACK for small tridiagonal matrix)
// ============================================================================

double DMRGGPU::lanczos_eigensolver(int site, double* d_theta) {
    int n = chi_L(site) * d_ * chi_R(site);
    int max_iter = std::min(100, n);
    double tol_lanczos = 1e-12;

    double* d_lanczos_v;
    HIP_CHECK(hipMalloc(&d_lanczos_v, max_iter * n * sizeof(double)));

    std::vector<double> h_alpha(max_iter);
    std::vector<double> h_beta(max_iter);

    // v[0] = theta / ||theta||
    double norm;
    ROCBLAS_CHECK(rocblas_dnrm2(rocblas_h_, n, d_theta, 1, &norm));

    if (norm < 1e-14) {
        std::vector<double> h_init(n);
        srand(42 + site);
        for (int i = 0; i < n; i++) h_init[i] = 2.0 * rand() / RAND_MAX - 1.0;
        HIP_CHECK(hipMemcpy(d_theta, h_init.data(), n * sizeof(double), hipMemcpyHostToDevice));
        ROCBLAS_CHECK(rocblas_dnrm2(rocblas_h_, n, d_theta, 1, &norm));
    }

    double inv_norm = 1.0 / norm;
    ROCBLAS_CHECK(rocblas_dscal(rocblas_h_, n, &inv_norm, d_theta, 1));
    HIP_CHECK(hipMemcpy(d_lanczos_v, d_theta, n * sizeof(double), hipMemcpyDeviceToDevice));

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        double* d_vi = d_lanczos_v + iter * n;

        // w = H|v_i> (GPU hipTensor contraction)
        apply_heff(site, d_vi, d_heff_result_);

        // alpha_i = <v_i|w> (GPU dot product)
        double alpha_i;
        ROCBLAS_CHECK(rocblas_ddot(rocblas_h_, n, d_vi, 1, d_heff_result_, 1, &alpha_i));
        h_alpha[iter] = alpha_i;

        // w = w - alpha_i * v_i
        double neg_alpha = -alpha_i;
        ROCBLAS_CHECK(rocblas_daxpy(rocblas_h_, n, &neg_alpha, d_vi, 1, d_heff_result_, 1));

        // w = w - beta_{i-1} * v_{i-1}
        if (iter > 0) {
            double neg_beta = -h_beta[iter - 1];
            double* d_vim1 = d_lanczos_v + (iter - 1) * n;
            ROCBLAS_CHECK(rocblas_daxpy(rocblas_h_, n, &neg_beta, d_vim1, 1, d_heff_result_, 1));
        }

        // FULL REORTHOGONALIZATION
        for (int j = 0; j <= iter; j++) {
            double* d_vj = d_lanczos_v + j * n;
            double overlap;
            ROCBLAS_CHECK(rocblas_ddot(rocblas_h_, n, d_vj, 1, d_heff_result_, 1, &overlap));
            double neg_overlap = -overlap;
            ROCBLAS_CHECK(rocblas_daxpy(rocblas_h_, n, &neg_overlap, d_vj, 1, d_heff_result_, 1));
        }

        // beta_i = ||w||
        double beta_i;
        ROCBLAS_CHECK(rocblas_dnrm2(rocblas_h_, n, d_heff_result_, 1, &beta_i));
        h_beta[iter] = beta_i;

        if (beta_i < tol_lanczos) {
            iter++;
            break;
        }

        // v_{i+1} = w / beta_i
        if (iter + 1 < max_iter) {
            double* d_vip1 = d_lanczos_v + (iter + 1) * n;
            double scale = 1.0 / beta_i;
            HIP_CHECK(hipMemcpy(d_vip1, d_heff_result_, n * sizeof(double), hipMemcpyDeviceToDevice));
            ROCBLAS_CHECK(rocblas_dscal(rocblas_h_, n, &scale, d_vip1, 1));
        }
    }

    int niter = iter;

    // Solve tridiagonal eigenvalue problem on CPU (tiny matrix, negligible cost)
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

    // Reconstruct ground state: |theta> = sum_i c[i] |v_i> (GPU BLAS)
    double* d_ritz_coeffs;
    HIP_CHECK(hipMalloc(&d_ritz_coeffs, niter * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_ritz_coeffs, h_Z.data(), niter * sizeof(double), hipMemcpyHostToDevice));

    const double one = 1.0, zero = 0.0;
    ROCBLAS_CHECK(rocblas_dgemv(
        rocblas_h_, rocblas_operation_none,
        n, niter, &one,
        d_lanczos_v, n,
        d_ritz_coeffs, 1,
        &zero, d_theta, 1
    ));

    // Normalize
    ROCBLAS_CHECK(rocblas_dnrm2(rocblas_h_, n, d_theta, 1, &norm));
    inv_norm = 1.0 / norm;
    ROCBLAS_CHECK(rocblas_dscal(rocblas_h_, n, &inv_norm, d_theta, 1));

    HIP_CHECK(hipFree(d_ritz_coeffs));
    HIP_CHECK(hipFree(d_lanczos_v));

    return energy;
}

// ============================================================================
// GPU SVD and MPS update (rocsolver_dgesvd + rocblas)
// ============================================================================

void DMRGGPU::svd_and_update_mps(int site, double* d_theta, char direction) {
    int cL = chi_L(site);
    int cR = chi_R(site);

    if (direction == 'R') {
        // theta[a,s,b] reshaped to M(cL*d, cR) -> U S Vh
        int m = cL * d_;
        int n_svd = cR;
        int k = std::min(m, n_svd);
        k = std::min(k, chi_max_);

        // Copy theta to SVD workspace (rocsolver overwrites input)
        HIP_CHECK(hipMemcpy(d_svd_A_, d_theta, m * n_svd * sizeof(double),
                            hipMemcpyDeviceToDevice));

        // GPU SVD via rocsolver
        rocsolver_dgesvd(rocblas_h_,
            rocblas_svect_singular, rocblas_svect_singular,
            m, n_svd,
            d_svd_A_, m,
            d_svd_S_,
            d_svd_U_, m,
            d_svd_Vh_, std::min(m, n_svd),
            d_svd_E_,
            rocblas_outofplace,
            d_svd_info_);

        // Read S to host for truncation decision (small copy, scalar-like)
        int full_k = std::min(m, n_svd);
        std::vector<double> h_S(full_k);
        HIP_CHECK(hipMemcpy(h_S.data(), d_svd_S_, full_k * sizeof(double),
                            hipMemcpyDeviceToHost));

        int new_k = std::min(k, full_k);
        for (int i = 0; i < new_k; i++) {
            if (h_S[i] < 1e-14) { new_k = i; break; }
        }
        if (new_k == 0) new_k = 1;

        int new_chi_R = new_k;

        // A[site] = U[:, :new_k] (first new_k columns of U)
        allocate_mps_tensor(site, cL, new_chi_R);
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], d_svd_U_,
                            m * new_chi_R * sizeof(double), hipMemcpyDeviceToDevice));

        // Compute S*Vh on GPU: SV[i,j] = S[i] * Vh[i,j]
        // Vh is (full_k, n_svd) column-major, lda=full_k
        // SV will be (new_k, n_svd), lda=new_k
        ROCBLAS_CHECK(rocblas_ddgmm(rocblas_h_, rocblas_side_left,
            new_k, n_svd,
            d_svd_Vh_, full_k,     // A: first new_k rows of each column
            d_svd_S_, 1,           // diagonal: S
            d_svd_work_, new_k));  // C: S*Vh output

        // Absorb SV into A[site+1] via GPU GEMM
        if (site + 1 < L_) {
            int next_cR = chi_R(site + 1);
            // new_A_next = SV @ old_A_next
            // SV: (new_k, cR), old_A_next: (cR, d*next_cR), result: (new_k, d*next_cR)
            double one = 1.0, zero = 0.0;

            ROCBLAS_CHECK(rocblas_dgemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                new_k, d_ * next_cR, cR,
                &one,
                d_svd_work_, new_k,            // SV
                d_mps_tensors_[site + 1], cR,  // old A_next
                &zero,
                d_T1_, new_k));                // new A_next (temp buffer)

            allocate_mps_tensor(site + 1, new_chi_R, next_cR);
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site + 1], d_T1_,
                                new_k * d_ * next_cR * sizeof(double),
                                hipMemcpyDeviceToDevice));
        }

        bond_dims_[site + 1] = new_chi_R;

    } else {  // direction == 'L'
        // theta[a,s,b] reshaped to M(cL, d*cR) -> U S Vh
        int m = cL;
        int n_svd = d_ * cR;
        int k = std::min(m, n_svd);
        k = std::min(k, chi_max_);

        HIP_CHECK(hipMemcpy(d_svd_A_, d_theta, m * n_svd * sizeof(double),
                            hipMemcpyDeviceToDevice));

        rocsolver_dgesvd(rocblas_h_,
            rocblas_svect_singular, rocblas_svect_singular,
            m, n_svd,
            d_svd_A_, m,
            d_svd_S_,
            d_svd_U_, m,
            d_svd_Vh_, std::min(m, n_svd),
            d_svd_E_,
            rocblas_outofplace,
            d_svd_info_);

        int full_k = std::min(m, n_svd);
        std::vector<double> h_S(full_k);
        HIP_CHECK(hipMemcpy(h_S.data(), d_svd_S_, full_k * sizeof(double),
                            hipMemcpyDeviceToHost));

        int new_k = std::min(k, full_k);
        for (int i = 0; i < new_k; i++) {
            if (h_S[i] < 1e-14) { new_k = i; break; }
        }
        if (new_k == 0) new_k = 1;

        int new_chi_L = new_k;

        // A[site] = Vh[:new_k, :] (first new_k rows of Vh)
        allocate_mps_tensor(site, new_chi_L, cR);
        if (new_chi_L == full_k) {
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site], d_svd_Vh_,
                                full_k * n_svd * sizeof(double), hipMemcpyDeviceToDevice));
        } else {
            // Extract first new_k rows from column-major (full_k, n_svd) via 2D copy
            HIP_CHECK(hipMemcpy2D(
                d_mps_tensors_[site], new_chi_L * sizeof(double),  // dst, dpitch
                d_svd_Vh_,            full_k * sizeof(double),     // src, spitch
                new_chi_L * sizeof(double),                        // width
                n_svd,                                             // height
                hipMemcpyDeviceToDevice));
        }

        // Compute U*S on GPU: US[i,j] = U[i,j] * S[j]
        // U is (m, full_k) column-major, lda=m. Use first new_k columns.
        ROCBLAS_CHECK(rocblas_ddgmm(rocblas_h_, rocblas_side_right,
            m, new_k,
            d_svd_U_, m,          // A: first new_k columns of U
            d_svd_S_, 1,          // diagonal: S
            d_svd_work_, m));     // C: U*S output

        // Absorb US into A[site-1] via GPU GEMM
        if (site > 0) {
            int prev_cL = chi_L(site - 1);
            // new_A_prev = old_A_prev @ US
            // old_A_prev viewed as (prev_cL*d, cL), US: (cL, new_k)
            // result: (prev_cL*d, new_k)
            double one = 1.0, zero = 0.0;

            ROCBLAS_CHECK(rocblas_dgemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                prev_cL * d_, new_k, m,   // m = cL
                &one,
                d_mps_tensors_[site - 1], prev_cL * d_,  // old A_prev
                d_svd_work_, m,                           // US
                &zero,
                d_T1_, prev_cL * d_));                    // new A_prev (temp)

            allocate_mps_tensor(site - 1, prev_cL, new_chi_L);
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site - 1], d_T1_,
                                prev_cL * d_ * new_k * sizeof(double),
                                hipMemcpyDeviceToDevice));
        }

        bond_dims_[site] = new_chi_L;
    }
}

// ============================================================================
// Site optimization
// ============================================================================

double DMRGGPU::optimize_site(int site, char direction) {
    form_theta(site, d_theta_);
    double energy = lanczos_eigensolver(site, d_theta_);
    svd_and_update_mps(site, d_theta_, direction);
    return energy;
}

// ============================================================================
// Sweep methods
// ============================================================================

double DMRGGPU::sweep_left_to_right() {
    double energy = 0.0;

    for (int site = 0; site < L_ - 1; site++) {
        energy = optimize_site(site, 'R');
        update_left_env(site);
    }
    // Optimize last site without SVD
    {
        int site = L_ - 1;
        form_theta(site, d_theta_);
        energy = lanczos_eigensolver(site, d_theta_);
        int sz = chi_L(site) * d_ * chi_R(site);
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], d_theta_, sz * sizeof(double),
                            hipMemcpyDeviceToDevice));
    }

    return energy;
}

double DMRGGPU::sweep_right_to_left() {
    double energy = 0.0;

    for (int site = L_ - 1; site >= 1; site--) {
        energy = optimize_site(site, 'L');
        update_right_env(site);
    }
    // Optimize first site without SVD
    {
        int site = 0;
        form_theta(site, d_theta_);
        energy = lanczos_eigensolver(site, d_theta_);
        int sz = chi_L(site) * d_ * chi_R(site);
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], d_theta_, sz * sizeof(double),
                            hipMemcpyDeviceToDevice));
    }

    return energy;
}

// ============================================================================
// Main algorithm
// ============================================================================

double DMRGGPU::run(int n_sweeps) {
    printf("=== GPU-Native DMRG (hipTensor + rocBLAS) ===\n");
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
        auto t_sweep = std::chrono::high_resolution_clock::now();

        double energy_LR = sweep_left_to_right();
        double energy_RL = sweep_right_to_left();

        auto t_sweep_end = std::chrono::high_resolution_clock::now();
        double sweep_time = std::chrono::duration<double>(t_sweep_end - t_sweep).count();

        energy_ = energy_RL;
        double dE = std::abs(energy_ - energy_prev);

        printf("Sweep %3d: E = %.12f, dE = %.2e, time = %.3f s\n",
               sweep, energy_, dE, sweep_time);

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

void DMRGGPU::load_mps_from_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open MPS file: " + filename);

    int L_file, d_file;
    file.read(reinterpret_cast<char*>(&L_file), sizeof(int));
    file.read(reinterpret_cast<char*>(&d_file), sizeof(int));
    if (L_file != L_ || d_file != d_)
        throw std::runtime_error("MPS file dimensions don't match");

    std::vector<int> bond_dims_file(L_ + 1);
    file.read(reinterpret_cast<char*>(bond_dims_file.data()), (L_ + 1) * sizeof(int));
    bond_dims_ = bond_dims_file;

    for (int i = 0; i < L_; i++) {
        allocate_mps_tensor(i, chi_L(i), chi_R(i));
        int size = chi_L(i) * d_ * chi_R(i);
        std::vector<double> h_A(size);
        file.read(reinterpret_cast<char*>(h_A.data()), size * sizeof(double));
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(double), hipMemcpyHostToDevice));
    }
    file.close();
}

void DMRGGPU::get_mps(std::vector<std::vector<double>>& h_mps) const {
    h_mps.resize(L_);
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        h_mps[i].resize(size);
        HIP_CHECK(hipMemcpy(h_mps[i].data(), d_mps_tensors_[i],
                            size * sizeof(double), hipMemcpyDeviceToHost));
    }
}
