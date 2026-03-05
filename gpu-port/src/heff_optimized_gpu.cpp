#include "heff_optimized_gpu.h"
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.hpp>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <iostream>

// Error checking macros
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << hipGetErrorString(err) << std::endl; \
            throw std::runtime_error("HIP error"); \
        } \
    } while(0)

#define HIPTENSOR_CHECK(call) \
    do { \
        hiptensorStatus_t status = call; \
        if (status != HIPTENSOR_STATUS_SUCCESS) { \
            std::cerr << "hipTensor error in " << __FILE__ << ":" << __LINE__ \
                      << " - status " << status << std::endl; \
            throw std::runtime_error("hipTensor error"); \
        } \
    } while(0)

//==============================================================================
// OptimizedHeff Implementation
//==============================================================================

OptimizedHeff::OptimizedHeff(
    int chi_L_in,
    int chi_R_in,
    int d_in,
    int D_mpo_in,
    hiptensorHandle_t* handle_in
)
    : chi_L(chi_L_in), chi_R(chi_R_in), d(d_in), D_mpo(D_mpo_in),
      handle(handle_in),
      d_T1(nullptr), d_T2(nullptr), d_T3(nullptr),
      is_initialized(false)
{
    try {
        initialize_descriptors();
        create_contractions();
        create_plans();
        allocate_intermediates();
        is_initialized = true;
    } catch (const std::exception& e) {
        std::cerr << "OptimizedHeff initialization failed: " << e.what() << std::endl;
        throw;
    }
}

OptimizedHeff::~OptimizedHeff() {
    // Free intermediate tensors
    if (d_T1) HIP_CHECK(hipFree(d_T1));
    if (d_T2) HIP_CHECK(hipFree(d_T2));
    if (d_T3) HIP_CHECK(hipFree(d_T3));

    // Free workspaces
    for (auto& [key, cache] : workspace_cache) {
        if (cache.is_allocated && cache.d_workspace) {
            HIP_CHECK(hipFree(cache.d_workspace));
        }
    }

    // Destroy plans
    if (is_initialized) {
        hiptensorDestroyPlan(plan_1);
        hiptensorDestroyPlan(plan_2);
        hiptensorDestroyPlan(plan_3);
        hiptensorDestroyPlan(plan_4);

        // Destroy contraction descriptors
        hiptensorDestroyOperationDescriptor(contraction_1);
        hiptensorDestroyOperationDescriptor(contraction_2);
        hiptensorDestroyOperationDescriptor(contraction_3);
        hiptensorDestroyOperationDescriptor(contraction_4);

        // Destroy tensor descriptors
        hiptensorDestroyTensorDescriptor(desc_L);
        hiptensorDestroyTensorDescriptor(desc_R);
        hiptensorDestroyTensorDescriptor(desc_W1);
        hiptensorDestroyTensorDescriptor(desc_W2);
        hiptensorDestroyTensorDescriptor(desc_theta);
        hiptensorDestroyTensorDescriptor(desc_T1);
        hiptensorDestroyTensorDescriptor(desc_T2);
        hiptensorDestroyTensorDescriptor(desc_T3);
        hiptensorDestroyTensorDescriptor(desc_result);
    }
}

void OptimizedHeff::initialize_descriptors() {
    // All tensors in column-major (Fortran) order
    // Extents are specified in logical order, strides will be set to nullptr
    // for default column-major layout

    // L[w, ap, a]: Left environment
    std::vector<int64_t> extent_L = {static_cast<int64_t>(D_mpo),
                                      static_cast<int64_t>(chi_L),
                                      static_cast<int64_t>(chi_L)};
    std::vector<int32_t> mode_L = {'w', 'p', 'a'};
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(
        *handle, &desc_L,
        extent_L.size(), extent_L.data(),
        nullptr,  // strides = nullptr means column-major
        HIP_R_64F,  // Double precision for < 1e-10 validation
        HIPTENSOR_OP_IDENTITY
    ));

    // R[y, b, bp]: Right environment
    std::vector<int64_t> extent_R = {static_cast<int64_t>(D_mpo),
                                      static_cast<int64_t>(chi_R),
                                      static_cast<int64_t>(chi_R)};
    std::vector<int32_t> mode_R = {'y', 'b', 'q'};
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(
        *handle, &desc_R,
        extent_R.size(), extent_R.data(),
        nullptr,
        HIP_R_64F,
        HIPTENSOR_OP_IDENTITY
    ));

    // W1[w, s1, s1p, x]: Left MPO tensor
    std::vector<int64_t> extent_W1 = {static_cast<int64_t>(D_mpo),
                                       static_cast<int64_t>(d),
                                       static_cast<int64_t>(d),
                                       static_cast<int64_t>(D_mpo)};
    std::vector<int32_t> mode_W1 = {'w', 's', 'r', 'x'};
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(
        *handle, &desc_W1,
        extent_W1.size(), extent_W1.data(),
        nullptr,
        HIP_R_64F,
        HIPTENSOR_OP_IDENTITY
    ));

    // W2[x, s2, s2p, y]: Right MPO tensor
    std::vector<int64_t> extent_W2 = {static_cast<int64_t>(D_mpo),
                                       static_cast<int64_t>(d),
                                       static_cast<int64_t>(d),
                                       static_cast<int64_t>(D_mpo)};
    std::vector<int32_t> mode_W2 = {'x', 't', 'u', 'y'};
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(
        *handle, &desc_W2,
        extent_W2.size(), extent_W2.data(),
        nullptr,
        HIP_R_64F,
        HIPTENSOR_OP_IDENTITY
    ));

    // theta[a, s1, s2, b]: Input wavefunction
    std::vector<int64_t> extent_theta = {static_cast<int64_t>(chi_L),
                                          static_cast<int64_t>(d),
                                          static_cast<int64_t>(d),
                                          static_cast<int64_t>(chi_R)};
    std::vector<int32_t> mode_theta = {'a', 's', 't', 'b'};
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(
        *handle, &desc_theta,
        extent_theta.size(), extent_theta.data(),
        nullptr,
        HIP_R_64F,
        HIPTENSOR_OP_IDENTITY
    ));

    // T1[w, ap, s1, s2, b]: First intermediate
    std::vector<int64_t> extent_T1 = {static_cast<int64_t>(D_mpo),
                                       static_cast<int64_t>(chi_L),
                                       static_cast<int64_t>(d),
                                       static_cast<int64_t>(d),
                                       static_cast<int64_t>(chi_R)};
    std::vector<int32_t> mode_T1 = {'w', 'p', 's', 't', 'b'};
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(
        *handle, &desc_T1,
        extent_T1.size(), extent_T1.data(),
        nullptr,
        HIP_R_64F,
        HIPTENSOR_OP_IDENTITY
    ));

    // T2[ap, s1p, s2, b, x]: Second intermediate
    // Note: Contraction over w, s1 gives x, s1p
    std::vector<int64_t> extent_T2 = {static_cast<int64_t>(chi_L),
                                       static_cast<int64_t>(d),
                                       static_cast<int64_t>(d),
                                       static_cast<int64_t>(chi_R),
                                       static_cast<int64_t>(D_mpo)};
    std::vector<int32_t> mode_T2 = {'p', 'r', 't', 'b', 'x'};
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(
        *handle, &desc_T2,
        extent_T2.size(), extent_T2.data(),
        nullptr,
        HIP_R_64F,
        HIPTENSOR_OP_IDENTITY
    ));

    // T3[ap, s1p, s2p, b, y]: Third intermediate
    std::vector<int64_t> extent_T3 = {static_cast<int64_t>(chi_L),
                                       static_cast<int64_t>(d),
                                       static_cast<int64_t>(d),
                                       static_cast<int64_t>(chi_R),
                                       static_cast<int64_t>(D_mpo)};
    std::vector<int32_t> mode_T3 = {'p', 'r', 'u', 'b', 'y'};
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(
        *handle, &desc_T3,
        extent_T3.size(), extent_T3.data(),
        nullptr,
        HIP_R_64F,
        HIPTENSOR_OP_IDENTITY
    ));

    // result[a, s1p, s2p, bp]: Output wavefunction
    std::vector<int64_t> extent_result = {static_cast<int64_t>(chi_L),
                                           static_cast<int64_t>(d),
                                           static_cast<int64_t>(d),
                                           static_cast<int64_t>(chi_R)};
    std::vector<int32_t> mode_result = {'p', 'r', 'u', 'q'};
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(
        *handle, &desc_result,
        extent_result.size(), extent_result.data(),
        nullptr,
        HIP_R_64F,
        HIPTENSOR_OP_IDENTITY
    ));
}

void OptimizedHeff::create_contractions() {
    // Contraction 1: T1[w, p, s, t, b] = L[w, p, a] * theta[a, s, t, b]
    // Contracted indices: {a}
    // Free indices from L: {w, p}
    // Free indices from theta: {s, t, b}
    // Output modes: {w, p, s, t, b}

    int32_t modesL_1[] = {'w', 'p', 'a'};
    int32_t modesTheta_1[] = {'a', 's', 't', 'b'};
    int32_t modesT1_1[] = {'w', 'p', 's', 't', 'b'};

    HIPTENSOR_CHECK(hiptensorCreateContraction(
        *handle,
        &contraction_1,
        desc_L, modesL_1, HIPTENSOR_OP_IDENTITY,
        desc_theta, modesTheta_1, HIPTENSOR_OP_IDENTITY,
        desc_T1, modesT1_1, HIPTENSOR_OP_IDENTITY,
        desc_T1, modesT1_1,
        HIPTENSOR_COMPUTE_DESC_64F
    ));

    // Contraction 2: T2[p, r, t, b, x] = W1[w, s, r, x] * T1[w, p, s, t, b]
    // Contracted indices: {w, s}
    // Output modes: {p, r, t, b, x}

    int32_t modesW1_2[] = {'w', 's', 'r', 'x'};
    int32_t modesT1_2[] = {'w', 'p', 's', 't', 'b'};
    int32_t modesT2_2[] = {'p', 'r', 't', 'b', 'x'};

    HIPTENSOR_CHECK(hiptensorCreateContraction(
        *handle,
        &contraction_2,
        desc_W1, modesW1_2, HIPTENSOR_OP_IDENTITY,
        desc_T1, modesT1_2, HIPTENSOR_OP_IDENTITY,
        desc_T2, modesT2_2, HIPTENSOR_OP_IDENTITY,
        desc_T2, modesT2_2,
        HIPTENSOR_COMPUTE_DESC_64F
    ));

    // Contraction 3: T3[p, r, u, b, y] = W2[x, t, u, y] * T2[p, r, t, b, x]
    // Contracted indices: {x, t}
    // Output modes: {p, r, u, b, y}

    int32_t modesW2_3[] = {'x', 't', 'u', 'y'};
    int32_t modesT2_3[] = {'p', 'r', 't', 'b', 'x'};
    int32_t modesT3_3[] = {'p', 'r', 'u', 'b', 'y'};

    HIPTENSOR_CHECK(hiptensorCreateContraction(
        *handle,
        &contraction_3,
        desc_W2, modesW2_3, HIPTENSOR_OP_IDENTITY,
        desc_T2, modesT2_3, HIPTENSOR_OP_IDENTITY,
        desc_T3, modesT3_3, HIPTENSOR_OP_IDENTITY,
        desc_T3, modesT3_3,
        HIPTENSOR_COMPUTE_DESC_64F
    ));

    // Contraction 4: result[p, r, u, q] = T3[p, r, u, b, y] * R[y, b, q]
    // Contracted indices: {y, b}
    // Output modes: {p, r, u, q}

    int32_t modesT3_4[] = {'p', 'r', 'u', 'b', 'y'};
    int32_t modesR_4[] = {'y', 'b', 'q'};
    int32_t modesResult_4[] = {'p', 'r', 'u', 'q'};

    HIPTENSOR_CHECK(hiptensorCreateContraction(
        *handle,
        &contraction_4,
        desc_T3, modesT3_4, HIPTENSOR_OP_IDENTITY,
        desc_R, modesR_4, HIPTENSOR_OP_IDENTITY,
        desc_result, modesResult_4, HIPTENSOR_OP_IDENTITY,
        desc_result, modesResult_4,
        HIPTENSOR_COMPUTE_DESC_64F
    ));
}

void OptimizedHeff::create_plans() {
    // Plan preferences: Use greedy algorithm for contraction path optimization
    hiptensorPlanPreference_t pref;
    HIPTENSOR_CHECK(hiptensorCreatePlanPreference(
        *handle,
        &pref,
        HIPTENSOR_ALGO_DEFAULT,  // or HIPTENSOR_ALGO_ACTOR_CRITIC for optimization
        HIPTENSOR_JIT_MODE_NONE
    ));

    // Plan 1: L × theta
    uint64_t workspace_size_1 = 0;
    HIPTENSOR_CHECK(hiptensorEstimateWorkspaceSize(
        *handle,
        contraction_1,
        pref,
        HIPTENSOR_WORKSPACE_RECOMMENDED,
        &workspace_size_1
    ));

    if (workspace_size_1 > 0) {
        WorkspaceCache& cache = workspace_cache["plan_1"];
        HIP_CHECK(hipMalloc(&cache.d_workspace, workspace_size_1));
        cache.size = workspace_size_1;
        cache.is_allocated = true;
    }

    HIPTENSOR_CHECK(hiptensorCreatePlan(
        *handle,
        &plan_1,
        contraction_1,
        pref,
        workspace_size_1
    ));

    // Plan 2: W1 × T1
    uint64_t workspace_size_2 = 0;
    HIPTENSOR_CHECK(hiptensorEstimateWorkspaceSize(
        *handle,
        contraction_2,
        pref,
        HIPTENSOR_WORKSPACE_RECOMMENDED,
        &workspace_size_2
    ));

    if (workspace_size_2 > 0) {
        WorkspaceCache& cache = workspace_cache["plan_2"];
        HIP_CHECK(hipMalloc(&cache.d_workspace, workspace_size_2));
        cache.size = workspace_size_2;
        cache.is_allocated = true;
    }

    HIPTENSOR_CHECK(hiptensorCreatePlan(
        *handle,
        &plan_2,
        contraction_2,
        pref,
        workspace_size_2
    ));

    // Plan 3: W2 × T2
    uint64_t workspace_size_3 = 0;
    HIPTENSOR_CHECK(hiptensorEstimateWorkspaceSize(
        *handle,
        contraction_3,
        pref,
        HIPTENSOR_WORKSPACE_RECOMMENDED,
        &workspace_size_3
    ));

    if (workspace_size_3 > 0) {
        WorkspaceCache& cache = workspace_cache["plan_3"];
        HIP_CHECK(hipMalloc(&cache.d_workspace, workspace_size_3));
        cache.size = workspace_size_3;
        cache.is_allocated = true;
    }

    HIPTENSOR_CHECK(hiptensorCreatePlan(
        *handle,
        &plan_3,
        contraction_3,
        pref,
        workspace_size_3
    ));

    // Plan 4: T3 × R
    uint64_t workspace_size_4 = 0;
    HIPTENSOR_CHECK(hiptensorEstimateWorkspaceSize(
        *handle,
        contraction_4,
        pref,
        HIPTENSOR_WORKSPACE_RECOMMENDED,
        &workspace_size_4
    ));

    if (workspace_size_4 > 0) {
        WorkspaceCache& cache = workspace_cache["plan_4"];
        HIP_CHECK(hipMalloc(&cache.d_workspace, workspace_size_4));
        cache.size = workspace_size_4;
        cache.is_allocated = true;
    }

    HIPTENSOR_CHECK(hiptensorCreatePlan(
        *handle,
        &plan_4,
        contraction_4,
        pref,
        workspace_size_4
    ));

    hiptensorDestroyPlanPreference(pref);
}

void OptimizedHeff::allocate_intermediates() {
    // T1: D_mpo × chi_L × d × d × chi_R
    size_t size_T1 = static_cast<size_t>(D_mpo) * chi_L * d * d * chi_R * sizeof(double);
    HIP_CHECK(hipMalloc(&d_T1, size_T1));

    // T2: chi_L × d × d × chi_R × D_mpo
    size_t size_T2 = static_cast<size_t>(chi_L) * d * d * chi_R * D_mpo * sizeof(double);
    HIP_CHECK(hipMalloc(&d_T2, size_T2));

    // T3: chi_L × d × d × chi_R × D_mpo
    size_t size_T3 = static_cast<size_t>(chi_L) * d * d * chi_R * D_mpo * sizeof(double);
    HIP_CHECK(hipMalloc(&d_T3, size_T3));
}

void OptimizedHeff::apply(
    const double* d_theta,
    double* d_result,
    const double* d_L,
    const double* d_R,
    const double* d_W1,
    const double* d_W2,
    hipStream_t stream
) {
    if (!is_initialized) {
        throw std::runtime_error("OptimizedHeff not initialized");
    }

    // Scaling factors for contractions
    const double alpha = 1.0;
    const double beta = 0.0;

    // Step 1: T1 = L × theta
    void* workspace_1 = workspace_cache.count("plan_1") > 0 ?
                        workspace_cache["plan_1"].d_workspace : nullptr;
    uint64_t workspace_size_1 = workspace_cache.count("plan_1") > 0 ?
                                workspace_cache["plan_1"].size : 0;

    HIPTENSOR_CHECK(hiptensorContract(
        *handle,
        plan_1,
        (void*)&alpha, (void*)d_L, (void*)d_theta,
        (void*)&beta, (void*)d_T1, (void*)d_T1,
        workspace_1, workspace_size_1,
        stream
    ));

    // Step 2: T2 = W1 × T1
    void* workspace_2 = workspace_cache.count("plan_2") > 0 ?
                        workspace_cache["plan_2"].d_workspace : nullptr;
    uint64_t workspace_size_2 = workspace_cache.count("plan_2") > 0 ?
                                workspace_cache["plan_2"].size : 0;

    HIPTENSOR_CHECK(hiptensorContract(
        *handle,
        plan_2,
        (void*)&alpha, (void*)d_W1, (void*)d_T1,
        (void*)&beta, (void*)d_T2, (void*)d_T2,
        workspace_2, workspace_size_2,
        stream
    ));

    // Step 3: T3 = W2 × T2
    void* workspace_3 = workspace_cache.count("plan_3") > 0 ?
                        workspace_cache["plan_3"].d_workspace : nullptr;
    uint64_t workspace_size_3 = workspace_cache.count("plan_3") > 0 ?
                                workspace_cache["plan_3"].size : 0;

    HIPTENSOR_CHECK(hiptensorContract(
        *handle,
        plan_3,
        (void*)&alpha, (void*)d_W2, (void*)d_T2,
        (void*)&beta, (void*)d_T3, (void*)d_T3,
        workspace_3, workspace_size_3,
        stream
    ));

    // Step 4: result = T3 × R
    void* workspace_4 = workspace_cache.count("plan_4") > 0 ?
                        workspace_cache["plan_4"].d_workspace : nullptr;
    uint64_t workspace_size_4 = workspace_cache.count("plan_4") > 0 ?
                                workspace_cache["plan_4"].size : 0;

    HIPTENSOR_CHECK(hiptensorContract(
        *handle,
        plan_4,
        (void*)&alpha, (void*)d_T3, (void*)d_R,
        (void*)&beta, (void*)d_result, (void*)d_result,
        workspace_4, workspace_size_4,
        stream
    ));
}

size_t OptimizedHeff::get_workspace_size() const {
    size_t total = 0;
    for (const auto& [key, cache] : workspace_cache) {
        if (cache.is_allocated) {
            total += cache.size;
        }
    }
    return total;
}

size_t OptimizedHeff::get_total_memory() const {
    size_t intermediates = 0;
    if (d_T1) intermediates += D_mpo * chi_L * d * d * chi_R * sizeof(double);
    if (d_T2) intermediates += chi_L * d * d * chi_R * D_mpo * sizeof(double);
    if (d_T3) intermediates += chi_L * d * d * chi_R * D_mpo * sizeof(double);

    return get_workspace_size() + intermediates;
}
