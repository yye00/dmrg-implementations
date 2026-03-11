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

// LAPACK tridiagonal eigensolver (kept for Lanczos - negligible cost)
extern "C" void dstev_(const char* jobz, const int* n, double* d, double* e,
                       double* z, const int* ldz, double* work, int* info);

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

    // Contraction intermediates: V and U buffers
    // Size: D_mpo * d * chi_max^2 (enough for all contraction steps)
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

    // W_matrices (allocated in set_mpo)
    d_W_matrices_.resize(L, nullptr);

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
    for (auto ptr : d_W_matrices_) if (ptr) hipFree(ptr);
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
// MPS initialization (host -> device copies)
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
    int D = D_mpo_, d = d_;
    for (int i = 0; i < L_; i++) {
        int size = D * d * d * D;
        HIP_CHECK(hipMalloc(&d_mpo_tensors_[i], size * sizeof(double)));
        HIP_CHECK(hipMemcpy(d_mpo_tensors_[i], h_mpo_tensors[i],
                            size * sizeof(double), hipMemcpyHostToDevice));

        // Precompute W_matrix[w*d+s, w'*d+s'] = W[w,s,s',w']
        // W stored as W[w + s*D + s'*D*d + w'*D*d*d]
        int wm_size = D * d * d * D;  // (D*d, d*D)
        std::vector<double> h_Wm(wm_size, 0.0);
        for (int w = 0; w < D; w++)
            for (int s = 0; s < d; s++)
                for (int sp = 0; sp < d; sp++)
                    for (int wp = 0; wp < D; wp++) {
                        int row = w * d + s;
                        int col = wp * d + sp;
                        // Column-major: h_Wm[row + col * (D*d)]
                        h_Wm[row + col * D * d] =
                            h_mpo_tensors[i][w + s*D + sp*D*d + wp*D*d*d];
                    }
        HIP_CHECK(hipMalloc(&d_W_matrices_[i], wm_size * sizeof(double)));
        HIP_CHECK(hipMemcpy(d_W_matrices_[i], h_Wm.data(),
                            wm_size * sizeof(double), hipMemcpyHostToDevice));
    }
}

// ============================================================================
// GEMM-based tensor contractions
// ============================================================================
//
// All contractions factored as loops over small MPO dims (D, d) dispatching
// rocBLAS dgemm. CPU does NO arithmetic — only control flow for kernel dispatch.
//
// H_eff application: result[a',s',b'] = sum L[a,w,a'] W[w,s,s',w'] R[b,w',b'] theta[a,s,b]
//
//   Step 1: V[a',b, ws] = sum_a L_w^T[a',a] * theta_s[a,b]     (D*d GEMMs)
//   Step 2: U[a',b, ws'] = sum_ws W_matrix[ws, ws'] * V[a',b, ws]  (1 GEMM)
//   Step 3: result[a',s',b'] = sum_{w',b} U[a',b, w'*d+s'] * R_w'[b,b']  (D*d GEMMs)
//

void DMRGGPU::apply_heff(int site, const double* d_theta_in, double* d_result) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    int D = D_mpo_, d = d_;
    double one = 1.0, zero = 0.0;

    double* L_env = d_L_envs_[site];      // (cL, D, cL) col-major
    double* R_env = d_R_envs_[site + 1];  // (cR, D, cR) col-major
    double* W_mat = d_W_matrices_[site];   // (D*d, d*D) col-major
    double* V = d_T1_;  // (cL*cR, D*d) buffer
    double* U = d_T2_;  // (cL*cR, d*D) buffer

    // Step 1: V_ws[a',b] = L_w^T[a',a] * theta_s[a,b]
    // L_w at &L[w*cL], lda = cL*D. theta_s at &theta[s*cL], lda = cL*d.
    // V_ws at &V[(w*d+s)*cL*cR], lda = cL.
    for (int w = 0; w < D; w++) {
        for (int s = 0; s < d; s++) {
            int ws = w * d + s;
            ROCBLAS_CHECK(rocblas_dgemm(rocblas_h_,
                rocblas_operation_transpose, rocblas_operation_none,
                cL, cR, cL,
                &one,
                L_env + w * cL, cL * D,            // L_w^T
                d_theta_in + s * cL, cL * d,        // theta_s
                &zero,
                V + ws * cL * cR, cL));             // V_ws
        }
    }

    // Step 2: U = V * W_matrix
    // V is (cL*cR, D*d), W_matrix is (D*d, d*D), U is (cL*cR, d*D)
    ROCBLAS_CHECK(rocblas_dgemm(rocblas_h_,
        rocblas_operation_none, rocblas_operation_none,
        cL * cR, d * D, D * d,
        &one,
        V, cL * cR,
        W_mat, D * d,
        &zero,
        U, cL * cR));

    // Step 3: result_s'[a',b'] = sum_w' U_{w'd+s'}[a',b] * R_w'[b,b']
    // U_{ws'} at &U[(w'*d+s')*cL*cR], lda = cL. Shape (cL, cR).
    // R_w' at &R[w'*cR], lda = cR*D. Shape (cR, cR).
    // result_s' at &result[s'*cL], lda = cL*d. Shape (cL, cR).
    for (int wp = 0; wp < D; wp++) {
        double beta = (wp == 0) ? 0.0 : 1.0;
        for (int sp = 0; sp < d; sp++) {
            int ws_out = wp * d + sp;
            ROCBLAS_CHECK(rocblas_dgemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR,
                &one,
                U + ws_out * cL * cR, cL,          // U_{ws'}
                R_env + wp * cR, cR * D,            // R_w'
                &beta,
                d_result + sp * cL, cL * d));       // result_s'
        }
    }
}

// ============================================================================
// GPU left environment update via rocBLAS dgemm
// ============================================================================
// L_new[b, w', b'] = sum_{a,s,w,s'} L[a,w,a'] * A[a,s,b] * W[w,s,s',w'] * A[a',s',b']
//
// Step 1: V_ws[a',b] = L_w^T @ A_s            (D*d GEMMs)
// Step 2: U = V * W_matrix                     (1 GEMM)
// Step 3: L_new_w'[b,b'] = sum_s' A_s'^T @ U_ws'  (D*d GEMMs)

void DMRGGPU::update_left_env(int site) {
    int chi_in = bond_dims_[site];
    int chi_out = bond_dims_[site + 1];
    int D = D_mpo_, d = d_;
    double one = 1.0, zero = 0.0;

    ensure_L_env_alloc(site + 1, chi_out);

    double* L_env = d_L_envs_[site];           // (chi_in, D, chi_in)
    double* A = d_mps_tensors_[site];           // (chi_in, d, chi_out)
    double* W_mat = d_W_matrices_[site];        // (D*d, d*D)
    double* L_new = d_L_envs_[site + 1];       // (chi_out, D, chi_out)
    double* V = d_T1_;
    double* U = d_T2_;

    // Step 1: V_ws[a',b] = L_w^T[a',a] * A_s[a,b]
    for (int w = 0; w < D; w++) {
        for (int s = 0; s < d; s++) {
            int ws = w * d + s;
            ROCBLAS_CHECK(rocblas_dgemm(rocblas_h_,
                rocblas_operation_transpose, rocblas_operation_none,
                chi_in, chi_out, chi_in,
                &one,
                L_env + w * chi_in, chi_in * D,     // L_w^T
                A + s * chi_in, chi_in * d,          // A_s
                &zero,
                V + ws * chi_in * chi_out, chi_in)); // V_ws
        }
    }

    // Step 2: U = V * W_matrix
    ROCBLAS_CHECK(rocblas_dgemm(rocblas_h_,
        rocblas_operation_none, rocblas_operation_none,
        chi_in * chi_out, d * D, D * d,
        &one,
        V, chi_in * chi_out,
        W_mat, D * d,
        &zero,
        U, chi_in * chi_out));

    // Step 3: L_new_w'[b,b'] = sum_s' A_s'^T[b',a'] * U_ws'[a',b]
    // A_s' at &A[s'*chi_in], lda = chi_in*d. Shape (chi_in, chi_out).
    // U_ws' at &U[(w'*d+s')*chi_in*chi_out], lda = chi_in. Shape (chi_in, chi_out).
    // L_new_w' at &L_new[w'*chi_out], lda = chi_out*D. Shape (chi_out, chi_out).
    for (int wp = 0; wp < D; wp++) {
        for (int sp = 0; sp < d; sp++) {
            double beta = (sp == 0) ? 0.0 : 1.0;
            int ws_out = wp * d + sp;
            ROCBLAS_CHECK(rocblas_dgemm(rocblas_h_,
                rocblas_operation_transpose, rocblas_operation_none,
                chi_out, chi_out, chi_in,
                &one,
                A + sp * chi_in, chi_in * d,              // A_s'^T
                U + ws_out * chi_in * chi_out, chi_in,    // U_ws'
                &beta,
                L_new + wp * chi_out, chi_out * D));      // L_new_w'
        }
    }
}

// ============================================================================
// GPU right environment update via rocBLAS dgemm
// ============================================================================
// R_new[a, w, a'] = sum_{s,s',w',b,b'} A[a,s,b] * W[w,s,s',w'] * R[b,w',b'] * A[a',s',b']
//
// Step 1: V_ws[a,b'] = A_s[a,b] * R_w'[b,b']      (D*d GEMMs, ws = w'*d+s)
// Step 2: U = V * W_matrix                          (1 GEMM)
// Step 3: R_new_w[a,a'] = sum_s' U_ws'[a,b'] * A_s'^T[b',a']  (D*d GEMMs)

void DMRGGPU::update_right_env(int site) {
    int chi_in = bond_dims_[site + 1];
    int chi_out = bond_dims_[site];
    int D = D_mpo_, d = d_;
    double one = 1.0, zero = 0.0;

    ensure_R_env_alloc(site, chi_out);

    double* A = d_mps_tensors_[site];           // (chi_out, d, chi_in)
    double* R_env = d_R_envs_[site + 1];        // (chi_in, D, chi_in)
    double* W_mat = d_W_matrices_[site];         // (D*d, d*D)
    double* R_new = d_R_envs_[site];             // (chi_out, D, chi_out)
    double* V = d_T1_;
    double* U = d_T2_;

    // Step 1: V_ws[a,b'] = A_s[a,b] * R_w'[b,b']
    // ws index = w'*d + s (iterate over w' and s)
    // A_s at &A[s*chi_out], lda = chi_out*d. Shape (chi_out, chi_in).
    // R_w' at &R[w'*chi_in], lda = chi_in*D. Shape (chi_in, chi_in).
    // V_ws at &V[ws*chi_out*chi_in], lda = chi_out. Shape (chi_out, chi_in).
    for (int wp = 0; wp < D; wp++) {
        for (int s = 0; s < d; s++) {
            int ws = wp * d + s;
            ROCBLAS_CHECK(rocblas_dgemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                chi_out, chi_in, chi_in,
                &one,
                A + s * chi_out, chi_out * d,               // A_s
                R_env + wp * chi_in, chi_in * D,             // R_w'
                &zero,
                V + ws * chi_out * chi_in, chi_out));        // V_ws
        }
    }

    // Step 2: U = V * W_matrix
    // V is (chi_out*chi_in, D*d), W_matrix is (D*d, d*D), U is (chi_out*chi_in, d*D)
    ROCBLAS_CHECK(rocblas_dgemm(rocblas_h_,
        rocblas_operation_none, rocblas_operation_none,
        chi_out * chi_in, d * D, D * d,
        &one,
        V, chi_out * chi_in,
        W_mat, D * d,
        &zero,
        U, chi_out * chi_in));

    // Step 3: R_new_w[a,a'] = sum_s' U_ws'[a,b'] * A_s'[a',b']^T
    // ws' = w*d+s'. U_ws' at &U[(w*d+s')*chi_out*chi_in], lda=chi_out. (chi_out, chi_in).
    // A_s' at &A[s'*chi_out], lda = chi_out*d. Shape (chi_out, chi_in).
    // R_new_w at &R_new[w*chi_out], lda = chi_out*D. Shape (chi_out, chi_out).
    for (int w = 0; w < D; w++) {
        for (int sp = 0; sp < d; sp++) {
            double beta = (sp == 0) ? 0.0 : 1.0;
            int ws_out = w * d + sp;
            ROCBLAS_CHECK(rocblas_dgemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_transpose,
                chi_out, chi_out, chi_in,
                &one,
                U + ws_out * chi_out * chi_in, chi_out,     // U_ws'
                A + sp * chi_out, chi_out * d,               // A_s'^T
                &beta,
                R_new + w * chi_out, chi_out * D));          // R_new_w
        }
    }
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

    // Build all R environments from right to left
    for (int i = L_ - 1; i >= 0; i--) {
        update_right_env(i);
    }
}

// ============================================================================
// Theta formation and Lanczos
// ============================================================================

void DMRGGPU::form_theta(int site, double* d_theta) {
    int size = chi_L(site) * d_ * chi_R(site);
    HIP_CHECK(hipMemcpy(d_theta, d_mps_tensors_[site],
                        size * sizeof(double), hipMemcpyDeviceToDevice));
}

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

        // w = H|v_i> (GPU GEMM-based contraction)
        apply_heff(site, d_vi, d_heff_result_);

        // alpha_i = <v_i|w>
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

    // Reconstruct ground state: |theta> = sum_i c[i] |v_i>
    double* d_ritz_coeffs;
    HIP_CHECK(hipMalloc(&d_ritz_coeffs, niter * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_ritz_coeffs, h_Z.data(), niter * sizeof(double), hipMemcpyHostToDevice));

    const double one = 1.0, zeroval = 0.0;
    ROCBLAS_CHECK(rocblas_dgemv(
        rocblas_h_, rocblas_operation_none,
        n, niter, &one,
        d_lanczos_v, n,
        d_ritz_coeffs, 1,
        &zeroval, d_theta, 1
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

        int new_chi_R = new_k;

        // A[site] = U[:, :new_k]
        allocate_mps_tensor(site, cL, new_chi_R);
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], d_svd_U_,
                            m * new_chi_R * sizeof(double), hipMemcpyDeviceToDevice));

        // S*Vh on GPU
        ROCBLAS_CHECK(rocblas_ddgmm(rocblas_h_, rocblas_side_left,
            new_k, n_svd,
            d_svd_Vh_, full_k,
            d_svd_S_, 1,
            d_svd_work_, new_k));

        // Absorb SV into A[site+1]
        if (site + 1 < L_) {
            int next_cR = chi_R(site + 1);
            double one = 1.0, zero = 0.0;

            ROCBLAS_CHECK(rocblas_dgemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                new_k, d_ * next_cR, cR,
                &one,
                d_svd_work_, new_k,
                d_mps_tensors_[site + 1], cR,
                &zero,
                d_T1_, new_k));

            allocate_mps_tensor(site + 1, new_chi_R, next_cR);
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site + 1], d_T1_,
                                new_k * d_ * next_cR * sizeof(double),
                                hipMemcpyDeviceToDevice));
        }

        bond_dims_[site + 1] = new_chi_R;

    } else {  // direction == 'L'
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

        // A[site] = Vh[:new_k, :]
        allocate_mps_tensor(site, new_chi_L, cR);
        if (new_chi_L == full_k) {
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site], d_svd_Vh_,
                                full_k * n_svd * sizeof(double), hipMemcpyDeviceToDevice));
        } else {
            HIP_CHECK(hipMemcpy2D(
                d_mps_tensors_[site], new_chi_L * sizeof(double),
                d_svd_Vh_,            full_k * sizeof(double),
                new_chi_L * sizeof(double),
                n_svd,
                hipMemcpyDeviceToDevice));
        }

        // U*S on GPU
        ROCBLAS_CHECK(rocblas_ddgmm(rocblas_h_, rocblas_side_right,
            m, new_k,
            d_svd_U_, m,
            d_svd_S_, 1,
            d_svd_work_, m));

        // Absorb US into A[site-1]
        if (site > 0) {
            int prev_cL = chi_L(site - 1);
            double one = 1.0, zero = 0.0;

            ROCBLAS_CHECK(rocblas_dgemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                prev_cL * d_, new_k, m,
                &one,
                d_mps_tensors_[site - 1], prev_cL * d_,
                d_svd_work_, m,
                &zero,
                d_T1_, prev_cL * d_));

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
    printf("=== GPU-Native DMRG (rocBLAS GEMM) ===\n");
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
