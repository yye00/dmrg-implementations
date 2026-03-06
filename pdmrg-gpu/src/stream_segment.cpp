#include "stream_segment.h"
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <hiptensor/hiptensor.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <fstream>
#include <complex>
#include <vector>
#include <algorithm>
#include <random>
#include <vector>

// Error checking macros
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error in %s:%d - %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err)); \
            throw std::runtime_error("HIP error"); \
        } \
    } while(0)

#define HIPBLAS_CHECK(call) \
    do { \
        hipblasStatus_t status = call; \
        if (status != HIPBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "hipBLAS error in %s:%d - status %d\n", \
                    __FILE__, __LINE__, status); \
            throw std::runtime_error("hipBLAS error"); \
        } \
    } while(0)

#define ROCBLAS_CHECK(call) \
    do { \
        rocblas_status status = call; \
        if (status != rocblas_status_success) { \
            fprintf(stderr, "rocBLAS error in %s:%d - status %d\n", \
                    __FILE__, __LINE__, status); \
            throw std::runtime_error("rocBLAS error"); \
        } \
    } while(0)

#define ROCSOLVER_CHECK(call) \
    do { \
        rocblas_status status = call; \
        if (status != rocblas_status_success) { \
            fprintf(stderr, "rocSOLVER error in %s:%d - status %d\n", \
                    __FILE__, __LINE__, status); \
            throw std::runtime_error("rocSOLVER error"); \
        } \
    } while(0)

#define HIPTENSOR_CHECK(call) \
    do { \
        hiptensorStatus_t status = call; \
        if (status != HIPTENSOR_STATUS_SUCCESS) { \
            fprintf(stderr, "hipTensor error in %s:%d - status %d\n", \
                    __FILE__, __LINE__, status); \
            throw std::runtime_error("hipTensor error"); \
        } \
    } while(0)

//==============================================================================
// BoundaryData Implementation
//==============================================================================

BoundaryData::BoundaryData()
    : d_psi_left(nullptr), d_psi_right(nullptr), d_V(nullptr),
      d_L_env(nullptr), d_R_env(nullptr), d_W_left(nullptr), d_W_right(nullptr),
      chi_L(0), chi_R(0), chi_bond(0), d(0), D_mpo(0),
      is_allocated(false)
{
}

BoundaryData::~BoundaryData() {
    free();
}

void BoundaryData::allocate(int chi_L_in, int chi_R_in, int chi_bond_in,
                             int d_in, int D_mpo_in) {
    if (is_allocated) {
        free();
    }

    chi_L = chi_L_in;
    chi_R = chi_R_in;
    chi_bond = chi_bond_in;
    d = d_in;
    D_mpo = D_mpo_in;

    // Allocate MPS tensors at boundary
    HIP_CHECK(hipMalloc(&d_psi_left, chi_L * d * chi_bond * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_psi_right, chi_bond * d * chi_R * sizeof(double)));

    // Allocate V = Lambda^-1
    HIP_CHECK(hipMalloc(&d_V, chi_bond * sizeof(double)));

    // Allocate environments
    // Convention: L_env[D_mpo, chi_L, chi_L], R_env[D_mpo, chi_R, chi_R]
    HIP_CHECK(hipMalloc(&d_L_env, D_mpo * chi_L * chi_L * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_R_env, D_mpo * chi_R * chi_R * sizeof(double)));

    // Allocate MPO tensors
    HIP_CHECK(hipMalloc(&d_W_left, D_mpo * d * d * D_mpo * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_W_right, D_mpo * d * d * D_mpo * sizeof(double)));

    is_allocated = true;
}

void BoundaryData::free() {
    if (!is_allocated) return;

    if (d_psi_left) HIP_CHECK(hipFree(d_psi_left));
    if (d_psi_right) HIP_CHECK(hipFree(d_psi_right));
    if (d_V) HIP_CHECK(hipFree(d_V));
    if (d_L_env) HIP_CHECK(hipFree(d_L_env));
    if (d_R_env) HIP_CHECK(hipFree(d_R_env));
    if (d_W_left) HIP_CHECK(hipFree(d_W_left));
    if (d_W_right) HIP_CHECK(hipFree(d_W_right));

    d_psi_left = nullptr;
    d_psi_right = nullptr;
    d_V = nullptr;
    d_L_env = nullptr;
    d_R_env = nullptr;
    d_W_left = nullptr;
    d_W_right = nullptr;

    is_allocated = false;
}

//==============================================================================
// StreamSegment Implementation
//==============================================================================

StreamSegment::StreamSegment(int segment_id, int start_site, int end_site,
                              int chi_max, int d, int D_mpo, hipStream_t stream, int total_chain_length)
    : id_(segment_id), start_site_(start_site), end_site_(end_site),
      num_sites_(end_site - start_site + 1),
      chain_length_(total_chain_length),
      chi_max_(chi_max), d_(d), D_mpo_(D_mpo), stream_(stream),
      d_mps_tensors_(nullptr), mps_chi_left_(nullptr), mps_chi_right_(nullptr),
      d_L_envs_(nullptr), d_R_envs_(nullptr), d_mpo_tensors_(nullptr),
      has_left_boundary_(segment_id > 0),
      has_right_boundary_(true),  // Will be set correctly by coordinator
      heff_(nullptr), svd_(nullptr), rocblas_h_(nullptr),
      d_workspace_(nullptr), workspace_size_(0),
      d_tau_(nullptr), tau_size_(0)
{
    if (num_sites_ < 1) {
        throw std::runtime_error("StreamSegment must have at least 1 site");
    }

    // Create rocBLAS handle for QR/LQ operations
    ROCBLAS_CHECK(rocblas_create_handle(&rocblas_h_));
    ROCBLAS_CHECK(rocblas_set_stream(rocblas_h_, stream_));

    // Create hipTensor handle for environment contractions
    HIPTENSOR_CHECK(hiptensorCreate(&hiptensor_h_));

    // Allocate memory for all tensors
    allocate_memory();

    // Initialize MPS and environments
    initialize_mps();
    initialize_environments();

    // Allocate tau workspace for QR/LQ (max size needed)
    tau_size_ = chi_max_;
    HIP_CHECK(hipMalloc(&d_tau_, tau_size_ * sizeof(double)));

    // Create OptimizedHeff for local two-site operations
    // Note: Will be created lazily when needed, since dimensions may vary

    // Create AccurateSVD for exact SVD operations
    svd_ = new AccurateSVD_GPU(1e-4, 0);  // epsilon=1e-4, max_depth=0 (no recursion)
}

StreamSegment::~StreamSegment() {
    free_memory();

    if (heff_) delete heff_;
    if (svd_) delete svd_;

    if (d_tau_) HIP_CHECK(hipFree(d_tau_));
    if (rocblas_h_) ROCBLAS_CHECK(rocblas_destroy_handle(rocblas_h_));
    if (hiptensor_h_) HIPTENSOR_CHECK(hiptensorDestroy(hiptensor_h_));
}

void StreamSegment::allocate_memory() {
    // Allocate arrays for pointers
    d_mps_tensors_ = new double*[num_sites_];
    mps_chi_left_ = new int[num_sites_];
    mps_chi_right_ = new int[num_sites_];

    d_L_envs_ = new double*[num_sites_ + 1];
    d_R_envs_ = new double*[num_sites_ + 1];
    d_mpo_tensors_ = new double*[num_sites_];

    // Initialize chi values to match CPU PDMRG exactly:
    // - chi_max for all internal bonds
    // - 1 only at chain edges (global site 0 and L-1)
    for (int i = 0; i < num_sites_; i++) {
        int global_site = start_site_ + i;

        // Match CPU: chi_L = 1 if gi == 0 else chi
        int chi_left = (global_site == 0) ? 1 : chi_max_;

        // Match CPU: chi_R = 1 if gi == L-1 else chi
        // Note: end_site_ is the last site index in this segment (inclusive)
        int chain_length = chain_length_;  // FIXED: Use total chain length
        int chi_right = (global_site == chain_length - 1) ? 1 : chi_max_;

        mps_chi_left_[i] = chi_left;
        mps_chi_right_[i] = chi_right;

        // Allocate MPS tensor: (chi_left, d, chi_right) in column-major
        HIP_CHECK(hipMalloc(&d_mps_tensors_[i], chi_left * d_ * chi_right * sizeof(double)));
    }

    // Initialize MPS tensors with random values matching CPU PDMRG
    // CPU uses: np.random.seed(42 + rank); np.random.randn(chi_L, d, chi_R)
    std::mt19937 rng(42 + id_);  // Match CPU seed (rank → segment_id)
    std::normal_distribution<double> dist(0.0, 1.0);  // Standard normal (mean=0, std=1)

    for (int i = 0; i < num_sites_; i++) {
        int chi_left = mps_chi_left_[i];
        int chi_right = mps_chi_right_[i];
        int tensor_size = chi_left * d_ * chi_right;

        // Generate random values on host
        std::vector<double> h_tensor(tensor_size);
        for (int j = 0; j < tensor_size; j++) {
            h_tensor[j] = dist(rng);
        }

        // Copy to device
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_tensor.data(),
                            tensor_size * sizeof(double), hipMemcpyHostToDevice));
    }

    // Allocate environments
    // L_envs[0] is left of start_site, L_envs[num_sites] is right of end_site
    for (int i = 0; i <= num_sites_; i++) {
        int chi_L = (i == 0) ? 1 : mps_chi_right_[i - 1];
        // L_env: (D_mpo, chi_L, chi_L) in column-major
        HIP_CHECK(hipMalloc(&d_L_envs_[i], D_mpo_ * chi_L * chi_L * sizeof(double)));

        int chi_R = (i == num_sites_) ? 1 : mps_chi_left_[i];
        // R_env: (D_mpo, chi_R, chi_R) in column-major
        HIP_CHECK(hipMalloc(&d_R_envs_[i], D_mpo_ * chi_R * chi_R * sizeof(double)));
    }

    // Allocate MPO tensors (will be set by coordinator)
    // Handle variable bond dimensions at boundaries:
    //   Global site 0 (chain start): D_left = 1, D_right = D_mpo
    //   Global site L-1 (chain end): D_left = D_mpo, D_right = 1
    //   Bulk sites: D_left = D_mpo, D_right = D_mpo
    for (int i = 0; i < num_sites_; i++) {
        int global_site = start_site_ + i;
        // Assuming chain_length is known via some parameter
        // For now, use conservative allocation (max size)
        // TODO: Pass chain_length to StreamSegment constructor to compute exact sizes
        int D_left = D_mpo_;   // Conservative: assume bulk
        int D_right = D_mpo_;  // Conservative: assume bulk
        size_t mpo_size = D_left * d_ * d_ * D_right;
        HIP_CHECK(hipMalloc(&d_mpo_tensors_[i], mpo_size * sizeof(double)));
    }

    // Allocate boundary data if needed
    if (has_left_boundary_) {
        int chi_bond = mps_chi_left_[0];
        left_boundary_.allocate(chi_bond, chi_bond, chi_bond, d_, D_mpo_);
    }

    if (has_right_boundary_) {
        int chi_bond = mps_chi_right_[num_sites_ - 1];
        right_boundary_.allocate(chi_bond, chi_bond, chi_bond, d_, D_mpo_);
    }
}

void StreamSegment::free_memory() {
    // Free MPS tensors
    if (d_mps_tensors_) {
        for (int i = 0; i < num_sites_; i++) {
            if (d_mps_tensors_[i]) HIP_CHECK(hipFree(d_mps_tensors_[i]));
        }
        delete[] d_mps_tensors_;
        d_mps_tensors_ = nullptr;
    }

    if (mps_chi_left_) delete[] mps_chi_left_;
    if (mps_chi_right_) delete[] mps_chi_right_;

    // Free environments
    if (d_L_envs_) {
        for (int i = 0; i <= num_sites_; i++) {
            if (d_L_envs_[i]) HIP_CHECK(hipFree(d_L_envs_[i]));
        }
        delete[] d_L_envs_;
        d_L_envs_ = nullptr;
    }

    if (d_R_envs_) {
        for (int i = 0; i <= num_sites_; i++) {
            if (d_R_envs_[i]) HIP_CHECK(hipFree(d_R_envs_[i]));
        }
        delete[] d_R_envs_;
        d_R_envs_ = nullptr;
    }

    // Free MPO tensors
    if (d_mpo_tensors_) {
        for (int i = 0; i < num_sites_; i++) {
            if (d_mpo_tensors_[i]) HIP_CHECK(hipFree(d_mpo_tensors_[i]));
        }
        delete[] d_mpo_tensors_;
        d_mpo_tensors_ = nullptr;
    }

    // Free workspace
    if (d_workspace_) {
        HIP_CHECK(hipFree(d_workspace_));
        d_workspace_ = nullptr;
    }
}

void StreamSegment::initialize_mps() {
    // Initialize MPS tensors with random values
    // TODO: Replace with proper initialization from initial state
    for (int i = 0; i < num_sites_; i++) {
        int chi_left = mps_chi_left_[i];
        int chi_right = mps_chi_right_[i];
        size_t size = chi_left * d_ * chi_right;

        std::vector<double> h_tensor(size);
        for (size_t j = 0; j < size; j++) {
            h_tensor[j] = ((double)rand() / RAND_MAX) * 0.1;  // Small random values
        }

        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_tensor.data(),
                           size * sizeof(double), hipMemcpyHostToDevice));
    }
}

void StreamSegment::initialize_environments() {
    // Initialize environments to identity
    for (int i = 0; i <= num_sites_; i++) {
        int chi_L = (i == 0) ? 1 : mps_chi_right_[i - 1];
        size_t size_L = D_mpo_ * chi_L * chi_L;

        std::vector<double> h_L_env(size_L, 0.0);
        // Set identity: L_env[w, a, a] = delta_{w,0}
        for (int a = 0; a < chi_L; a++) {
            h_L_env[0 + a * D_mpo_ + a * D_mpo_ * chi_L] = 1.0;
        }
        HIP_CHECK(hipMemcpy(d_L_envs_[i], h_L_env.data(),
                           size_L * sizeof(double), hipMemcpyHostToDevice));

        int chi_R = (i == num_sites_) ? 1 : mps_chi_left_[i];
        size_t size_R = D_mpo_ * chi_R * chi_R;

        std::vector<double> h_R_env(size_R, 0.0);
        // Set identity: R_env[w, b, b] = delta_{w,0}
        for (int b = 0; b < chi_R; b++) {
            h_R_env[0 + b * D_mpo_ + b * D_mpo_ * chi_R] = 1.0;
        }
        HIP_CHECK(hipMemcpy(d_R_envs_[i], h_R_env.data(),
                           size_R * sizeof(double), hipMemcpyHostToDevice));
    }
}

//==============================================================================
// Sweep Operations
//==============================================================================

void StreamSegment::sweep_left_to_right() {
    // QR sweep to move orthogonality center left-to-right
    // Each site is left-canonized (QR decomposition), R absorbed to right
    //
    // For site i with tensor psi[chi_L, d, chi_R]:
    //   1. Reshape to matrix M[chi_L*d, chi_R]
    //   2. Compute QR: M = Q*R
    //   3. Store Q reshaped as psi[chi_L, d, k] at site i
    //   4. Contract R into next site: psi[i+1] = R * psi[i+1]

    for (int i = 0; i < num_sites_ - 1; i++) {
        int chi_L = mps_chi_left_[i];
        int chi_R = mps_chi_right_[i];
        int m = chi_L * d_;     // Rows of reshaped matrix
        int n = chi_R;          // Columns of reshaped matrix
        int k = std::min(m, n); // Rank of Q and R

        double* d_psi = d_mps_tensors_[i];

        // Allocate temporary workspace for QR
        double* d_M_qr;
        HIP_CHECK(hipMalloc(&d_M_qr, m * n * sizeof(double)));

        // Copy MPS tensor to workspace (rocsolver modifies in-place)
        HIP_CHECK(hipMemcpy(d_M_qr, d_psi, m * n * sizeof(double), hipMemcpyDeviceToDevice));

        // Compute QR factorization: M = Q*R
        // After this call: R is in upper triangle of d_M_qr, Q is implicit (stored with tau)
        ROCSOLVER_CHECK(rocsolver_dgeqrf(
            rocblas_h_,
            m, n,           // Matrix dimensions
            d_M_qr, m,      // Matrix (column-major), leading dimension
            d_tau_          // Output: tau vector for implicit Q
        ));

        // Extract R from upper triangle (k x n matrix)
        double* d_R;
        HIP_CHECK(hipMalloc(&d_R, k * n * sizeof(double)));

        // Copy R: upper triangle of d_M_qr to d_R
        // For column-major storage, R[i,j] = d_M_qr[i + j*m] for i <= j
        std::vector<double> h_M_qr(m * n);
        HIP_CHECK(hipMemcpy(h_M_qr.data(), d_M_qr, m * n * sizeof(double), hipMemcpyDeviceToHost));
        std::vector<double> h_R(k * n, 0.0);
        for (int col = 0; col < n; col++) {
            for (int row = 0; row < std::min(k, col + 1); row++) {
                h_R[row + col * k] = h_M_qr[row + col * m];
            }
        }
        HIP_CHECK(hipMemcpy(d_R, h_R.data(), k * n * sizeof(double), hipMemcpyHostToDevice));

        // Generate explicit Q matrix (m x k)
        // Note: dorgqr overwrites d_M_qr with Q
        ROCSOLVER_CHECK(rocsolver_dorgqr(
            rocblas_h_,
            m, k, k,        // m, n, k for Q generation
            d_M_qr, m,      // Matrix to overwrite with Q
            d_tau_          // tau from QR factorization
        ));

        // Update site i with Q (reshaped to chi_L x d x k)
        // Q has shape (chi_L*d, k) = (m, k)
        // Need to store first m*k elements
        if (k != chi_R) {
            // Bond dimension changed - reallocate
            HIP_CHECK(hipFree(d_psi));
            HIP_CHECK(hipMalloc(&d_psi, chi_L * d_ * k * sizeof(double)));
            d_mps_tensors_[i] = d_psi;
            mps_chi_right_[i] = k;
        }
        HIP_CHECK(hipMemcpy(d_psi, d_M_qr, chi_L * d_ * k * sizeof(double), hipMemcpyDeviceToDevice));

        // Contract R into next site: psi[i+1] = R * psi[i+1]
        // R is (k, chi_R), psi[i+1] is (chi_R, d, chi_R_next)
        // Result is (k, d, chi_R_next)
        int chi_R_next = mps_chi_right_[i + 1];
        double* d_psi_next = d_mps_tensors_[i + 1];
        double* d_psi_next_new;
        HIP_CHECK(hipMalloc(&d_psi_next_new, k * d_ * chi_R_next * sizeof(double)));

        // Contract: C[k, d*chi_R_next] = R[k, chi_R] * psi_next[chi_R, d*chi_R_next]
        double alpha = 1.0, beta = 0.0;
        ROCBLAS_CHECK(rocblas_dgemm(
            rocblas_h_,
            rocblas_operation_none,
            rocblas_operation_none,
            k, d_ * chi_R_next, chi_R,  // M, N, K
            &alpha,
            d_R, k,                     // A (k x chi_R)
            d_psi_next, chi_R,          // B (chi_R x d*chi_R_next)
            &beta,
            d_psi_next_new, k           // C (k x d*chi_R_next)
        ));

        // Update next site
        HIP_CHECK(hipFree(d_psi_next));
        d_mps_tensors_[i + 1] = d_psi_next_new;
        mps_chi_left_[i + 1] = k;

        // Cleanup
        HIP_CHECK(hipFree(d_M_qr));
        HIP_CHECK(hipFree(d_R));
    }

    // Synchronize stream after sweep
    HIP_CHECK(hipStreamSynchronize(stream_));
}

void StreamSegment::sweep_right_to_left() {
    // LQ sweep to move orthogonality center right-to-left
    // Each site is right-canonized (LQ decomposition), L absorbed to left
    //
    // For site i with tensor psi[chi_L, d, chi_R]:
    //   1. Reshape to matrix M[chi_L, d*chi_R]
    //   2. Compute LQ: M = L*Q
    //   3. Store Q reshaped as psi[k, d, chi_R] at site i
    //   4. Contract L into previous site: psi[i-1] = psi[i-1] * L

    for (int i = num_sites_ - 1; i > 0; i--) {
        int chi_L = mps_chi_left_[i];
        int chi_R = mps_chi_right_[i];
        int m = chi_L;          // Rows of reshaped matrix
        int n = d_ * chi_R;     // Columns of reshaped matrix
        int k = std::min(m, n); // Rank of L and Q

        double* d_psi = d_mps_tensors_[i];

        // Allocate temporary workspace for LQ
        double* d_M_lq;
        HIP_CHECK(hipMalloc(&d_M_lq, m * n * sizeof(double)));

        // Copy MPS tensor to workspace (rocsolver modifies in-place)
        HIP_CHECK(hipMemcpy(d_M_lq, d_psi, m * n * sizeof(double), hipMemcpyDeviceToDevice));

        // Compute LQ factorization: M = L*Q
        // After this call: L is in lower triangle of d_M_lq, Q is implicit (stored with tau)
        ROCSOLVER_CHECK(rocsolver_dgelqf(
            rocblas_h_,
            m, n,           // Matrix dimensions
            d_M_lq, m,      // Matrix (column-major), leading dimension
            d_tau_          // Output: tau vector for implicit Q
        ));

        // Extract L from lower triangle (m x k matrix)
        double* d_L;
        HIP_CHECK(hipMalloc(&d_L, m * k * sizeof(double)));

        // Copy L: lower triangle of d_M_lq to d_L
        // For column-major storage, L[i,j] = d_M_lq[i + j*m] for i >= j
        std::vector<double> h_M_lq(m * n);
        HIP_CHECK(hipMemcpy(h_M_lq.data(), d_M_lq, m * n * sizeof(double), hipMemcpyDeviceToHost));
        std::vector<double> h_L(m * k, 0.0);
        for (int col = 0; col < k; col++) {
            for (int row = col; row < m; row++) {
                h_L[row + col * m] = h_M_lq[row + col * m];
            }
        }
        HIP_CHECK(hipMemcpy(d_L, h_L.data(), m * k * sizeof(double), hipMemcpyHostToDevice));

        // Generate explicit Q matrix (k x n)
        // Note: dorglq overwrites d_M_lq with Q
        ROCSOLVER_CHECK(rocsolver_dorglq(
            rocblas_h_,
            k, n, k,        // m, n, k for Q generation
            d_M_lq, m,      // Matrix to overwrite with Q (note: lda is still m)
            d_tau_          // tau from LQ factorization
        ));

        // Update site i with Q (reshaped to k x d x chi_R)
        // Q has shape (k, d*chi_R) = (k, n)
        // Need to extract first k rows and n columns
        if (k != chi_L) {
            // Bond dimension changed - reallocate
            HIP_CHECK(hipFree(d_psi));
            HIP_CHECK(hipMalloc(&d_psi, k * d_ * chi_R * sizeof(double)));
            d_mps_tensors_[i] = d_psi;
            mps_chi_left_[i] = k;
        }

        // Copy Q (first k rows of d_M_lq, which has leading dimension m)
        // Need to copy column by column
        std::vector<double> h_Q(k * n);
        HIP_CHECK(hipMemcpy(h_M_lq.data(), d_M_lq, m * n * sizeof(double), hipMemcpyDeviceToHost));
        for (int col = 0; col < n; col++) {
            for (int row = 0; row < k; row++) {
                h_Q[row + col * k] = h_M_lq[row + col * m];
            }
        }
        HIP_CHECK(hipMemcpy(d_psi, h_Q.data(), k * d_ * chi_R * sizeof(double), hipMemcpyHostToDevice));

        // Contract L into previous site: psi[i-1] = psi[i-1] * L
        // psi[i-1] is (chi_L_prev, d, chi_L), L is (chi_L, k)
        // Result is (chi_L_prev, d, k)
        int chi_L_prev = mps_chi_left_[i - 1];
        double* d_psi_prev = d_mps_tensors_[i - 1];
        double* d_psi_prev_new;
        HIP_CHECK(hipMalloc(&d_psi_prev_new, chi_L_prev * d_ * k * sizeof(double)));

        // Contract: C[chi_L_prev*d, k] = psi_prev[chi_L_prev*d, chi_L] * L[chi_L, k]
        double alpha = 1.0, beta = 0.0;
        ROCBLAS_CHECK(rocblas_dgemm(
            rocblas_h_,
            rocblas_operation_none,
            rocblas_operation_none,
            chi_L_prev * d_, k, chi_L,  // M, N, K
            &alpha,
            d_psi_prev, chi_L_prev * d_, // A (chi_L_prev*d x chi_L)
            d_L, chi_L,                  // B (chi_L x k)
            &beta,
            d_psi_prev_new, chi_L_prev * d_  // C (chi_L_prev*d x k)
        ));

        // Update previous site
        HIP_CHECK(hipFree(d_psi_prev));
        d_mps_tensors_[i - 1] = d_psi_prev_new;
        mps_chi_right_[i - 1] = k;

        // Cleanup
        HIP_CHECK(hipFree(d_M_lq));
        HIP_CHECK(hipFree(d_L));
    }

    // Synchronize stream after sweep
    HIP_CHECK(hipStreamSynchronize(stream_));
}

//==============================================================================
// Boundary Access
//==============================================================================

BoundaryData* StreamSegment::get_left_boundary() {
    return has_left_boundary_ ? &left_boundary_ : nullptr;
}

BoundaryData* StreamSegment::get_right_boundary() {
    return has_right_boundary_ ? &right_boundary_ : nullptr;
}

//==============================================================================
// Environment Rebuilding
//==============================================================================

void StreamSegment::rebuild_right_boundary_env() {
    if (!has_right_boundary_) return;

    // Rebuild L_env at the right boundary (position num_sites_)
    // by contracting MPS and MPO from left to right
    //
    // L_new[b, wp, b'] = sum_{a, a', w, s, s'}
    //     L[a, w, a'] * A[a, s, b] * W[w, s, s', wp] * A[a', s', b']
    //
    // For real tensors (no complex conjugate needed), decompose into steps:
    // 1. temp1[w, a', s, b] = L[a, w, a'] * A[a, s, b]
    // 2. temp2[a', s', b, wp] = temp1[w, a', s, b] * W[w, s, s', wp]
    // 3. L_new[b, wp, b'] = temp2[a', s', b, wp] * A[a', s', b']

    // Copy current L_env[0] (identity) as starting point
    // Then contract through all sites to build L_env[num_sites_]

    // For now, we only contract the last site to build the right boundary env
    // Full segment contraction can be added later if needed

    int site_idx = num_sites_ - 1;  // Rightmost site
    int chi_L = mps_chi_left_[site_idx];
    int chi_R = mps_chi_right_[site_idx];

    double* d_A = d_mps_tensors_[site_idx];
    double* d_W = d_mpo_tensors_[site_idx];
    double* d_L_in = d_L_envs_[site_idx];   // L_env to the left of this site
    double* d_L_out = d_L_envs_[num_sites_]; // L_env at right boundary

    // Allocate workspace for intermediate tensors
    // temp1: (D_mpo, chi_L, d, chi_R)
    // temp2: (chi_L, d, chi_R, D_mpo)
    // L_out: (D_mpo, chi_R, chi_R)

    size_t temp_size = D_mpo_ * chi_L * d_ * chi_R;
    double* d_temp1;
    double* d_temp2;
    HIP_CHECK(hipMalloc(&d_temp1, temp_size * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_temp2, temp_size * sizeof(double)));

    // Full hipTensor implementation for environment contraction
    // L_new[b, wp, b'] = sum_{a, a', w, s, s'}
    //     L[a, w, a'] * A[a, s, b] * W[w, s, s', wp] * A[a', s', b']
    //
    // Decomposed into 3 sequential contractions:
    // 1. temp1[w, a', s, b] = L[a, w, a'] * A[a, s, b]  (contract over a)
    // 2. temp2[a', s', b, wp] = temp1[w, a', s, b] * W[w, s, s', wp]  (contract over w, s)
    // 3. L_new[b, wp, b'] = temp2[a', s', b, wp] * A[a', s', b']  (contract over a', s')

    // Step 1: temp1[w, a', s, b] = sum_a L[a, w, a'] * A[a, s, b]
    // Mode labels: a=0, w=1, astar=2, s=3, b=4
    // L col-major (a, w, a'):        modes {0, 1, 2}
    // A col-major (a, s, b):         modes {0, 3, 4}
    // temp1 col-major (w, a', s, b): modes {1, 2, 3, 4}
    // Contract over mode 0 (a)
    {
        int64_t extentL[] = {(int64_t)chi_L, (int64_t)D_mpo_, (int64_t)chi_L};
        int64_t extentA[] = {(int64_t)chi_L, (int64_t)d_, (int64_t)chi_R};
        int64_t extent_temp1[] = {(int64_t)D_mpo_, (int64_t)chi_L, (int64_t)d_, (int64_t)chi_R};

        int32_t modesL[] = {0, 1, 2};
        int32_t modesA[] = {0, 3, 4};
        int32_t modes_temp1[] = {1, 2, 3, 4};

        hiptensor_contract(d_L_in, 3, extentL, modesL,
                           d_A, 3, extentA, modesA,
                           d_temp1, 4, extent_temp1, modes_temp1,
                           1.0, 0.0);
    }

    // Step 2: temp2[a', s', b, wp] = sum_{w,s} temp1[w, a', s, b] * W[w, s, s', wp]
    // temp1[w, a', s, b]: modes {1, 2, 3, 4}
    // W[w, s, s', wp]: modes {1, 3, 5, 6}
    // temp2[a', s', b, wp]: modes {2, 5, 4, 6}
    // Contract over modes 1(w) and 3(s)
    {
        int64_t extent_temp1[] = {(int64_t)D_mpo_, (int64_t)chi_L, (int64_t)d_, (int64_t)chi_R};
        int64_t extentW[] = {(int64_t)D_mpo_, (int64_t)d_, (int64_t)d_, (int64_t)D_mpo_};
        int64_t extent_temp2[] = {(int64_t)chi_L, (int64_t)d_, (int64_t)chi_R, (int64_t)D_mpo_};

        int32_t modes_temp1[] = {0, 1, 2, 3};
        int32_t modesW[] = {0, 2, 4, 5};
        int32_t modes_temp2[] = {1, 4, 3, 5};

        hiptensor_contract(d_temp1, 4, extent_temp1, modes_temp1,
                           d_W, 4, extentW, modesW,
                           d_temp2, 4, extent_temp2, modes_temp2,
                           1.0, 0.0);
    }

    HIP_CHECK(hipFree(d_temp1));

    // Step 3: L_new[b, wp, b'] = sum_{a',s'} temp2[a', s', b, wp] * A[a', s', b']
    // temp2[a', s', b, wp]: modes {2, 5, 4, 6}
    // A[a', s', b']: modes {2, 5, 7}
    // L_new[b, wp, b']: modes {4, 6, 7}
    // Contract over modes 2(a') and 5(s')
    {
        int64_t extent_temp2[] = {(int64_t)chi_L, (int64_t)d_, (int64_t)chi_R, (int64_t)D_mpo_};
        int64_t extentA[] = {(int64_t)chi_L, (int64_t)d_, (int64_t)chi_R};
        int64_t extent_Lnew[] = {(int64_t)chi_R, (int64_t)D_mpo_, (int64_t)chi_R};

        int32_t modes_temp2[] = {1, 4, 3, 5};
        int32_t modesA[] = {1, 4, 6};
        int32_t modes_Lnew[] = {3, 5, 6};

        hiptensor_contract(d_temp2, 4, extent_temp2, modes_temp2,
                           d_A, 3, extentA, modesA,
                           d_L_out, 3, extent_Lnew, modes_Lnew,
                           1.0, 0.0);
    }

    HIP_CHECK(hipFree(d_temp2));
}

void StreamSegment::rebuild_left_boundary_env() {
    if (!has_left_boundary_) return;

    // Rebuild R_env at the left boundary (position 0)
    // by contracting MPS and MPO from right to left
    //
    // R_new[a, w, a'] = sum_{b, b', wp, s, s'}
    //     R[b, wp, b'] * A[a, s, b] * W[w, s, s', wp] * A[a', s', b']
    //
    // Similar to rebuild_right_boundary_env but contracting from right to left

    int site_idx = 0;  // Leftmost site
    int chi_L = mps_chi_left_[site_idx];
    int chi_R = mps_chi_right_[site_idx];

    double* d_A = d_mps_tensors_[site_idx];
    double* d_W = d_mpo_tensors_[site_idx];
    double* d_R_in = d_R_envs_[site_idx + 1];  // R_env to the right of this site
    double* d_R_out = d_R_envs_[0];            // R_env at left boundary

    // Allocate workspace
    size_t temp_size = D_mpo_ * chi_L * d_ * chi_R;
    double* d_temp1;
    double* d_temp2;
    HIP_CHECK(hipMalloc(&d_temp1, temp_size * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_temp2, temp_size * sizeof(double)));

    // Full hipTensor implementation (mirror of rebuild_right_boundary_env)
    // R_new[a, w, a'] = sum_{b, b', wp, s, s'}
    //     R[b, wp, b'] * A[a, s, b] * W[w, s, s', wp] * A[a', s', b']
    //
    // This is analogous but contracts from right to left

    // Step 1: temp1[wp, b', s, a] = sum_b R[b, wp, b'] * A[a, s, b]
    // R col-major (b, wp, b'): modes {0, 1, 2}
    // A col-major (a, s, b):   modes {3, 4, 0}
    // temp1 col-major (wp, b', s, a): modes {1, 2, 4, 3}
    // Contract over mode 0 (b)
    {
        int64_t extentR[] = {(int64_t)chi_R, (int64_t)D_mpo_, (int64_t)chi_R};
        int64_t extentA[] = {(int64_t)chi_L, (int64_t)d_, (int64_t)chi_R};
        int64_t extent_temp1[] = {(int64_t)D_mpo_, (int64_t)chi_R, (int64_t)d_, (int64_t)chi_L};

        int32_t modesR[] = {0, 1, 2};
        int32_t modesA[] = {3, 4, 0};
        int32_t modes_temp1[] = {1, 2, 4, 3};

        hiptensor_contract(d_R_in, 3, extentR, modesR,
                           d_A, 3, extentA, modesA,
                           d_temp1, 4, extent_temp1, modes_temp1,
                           1.0, 0.0);
    }

    // Step 2: temp2[b', s', a, w] = sum_{wp,s} temp1[wp, b', s, a] * W[w, s, s', wp]
    // temp1 modes: {1(wp), 2(b'), 4(s), 3(a)}
    // W modes: {5(w), 4(s), 6(s'), 1(wp)}
    // temp2 modes: {2(b'), 6(s'), 3(a), 5(w)}
    // Contract over modes 1(wp) and 4(s)
    {
        int64_t extent_temp1[] = {(int64_t)D_mpo_, (int64_t)chi_R, (int64_t)d_, (int64_t)chi_L};
        int64_t extentW[] = {(int64_t)D_mpo_, (int64_t)d_, (int64_t)d_, (int64_t)D_mpo_};
        int64_t extent_temp2[] = {(int64_t)chi_R, (int64_t)d_, (int64_t)chi_L, (int64_t)D_mpo_};

        int32_t modes_temp1[] = {1, 2, 4, 3};
        int32_t modesW[] = {5, 4, 6, 1};
        int32_t modes_temp2[] = {2, 6, 3, 5};

        hiptensor_contract(d_temp1, 4, extent_temp1, modes_temp1,
                           d_W, 4, extentW, modesW,
                           d_temp2, 4, extent_temp2, modes_temp2,
                           1.0, 0.0);
    }

    HIP_CHECK(hipFree(d_temp1));

    // Step 3: R_new[a, w, a'] = sum_{b',s'} temp2[b', s', a, w] * A[a', s', b']
    // temp2 modes: {2(b'), 6(s'), 3(a), 5(w)}
    // A modes: {7(a'), 6(s'), 2(b')}
    // R_new modes: {3(a), 5(w), 7(a')}
    // Contract over modes 2(b') and 6(s')
    {
        int64_t extent_temp2[] = {(int64_t)chi_R, (int64_t)d_, (int64_t)chi_L, (int64_t)D_mpo_};
        int64_t extentA[] = {(int64_t)chi_L, (int64_t)d_, (int64_t)chi_R};
        int64_t extent_Rnew[] = {(int64_t)chi_L, (int64_t)D_mpo_, (int64_t)chi_L};

        int32_t modes_temp2[] = {2, 6, 3, 5};
        int32_t modesA[] = {7, 6, 2};
        int32_t modes_Rnew[] = {3, 5, 7};

        hiptensor_contract(d_temp2, 4, extent_temp2, modes_temp2,
                           d_A, 3, extentA, modesA,
                           d_R_out, 3, extent_Rnew, modes_Rnew,
                           1.0, 0.0);
    }

    HIP_CHECK(hipFree(d_temp2));
}

//==============================================================================
// V Matrix Updates
//==============================================================================

void StreamSegment::recompute_boundary_v(bool left_boundary) {
    // Recompute V = 1/S at boundary after canonization
    // The boundary tensors contract to form the bond matrix Lambda
    //
    // NOTE: This is called after sweeps, which may have changed bond dimensions.
    // We need to use the current MPS bond dimensions, not the cached boundary dimensions.

    if (left_boundary && has_left_boundary_) {
        // Get current bond dimension at left edge
        int site_idx = 0;
        int chi_bond_current = mps_chi_left_[site_idx];  // Left bond of leftmost site

        // Check if boundary needs reallocation
        if (left_boundary_.chi_bond != chi_bond_current) {
            // Bond dimension changed - will be reallocated by extract_boundary_tensors()
            // Skip V update for now
            return;
        }

        // TODO: Implement proper V computation from SVD
        // For now: initialize V to ones (will be updated during merge)
        std::vector<double> h_V(chi_bond_current, 1.0);
        HIP_CHECK(hipMemcpy(left_boundary_.d_V, h_V.data(),
                           chi_bond_current * sizeof(double), hipMemcpyHostToDevice));

    } else if (!left_boundary && has_right_boundary_) {
        // Get current bond dimension at right edge
        int site_idx = num_sites_ - 1;
        int chi_bond_current = mps_chi_right_[site_idx];  // Right bond of rightmost site

        // Check if boundary needs reallocation
        if (right_boundary_.chi_bond != chi_bond_current) {
            // Bond dimension changed - will be reallocated by extract_boundary_tensors()
            // Skip V update for now
            return;
        }

        // TODO: Implement proper V computation from SVD
        // For now: initialize V to ones (will be updated during merge)
        std::vector<double> h_V(chi_bond_current, 1.0);
        HIP_CHECK(hipMemcpy(right_boundary_.d_V, h_V.data(),
                           chi_bond_current * sizeof(double), hipMemcpyHostToDevice));
    }
}

//==============================================================================
// Boundary Tensor Extraction
//==============================================================================

void StreamSegment::extract_boundary_tensors() {
    // Extract MPS tensors and environments at segment boundaries
    // and copy them into BoundaryData structures for merge operations
    //
    // Left boundary: site 0 is the rightmost tensor of the left segment's boundary
    // Right boundary: site num_sites-1 is the leftmost tensor of the right segment's boundary

    if (has_left_boundary_) {
        // Left boundary data (for merging with left neighbor)
        int site_idx = 0;  // Leftmost site in this segment
        int chi_left_site = mps_chi_left_[site_idx];   // Left bond (= chi_bond at boundary)
        int chi_right_site = mps_chi_right_[site_idx]; // Right bond

        // For BoundaryData at left edge:
        // - psi_right is this segment's leftmost site: (chi_left_site, d, chi_right_site)
        // - R_env should be to the RIGHT of psi_right, with dimension chi_right_site
        // - chi_R in BoundaryData = chi_right_site
        // - chi_bond = chi_left_site

        // Check if BoundaryData needs reallocation
        if (left_boundary_.chi_bond != chi_left_site ||
            left_boundary_.chi_L != chi_left_site ||
            left_boundary_.chi_R != chi_right_site) {
            // Reallocate with correct dimensions
            left_boundary_.free();
            left_boundary_.allocate(chi_left_site, chi_right_site, chi_left_site, d_, D_mpo_);
        }

        // Copy MPS tensor (this site is psi_right in the boundary merge)
        size_t mps_size = chi_left_site * d_ * chi_right_site * sizeof(double);
        HIP_CHECK(hipMemcpy(left_boundary_.d_psi_right, d_mps_tensors_[site_idx],
                           mps_size, hipMemcpyDeviceToDevice));

        // For psi_left, we would need the rightmost tensor from the left neighbor
        // This will be set during the merge coordination by the coordinator

        // Copy R_env: environment to the RIGHT of site_idx
        // This is R_env at position site_idx+1 (not site_idx!)
        size_t env_size = D_mpo_ * chi_right_site * chi_right_site * sizeof(double);
        HIP_CHECK(hipMemcpy(left_boundary_.d_R_env, d_R_envs_[site_idx + 1],
                           env_size, hipMemcpyDeviceToDevice));

        // Copy MPO tensor
        size_t mpo_size = D_mpo_ * d_ * d_ * D_mpo_ * sizeof(double);
        HIP_CHECK(hipMemcpy(left_boundary_.d_W_right, d_mpo_tensors_[site_idx],
                           mpo_size, hipMemcpyDeviceToDevice));
    }

    if (has_right_boundary_) {
        // Right boundary data (for merging with right neighbor)
        int site_idx = num_sites_ - 1;  // Rightmost site in this segment
        int chi_left_site = mps_chi_left_[site_idx];   // Left bond of rightmost site
        int chi_right_site = mps_chi_right_[site_idx]; // Right bond (= chi_bond at boundary)

        // For BoundaryData at right edge:
        // - psi_left is this segment's rightmost site: (chi_left_site, d, chi_right_site)
        // - L_env should be to the LEFT of psi_left, with dimension chi_left_site
        // - chi_L in BoundaryData = chi_left_site
        // - chi_bond = chi_right_site

        // Check if BoundaryData needs reallocation
        if (right_boundary_.chi_bond != chi_right_site ||
            right_boundary_.chi_L != chi_left_site ||
            right_boundary_.chi_R != chi_right_site) {
            // Reallocate with correct dimensions
            right_boundary_.free();
            right_boundary_.allocate(chi_left_site, chi_right_site, chi_right_site, d_, D_mpo_);
        }

        // Copy MPS tensor (this site is psi_left in the boundary merge)
        size_t mps_size = chi_left_site * d_ * chi_right_site * sizeof(double);
        HIP_CHECK(hipMemcpy(right_boundary_.d_psi_left, d_mps_tensors_[site_idx],
                           mps_size, hipMemcpyDeviceToDevice));

        // For psi_right, we would need the leftmost tensor from the right neighbor
        // This will be set during the merge coordination by the coordinator

        // Copy L_env: environment to the LEFT of site_idx
        // This is L_env at position site_idx (not num_sites_!)
        size_t env_size = D_mpo_ * chi_left_site * chi_left_site * sizeof(double);
        HIP_CHECK(hipMemcpy(right_boundary_.d_L_env, d_L_envs_[site_idx],
                           env_size, hipMemcpyDeviceToDevice));

        // Copy MPO tensor
        size_t mpo_size = D_mpo_ * d_ * d_ * D_mpo_ * sizeof(double);
        HIP_CHECK(hipMemcpy(right_boundary_.d_W_left, d_mpo_tensors_[site_idx],
                           mpo_size, hipMemcpyDeviceToDevice));
    }
}

//==============================================================================
// Getters
//==============================================================================

double* StreamSegment::get_mps_tensor(int site) {
    int local_idx = site - start_site_;
    if (local_idx < 0 || local_idx >= num_sites_) return nullptr;
    return d_mps_tensors_[local_idx];
}

double* StreamSegment::get_L_env(int site) {
    int local_idx = site - start_site_;
    if (local_idx < 0 || local_idx > num_sites_) return nullptr;
    return d_L_envs_[local_idx];
}

double* StreamSegment::get_R_env(int site) {
    int local_idx = site - start_site_;
    if (local_idx < 0 || local_idx > num_sites_) return nullptr;
    return d_R_envs_[local_idx];
}

double* StreamSegment::get_mpo_tensor(int site) {
    int local_idx = site - start_site_;
    if (local_idx < 0 || local_idx >= num_sites_) return nullptr;
    return d_mpo_tensors_[local_idx];
}

//==============================================================================
// hipTensor Helper
//==============================================================================

void StreamSegment::hiptensor_contract(
    const double* A, int nmodeA, const int64_t* extentA, const int32_t* modesA,
    const double* B, int nmodeB, const int64_t* extentB, const int32_t* modesB,
    double* C, int nmodeC, const int64_t* extentC, const int32_t* modesC,
    double alpha, double beta)
{
    // Create tensor descriptors
    hiptensorTensorDescriptor_t descA, descB, descC;
    printf("DEBUG: Creating descA\n");
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(hiptensor_h_, &descA, nmodeA, extentA,
                    nullptr, HIPTENSOR_R_64F, 8));
    printf("DEBUG: Creating descB\n");
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(hiptensor_h_, &descB, nmodeB, extentB,
                    nullptr, HIPTENSOR_R_64F, 8));
    printf("DEBUG: Creating descC\n");
    HIPTENSOR_CHECK(hiptensorCreateTensorDescriptor(hiptensor_h_, &descC, nmodeC, extentC,
                    nullptr, HIPTENSOR_R_64F, 8));

    // Create contraction operation descriptor
    hiptensorOperationDescriptor_t opDesc;
    printf("DEBUG: Creating opDesc\n");
    HIPTENSOR_CHECK(hiptensorCreateContraction(hiptensor_h_, &opDesc,
        descA, modesA, HIPTENSOR_OP_IDENTITY,
        descB, modesB, HIPTENSOR_OP_IDENTITY,
        descC, modesC, HIPTENSOR_OP_IDENTITY,
        descC, modesC,
        HIPTENSOR_COMPUTE_DESC_64F));

    // Create plan preference
    hiptensorPlanPreference_t pref;
    printf("DEBUG: Creating plan preference\n");
    printf("DEBUG: Creating plan\n");
    HIPTENSOR_CHECK(hiptensorCreatePlanPreference(hiptensor_h_, &pref,
        HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE));

    // Estimate workspace size
    uint64_t workspaceSize = 0;
    printf("DEBUG: Estimating workspace\n");
    HIPTENSOR_CHECK(hiptensorEstimateWorkspaceSize(hiptensor_h_, opDesc, pref,
        HIPTENSOR_WORKSPACE_DEFAULT, &workspaceSize));

    // Allocate workspace
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        HIP_CHECK(hipMalloc(&workspace, workspaceSize));
    }

    // Create plan
    hiptensorPlan_t plan;
    printf("DEBUG: Creating plan\n");
    HIPTENSOR_CHECK(hiptensorCreatePlan(hiptensor_h_, &plan, opDesc, pref, workspaceSize));

    // Execute contraction
    printf("DEBUG hiptensorContract: nmodeA=%d, nmodeB=%d, nmodeC=%d\n", nmodeA, nmodeB, nmodeC);
    printf("  extentA="); for(int i=0; i<nmodeA; i++) printf("%ld ", extentA[i]); printf("\n");
    printf("  extentB="); for(int i=0; i<nmodeB; i++) printf("%ld ", extentB[i]); printf("\n");
    printf("  extentC="); for(int i=0; i<nmodeC; i++) printf("%ld ", extentC[i]); printf("\n");
    printf("  modesA="); for(int i=0; i<nmodeA; i++) printf("%d ", modesA[i]); printf("\n");
    printf("  modesB="); for(int i=0; i<nmodeB; i++) printf("%d ", modesB[i]); printf("\n");
    printf("  modesC="); for(int i=0; i<nmodeC; i++) printf("%d ", modesC[i]); printf("\n");
    printf("  alpha=%.6f, beta=%.6f\n", alpha, beta);
    HIPTENSOR_CHECK(hiptensorContract(hiptensor_h_, plan,
        &alpha, A, B, &beta, C, C,
        workspace, workspaceSize, stream_));

    // Cleanup
    if (workspace) HIP_CHECK(hipFree(workspace));
    HIPTENSOR_CHECK(hiptensorDestroyPlan(plan));
    HIPTENSOR_CHECK(hiptensorDestroyPlanPreference(pref));
    HIPTENSOR_CHECK(hiptensorDestroyOperationDescriptor(opDesc));
    HIPTENSOR_CHECK(hiptensorDestroyTensorDescriptor(descA));
    HIPTENSOR_CHECK(hiptensorDestroyTensorDescriptor(descB));
    HIPTENSOR_CHECK(hiptensorDestroyTensorDescriptor(descC));
}

bool StreamSegment::load_mps_from_binary(const char* filename, const int* bond_dims) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        printf("[Stream %d] ERROR: Cannot open MPS file %s\\n", id_, filename);
        return false;
    }

    std::vector<std::vector<std::complex<double>>> all_tensors(chain_length_);
    for (int i = 0; i < chain_length_; i++) {
        int chi_L = bond_dims[i];
        int chi_R = bond_dims[i + 1];
        size_t size = chi_L * d_ * chi_R;
        all_tensors[i].resize(size);
        file.read(reinterpret_cast<char*>(all_tensors[i].data()),
                  size * sizeof(std::complex<double>));
        if (!file) {
            printf("[Stream %d] ERROR: Failed to read MPS tensor %d\\n", id_, i);
            return false;
        }
    }
    file.close();

    printf("[Stream %d] Loading MPS tensors from %s for sites %d-%d\\n",
           id_, filename, start_site_, end_site_);

    for (int local_idx = 0; local_idx < num_sites_; local_idx++) {
        int global_site = start_site_ + local_idx;
        int chi_L = bond_dims[global_site];
        int chi_R = bond_dims[global_site + 1];
        size_t size = chi_L * d_ * chi_R;

        const auto& tensor_data = all_tensors[global_site];
        std::vector<double> real_data(size);
        for (size_t j = 0; j < size; j++) {
            real_data[j] = tensor_data[j].real();
        }

        double* d_tensor = d_mps_tensors_[local_idx];
        hipMemcpyAsync(d_tensor, real_data.data(), size * sizeof(double),
                      hipMemcpyHostToDevice, stream_);

    }

    hipStreamSynchronize(stream_);
    printf("[Stream %d] Successfully loaded %d MPS tensors\\n", id_, num_sites_);
    // Initialize all environments to identity
    // BUG FIX: Do not initialize to identity for loaded MPS
    //     initialize_environments();
    return true;
}
