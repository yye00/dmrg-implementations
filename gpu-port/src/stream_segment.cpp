#include "stream_segment.h"
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocsolver/rocsolver.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <cmath>

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
                              int chi_max, int d, int D_mpo, hipStream_t stream)
    : id_(segment_id), start_site_(start_site), end_site_(end_site),
      num_sites_(end_site - start_site + 1),
      chi_max_(chi_max), d_(d), D_mpo_(D_mpo), stream_(stream),
      d_mps_tensors_(nullptr), mps_chi_left_(nullptr), mps_chi_right_(nullptr),
      d_L_envs_(nullptr), d_R_envs_(nullptr), d_mpo_tensors_(nullptr),
      has_left_boundary_(segment_id > 0),
      has_right_boundary_(true),  // Will be set correctly by coordinator
      heff_(nullptr), svd_(nullptr),
      d_workspace_(nullptr), workspace_size_(0)
{
    if (num_sites_ < 1) {
        throw std::runtime_error("StreamSegment must have at least 1 site");
    }

    // Allocate memory for all tensors
    allocate_memory();

    // Initialize MPS and environments
    initialize_mps();
    initialize_environments();

    // Create OptimizedHeff for local two-site operations
    // Note: Will be created lazily when needed, since dimensions may vary

    // Create AccurateSVD for exact SVD operations
    svd_ = new AccurateSVD_GPU(1e-4, 0);  // epsilon=1e-4, max_depth=0 (no recursion)
}

StreamSegment::~StreamSegment() {
    free_memory();

    if (heff_) delete heff_;
    if (svd_) delete svd_;
}

void StreamSegment::allocate_memory() {
    // Allocate arrays for pointers
    d_mps_tensors_ = new double*[num_sites_];
    mps_chi_left_ = new int[num_sites_];
    mps_chi_right_ = new int[num_sites_];

    d_L_envs_ = new double*[num_sites_ + 1];
    d_R_envs_ = new double*[num_sites_ + 1];
    d_mpo_tensors_ = new double*[num_sites_];

    // Initialize chi values (will grow from boundaries)
    // For now, use simple pattern: chi = min(d^site, chi_max)
    for (int i = 0; i < num_sites_; i++) {
        int site_from_left = start_site_ + i;
        int site_from_right = end_site_ - i;

        // Left chi: grows exponentially from left, capped at chi_max
        int chi_left = (i == 0) ? 1 : std::min(chi_max_, (int)std::pow(d_, i));

        // Right chi: similar from right
        int chi_right = (i == num_sites_ - 1) ? 1 : std::min(chi_max_, (int)std::pow(d_, num_sites_ - 1 - i));

        mps_chi_left_[i] = chi_left;
        mps_chi_right_[i] = chi_right;

        // Allocate MPS tensor: (chi_left, d, chi_right) in column-major
        HIP_CHECK(hipMalloc(&d_mps_tensors_[i], chi_left * d_ * chi_right * sizeof(double)));
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
    for (int i = 0; i < num_sites_; i++) {
        HIP_CHECK(hipMalloc(&d_mpo_tensors_[i], D_mpo_ * d_ * d_ * D_mpo_ * sizeof(double)));
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

    for (int i = 0; i < num_sites_ - 1; i++) {
        int chi_left = mps_chi_left_[i];
        int chi_right = mps_chi_right_[i];

        // TODO: Implement QR decomposition on GPU
        // M = QR where M is (chi_left * d, chi_right)
        // Update this site with Q, absorb R into next site

        // For now: placeholder
    }
}

void StreamSegment::sweep_right_to_left() {
    // LQ sweep to move orthogonality center right-to-left
    // Each site is right-canonized (LQ decomposition), L absorbed to left

    for (int i = num_sites_ - 1; i > 0; i--) {
        int chi_left = mps_chi_left_[i];
        int chi_right = mps_chi_right_[i];

        // TODO: Implement LQ decomposition on GPU
        // M = LQ where M is (chi_left, d * chi_right)
        // Update this site with Q, absorb L into previous site

        // For now: placeholder
    }
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

    // TODO: Implement environment contraction
    // L_env[i+1] = contract(L_env[i], MPS[i], MPO[i], MPS[i]*)

    // For now: placeholder
}

void StreamSegment::rebuild_left_boundary_env() {
    if (!has_left_boundary_) return;

    // Rebuild R_env at the left boundary (position 0)
    // by contracting MPS and MPO from right to left

    // TODO: Implement environment contraction
    // R_env[i] = contract(R_env[i+1], MPS[i], MPO[i], MPS[i]*)

    // For now: placeholder
}

//==============================================================================
// V Matrix Updates
//==============================================================================

void StreamSegment::recompute_boundary_v(bool left_boundary) {
    // Recompute V = 1/S at boundary after canonization
    // The boundary tensors contract to form the bond matrix Lambda

    if (left_boundary && has_left_boundary_) {
        // Contract psi_left and psi_right to get bond matrix
        // Compute SVD: M = U S V^T
        // Set V = 1 / clip(S, 1e-12, inf)

        // TODO: Implement V computation
        // For now: initialize V to ones (will be updated during merge)
        int chi_bond = left_boundary_.chi_bond;
        std::vector<double> h_V(chi_bond, 1.0);
        HIP_CHECK(hipMemcpy(left_boundary_.d_V, h_V.data(),
                           chi_bond * sizeof(double), hipMemcpyHostToDevice));

    } else if (!left_boundary && has_right_boundary_) {
        // Similar for right boundary
        int chi_bond = right_boundary_.chi_bond;
        std::vector<double> h_V(chi_bond, 1.0);
        HIP_CHECK(hipMemcpy(right_boundary_.d_V, h_V.data(),
                           chi_bond * sizeof(double), hipMemcpyHostToDevice));
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
