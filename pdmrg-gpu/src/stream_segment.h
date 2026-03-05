#pragma once

#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>
#include "heff_optimized_gpu.h"
#include "accurate_svd_gpu.h"

/**
 * BoundaryData: Manages data at segment boundaries for merge operations
 *
 * At each boundary between segments, we store:
 * - MPS tensors (psi_left, psi_right)
 * - V = Lambda^-1 bridge matrix
 * - Environments (L_env, R_env) for two-site optimization
 * - MPO tensors at boundary sites
 */
struct BoundaryData {
    // MPS tensors at boundary
    double* d_psi_left;   // (chi_L, d, chi_bond) - right edge of left segment
    double* d_psi_right;  // (chi_bond, d, chi_R) - left edge of right segment

    // V = Lambda^-1 bridge matrix (critical for exact boundary reconciliation)
    double* d_V;          // (chi_bond,) - current V = 1/S values

    // Environments for two-site optimization at boundary
    double* d_L_env;      // (chi_L, D_mpo, chi_L) or (D_mpo, chi_L, chi_L) depending on convention
    double* d_R_env;      // (chi_R, D_mpo, chi_R) or (D_mpo, chi_R, chi_R)

    // MPO tensors at boundary sites
    double* d_W_left;     // (D_mpo, d, d, D_mpo)
    double* d_W_right;    // (D_mpo, d, d, D_mpo)

    // Dimensions
    int chi_L;      // Left bond dimension
    int chi_R;      // Right bond dimension
    int chi_bond;   // Bond dimension at boundary
    int d;          // Physical dimension
    int D_mpo;      // MPO bond dimension

    // Memory management
    bool is_allocated;

    BoundaryData();
    ~BoundaryData();

    void allocate(int chi_L, int chi_R, int chi_bond, int d, int D_mpo);
    void free();
};

/**
 * StreamSegment: Manages a contiguous segment of the MPS chain on a single HIP stream
 *
 * Maps to one MPI rank in CPU PDMRG. Each segment:
 * - Owns a contiguous range of sites [start_site, end_site]
 * - Performs local DMRG sweeps independently
 * - Maintains boundary data for merging with neighbors
 * - Uses OptimizedHeff from Phase 1 for local H_eff operations
 *
 * Example for 8-site chain with 4 streams:
 *   Stream 0: sites [0,1]   - has right_boundary
 *   Stream 1: sites [2,3]   - has left_boundary, right_boundary
 *   Stream 2: sites [4,5]   - has left_boundary, right_boundary
 *   Stream 3: sites [6,7]   - has left_boundary
 */
class StreamSegment {
public:
    /**
     * Constructor
     *
     * @param segment_id    Unique ID for this segment (0 to n_streams-1)
     * @param start_site    First site in this segment (global indexing)
     * @param end_site      Last site in this segment (inclusive)
     * @param chi_max       Maximum bond dimension
     * @param d             Physical dimension
     * @param D_mpo         MPO bond dimension
     * @param stream        HIP stream for asynchronous operations
     */
    StreamSegment(int segment_id, int start_site, int end_site,
                  int chi_max, int d, int D_mpo, hipStream_t stream);

    ~StreamSegment();

    // ============================================================================
    // Local DMRG Sweeps (run independently on this stream)
    // ============================================================================

    /**
     * Sweep left-to-right within this segment
     * Uses QR decomposition to move orthogonality center
     * Does NOT optimize (skip_opt=True equivalent from CPU)
     */
    void sweep_left_to_right();

    /**
     * Sweep right-to-left within this segment
     */
    void sweep_right_to_left();

    // ============================================================================
    // Boundary Data Access
    // ============================================================================

    /**
     * Get left boundary data (for merging with left neighbor)
     * Returns nullptr if this is the leftmost segment
     */
    BoundaryData* get_left_boundary();

    /**
     * Get right boundary data (for merging with right neighbor)
     * Returns nullptr if this is the rightmost segment
     */
    BoundaryData* get_right_boundary();

    // ============================================================================
    // Environment Rebuilding (before merge operations)
    // ============================================================================

    /**
     * Rebuild L_env at right boundary for merge with right neighbor
     *
     * After local sweeps, we need to rebuild the left environment at the
     * right boundary to ensure correct two-site optimization during merge.
     */
    void rebuild_right_boundary_env();

    /**
     * Rebuild R_env at left boundary for merge with left neighbor
     */
    void rebuild_left_boundary_env();

    // ============================================================================
    // V Matrix Updates (after canonization)
    // ============================================================================

    /**
     * Recompute V at boundary after QR canonization sweeps
     *
     * When the boundary tensors contract, they should equal the current bond
     * matrix. We compute V = 1/S from this to maintain exact reconciliation.
     *
     * @param left_boundary  If true, update left boundary V; else right boundary V
     */
    void recompute_boundary_v(bool left_boundary);

    /**
     * Extract boundary MPS tensors into BoundaryData struct
     *
     * Copies edge MPS tensors, environments, and MPO tensors from the segment
     * into the BoundaryData structures. Must be called before merge operations
     * to ensure boundary data is up to date.
     */
    void extract_boundary_tensors();

    // ============================================================================
    // Getters
    // ============================================================================

    int get_id() const { return id_; }
    int get_start_site() const { return start_site_; }
    int get_end_site() const { return end_site_; }
    int get_num_sites() const { return end_site_ - start_site_ + 1; }
    hipStream_t get_stream() const { return stream_; }

    /**
     * Get pointer to MPS tensor at site i (global indexing)
     * Returns nullptr if site is not in this segment
     */
    double* get_mps_tensor(int site);

    /**
     * Get pointer to L_env at site i
     */
    double* get_L_env(int site);

    /**
     * Get pointer to R_env at site i
     */
    double* get_R_env(int site);

    /**
     * Get pointer to MPO tensor at site i
     */
    double* get_mpo_tensor(int site);

private:
    // Segment identification
    int id_;
    int start_site_;
    int end_site_;
    int num_sites_;

    // Dimensions
    int chi_max_;
    int d_;
    int D_mpo_;

    // HIP stream for async operations
    hipStream_t stream_;

    // MPS tensors: array of num_sites tensors
    // Each tensor has shape (chi_left, d, chi_right) stored in column-major
    // Actual chi values vary by position; chi_max is the maximum
    double** d_mps_tensors_;     // Array of device pointers
    int* mps_chi_left_;          // chi_left for each site
    int* mps_chi_right_;         // chi_right for each site

    // Environments
    double** d_L_envs_;          // Left environments [0..num_sites]
    double** d_R_envs_;          // Right environments [0..num_sites]

    // MPO tensors (constant during sweeps)
    double** d_mpo_tensors_;     // Array of MPO tensors

    // Boundary data
    BoundaryData left_boundary_;
    BoundaryData right_boundary_;
    bool has_left_boundary_;
    bool has_right_boundary_;

    // OptimizedHeff for local two-site operations (from Phase 1)
    OptimizedHeff* heff_;

    // AccurateSVD for exact SVD operations
    AccurateSVD_GPU* svd_;

    // rocBLAS handle for QR/LQ operations
    rocblas_handle rocblas_h_;

    // hipTensor handle for environment contractions
    hiptensorHandle_t hiptensor_h_;

    // Workspace for intermediate operations
    double* d_workspace_;
    size_t workspace_size_;
    double* d_tau_;           // Tau vector for QR/LQ
    size_t tau_size_;

    // ============================================================================
    // Helper Methods
    // ============================================================================

    /**
     * Initialize MPS tensors with random values (or from initial state)
     */
    void initialize_mps();

    /**
     * Initialize environments for local sweeps
     */
    void initialize_environments();

    /**
     * Allocate memory for all tensors
     */
    void allocate_memory();

    /**
     * Free all allocated memory
     */
    void free_memory();

    /**
     * Helper: Perform hipTensor contraction
     * C = alpha * A * B + beta * C
     */
    void hiptensor_contract(
        const double* A, int nmodeA, const int64_t* extentA, const int32_t* modesA,
        const double* B, int nmodeB, const int64_t* extentB, const int32_t* modesB,
        double* C, int nmodeC, const int64_t* extentC, const int32_t* modesC,
        double alpha, double beta);
};
